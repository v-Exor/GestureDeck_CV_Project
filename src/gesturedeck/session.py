"""Background camera/inference session shared with the desktop interface.

No Tk calls are made here. Frames use a latest-only mailbox, so a slow UI cannot
queue stale video. The existing processor remains the sole gesture decision maker.
"""

from contextlib import ExitStack
from collections import deque
from dataclasses import dataclass
from enum import Enum
import logging
from math import ceil
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter

from gesturedeck.config import Config
from gesturedeck.controller import SlideController
from gesturedeck.feedback import AudioFeedback
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor, Observation

LOGGER = logging.getLogger(__name__)


class Mode(str, Enum):
    DRY_RUN = "Recognition / dry run"
    CAMERA = "Camera preview only"
    DEMO = "Interface demo / no camera"
    LIVE = "PowerPoint / live keys"


@dataclass
class Snapshot:
    frame: object = None
    observation: Observation = Observation()
    active: bool = False
    fps: float = 0.0
    inference_ms: float | None = None
    last_action: str = "No commands yet"
    status: str = "Ready to start"
    phase: str = "Running"


class Session:
    def __init__(self, config: Config, mode: Mode = Mode.DRY_RUN) -> None:
        self.config = config
        self.mode = Mode(mode)
        self.frames = Queue(maxsize=1)
        self.events = Queue(maxsize=100)
        self.stopped = Event()
        self.pause_requested = Event()
        self._gate = Lock()
        self._demo_lock = Lock()
        self._demo_observation = Observation()
        self._demo_sequence = deque()
        self.thread: Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("Create a new session to restart")
        self.thread = Thread(target=self._run, name="GestureDeck-camera", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        # The same lock guards delivery, so no new key can begin after this returns.
        with self._gate:
            self.stopped.set()

    def pause(self) -> None:
        with self._gate:
            self.pause_requested.set()

    def set_demo(self, label: str | None, low_confidence: bool = False) -> None:
        if self.mode != Mode.DEMO:
            raise ValueError("Simulated input is available only in demo mode")
        with self._demo_lock:
            self._demo_sequence.clear()
            self._demo_observation = Observation(label, 0.25 if low_confidence else 0.99,
                                                 label is not None, (0.5, 0.5) if label else None)

    def set_demo_swipe(self, direction: str) -> None:
        if self.mode != Mode.DEMO:
            raise ValueError("Simulated input is available only in demo mode")
        if direction not in ("next", "previous"):
            raise ValueError("Swipe direction must be next or previous")
        xs = [0.80 - i * 0.05 for i in range(11)]
        if direction == "previous":
            xs.reverse()
        with self._demo_lock:
            self._demo_observation = Observation()
            self._demo_sequence = deque(
                [Observation("Open_Palm", 0.99, True, (x, 0.5)) for x in xs]
                + [Observation()] * 8)

    def _publish(self, snapshot: Snapshot) -> None:
        self._put_latest(self.frames, snapshot)

    @staticmethod
    def _put_latest(mailbox: Queue, item) -> None:
        try:
            mailbox.put_nowait(item)
        except Full:
            try:
                mailbox.get_nowait()
            except Empty:
                pass
            mailbox.put_nowait(item)

    def _log(self, text: str) -> None:
        LOGGER.info(text)
        self._put_latest(self.events, text)

    def _dispatch(self, controller: SlideController, action: Action):
        with self._gate:
            if not self.stopped.is_set() and not self.pause_requested.is_set():
                delivered = controller.execute(action)
                # The controller already logs to the terminal; mirror once in UI.
                self._put_latest(self.events, controller.last_action)
                return delivered

    def _run(self) -> None:
        try:
            self._capture()
        except Exception as exc:
            LOGGER.exception("Desktop session stopped")
            self._log(f"Error: {exc}")
            self._publish(Snapshot(status=str(exc), phase="Error"))
        finally:
            self.stopped.set()
            self._log("Session closed. Camera and model resources released.")

    def _capture(self) -> None:
        import cv2
        import numpy as np
        from gesturedeck.preview import draw_landmarks

        processor = GestureProcessor(self.config)
        with ExitStack() as resources:
            feedback = resources.enter_context(AudioFeedback(self.config.audio_feedback and self.mode == Mode.LIVE))
            recognizer = camera = None
            if self.mode != Mode.DEMO:
                from gesturedeck.camera import Webcam
                if self.mode != Mode.CAMERA:
                    from gesturedeck.recognizer import GestureRecognizer
                    recognizer = resources.enter_context(GestureRecognizer(self.config))
                camera = resources.enter_context(Webcam(self.config))
            if self.stopped.is_set():
                return
            controller = SlideController(dry_run=self.mode != Mode.LIVE)
            recorder = None
            if self.config.record_directory:
                if self.mode == Mode.DEMO:
                    raise ValueError("Recording requires real webcam input; demo recording is disabled")
                from gesturedeck.recording import SessionRecorder
                recorder = resources.enter_context(SessionRecorder(self.config, self.mode))
                self._log(f"RECORDING webcam + landmarks locally: {recorder.path}")
            ready_at = perf_counter() + (self.config.countdown_seconds if self.mode == Mode.LIVE else 0)
            self._log(f"Started: {self.mode.value}. Control activates after countdown; fist pauses/resumes.")
            previous_time = None
            fps = 0.0
            last_active = None
            last_countdown = None
            last_diagnostic = float("-inf")
            reported_camera_size = None
            while not self.stopped.is_set():
                now = perf_counter()
                if previous_time is not None:
                    measured = 1 / max(now - previous_time, 1e-6)
                    fps = measured if fps == 0 else 0.9 * fps + 0.1 * measured
                previous_time = now
                frame = (np.full((360, 640, 3), (242, 246, 251), dtype=np.uint8)
                         if camera is None else camera.read())
                captured_at = perf_counter()
                if camera is not None:
                    output_size = (frame.shape[1], frame.shape[0])
                    source_size = getattr(camera, "source_size", None) or output_size
                    sizes = (source_size, output_size)
                    if sizes != reported_camera_size:
                        self._log(f"Camera input {source_size[0]}x{source_size[1]} -> "
                                  f"processing {output_size[0]}x{output_size[1]}. "
                                  "Center crop and resize only when needed; no stretching.")
                        reported_camera_size = sizes
                with self._gate:
                    if self.pause_requested.is_set():
                        processor.pause()
                        self.pause_requested.clear()
                        self._log("Manually PAUSED. Lower hand, then hold a fist to resume.")
                observation, hands = Observation(), []
                inference_ms = None
                remaining = max(0, ceil(ready_at - perf_counter()))
                phase = f"Focus slideshow: {remaining}s" if remaining else "Running"
                status = processor.status
                action = None
                decision_at = None
                delivered = None
                inference_ran = False
                if remaining:
                    status = "Focus the PowerPoint slideshow. No keys during countdown."
                    if remaining != last_countdown:
                        self._log(f"Live control countdown: {remaining}s. Focus your slideshow.")
                        last_countdown = remaining
                elif self.mode != Mode.CAMERA:
                    if recognizer is not None:
                        start = perf_counter()
                        observation, hands = recognizer.recognize(frame)
                        inference_ran = True
                        inference_ms = (perf_counter() - start) * 1000
                    else:
                        with self._demo_lock:
                            observation = (self._demo_sequence.popleft() if self._demo_sequence
                                           else self._demo_observation)
                        cv2.putText(frame, "SIMULATED INPUT", (40, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65, (30, 92, 221), 2, cv2.LINE_AA)
                        cv2.putText(frame, observation.gesture or "No hand / release", (40, 180),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (45, 42, 40), 2, cv2.LINE_AA)
                        cv2.putText(frame, "Choose a pose below to test the gesture rules.", (40, 235),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (105, 100, 95), 1, cv2.LINE_AA)
                    decision_at = perf_counter()
                    action = processor.update(observation, decision_at)
                    if action is not None and action != Action.TOGGLE:
                        delivered = self._dispatch(controller, action)
                        if delivered is False:
                            feedback.notify("blocked")
                        elif delivered and action == Action.EXIT:
                            feedback.notify("exit")
                    status = processor.status
                    if self.config.diagnostics and now - last_diagnostic >= 0.5:
                        self._log(f"Hand: pose={observation.gesture} score={observation.confidence:.2f} "
                                  f"open={observation.open_hand} fingers={observation.extended_fingers} "
                                  f"size={observation.hand_scale} position={observation.position} "
                                  f"facing={observation.palm_facing}; {status}")
                        last_diagnostic = now
                else:
                    status = "Camera preview only. Recognition and keyboard control are off."
                active = processor.active and not remaining and self.mode != Mode.CAMERA
                if not remaining and self.mode != Mode.CAMERA and active != last_active:
                    self._log("Gesture control ACTIVE." if active else "Gesture control PAUSED.")
                    if action != Action.EXIT:
                        feedback.notify("active" if active else "paused")
                    last_active = active
                if recorder is not None:
                    recorder.write(frame, observation, captured_at=captured_at, active=active,
                                   status=status, phase=phase, action=action, delivered=delivered,
                                   inference_ms=inference_ms, inference_ran=inference_ran,
                                   last_action=controller.last_action, fps=fps,
                                   navigation_state=processor.diagnostic_state(perf_counter()), decision_at=decision_at)
                    phase += f" | REC {recorder.frame_count}"
                draw_landmarks(frame, hands)
                self._publish(Snapshot(frame, observation, active, fps, inference_ms,
                                       controller.last_action, status, phase))
                self.stopped.wait(0.05 if self.mode == Mode.DEMO or remaining else 0.005)
