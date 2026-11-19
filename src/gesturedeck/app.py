"""CLI and the small application loop. Run with python -m gesturedeck."""

import argparse
from contextlib import ExitStack
import logging
from pathlib import Path
from time import perf_counter, sleep

from gesturedeck.config import load_config
from gesturedeck.controller import ControllerError, SlideController
from gesturedeck.feedback import AudioFeedback
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor, Observation

LOGGER = logging.getLogger(__name__)
WINDOW = "GestureDeck"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="GestureDeck touchless slide navigation (dry-run by default)")
    parser.add_argument("--config", type=Path, help="Optional TOML settings file")
    parser.add_argument("--model", dest="model_path", help="Path to official .task bundle")
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--camera-backend", choices=("auto", "dshow", "msmf"))
    parser.add_argument("--confidence", dest="confidence_threshold", type=float)
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", default=None)
    parser.add_argument("--countdown", dest="countdown_seconds", type=int)
    parser.add_argument("--silent", dest="audio_feedback", action="store_false", default=None,
                        help="Disable fullscreen state tones")
    parser.add_argument("--diagnostics", action="store_true", default=None,
                        help="Log hand-tracking and swipe rejection details")
    parser.add_argument("--navigation", dest="navigation_mode", choices=("swipe", "palm_turn"),
                        help="Navigation method; palm_turn is experimental and uses the right hand")
    parser.add_argument("--swipe-model", dest="swipe_model_path",
                        help="Experimental trained temporal swipe model (.npz); test with --dry-run first")
    parser.add_argument("--experimental-swipes", dest="swipe_trial", action="store_true", default=None,
                        help="Explicitly try an unqualified swipe model; requires --swipe-model")
    parser.add_argument("--record", dest="record_directory", metavar="DIRECTORY",
                        help="Record webcam video, landmarks and events locally (requires --gui)")
    parser.add_argument("--record-slides", dest="record_slideshow", action="store_true", default=None,
                        help="Also save timed PowerPoint slideshow snapshots locally; requires recording")
    parser.add_argument("--gui", action="store_true", help="Open the orange-and-white desktop interface")
    parser.add_argument("--demo", action="store_true", help="Open desktop demo with simulated input; no camera or real keys")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--live", action="store_true", help="Explicitly allow keys to the focused app after countdown")
    mode.add_argument("--dry-run", action="store_true", help="Log commands only (default)")
    mode.add_argument("--camera-only", action="store_true", help="Preview webcam without loading the model")
    args = parser.parse_args(argv)
    if args.demo and (args.live or args.camera_only):
        parser.error("--demo cannot be combined with --live or --camera-only")
    if args.record_directory and (not args.gui or args.demo):
        parser.error("--record requires --gui and real webcam input (no demo)")
    return args


def focus_countdown(seconds: int) -> None:
    LOGGER.warning("LIVE mode: keys are allowed only while the PowerPoint slideshow has focus.")
    LOGGER.warning("Focus your PowerPoint slideshow now. Ctrl+C cancels. Keep hands away until ready.")
    for remaining in range(seconds, 0, -1):
        LOGGER.warning("Controls available in %d...", remaining)
        sleep(1)
    LOGGER.warning("Countdown finished: ACTIVE. Swipe to navigate; hold fist to pause/resume.")


def run(config, *, live=False, camera_only=False) -> int:
    # Lazy CV imports keep --help and pure gesture tests usable without hardware libraries.
    import cv2
    from gesturedeck.camera import Webcam
    from gesturedeck.preview import draw_landmarks, render_preview
    from gesturedeck.recognizer import GestureRecognizer

    processor = GestureProcessor(config)
    fps = 0.0
    inference_ms = 0.0
    last_frame_time = None
    try:
        with ExitStack() as resources:
            feedback = resources.enter_context(AudioFeedback(config.audio_feedback and live))
            recognizer = (None if camera_only else
                          resources.enter_context(GestureRecognizer(config)))
            camera = resources.enter_context(Webcam(config))
            controller = SlideController(dry_run=not live)
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            # Show the window before countdown; no subsequent window creation
            # should steal focus after the user chooses their slideshow.
            first_frame = camera.read()
            cv2.imshow(WINDOW, render_preview(first_frame, [
                "GestureDeck_CV | STARTING", "Preparing; see terminal for live-mode countdown",
            ]))
            cv2.waitKey(1)
            if live:
                focus_countdown(config.countdown_seconds)
            last_active = processor.active
            last_diagnostic = float("-inf")
            if not camera_only:
                feedback.notify("active")
            while True:
                frame = camera.read()
                if last_frame_time is not None:
                    elapsed = perf_counter() - last_frame_time
                    current_fps = 1 / max(elapsed, 1e-6)
                    fps = current_fps if fps == 0 else 0.9 * fps + 0.1 * current_fps
                last_frame_time = perf_counter()
                observation, hands = Observation(), []
                if recognizer is not None:
                    start = perf_counter()
                    observation, hands = recognizer.recognize(frame)
                    inference_ms = (perf_counter() - start) * 1000
                    action = processor.update(observation, perf_counter())
                    if config.diagnostics and perf_counter() - last_diagnostic >= 0.5:
                        LOGGER.info("Hand: pose=%s score=%.2f open=%s fingers=%s size=%s position=%s; %s",
                                    observation.gesture, observation.confidence, observation.open_hand,
                                    observation.extended_fingers, observation.hand_scale,
                                    observation.position, processor.status)
                        last_diagnostic = perf_counter()
                    if action == Action.TOGGLE:
                        LOGGER.info("Control state: %s", "ACTIVE" if processor.active else "PAUSED")
                    elif action is not None:
                        delivered = controller.execute(action)
                        if delivered is False:
                            feedback.notify("blocked")
                        elif delivered and action == Action.EXIT:
                            feedback.notify("exit")
                    if processor.active != last_active:
                        if action != Action.EXIT:
                            feedback.notify("active" if processor.active else "paused")
                        last_active = processor.active
                draw_landmarks(frame, hands)
                mode = "CAMERA ONLY" if camera_only else ("LIVE KEYS" if live else "DRY RUN")
                label = observation.gesture or ("Unknown" if observation.hand_present else "No hand")
                lines = [
                    f"GestureDeck_CV | {mode} | {'ACTIVE' if processor.active and not camera_only else 'PAUSED'}",
                    f"Gesture: {label} | Score: {observation.confidence:.2f} | Threshold: {config.confidence_threshold:.2f}",
                    f"FPS ~{fps:.1f} | Inference: {inference_ms:.1f} ms | Hands: {len(hands)}",
                    f"Status: {processor.status}",
                    f"Last action: {controller.last_action}",
                    "Exit: Q/Esc in preview, close window, or Ctrl+C in terminal",
                ]
                cv2.imshow(WINDOW, render_preview(frame, lines))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
    finally:
        cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config = load_config(
            args.config, model_path=args.model_path, camera_index=args.camera_index,
            camera_backend=args.camera_backend, confidence_threshold=args.confidence_threshold,
            mirror=args.mirror, countdown_seconds=args.countdown_seconds,
            audio_feedback=args.audio_feedback,
            diagnostics=args.diagnostics,
            navigation_mode=args.navigation_mode, record_directory=args.record_directory,
            swipe_model_path=args.swipe_model_path,
            swipe_trial=args.swipe_trial,
            record_slideshow=args.record_slideshow,
        )
        if config.record_directory and (not args.gui or args.demo):
            raise ValueError("record_directory requires --gui and real webcam input")
        if config.record_slideshow and not config.record_directory:
            raise ValueError("--record-slides requires a recording directory (--record)")
        if args.gui or args.demo:
            from gesturedeck.desktop import run_desktop
            from gesturedeck.session import Mode
            mode = (Mode.DEMO if args.demo else Mode.LIVE if args.live else
                    Mode.CAMERA if args.camera_only else Mode.DRY_RUN)
            return run_desktop(config, mode)
        return run(config, live=args.live, camera_only=args.camera_only)
    except KeyboardInterrupt:
        LOGGER.info("Stopped by user; resources released.")
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, ControllerError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except Exception as exc:
        # OpenCV/native task errors vary by library version. Keep a traceback for
        # unexpected failures, and ExitStack/finally still release all resources.
        LOGGER.exception("Application stopped: %s", exc)
        return 1
