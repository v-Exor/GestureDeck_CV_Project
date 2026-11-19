import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.mapping import Action
from gesturedeck.session import Mode, Session, Snapshot


def test_frame_mailbox_keeps_only_latest():
    session = Session(Config())
    for i in range(20):
        session._publish(Snapshot(status=str(i)))
    assert session.frames.qsize() == 1
    assert session.frames.get_nowait().status == "19"


@pytest.mark.parametrize("operation", ["stop", "pause"])
def test_stop_and_pause_block_pending_keyboard_delivery(operation):
    session = Session(Config(), Mode.LIVE)
    calls = []

    class FakeController:
        def execute(self, action):
            calls.append(action)

    getattr(session, operation)()
    session._dispatch(FakeController(), Action.NEXT)
    assert calls == []


def test_simulated_input_is_rejected_outside_demo():
    with pytest.raises(ValueError, match="only in demo"):
        Session(Config(), Mode.LIVE).set_demo("Victory")


def test_background_error_is_visible_and_marks_stopped(monkeypatch):
    session = Session(Config())

    def fail():
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(session, "_capture", fail)
    session.start()
    session.thread.join(timeout=2)
    assert not session.thread.is_alive()
    assert session.stopped.is_set()
    snapshot = session.frames.get_nowait()
    assert snapshot.phase == "Error" and "unavailable" in snapshot.status


def test_demo_uses_real_processor_and_dry_controller_without_camera(monkeypatch):
    session = Session(Config(), Mode.DEMO)
    clock = [0.0]
    actual_actions = []
    controller_modes = []

    class StopClock:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, _):
            clock[0] = round(clock[0] + 0.05, 3)
            t = clock[0]
            if t == 0.05:
                session.set_demo_swipe("next")
            elif t == 1.3:
                session.set_demo_swipe("previous")
            elif t == 2.5:
                session.set_demo("Closed_Fist")
            elif t == 4.0:
                session.set_demo(None)
            elif t == 4.5:
                session.pause()
            if t > 6:
                self.set()
            return self.stopped

    from gesturedeck.controller import SlideController

    class TrackingController(SlideController):
        def __init__(self, dry_run):
            assert dry_run
            controller_modes.append(dry_run)
            super().__init__(dry_run=True, press=lambda _: pytest.fail("Demo sent a key"))

        def execute(self, action):
            actual_actions.append(action)
            super().execute(action)

    session.stopped = StopClock()
    session.set_demo(None)
    monkeypatch.setattr("gesturedeck.session.perf_counter", lambda: clock[0])
    monkeypatch.setattr("gesturedeck.session.SlideController", TrackingController)
    session._capture()
    assert controller_modes == [True]
    assert actual_actions == [Action.NEXT, Action.PREVIOUS]
    assert not session.frames.get_nowait().active


def test_live_countdown_prevents_early_inference_and_closes_resources(monkeypatch):
    session = Session(Config(countdown_seconds=3, audio_feedback=False), Mode.LIVE)
    clock = [0.0]
    inference_times = []
    closed = []
    actions = []

    class StopClock:
        stopped = False

        def is_set(self):
            return self.stopped

        def set(self):
            self.stopped = True

        def wait(self, _):
            clock[0] += 0.25
            if clock[0] > 3.25:
                self.set()

    class Camera:
        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            closed.append("camera")

        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class Recognizer(Camera):
        def recognize(self, _):
            from gesturedeck.processor import Observation
            inference_times.append(clock[0])
            return Observation(), []

        def __exit__(self, *_):
            closed.append("recognizer")

    class Controller:
        last_action = "fake controller"

        def __init__(self, dry_run):
            assert not dry_run

        def execute(self, action):
            actions.append(action)

    session.stopped = StopClock()
    monkeypatch.setattr("gesturedeck.session.perf_counter", lambda: clock[0])
    monkeypatch.setattr("gesturedeck.camera.Webcam", Camera)
    monkeypatch.setattr("gesturedeck.recognizer.GestureRecognizer", Recognizer)
    monkeypatch.setattr("gesturedeck.session.SlideController", Controller)
    session._capture()
    assert inference_times and min(inference_times) >= 3
    assert actions == []
    assert closed == ["camera", "recognizer"]


def test_pause_waits_for_inflight_delivery_and_blocks_later_keys():
    from threading import Event, Thread
    session = Session(Config(audio_feedback=False), Mode.LIVE)
    entered, finish = Event(), Event()
    calls = []

    class Controller:
        last_action = 'fake next'
        def execute(self, action):
            calls.append(action)
            entered.set()
            assert finish.wait(2)
            return True

    delivery = Thread(target=session._dispatch, args=(Controller(), Action.NEXT))
    delivery.start()
    try:
        assert entered.wait(1)
        pausing = Thread(target=session.pause)
        pausing.start()
        assert not session.pause_requested.is_set()
    finally:
        finish.set()
        delivery.join(timeout=2)
    pausing.join(timeout=2)
    session._dispatch(Controller(), Action.PREVIOUS)
    assert calls == [Action.NEXT]
    assert session.pause_requested.is_set()


def test_demo_swipe_is_one_timed_sequence_followed_by_release():
    from gesturedeck.processor import GestureProcessor
    session = Session(Config(), Mode.DEMO)
    processor = GestureProcessor(session.config)
    session.set_demo_swipe('next')
    actions = []
    for i, observation in enumerate(session._demo_sequence):
        action = processor.update(observation, i * 0.05)
        if action is not None:
            actions.append(action)
    assert actions == [Action.NEXT]
    assert not session._demo_sequence[-1].hand_present


def test_demo_exit_runs_through_real_processor_and_controller(monkeypatch):
    session = Session(Config(audio_feedback=False), Mode.DEMO)
    clock = [0.0]
    events = []

    class Clock:
        stopped = False
        def is_set(self): return self.stopped
        def set(self): self.stopped = True
        def wait(self, _):
            clock[0] = round(clock[0] + 0.05, 3)
            if clock[0] > 3: self.set()

    from gesturedeck.controller import SlideController
    class Controller(SlideController):
        def execute(self, action):
            events.append(action)
            return super().execute(action)

    session.stopped = Clock()
    session.set_demo('Victory')
    monkeypatch.setattr('gesturedeck.session.perf_counter', lambda: clock[0])
    monkeypatch.setattr('gesturedeck.session.SlideController', Controller)
    session._capture()
    assert events == [Action.EXIT]
    snapshot = session.frames.get_nowait()
    assert not snapshot.active
    assert snapshot.last_action == 'DRY RUN: exit (esc)'
