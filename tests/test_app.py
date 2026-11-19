from types import SimpleNamespace

import numpy as np
import pytest

from gesturedeck.app import run
from gesturedeck.config import Config
from gesturedeck.controller import ControllerError
from gesturedeck.mapping import Action
from gesturedeck.processor import Observation


@pytest.mark.parametrize("fail", [False, True])
def test_live_countdown_precedes_inference_and_resources_always_close(monkeypatch, fail):
    """Fake the camera, windows, inference, and keyboard; use the actual loop."""
    events = []

    class Camera:
        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("camera closed")

        def read(self):
            return np.zeros((240, 320, 3), dtype=np.uint8)

    class Recognizer:
        def __init__(self, _):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            events.append("recognizer closed")

        def recognize(self, frame):
            events.append("inference")
            return Observation("Victory", 0.99, True), []

    class Controller:
        last_action = "test"

        def __init__(self, dry_run):
            assert dry_run is False

        def execute(self, action):
            events.append("fake key")
            if fail:
                raise ControllerError("fake fail-safe")

    monkeypatch.setattr("gesturedeck.camera.Webcam", Camera)
    monkeypatch.setattr("gesturedeck.recognizer.GestureRecognizer", Recognizer)
    monkeypatch.setattr("gesturedeck.app.SlideController", Controller)
    monkeypatch.setattr("gesturedeck.app.GestureProcessor", lambda _: SimpleNamespace(
        update=lambda *_: Action.NEXT, active=True, status="test"))
    monkeypatch.setattr("gesturedeck.app.focus_countdown", lambda _: events.append("countdown"))
    monkeypatch.setattr("cv2.namedWindow", lambda *_: events.append("window created"))
    monkeypatch.setattr("cv2.imshow", lambda *_: None)
    monkeypatch.setattr("cv2.waitKey", lambda *_: ord("q"))
    monkeypatch.setattr("cv2.destroyAllWindows", lambda: events.append("windows closed"))
    if fail:
        with pytest.raises(ControllerError):
            run(Config(audio_feedback=False), live=True)
    else:
        assert run(Config(audio_feedback=False), live=True) == 0
    assert events.index("window created") < events.index("countdown")
    assert events.index("countdown") < events.index("inference") < events.index("fake key")
    assert events[-3:] == ["camera closed", "recognizer closed", "windows closed"]
