import numpy as np
import pytest

from gesturedeck.camera import CameraError, Webcam
from gesturedeck.config import Config


class FakeCapture:
    def __init__(self, opened=True, frame=None):
        self.opened = opened
        self.frame = frame
        self.released = False

    def isOpened(self):
        return self.opened

    def set(self, *_):
        return True

    def read(self):
        return self.frame is not None, self.frame

    def release(self):
        self.released = True


def test_unavailable_camera_released(monkeypatch):
    capture = FakeCapture(opened=False)
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with pytest.raises(CameraError, match="Cannot open"):
        with Webcam(Config()):
            pytest.fail("Unavailable camera must not enter")
    assert capture.released


def test_disconnected_camera_released(monkeypatch):
    capture = FakeCapture()
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with pytest.raises(CameraError, match="disconnected"):
        with Webcam(Config()) as camera:
            camera.read()
    assert capture.released


def test_mirror_and_cleanup_on_processing_error(monkeypatch):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :320] = (1, 2, 3)
    capture = FakeCapture(frame=frame)
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with pytest.raises(RuntimeError, match="processing"):
        with Webcam(Config()) as camera:
            np.testing.assert_array_equal(camera.read(), frame[:, ::-1])
            raise RuntimeError("processing")
    assert capture.released


def test_native_target_frames_are_unchanged_without_mirror(monkeypatch):
    frame = np.random.default_rng(4).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    capture = FakeCapture(frame=frame)
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with Webcam(Config(mirror=False)) as camera:
        assert camera.read() is frame
        assert camera.source_size == (640, 480)
        assert camera.crop_box == (0, 0, 640, 480)


@pytest.mark.parametrize("width,height,crop", [
    (1920, 1080, (240, 0, 1440, 1080)),
    (480, 640, (0, 140, 480, 360)),
    (1280, 960, (0, 0, 1280, 960)),
])
def test_driver_resolution_is_cropped_without_stretching(monkeypatch, width, height, crop):
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    left, top, cw, ch = crop
    # Different-colored discarded edges reveal any accidental full-frame squeeze.
    frame[:top] = 0; frame[top+ch:] = 0
    frame[:, :left] = 0; frame[:, left+cw:] = 0
    capture = FakeCapture(frame=frame)
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with Webcam(Config(mirror=False)) as camera:
        result = camera.read()
        assert result.shape == (480, 640, 3)
        assert camera.crop_box == crop
        np.testing.assert_array_equal(result, np.full_like(result, 200))


def test_resolution_change_midstream_still_delivers_target_and_mirrors_once(monkeypatch):
    capture = FakeCapture(frame=np.zeros((480, 640, 3), dtype=np.uint8))
    monkeypatch.setattr("gesturedeck.camera.cv2.VideoCapture", lambda *_: capture)
    with Webcam(Config()) as camera:
        assert camera.read().shape == (480, 640, 3)
        capture.frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        capture.frame[:, 240:960] = (0, 0, 255)
        capture.frame[:, 960:1680] = (255, 0, 0)
        result = camera.read()
        assert result.shape == (480, 640, 3)
        assert camera.source_size == (1920, 1080)
        np.testing.assert_array_equal(result[240, 100], (255, 0, 0))
        np.testing.assert_array_equal(result[240, 540], (0, 0, 255))
