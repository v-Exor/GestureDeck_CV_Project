import io
import json

from PIL import Image

from gesturedeck.slideshow_recording import SlideshowRecorder


def test_unfocused_other_apps_are_never_captured(tmp_path, monkeypatch):
    monkeypatch.setattr("gesturedeck.slideshow_recording.focused_powerpoint_window", lambda: None)
    monkeypatch.setattr("gesturedeck.slideshow_recording.ImageGrab.grab",
                        lambda **_: (_ for _ in ()).throw(AssertionError("Other app captured")))
    recorder = SlideshowRecorder(tmp_path, 0)
    stream = io.StringIO()
    recorder.capture(stream)
    row = json.loads(stream.getvalue())
    assert not row["slideshow_focused"] and row["image"] is None
    assert recorder.samples == 1 and recorder.images == 0


def test_capture_is_bound_to_validated_window_and_has_shared_timestamp(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("gesturedeck.slideshow_recording.focused_powerpoint_window", lambda: 123)
    monkeypatch.setattr("gesturedeck.slideshow_recording.powerpoint_window_bounds", lambda _: (0, 0, 320, 180))
    monkeypatch.setattr("gesturedeck.slideshow_recording.perf_counter", lambda: 12.5)
    def grab(**kwargs):
        calls.append(kwargs)
        return Image.new("RGB", (320, 180), "purple")
    monkeypatch.setattr("gesturedeck.slideshow_recording.ImageGrab.grab", grab)
    (tmp_path / "slides").mkdir()
    recorder = SlideshowRecorder(tmp_path, 10)
    stream = io.StringIO()
    recorder.capture(stream)
    row = json.loads(stream.getvalue())
    assert calls == [{"bbox": (0, 0, 320, 180), "all_screens": True}]
    assert row["elapsed_seconds"] == 2.5 and row["slideshow_focused"]
    assert (tmp_path / row["image"]).is_file()


def test_focus_switch_during_visible_capture_discards_image(tmp_path, monkeypatch):
    handles = iter((123, None))
    monkeypatch.setattr("gesturedeck.slideshow_recording.focused_powerpoint_window", lambda: next(handles))
    monkeypatch.setattr("gesturedeck.slideshow_recording.powerpoint_window_bounds", lambda _: (0, 0, 320, 180))
    monkeypatch.setattr("gesturedeck.slideshow_recording.ImageGrab.grab", lambda **_: Image.new("RGB", (320, 180)))
    recorder = SlideshowRecorder(tmp_path, 0)
    stream = io.StringIO()
    recorder.capture(stream)
    row = json.loads(stream.getvalue())
    assert row["image"] is None and row["discarded"] == "foreground_changed_during_capture"
    assert recorder.images == 0 and not list(tmp_path.glob("**/*.jpg"))


def test_capture_failure_is_reported_without_unhandled_thread_error(tmp_path, monkeypatch):
    def fail(): raise RuntimeError("capture unavailable")
    monkeypatch.setattr("gesturedeck.slideshow_recording.focused_powerpoint_window", fail)
    recorder = SlideshowRecorder(tmp_path, 0)
    recorder.start()
    recorder.thread.join(timeout=1)
    recorder.close()
    assert recorder.summary()["status"] == "error"
    assert "capture unavailable" in recorder.summary()["error"]
