"""Opt-in snapshots of the validated PowerPoint slideshow, on a separate worker."""

import json
from threading import Event, Thread
from time import perf_counter

from PIL import ImageGrab

from gesturedeck.focus import focused_powerpoint_window, powerpoint_window_bounds


class SlideshowRecorder:
    def __init__(self, path, started):
        self.path, self.started = path, started
        self.samples = 0
        self.images = 0
        self.error = None
        self.stopped = Event()
        self.thread = None

    def start(self):
        (self.path / "slides").mkdir()
        self.thread = Thread(target=self._run, name="GestureDeck-slide-recording", daemon=True)
        self.thread.start()

    def capture(self, stream):
        elapsed = perf_counter() - self.started
        window = focused_powerpoint_window()
        row = {"sample_index": self.samples, "elapsed_seconds": elapsed,
               "slideshow_focused": window is not None, "image": None}
        if window is not None:
            # PrintWindow (ImageGrab's window= path) returned a cached initial
            # slide on this machine. Capture the visible slideshow rectangle.
            # Discard the in-memory image if foreground changes during capture.
            bounds = powerpoint_window_bounds(window)
            picture = ImageGrab.grab(bbox=bounds, all_screens=True)
            if focused_powerpoint_window() != window:
                row["slideshow_focused"] = False
                row["discarded"] = "foreground_changed_during_capture"
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                self.samples += 1
                return
            if picture.width < 1 or picture.height < 1:
                raise RuntimeError("PowerPoint returned an empty window capture")
            relative = f"slides/{self.samples:07d}.jpg"
            picture.convert("RGB").save(self.path / relative, quality=85)
            row["image"] = relative
            row["capture_seconds"] = perf_counter() - self.started - elapsed
            row["capture_method"] = "visible_screen_rectangle"
            self.images += 1
        stream.write(json.dumps(row) + "\n")
        stream.flush()
        self.samples += 1

    def _run(self):
        try:
            with (self.path / "slides.jsonl").open("x", encoding="utf-8") as stream:
                while not self.stopped.is_set():
                    self.capture(stream)
                    self.stopped.wait(0.25)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def close(self):
        self.stopped.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                self.error = "PowerPoint snapshot worker did not stop within two seconds"

    def summary(self):
        return {"status": "error" if self.error else "stopped" if self.stopped.is_set() else "recording",
                "samples": self.samples, "images": self.images, "error": self.error,
                "index": "slides.jsonl", "interval_seconds": 0.25,
                "capture_method": "visible_screen_rectangle_with_foreground_recheck",
                "scope": "validated foreground PowerPoint slideshow window only"}
