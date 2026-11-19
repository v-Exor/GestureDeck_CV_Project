import json
from time import perf_counter

import cv2
import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.mapping import Action
from gesturedeck.processor import Observation
from gesturedeck.recording import SessionRecorder


def test_real_video_roundtrip_preserves_frame_order_and_metadata(tmp_path):
    recorder = SessionRecorder(Config(record_directory=str(tmp_path), navigation_mode="palm_turn"), "dry run")
    with recorder:
        for i in range(4):
            recorder.write(np.full((240, 320, 3), i * 60, dtype=np.uint8),
                           Observation(palm_facing=float("nan")), captured_at=perf_counter(),
                           active=True, status="trial", phase="Running",
                           action=Action.NEXT if i == 2 else None, delivered=True if i == 2 else None)
    rows = [json.loads(line) for line in (recorder.path / "frames.jsonl").read_text().splitlines()]
    assert [row["frame_index"] for row in rows] == [0, 1, 2, 3]
    assert rows[2]["predicted_action"] == "next" and rows[2]["delivered"] is True
    assert all(row["ground_truth"] is None and row["observation"]["palm_facing"] is None for row in rows)
    assert rows[-1]["elapsed_seconds"] >= rows[0]["elapsed_seconds"] >= 0
    capture = cv2.VideoCapture(str(recorder.path / "webcam.avi"))
    means = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            means.append(frame.mean())
    finally:
        capture.release()
    assert len(means) == len(rows)
    assert means == pytest.approx([0, 60, 120, 180], abs=3)
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["status"] == "completed" and manifest["frame_count"] == 4
    assert not manifest["prediction_labels_are_ground_truth"]
    assert (recorder.path / "annotations.csv").read_text().strip() == "start_frame,end_frame,label,notes"


def test_failed_encoder_is_visible_and_manifest_marks_error(tmp_path, monkeypatch):
    class Writer:
        def isOpened(self): return False
        def release(self): pass
    monkeypatch.setattr(cv2, "VideoWriter", lambda *args: Writer())
    recorder = SessionRecorder(Config(record_directory=str(tmp_path)), "live")
    with pytest.raises(RuntimeError, match="Recording failed"):
        with recorder:
            recorder.write(np.zeros((240, 320, 3), dtype=np.uint8), Observation(),
                           captured_at=perf_counter(), active=True, status="", phase="Running")
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["status"] == "error" and "Recording failed" in manifest["error"]


def test_session_records_raw_frames_and_dispatched_turns_together(tmp_path, monkeypatch):
    from gesturedeck.session import Mode, Session
    from gesturedeck.recognizer import observation_from_result
    from hand_fixture import model_result

    clock = [0.0]
    config = Config(record_directory=str(tmp_path), navigation_mode="palm_turn", audio_feedback=False)
    session = Session(config, Mode.DRY_RUN)

    class Clock:
        stopped = False
        def is_set(self): return self.stopped
        def set(self): self.stopped = True
        def wait(self, _):
            clock[0] = round(clock[0] + 0.05, 3)
            if clock[0] > 2.8: self.set()

    class Camera:
        def __init__(self, _): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def read(self): return np.zeros((240, 320, 3), dtype=np.uint8)

    class Recognizer(Camera):
        def recognize(self, _):
            result = model_result()
            if 0.8 <= clock[0] < 2.0:
                for point in result.hand_landmarks[0]: point.x = 1 - point.x
            return observation_from_result(result), []

    session.stopped = Clock()
    monkeypatch.setattr("gesturedeck.session.perf_counter", lambda: clock[0])
    monkeypatch.setattr("gesturedeck.recording.perf_counter", lambda: clock[0])
    monkeypatch.setattr("gesturedeck.camera.Webcam", Camera)
    monkeypatch.setattr("gesturedeck.recognizer.GestureRecognizer", Recognizer)
    monkeypatch.setattr("gesturedeck.preview.draw_landmarks", lambda frame, _: frame.fill(255))
    session._capture()
    directory = next(tmp_path.iterdir())
    rows = [json.loads(line) for line in (directory / "frames.jsonl").read_text().splitlines()]
    assert [row["predicted_action"] for row in rows if row["predicted_action"]] == ["next", "previous"]
    assert all(row["delivered"] for row in rows if row["predicted_action"])
    assert all(row["inference_ran"] for row in rows)
    assert all(row["decision_elapsed_seconds"] is not None for row in rows)
    assert len(rows[0]["observation"]["landmarks"]) == 21
    capture = cv2.VideoCapture(str(directory / "webcam.avi"))
    try:
        ok, frame = capture.read()
        assert ok and frame.mean() == pytest.approx(0, abs=3)  # No white overlay in saved video.
    finally:
        capture.release()
