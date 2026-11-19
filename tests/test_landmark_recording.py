import hashlib
import json
from time import perf_counter

import cv2
import numpy as np
import pytest

from gesturedeck.annotate_recording import export_recording
from gesturedeck.config import Config
from gesturedeck.preview import draw_landmarks, draw_recorded_landmarks
from gesturedeck.processor import Observation
from gesturedeck.recording import SessionRecorder
from hand_fixture import model_result


def coordinates(x):
    return tuple((p.x, p.y, p.z) for p in model_result(x=x).hand_landmarks[0])


def decoded_video(path):
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        assert capture.isOpened()
        while True:
            ok, frame = capture.read()
            if not ok:
                return frames
            frames.append(frame)
    finally:
        capture.release()


def test_recorded_joints_match_live_preview_pixels_without_another_mirror():
    expected = np.zeros((240, 320, 3), np.uint8)
    actual = expected.copy()
    draw_landmarks(expected, model_result(x=.25).hand_landmarks)
    assert draw_recorded_landmarks(actual, coordinates(.25))
    assert np.array_equal(actual, expected)
    assert actual[:, :160].sum() > 0 and actual[:, 160:].sum() == 0


@pytest.mark.parametrize("landmarks", [None, (), ((.2, .2, 0),),
                                    ((float('nan'), .2, 0),) * 21,
                                    ((None, .2, 0),) * 21])
def test_missing_or_invalid_landmarks_leave_the_frame_unmarked(landmarks):
    frame = np.full((240, 320, 3), 40, np.uint8)
    assert not draw_recorded_landmarks(frame, landmarks)
    assert np.all(frame == 40)


def record_three_frames(directory):
    recorder = SessionRecorder(Config(record_directory=str(directory)), "Recognition / dry run")
    source = np.full((240, 320, 3), 40, np.uint8)
    with recorder:
        for points in (coordinates(.25), coordinates(.75), ()):
            recorder.write(source, Observation(landmarks=points), captured_at=perf_counter(),
                           active=True, status="test", phase="Running", inference_ran=True)
            assert np.all(source == 40)  # Drawing must never contaminate inference/raw input.
    return recorder.path


def test_saved_joint_video_moves_with_the_hand_and_clears_when_tracking_is_lost(tmp_path):
    folder = record_three_frames(tmp_path)
    raw = decoded_video(folder / "webcam.avi")
    annotated = decoded_video(folder / "webcam_landmarks.avi")
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines()]
    manifest = json.loads((folder / "manifest.json").read_text())
    assert len(raw) == len(annotated) == len(rows) == manifest['frame_count'] == 3
    assert manifest['annotated_frame_count'] == 3 and manifest['annotated_hand_frames'] == 2
    assert manifest['schema_version'] == 5 and manifest['status'] == 'completed'
    assert manifest['video'] == 'webcam.avi' and manifest['annotated_video'] == 'webcam_landmarks.avi'
    assert all(frame.mean() == pytest.approx(40, abs=3) for frame in raw)
    for i, region in ((0, slice(0, 160)), (1, slice(160, 320))):
        difference = np.abs(annotated[i].astype(float) - raw[i].astype(float)).mean(axis=2)
        assert difference[:, region].sum() > 1000
        other_half = slice(160, 320) if i == 0 else slice(0, 160)
        assert difference[:, other_half].mean() < 3
    assert np.abs(annotated[2].astype(float) - raw[2].astype(float)).mean() < 3


def test_joint_encoder_failure_marks_error_and_releases_both_writers(tmp_path, monkeypatch):
    writers = []
    class Writer:
        released = False
        def __init__(self, opened): self.opened = opened
        def isOpened(self): return self.opened
        def release(self): self.released = True
    def create_writer(*args):
        writer = Writer(not writers)
        writers.append(writer)
        return writer
    monkeypatch.setattr(cv2, 'VideoWriter', create_writer)
    recorder = SessionRecorder(Config(record_directory=str(tmp_path)), 'dry run')
    with pytest.raises(RuntimeError, match='hand-landmark video writer'):
        with recorder:
            recorder.write(np.zeros((240, 320, 3), np.uint8), Observation(), captured_at=perf_counter(),
                           active=True, status='', phase='Running')
    assert len(writers) == 2 and all(writer.released for writer in writers)
    manifest = json.loads((recorder.path / 'manifest.json').read_text())
    assert manifest['status'] == 'error'


def test_legacy_export_preserves_sources_and_refuses_overwrite_and_path_escape(tmp_path):
    folder = record_three_frames(tmp_path / 'recordings')
    manifest_path = folder / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['schema_version'] = 4
    for key in ('annotated_video', 'annotated_pixels', 'annotated_frame_count', 'annotated_hand_frames'):
        manifest.pop(key)
    manifest_path.write_text(json.dumps(manifest))
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir() if p.is_file()}
    result = export_recording(folder, root=tmp_path)
    assert result['frame_count'] == 3 and result['hand_landmark_frames'] == 2
    assert not result['camera_opened'] and not result['raw_files_modified']
    assert len(decoded_video(result['video'])) == 3
    for name, expected in source_hashes.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    with pytest.raises(ValueError, match='already exists'):
        export_recording(folder, root=tmp_path)
    with pytest.raises(ValueError, match='Output must'):
        export_recording(folder, tmp_path / 'outside.avi', root=tmp_path)


def test_export_rejects_active_recordings(tmp_path):
    folder = tmp_path / 'recordings/active'
    folder.mkdir(parents=True)
    (folder / 'manifest.json').write_text(json.dumps({'status': 'recording'}))
    with pytest.raises(ValueError, match='Stop the recording'):
        export_recording(folder, root=tmp_path)
