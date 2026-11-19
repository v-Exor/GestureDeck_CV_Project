from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.learned_swipe import (
    FEATURE_SIZE, MotionModel, frame_vector, sequence_features,
)
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor
from gesturedeck.recognizer import observation_from_result
from hand_fixture import model_result


def observation(x, *, extended=(0, 1, 2, 3)):
    return observation_from_result(model_result(x=x, extended=extended))


def test_temporal_features_preserve_direction_and_remove_static_offset():
    times = np.arange(16) / 30
    frames = [frame_vector(observation(.8 - i * .02).landmarks) for i in range(16)]
    feature = sequence_features(times, frames)
    shifted = np.array(frames)
    shifted[:, :2] -= .05
    np.testing.assert_allclose(feature, sequence_features(times, shifted), atol=1e-5)
    reverse = sequence_features(times, frames[::-1])
    assert feature.reshape(-1)[15 * 16] < 0
    assert reverse[15 * 16] > 0


def test_bad_landmarks_and_large_tracking_gap_cannot_become_features():
    assert not frame_vector([]).any()
    assert not frame_vector(np.full((21, 3), np.nan)).any()
    frames = [frame_vector(observation(.5).landmarks)] * 16
    times = np.arange(16) / 30
    times[8:] += .2
    assert sequence_features(times, frames) is None
    assert sequence_features(times[::-1], frames) is None


def test_mirror_normalization_matches_between_camera_settings():
    points = np.array(observation(.5).landmarks)
    mirrored = points.copy()
    mirrored[:, 0] = 1 - mirrored[:, 0]
    np.testing.assert_allclose(frame_vector(points), frame_vector(mirrored, flip_x=True), atol=1e-7)


def test_model_round_trip_and_nonfinite_rejection(tmp_path):
    rng = np.random.default_rng(3)
    model = MotionModel(np.zeros(FEATURE_SIZE), np.ones(FEATURE_SIZE),
                        rng.normal(size=(FEATURE_SIZE, 4)), np.zeros(4),
                        rng.normal(size=(1 + FEATURE_SIZE + 4, 3)), .8, {"experimental": True})
    path = tmp_path / "model.npz"
    model.save(path)
    loaded = MotionModel.load(path)
    sample = rng.normal(size=FEATURE_SIZE)
    np.testing.assert_allclose(model.scores(sample), loaded.scores(sample))
    model.weights[0, 0] = np.nan
    model.save(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        MotionModel.load(path)


class AlwaysLeftModel:
    threshold = .9
    metadata = {"validation_target_met": True}

    def scores(self, samples):
        return np.tile([0, 1, 0], (len(samples), 1))


def test_learned_path_accepts_motion_without_three_fingers_and_latches(monkeypatch):
    monkeypatch.setattr(MotionModel, "load", lambda _: AlwaysLeftModel())
    processor = GestureProcessor(Config(swipe_model_path="unit-test-only.npz"))
    events = []
    for i in range(25):
        event = processor.update(observation(.85-i*.015, extended=(0,)), 1+i*.04)
        if event:
            events.append(event)
    assert events == [Action.NEXT]
    assert processor._needs_release


def test_direction_disagreement_and_stationary_hand_do_not_send_commands(monkeypatch):
    monkeypatch.setattr(MotionModel, "load", lambda _: AlwaysLeftModel())
    for xs in ([.3+i*.01 for i in range(30)], [.5]*30):
        processor = GestureProcessor(Config(swipe_model_path="unit-test-only.npz"))
        assert all(processor.update(observation(x), 1+i*.04) is None for i, x in enumerate(xs))


def test_learned_model_preserves_fist_pause_and_victory_exit(monkeypatch):
    monkeypatch.setattr(MotionModel, "load", lambda _: AlwaysLeftModel())
    for pose, duration, expected in (("Closed_Fist", 1.2, Action.TOGGLE), ("Victory", 2.2, Action.EXIT)):
        processor = GestureProcessor(Config(swipe_model_path="unit-test-only.npz"))
        events = []
        for i in range(round(duration/.04)):
            sample = replace(observation(.85-i*.003), gesture=pose, confidence=.99)
            event = processor.update(sample, 1+i*.04)
            if event:
                events.append(event)
        assert events == [expected]
        assert not processor.active


def test_missing_model_fails_instead_of_silently_using_original_rules(tmp_path):
    with pytest.raises(FileNotFoundError):
        GestureProcessor(Config(swipe_model_path=str(tmp_path / "missing.npz")))


def test_failed_validation_model_cannot_be_used_for_camera_control(monkeypatch):
    model = AlwaysLeftModel()
    model.metadata = {"validation_target_met": False}
    monkeypatch.setattr(MotionModel, "load", lambda _: model)
    with pytest.raises(ValueError, match="failed its validation"):
        GestureProcessor(Config(swipe_model_path="unit-test-only.npz"))


def test_train_validation_test_keep_all_subject_videos_together(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("train_swipe_test", scripts / "train_swipe_model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train = [f"camera_{subject}_R_#{subject*4+video}" for subject in range(10) for video in range(4)]
    test = ["camera_99_R_#999"]
    assignments = module.split_videos(train, test)
    for subject in range(10):
        assert len({assignments[v] for v in train if module.subject_id(v) == f"camera_{subject}"}) == 1
    with pytest.raises(ValueError, match="overlapping"):
        module.split_videos(train, [train[0]])
