import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.learned_swipe import FEATURE_SIZE, MotionModel, TemporalMLP
from gesturedeck.processor import GestureProcessor
from gesturedeck.session import Mode, Session
from test_learned_swipe import AlwaysLeftModel


def load_restore_module():
    path = Path(__file__).resolve().parents[1] / "scripts/restore_checkpoint.py"
    spec = importlib.util.spec_from_file_location("restore_trial_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def checkpoint_fixture(root):
    snapshot = root / "backups/checkpoint"
    baseline = {"src/app.py": b"before", "Start.cmd": b"original launcher"}
    for relative, contents in baseline.items():
        target = snapshot / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
        current = root / relative
        current.parent.mkdir(parents=True, exist_ok=True)
        current.write_bytes(b"after")
    manifest = {"files": {name: hashlib.sha256(contents).hexdigest() for name, contents in baseline.items()}}
    (snapshot / "checkpoint.json").write_text(json.dumps(manifest))
    (snapshot / "trial_files.json").write_text(json.dumps(["new-model.npz"]))
    (root / "new-model.npz").write_bytes(b"trial")
    (root / "personal.txt").write_text("keep")
    return snapshot


def test_restore_preserves_current_edits_and_does_not_touch_unrelated_files(tmp_path):
    module = load_restore_module()
    snapshot = checkpoint_fixture(tmp_path)
    archive, changed = module.restore(tmp_path, snapshot)
    assert set(changed) == {"src/app.py", "Start.cmd"}
    assert (tmp_path / "src/app.py").read_bytes() == b"before"
    assert (archive / "src/app.py").read_bytes() == b"after"
    assert (archive / "trial-files/new-model.npz").read_bytes() == b"trial"
    assert not (tmp_path / "new-model.npz").exists()
    assert (tmp_path / "personal.txt").read_text() == "keep"
    assert (snapshot / "restored.json").is_file()
    _, changed_again = module.restore(tmp_path, snapshot)
    assert changed_again == []


def test_corrupted_checkpoint_is_rejected_before_any_project_write(tmp_path):
    module = load_restore_module()
    snapshot = checkpoint_fixture(tmp_path)
    (snapshot / "src/app.py").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="verification failed"):
        module.restore(tmp_path, snapshot)
    assert (tmp_path / "src/app.py").read_bytes() == b"after"
    assert (tmp_path / "new-model.npz").read_bytes() == b"trial"


def test_restore_rejects_paths_outside_project(tmp_path):
    module = load_restore_module()
    snapshot = checkpoint_fixture(tmp_path)
    (snapshot / "trial_files.json").write_text(json.dumps(["../outside.txt"]))
    with pytest.raises(ValueError, match="escapes project"):
        module.restore(tmp_path, snapshot)
    assert (tmp_path / "src/app.py").read_bytes() == b"after"


def test_neural_model_round_trip_and_output_validation(tmp_path):
    rng = np.random.default_rng(4)
    model = TemporalMLP(np.zeros(FEATURE_SIZE), np.ones(FEATURE_SIZE),
                        rng.normal(size=(FEATURE_SIZE, 6)), np.zeros(6),
                        rng.normal(size=(6, 3)), np.zeros(3), .8, {"experimental": True})
    path = tmp_path / "trial.npz"
    model.save(path)
    loaded = MotionModel.load(path)
    samples = rng.normal(size=(8, FEATURE_SIZE))
    np.testing.assert_allclose(model.scores(samples), loaded.scores(samples))
    np.testing.assert_allclose(loaded.scores(samples).sum(axis=1), 1.)
    model.w1[0, 0] = np.nan
    model.save(path)
    with pytest.raises(ValueError, match="Nonfinite"):
        MotionModel.load(path)


def test_failed_candidate_requires_explicit_trial_opt_in(monkeypatch):
    model = AlwaysLeftModel()
    model.metadata = {"validation_target_met": False}
    monkeypatch.setattr(MotionModel, "load", lambda _: model)
    with pytest.raises(ValueError, match="failed its validation"):
        GestureProcessor(Config(swipe_model_path="unit-test.npz"))
    processor = GestureProcessor(Config(swipe_model_path="unit-test.npz", swipe_trial=True))
    assert processor._learned_swipe
    assert not GestureProcessor(Config())._learned_swipe
    with pytest.raises(ValueError, match="requires"):
        Config(swipe_trial=True)
