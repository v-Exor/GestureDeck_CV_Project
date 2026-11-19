import csv
import hashlib
import json
from time import perf_counter

import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.learned_swipe import MotionModel
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor, Observation
from gesturedeck.recording import SessionRecorder
from test_learned_swipe import AlwaysLeftModel, observation


@pytest.mark.parametrize("mode,delivered,action,outcome", [
    ("Recognition / dry run", True, Action.NEXT, "dry_run_logged"),
    ("PowerPoint / live keys", True, Action.PREVIOUS, "key_sender_succeeded"),
    ("PowerPoint / live keys", False, Action.NEXT, "blocked"),
    ("PowerPoint / live keys", None, Action.NEXT, "not_dispatched"),
    ("PowerPoint / live keys", None, Action.TOGGLE, "control_state_changed"),
])
def test_command_audit_distinguishes_key_sending_and_dry_run(tmp_path, mode, delivered, action, outcome):
    recorder = SessionRecorder(Config(record_directory=str(tmp_path)), mode)
    with recorder:
        now = perf_counter()
        recorder.write(np.zeros((240, 320, 3), np.uint8), Observation(), captured_at=now,
                       active=True, status="test", phase="Running", action=action,
                       delivered=delivered, decision_at=now+.01)
    with (recorder.path / "commands.csv").open(newline="") as stream:
        events = list(csv.DictReader(stream))
    assert len(events) == 1
    assert events[0]["action"] == action.value
    assert events[0]["outcome"] == outcome
    assert float(events[0]["decision_seconds"]) > float(events[0]["capture_seconds"])
    assert events[0]["frame_index"] == "0"
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["command_count"] == 1
    assert len((recorder.path / "attempts_review.csv").read_text().splitlines()) == 1
    assert not manifest["prediction_labels_are_ground_truth"]


def test_recording_preserves_model_bytes_and_no_command_is_invented(tmp_path):
    source = tmp_path / "source.npz"
    source.write_bytes(b"test model snapshot bytes")
    recorder = SessionRecorder(Config(record_directory=str(tmp_path / "recordings"),
                                      swipe_model_path=str(source), swipe_trial=True), "dry run")
    with recorder:
        recorder.write(np.zeros((240, 320, 3), np.uint8), Observation(), captured_at=perf_counter(),
                       active=True, status="uncertain", phase="Running")
    source.write_bytes(b"new model after recording")
    saved = recorder.path / "swipe_model.npz"
    assert saved.read_bytes() == b"test model snapshot bytes"
    manifest = json.loads((recorder.path / "manifest.json").read_text())
    assert manifest["swipe_model_snapshot"]["sha256"] == hashlib.sha256(saved.read_bytes()).hexdigest()
    assert manifest["command_count"] == 0


def test_accepting_a_swipe_preserves_its_scores_for_the_recording(monkeypatch):
    monkeypatch.setattr(MotionModel, "load", lambda _: AlwaysLeftModel())
    processor = GestureProcessor(Config(swipe_model_path="unit-test-only.npz"))
    decision = None
    for i in range(25):
        now = 1+i*.04
        action = processor.update(observation(.85-i*.015), now)
        if action:
            decision = processor.diagnostic_state(now+.001)
            break
    assert action == Action.NEXT
    assert decision["swipe_engine"] == "learned"
    audit = decision["swipe_diagnostics"]
    assert audit["state"] == "accepted"
    assert audit["current_observation_scored"]
    assert audit["windows"][audit["accepted_window"]]["scores"]["left"] == 1.
    assert audit["score_age_seconds"] == pytest.approx(.001)
    json.dumps(decision, allow_nan=False)
    processor.update(observation(.85-i*.015), now+.02)
    assert not processor.diagnostic_state(now+.021)["swipe_diagnostics"]["current_observation_scored"]
    processor.update(Observation(), now+.04)
    assert processor.diagnostic_state(now+.041)["swipe_diagnostics"]["windows"] == []
