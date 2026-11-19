"""Experimental supervised sequence classifier. Does not train at runtime.

Uses NumPy only; the MediaPipe landmark model remains unchanged. Scores are
classifier scores, not calibrated probabilities or measured accuracy.
"""

from collections import deque
import json
from pathlib import Path

import numpy as np

from gesturedeck.mapping import Action

SCHEMA = 1
STEPS = 16
FRAME_SIZE = 16
FEATURE_SIZE = STEPS * FRAME_SIZE + 1
WINDOWS = (0.4, 0.7, 1.0, 1.4)
PALM = (0, 5, 9, 13, 17)
SHAPE = (0, 4, 8, 12, 16, 20)
LABELS = ("other", "left", "right")


def frame_vector(landmarks, *, flip_x=False, aspect=4 / 3):
    """Normalized 2D positions; finger extension is deliberately not a gate."""
    p = np.asarray(landmarks, dtype=np.float64)
    if p.shape not in ((21, 2), (21, 3)) or not np.isfinite(p[:, :2]).all():
        return np.zeros(FRAME_SIZE)
    p = p[:, :2].copy()
    if flip_x:
        p[:, 0] = 1 - p[:, 0]
    p[:, 1] /= aspect
    center = p[list(PALM)].mean(axis=0)
    scale = max(np.linalg.norm(p[5] - p[17]), np.linalg.norm(p[0] - p[9]))
    if scale < 0.02 or not (0 <= center[0] <= 1 and 0 <= center[1] <= 1 / aspect):
        return np.zeros(FRAME_SIZE)
    shape = np.clip((p[list(SHAPE)] - center) / scale, -4, 4).ravel()
    return np.r_[center, scale, 1., shape]


def sequence_features(times, frames):
    """Preserve horizontal trajectory, hand shape, scale change and duration."""
    t = np.asarray(times, dtype=np.float64)
    f = np.asarray(frames, dtype=np.float64)
    if len(t) < 4 or f.shape != (len(t), FRAME_SIZE) or not np.isfinite(f).all():
        return None
    if not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        return None
    valid = f[:, 3] > 0
    if valid.mean() < .8 or valid.sum() < 4 or not valid[-1]:
        return None
    tv = t[valid]
    if tv[0] - t[0] > .12 or np.max(np.diff(tv)) > .12:
        return None
    duration = t[-1] - t[0]
    if not .25 <= duration <= 1.6:
        return None
    target = np.linspace(t[0], t[-1], STEPS)
    out = np.column_stack([np.interp(target, tv, f[valid, i]) for i in range(FRAME_SIZE)])
    scale = np.median(f[valid, 2])
    out[:, :2] = (out[:, :2] - out[0, :2]) / scale
    out[:, 2] = np.log(np.maximum(out[:, 2], .001) / scale)
    out[:, 3] = np.interp(target, t, valid.astype(float))
    return np.r_[np.clip(out, -6, 6).ravel(), duration].astype(np.float32)


class MotionModel:
    def __init__(self, mean, std, projection, phase, weights, threshold, metadata):
        self.mean, self.std = mean, std
        self.projection, self.phase, self.weights = projection, phase, weights
        self.threshold, self.metadata = float(threshold), metadata

    def design(self, features):
        x = np.clip((np.atleast_2d(features) - self.mean) / self.std, -5, 5)
        nonlinear = np.cos(x @ self.projection + self.phase) * np.sqrt(2 / len(self.phase))
        return np.column_stack((np.ones(len(x)), x * .05, nonlinear))

    def scores(self, features):
        return self.design(features) @ self.weights

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, schema=SCHEMA, mean=self.mean, std=self.std,
                                projection=self.projection, phase=self.phase, weights=self.weights,
                                threshold=self.threshold, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema"]) == 2:
                return TemporalMLP.load(path)
            if int(data["schema"]) != SCHEMA:
                raise ValueError("Unsupported learned swipe model schema")
            model = cls(*(data[key].copy() for key in ("mean", "std", "projection", "phase", "weights")),
                        float(data["threshold"]), json.loads(str(data["metadata"])))
        n = len(model.phase)
        if (model.mean.shape != (FEATURE_SIZE,) or model.std.shape != (FEATURE_SIZE,)
                or model.projection.shape != (FEATURE_SIZE, n)
                or model.weights.shape != (1 + FEATURE_SIZE + n, 3)
                or np.any(model.std <= 0) or not np.isfinite(model.threshold)
                or model.threshold < 0):
            raise ValueError("Invalid learned swipe model dimensions/settings")
        if not all(np.isfinite(a).all() for a in (model.mean, model.std, model.projection,
                                                model.phase, model.weights)):
            raise ValueError("Nonfinite learned swipe model parameters")
        return model


class TemporalMLP:
    """Small learned nonlinear classifier over ordered sequence features."""

    def __init__(self, mean, std, w1, b1, w2, b2, threshold, metadata):
        self.mean, self.std = mean, std
        self.w1, self.b1, self.w2, self.b2 = w1, b1, w2, b2
        self.threshold, self.metadata = float(threshold), metadata

    def scores(self, features):
        x = np.clip((np.atleast_2d(features) - self.mean) / self.std, -5, 5)
        logits = np.maximum(x @ self.w1 + self.b1, 0) @ self.w2 + self.b2
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, schema=2, mean=self.mean, std=self.std,
                                w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2,
                                threshold=self.threshold, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["schema"]) != 2:
                raise ValueError("Unsupported temporal model schema")
            model = cls(*(data[k].copy() for k in ("mean", "std", "w1", "b1", "w2", "b2")),
                        float(data["threshold"]), json.loads(str(data["metadata"])))
        hidden = len(model.b1)
        if (model.mean.shape != (FEATURE_SIZE,) or model.std.shape != (FEATURE_SIZE,)
                or model.w1.shape != (FEATURE_SIZE, hidden) or model.w2.shape != (hidden, 3)
                or model.b2.shape != (3,) or hidden < 1 or np.any(model.std <= 0)
                or not 0 <= model.threshold <= 1):
            raise ValueError("Invalid temporal model dimensions/settings")
        if not all(np.isfinite(a).all() for a in (model.mean, model.std, model.w1,
                                                model.b1, model.w2, model.b2)):
            raise ValueError("Nonfinite temporal model parameters")
        return model


class LearnedSwipeDetector:
    def __init__(self, config):
        self.config = config
        self.model = MotionModel.load(config.swipe_model_path)
        if self.model.metadata.get("validation_target_met") is False and not config.swipe_trial:
            raise ValueError("This experimental swipe model failed its validation precision target. "
                             "Use the original launcher; see docs/LEARNED_SWIPES.md for the training report.")
        self.points = deque()
        self.status = "Learned swipe model: move your hand sideways"
        self.last_inference = float("-inf")
        self.last_evaluation = None
        self.details = {"state": "waiting", "windows": []}

    def reset(self, *, keep_diagnostics=False):
        self.points.clear()
        self.last_inference = float("-inf")
        if not keep_diagnostics:
            self.last_evaluation = None
            self.details = {"state": "reset", "windows": []}

    def diagnostic_state(self, now):
        return {**self.details, "threshold": self.model.threshold,
                "score_age_seconds": (max(0, now - self.last_evaluation)
                                      if self.last_evaluation is not None else None),
                "scores_are_calibrated_accuracy": False}

    def interrupt(self, now):
        self.reset()

    def tracking_lost(self, now):
        """Bridge brief missing frames without treating them as evidence."""
        if not self.points or now - self.points[-1][0] > self.config.swipe_tracking_grace_seconds:
            self.reset()
        else:
            self.details = {**self.details, "state": "brief_tracking_gap"}

    def observe(self, observation, now):
        f = frame_vector(observation.landmarks, flip_x=not self.config.mirror,
                         aspect=self.config.width / self.config.height)
        if f[3] == 0:
            self.reset()
            self.details["state"] = "invalid_or_small_hand"
            self.status = "Learned swipe: keep the whole hand visible"
            return None
        if self.points and (now - self.points[-1][0] > self.config.swipe_tracking_grace_seconds
                            or np.linalg.norm(f[:2] - self.points[-1][1][:2]) > self.config.swipe_max_step):
            self.reset()
        self.points.append((now, f))
        while self.points and now - self.points[0][0] > 1.5:
            self.points.popleft()
        if now - self.last_inference < .06:
            return None
        self.last_inference = now
        candidates, movements, durations = [], [], []
        for window in WINDOWS:
            chosen = [(t, p) for t, p in self.points if t >= now - window]
            if len(chosen) < 4 or chosen[-1][0] - chosen[0][0] < window * .85:
                continue
            features = sequence_features([t for t, _ in chosen], [p for _, p in chosen])
            if features is not None:
                candidates.append(features)
                movements.append(chosen[-1][1][0] - chosen[0][1][0])
                durations.append(chosen[-1][0] - chosen[0][0])
        if not candidates:
            self.details = {"state": "collecting_history", "windows": []}
            self.last_evaluation = None
            self.status = "Learned swipe: collecting motion history"
            return None
        scores = self.model.scores(candidates)
        winner = np.argmax(scores, axis=1)
        self.last_evaluation = now
        windows = []
        for i, label in enumerate(winner):
            dx = movements[i]
            direction_ok = (label == 1 and dx < -.025) or (label == 2 and dx > .025)
            reason = ("other_motion" if label == 0 else "direction_or_jitter_guard" if not direction_ok else
                      "below_threshold" if scores[i, label] < self.model.threshold else "eligible")
            windows.append({"duration_seconds": float(durations[i]),
                            "scores": {name: float(value) for name, value in zip(LABELS, scores[i])},
                            "predicted_label": LABELS[int(label)], "horizontal_displacement": float(dx),
                            "reason": reason})
        self.details = {"state": "evaluated", "windows": windows}
        for index in np.argsort(-scores.max(axis=1)):
            label = int(winner[index])
            # Reject direction disagreement and tiny jitter independently of ML.
            dx = movements[index]
            direction_ok = (label == 1 and dx < -.025) or (label == 2 and dx > .025)
            if direction_ok and scores[index, label] >= self.model.threshold:
                action = Action.NEXT if label == 1 else Action.PREVIOUS
                self.details["state"] = "accepted"
                self.details["accepted_window"] = int(index)
                self.reset(keep_diagnostics=True)
                self.status = f"Learned swipe accepted: {action.value}; lower hand"
                return action
        best = int(np.argmax(scores.max(axis=1)))
        label = LABELS[int(winner[best])]
        self.status = (f"Learned swipe: {label}/uncertain; score {scores[best].max():.2f}, "
                       f"threshold {self.model.threshold:.2f}")
        return None
