"""Train an opt-in neural temporal-feature trial; preserve the original baseline.

Uses the existing real IPN HandS cache, NumPy, fixed groups, and early stopping.
Validation chooses an experimental operating point. No test data fits weights.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from gesturedeck.learned_swipe import FEATURE_SIZE, FRAME_SIZE, STEPS, LABELS, TemporalMLP
from train_swipe_model import metrics, predict, subject_id


def fscore(report, beta=.5):
    p, r = report["swipe_precision"], report["swipe_recall"]
    return (1 + beta * beta) * p * r / (beta * beta * p + r) if p and r else 0.


def mirror_sequences(x, y):
    mirrored = x.copy()
    frames = mirrored[:, :STEPS*FRAME_SIZE].reshape(-1, STEPS, FRAME_SIZE)
    frames[:, :, [0, 4, 6, 8, 10, 12, 14]] *= -1
    labels = np.where(y == 1, 2, np.where(y == 2, 1, 0))
    return mirrored, labels


def train(cache, output, report_path, epochs=100, mirror_augment=False):
    with np.load(cache, allow_pickle=False) as data:
        x, y, split, videos = (data[k] for k in ("x", "y", "split", "video"))
        metadata = json.loads(str(data["metadata"]))
    if x.shape != (len(y), FEATURE_SIZE) or not np.isfinite(x).all():
        raise ValueError("Invalid feature cache")
    groups = {s: {subject_id(str(v)) for v in videos[split == s]}
              for s in ("train", "validation", "test")}
    if groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["validation"] & groups["test"]:
        raise ValueError("Subject leakage between splits")
    masks = {s: split == s for s in groups}
    if any(len(np.unique(y[mask])) != 3 for mask in masks.values()):
        raise ValueError("Each split must contain all three classes")
    rng = np.random.default_rng(20261020)
    training_x, yt = x[masks["train"]], y[masks["train"]]
    if mirror_augment:
        reflected, reflected_y = mirror_sequences(training_x, yt)
        training_x = np.concatenate((training_x, reflected))
        yt = np.concatenate((yt, reflected_y))
    mean = training_x.mean(axis=0)
    std = np.maximum(training_x.std(axis=0), .05)
    xt = np.clip((training_x - mean) / std, -5, 5).astype(np.float32)
    xv, yv = x[masks["validation"]], y[masks["validation"]]
    if mirror_augment:
        reflected, reflected_y = mirror_sequences(xv, yv)
        xv, yv = np.concatenate((xv, reflected)), np.concatenate((yv, reflected_y))
    hidden = 64
    metadata.update({"experimental": True, "architecture": "temporal features / 64 ReLU units / 3 classes",
                     "horizontal_mirror_augmentation": mirror_augment,
                     "validation_target_met": False,
                     "local_camera_validated": False})
    model = TemporalMLP(mean, std,
                        rng.normal(0, np.sqrt(2 / FEATURE_SIZE), (FEATURE_SIZE, hidden)).astype(np.float32),
                        np.zeros(hidden, np.float32),
                        rng.normal(0, np.sqrt(2 / hidden), (hidden, 3)).astype(np.float32),
                        np.zeros(3, np.float32), .8, metadata)
    params = [model.w1, model.b1, model.w2, model.b2]
    first = [np.zeros_like(a) for a in params]
    second = [np.zeros_like(a) for a in params]
    class_weights = len(yt) / (3 * np.bincount(yt, minlength=3))
    step, best, best_epoch, history = 0, -1., 0, []
    best_params = None
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(yt))
        for start in range(0, len(order), 256):
            ids = order[start:start+256]
            batch, target = xt[ids], yt[ids]
            z = batch @ model.w1 + model.b1
            h = np.maximum(z, 0)
            # Small dropout reduces reliance on a single shape/time feature.
            keep = (rng.random(h.shape) >= .1).astype(np.float32) / .9
            dropped = h * keep
            logits = dropped @ model.w2 + model.b2
            e = np.exp(logits - logits.max(axis=1, keepdims=True))
            error = e / e.sum(axis=1, keepdims=True)
            error[np.arange(len(target)), target] -= 1
            error *= (class_weights[target] / len(target))[:, None]
            dh = (error @ model.w2.T) * keep * (z > 0)
            gradients = [batch.T @ dh + 1e-4 * model.w1, dh.sum(axis=0),
                         dropped.T @ error + 1e-4 * model.w2, error.sum(axis=0)]
            step += 1
            for i, (parameter, gradient) in enumerate(zip(params, gradients)):
                first[i] *= .9
                first[i] += .1 * gradient
                second[i] *= .999
                second[i] += .001 * gradient * gradient
                parameter -= .001 * (first[i] / (1 - .9 ** step)) / (np.sqrt(second[i] / (1 - .999 ** step)) + 1e-8)
        if epoch % 2 == 0:
            scores = model.scores(xv)
            candidates = [(fscore(m := metrics(yv, predict(scores.copy(), t))), float(t), m)
                          for t in np.linspace(.5, .99, 50)]
            score, threshold, result = max(candidates, key=lambda item: item[0])
            history.append({"epoch": epoch, "threshold": threshold, "f0_5": score,
                            "precision": result["swipe_precision"], "recall": result["swipe_recall"]})
            if score > best + .001:
                best, best_epoch = score, epoch
                best_params = [a.copy() for a in params]
                model.threshold = threshold
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: validation precision={result['swipe_precision']:.3f}, "
                      f"recall={result['swipe_recall']:.3f}, F0.5={score:.3f}", flush=True)
            if epoch - best_epoch >= 20:
                break
    if best_params is None:
        raise ValueError("Training did not produce a valid checkpoint")
    for parameter, saved in zip(params, best_params):
        parameter[:] = saved
    scores = model.scores(xv)
    curve = [{"threshold": float(t), **metrics(yv, predict(scores.copy(), t))}
             for t in np.linspace(.5, .99, 50)]
    chosen = max(curve, key=fscore)
    model.threshold = chosen["threshold"]
    # Trial permits evaluation without relabeling a failed model as qualified.
    model.metadata.update({"selected_epoch": best_epoch, "seed": 20261020,
                           "validation_target_met": chosen["swipe_precision"] >= .95 and chosen["predicted_swipe_windows"] >= 10,
                           "operating_point": "validation F0.5 maximum, prioritizes precision; experimental"})
    model.save(output)
    report = {"model": model.metadata["architecture"], "dataset": copy.deepcopy(metadata),
              "threshold": model.threshold, "selected_epoch": best_epoch,
              "validation": metrics(yv, predict(scores.copy(), model.threshold)),
              "validation_threshold_curve": curve, "training_history": history,
              "test_exploratory": metrics(y[masks["test"]], predict(model.scores(x[masks["test"]]), model.threshold)),
              "test_note": "This public test split was inspected in the earlier experiment. Fresh local tests are needed for final accuracy claims.",
              "scope": "Sampled overlapping windows, not gesture-event or PowerPoint accuracy",
              "model_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
              "cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest(),
              "local_camera_validated": False}
    report["validation_views"] = "original + mirrored, correlated views" if mirror_augment else "original"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("threshold", "selected_epoch", "validation", "test_exploratory")}, indent=2), flush=True)
    print(f"Saved opt-in trial: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("datasets/ipn_hands/windows.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/swipe_motion_trial.npz"))
    parser.add_argument("--report", type=Path, default=Path("models/swipe_motion_trial_report.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mirror-augment", action="store_true")
    args = parser.parse_args()
    train(args.cache, args.output, args.report, args.epochs, args.mirror_augment)
