"""Train/evaluate a small supervised temporal model on official IPN HandS data.

The official test split is never used for fitting or threshold selection.
Whole subject/session groups stay together. No synthetic training data.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

from download_ipn_hands import FILES
from gesturedeck.learned_swipe import (
    FEATURE_SIZE, FRAME_SIZE, LABELS, WINDOWS, MotionModel, frame_vector, sequence_features,
)


def subject_id(video):
    return video.rsplit("_", 2)[0]


def checked_archive(path, expected):
    with path.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"Unrecognized dataset bytes: {path}")
    return zipfile.ZipFile(path)


def split_videos(train, test):
    train_groups = {subject_id(v) for v in train}
    test_groups = {subject_id(v) for v in test}
    if train_groups & test_groups or set(train) & set(test):
        raise ValueError("Official split contains overlapping subject/video groups")
    ordered = sorted(train_groups, key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    validation_groups = set(ordered[:max(1, len(ordered) // 5)])
    return {v: ("validation" if subject_id(v) in validation_groups else "train") for v in train} | {
        v: "test" for v in test}


def prepare(directory, cache):
    rng = np.random.default_rng(20261020)
    features, labels, splits, videos = [], [], [], []
    audit = {7: [], 8: []}
    stats = Counter()
    with checked_archive(directory / FILES["annotations"][0], FILES["annotations"][2]) as annotations, \
            checked_archive(directory / FILES["hands"][0], FILES["hands"][2]) as landmarks:
        train = annotations.read("splits/train_list.txt").decode().split()
        test = annotations.read("splits/test_list.txt").decode().split()
        assignments = split_videos(train, test)
        members = {Path(name).stem: name for name in landmarks.namelist() if name.endswith(".json")}
        if set(assignments) != set(members):
            raise ValueError("Annotation and landmark videos do not match")
        for index, video in enumerate(sorted(assignments)):
            frames = json.loads(landmarks.read(members[video]))
            data = np.zeros((len(frames), FRAME_SIZE))
            for name, frame in frames.items():
                number = int(Path(name).stem.rsplit("_", 1)[1])
                if not 1 <= number <= len(data):
                    raise ValueError(f"Invalid frame index: {name}")
                hands = frame.get("hand_landmarks", [])
                if len(hands) == 1:
                    points = hands[0].get("landmarks", {})
                    if all(str(i) in points for i in range(21)):
                        data[number - 1] = frame_vector([points[str(i)] for i in range(21)])
            target = np.full(len(data), -1, dtype=np.int16)
            for line in annotations.read(f"temporal_annotations/{video}.txt").decode().splitlines():
                if not line.strip():
                    continue
                label, first, last = map(int, line.split(","))
                if label not in range(15) or not 1 <= first <= last <= len(data) or np.any(target[first-1:last] != -1):
                    raise ValueError(f"Invalid/overlapping temporal annotation: {video}: {line}")
                target[first-1:last] = label
                if label in audit and assignments[video] == "train":
                    valid = data[first-1:last]
                    valid = valid[valid[:, 3] > 0]
                    if len(valid) >= 4:
                        audit[label].append(float(valid[-1, 0] - valid[0, 0]))
            if np.any(target == -1):
                raise ValueError(f"Missing annotation coverage: {video}")
            for end in range(12, len(data) + 1, 6):
                for window in WINDOWS:
                    size = round(window * 30)
                    start = end - size
                    if start < 0:
                        continue
                    y = target[start:end]
                    counts = [np.count_nonzero(y == label) for label in (7, 8)]
                    if max(counts) >= len(y) * .6:
                        source_label = 7 + int(counts[1] > counts[0])
                    elif max(counts):
                        continue  # Ambiguous boundaries are not negative training labels.
                    else:
                        source_label = 0
                        if rng.random() > .12:
                            continue
                    sample = sequence_features(np.arange(start, end) / 30, data[start:end])
                    if sample is None:
                        stats[f"rejected_quality_{source_label}"] += 1
                        continue
                    features.append(sample)
                    labels.append(source_label)
                    splits.append(assignments[video])
                    videos.append(video)
            stats["frames"] += len(data)
            stats["tracked_frames"] += int(np.count_nonzero(data[:, 3]))
            if index % 20 == 0:
                print(f"Prepared {index+1}/200 videos; {len(labels):,} windows", flush=True)
    medians = {str(k): float(np.median(v)) for k, v in audit.items()}
    if medians["7"] * medians["8"] >= 0 or min(abs(v) for v in medians.values()) < .025:
        raise ValueError(f"Source direction labels need manual auditing: {medians}")
    mapping = {0: 0, **{int(k): (1 if dx < 0 else 2) for k, dx in medians.items()}}
    metadata = {"dataset": "IPN HandS / MediaPipe Hands", "license": "CC BY 4.0",
                "source": "https://github.com/GibranBenitez/IPN-hand/blob/master/README_HandS.md",
                "files_sha256": {k: v[2] for k, v in FILES.items()},
                "annotation_checksum_note": "Observed official download differs from README's published annotation hash",
                "source_class_to_image_direction": {str(k): LABELS[v] for k, v in mapping.items()},
                "training_source_class_median_dx": medians,
                "split_groups": {s: sorted({subject_id(v) for v in assignments if assignments[v] == s})
                                 for s in ("train", "validation", "test")},
                "quality_counts": dict(stats), "sample_stride_frames": 6,
                "negative_sampling_probability": .12,
                "window_label": "source swipe annotation covers at least 60%; other has zero swipe overlap"}
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x=np.asarray(features), y=np.asarray([mapping[v] for v in labels]),
                        split=np.asarray(splits), video=np.asarray(videos), metadata=json.dumps(metadata))
    print(f"Cached {len(labels):,} real-data windows; direction audit: {medians}", flush=True)


def predict(scores, threshold):
    result = scores.argmax(axis=1)
    result[scores.max(axis=1) < threshold] = 0
    return result


def metrics(y, prediction):
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (y, prediction), 1)
    tp = sum(int(matrix[i, i]) for i in (1, 2))
    predicted = int(np.count_nonzero(prediction))
    actual = int(np.count_nonzero(y))
    return {"labels": list(LABELS), "confusion_matrix_true_rows": matrix.tolist(),
            "swipe_precision": tp / predicted if predicted else 0.,
            "swipe_recall": tp / actual if actual else 0.,
            "false_positive_other_windows": int(matrix[0, 1:].sum()),
            "wrong_direction_windows": int(matrix[1, 2] + matrix[2, 1]),
            "predicted_swipe_windows": predicted, "true_swipe_windows": actual,
            "windows": len(y)}


def train(cache, output, report_path):
    with np.load(cache, allow_pickle=False) as data:
        x, y, split = data["x"], data["y"], data["split"]
        metadata = json.loads(str(data["metadata"]))
    if x.shape != (len(y), FEATURE_SIZE) or not np.isfinite(x).all():
        raise ValueError("Invalid feature cache")
    train_mask, val_mask, test_mask = (split == key for key in ("train", "validation", "test"))
    if any(len(np.unique(y[mask])) != 3 for mask in (train_mask, val_mask, test_mask)):
        raise ValueError("Each split needs both swipe directions and negatives")
    rng = np.random.default_rng(42)
    mean, std = x[train_mask].mean(axis=0), np.maximum(x[train_mask].std(axis=0), .05)
    projection = rng.normal(0, 1 / np.sqrt(FEATURE_SIZE), (FEATURE_SIZE, 384))
    phase = rng.uniform(0, 2 * np.pi, 384)
    model = MotionModel(mean, std, projection, phase, np.zeros((1+FEATURE_SIZE+384, 3)), 0., metadata)
    a = model.design(x[train_mask])
    target = np.eye(3)[y[train_mask]]
    weights = len(target) / (3 * np.bincount(y[train_mask], minlength=3)[y[train_mask]])
    penalty = np.eye(a.shape[1]) * .5
    penalty[0, 0] = 1e-6
    print(f"Fitting supervised sequence classifier on {len(a):,} training windows...", flush=True)
    model.weights = np.linalg.solve(a.T @ (a * weights[:, None]) + penalty,
                                    a.T @ (target * weights[:, None]))
    scores = model.scores(x[val_mask])
    choices = []
    validation_curve = []
    for threshold in np.linspace(.4, 1.6, 121):
        result = metrics(y[val_mask], predict(scores.copy(), threshold))
        validation_curve.append({"threshold": float(threshold), **result})
        if result["swipe_precision"] >= .95 and result["predicted_swipe_windows"] >= 10:
            choices.append((result["swipe_recall"], float(threshold)))
    if choices:
        model.threshold = max(choices)[1]
        threshold_note = "Maximum validation recall among thresholds reaching 95% window precision"
    else:
        model.threshold = 1.6
        threshold_note = "Validation precision target not reached; conservative experimental threshold; not approved for live use"
    report = {"model": "class-balanced ridge with random Fourier temporal features", "seed": 42,
              "scope": "Sampled overlapping window classification; NOT independent gesture-event or slide accuracy",
              "threshold": model.threshold, "threshold_selection": threshold_note,
              "dataset": metadata,
              "counts": {s: dict(Counter(LABELS[int(v)] for v in y[split == s]))
                         for s in ("train", "validation", "test")},
              "validation": metrics(y[val_mask], predict(scores.copy(), model.threshold)),
              "validation_threshold_curve": validation_curve,
              "test": metrics(y[test_mask], predict(model.scores(x[test_mask]), model.threshold)),
              "limitations": ["No local webcam swipe validation yet", "Not an event-level continuous evaluation",
                              "Source Throw gestures may differ from user's open-palm swipes",
                              "MediaPipe Hands extraction differs from the installed Tasks bundle",
                              "Source annotation ZIP differs from published hash; observed download is pinned"]}
    metadata["threshold_selection"] = threshold_note
    metadata["experimental"] = True
    metadata["validation_target_met"] = bool(choices)
    model.save(output)
    report["model_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("threshold", "threshold_selection", "counts", "validation", "test")}, indent=2), flush=True)
    print(f"Saved {output}; report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("datasets/ipn_hands"))
    parser.add_argument("--cache", type=Path, default=Path("datasets/ipn_hands/windows.npz"))
    parser.add_argument("--output", type=Path, default=Path("models/swipe_motion.npz"))
    parser.add_argument("--report", type=Path, default=Path("models/swipe_motion_report.json"))
    args = parser.parse_args()
    if not args.cache.exists():
        prepare(args.dataset, args.cache)
    train(args.cache, args.output, args.report)


if __name__ == "__main__":
    main()
