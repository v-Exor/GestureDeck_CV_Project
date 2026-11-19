"""Recompute validation metrics without training or evaluating the test split."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gesturedeck.learned_swipe import MotionModel
from train_swipe_model import metrics, predict, subject_id


def evaluate(cache, model_path):
    with np.load(cache, allow_pickle=False) as data:
        split, video = data['split'], data['video']
        groups = {s: {subject_id(str(v)) for v in video[split == s]}
                  for s in ('train', 'validation', 'test')}
        if any(groups[a] & groups[b] for a, b in (('train', 'validation'),
                                                ('train', 'test'), ('validation', 'test'))):
            raise ValueError('Subject group leakage')
        mask = split == 'validation'
        x, y = data['x'][mask], data['y'][mask]
    model = MotionModel.load(model_path)
    result = metrics(y, predict(model.scores(x), model.threshold))
    matrix = np.asarray(result['confusion_matrix_true_rows'])
    tp = np.diag(matrix)
    f1 = 2 * tp / (matrix.sum(axis=0) + matrix.sum(axis=1))
    return {
        'split': 'validation', 'threshold': model.threshold, **result,
        'accuracy': float(tp.sum() / matrix.sum()), 'macro_f1': float(f1.mean()),
        'model_sha256': hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
        'cache_sha256': hashlib.sha256(Path(cache).read_bytes()).hexdigest(),
        'group_leakage_found': False, 'test_predictions_computed': False,
        'scope': 'Overlapping sampled windows; model and threshold selected on this validation set',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('datasets/ipn_hands/windows.npz'))
    parser.add_argument('--model', type=Path, default=Path('models/swipe_motion_trial.npz'))
    parser.add_argument('--output', type=Path, default=Path('recordings/milestone2_20261024/validation.json'))
    args = parser.parse_args()
    result = evaluate(args.cache, args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
