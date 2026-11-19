"""CPU image inference for checkpoint review; no webcam, GUI, or slide keys."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2

from gesturedeck.config import Config
from gesturedeck.recognizer import GestureRecognizer

ROOT = Path(__file__).resolve().parents[2]
CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
               (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
               (9, 13), (13, 14), (14, 15), (15, 16), (13, 17),
               (0, 17), (17, 18), (18, 19), (19, 20))


def run_image(input_path, output_dir, model_path=None):
    """Use the application's BGR to RGB inference path and save actual results."""
    input_path, output_dir = Path(input_path), Path(output_dir).resolve()
    if not output_dir.is_relative_to((ROOT / 'recordings').resolve()):
        raise ValueError('Save derived capture data beneath this project recordings/ folder.')
    bgr = cv2.imread(str(input_path))
    if bgr is None:
        raise ValueError(f'Cannot decode input image: {input_path}')
    config = Config(model_path=str(model_path or ROOT / 'models/gesture_recognizer.task'),
                    mirror=False)
    with GestureRecognizer(config) as recognizer:
        started = perf_counter()
        observation, hands = recognizer.recognize(bgr)
        inference_ms = (perf_counter() - started) * 1000
    annotated = bgr.copy()
    height, width = bgr.shape[:2]
    for hand in hands:
        points = [(round(p.x * width), round(p.y * height)) for p in hand]
        for a, b in CONNECTIONS:
            cv2.line(annotated, points[a], points[b], (40, 230, 40), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(annotated, point, 3, (0, 0, 0), -1, cv2.LINE_AA)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f'{input_path.stem}_annotated.png'
    if not cv2.imwrite(str(target), annotated):
        raise OSError(f'Cannot write output: {target}')
    result = {
        'input': input_path.name, 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'image_width': width, 'image_height': height, 'hand_count': len(hands),
        'landmark_count': sum(len(hand) for hand in hands),
        'predicted_pose': observation.gesture if observation.hand_present else 'No hand detected',
        'pose_score': observation.confidence if observation.hand_present else None,
        'inference_ms': round(inference_ms, 3), 'output': target.name,
        'scope': 'Single image hand landmarks and static pose; no temporal swipe accuracy claim',
        'preprocessing': 'Shared recognizer BGR to RGB conversion; original dimensions; no mirroring',
    }
    (output_dir / f'{input_path.stem}_result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    print(json.dumps(run_image(args.input, args.output, args.model), indent=2))


if __name__ == '__main__':
    main()
