"""Current MediaPipe Tasks API, synchronous VIDEO mode, explicit CPU delegate."""

from pathlib import Path
from time import perf_counter_ns

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from gesturedeck.config import Config
from gesturedeck.hand import hand_geometry, palm_facing_score
from gesturedeck.processor import Observation


class VideoTimestamp:
    """Monotonic milliseconds; even two frames in one millisecond increase."""

    def __init__(self) -> None:
        self._start = perf_counter_ns()
        self._last = -1

    def next(self) -> int:
        elapsed = (perf_counter_ns() - self._start) // 1_000_000
        self._last = max(elapsed, self._last + 1)
        return self._last


def observation_from_result(result, *, mirror=True, aspect_ratio=4 / 3) -> Observation:
    if not result.hand_landmarks:
        return Observation()
    categories = result.gestures[0] if result.gestures else []
    top = max(categories, key=lambda category: category.score) if categories else None
    hand = result.hand_landmarks[0]
    position, scale, extended = hand_geometry(hand, aspect_ratio)
    if position is not None and not mirror:
        position = (1 - position[0], position[1])
    sides = getattr(result, "handedness", [])
    side = sides[0][0].category_name if sides and sides[0] else None
    def coordinates(points):
        return tuple((float(p.x), float(p.y), float(getattr(p, "z", 0))) for p in points) if len(points) == 21 else ()
    world = getattr(result, "hand_world_landmarks", [])
    return Observation(top.category_name if top else None, float(top.score) if top else 0.0,
                       True, position, side, extended >= 3, scale, extended,
                       palm_facing_score(hand, mirror=mirror, aspect_ratio=aspect_ratio),
                       coordinates(hand), coordinates(world[0]) if world else ())


class GestureRecognizer:
    def __init__(self, config: Config) -> None:
        self._mirror = config.mirror
        path = Path(config.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model not found: {path.resolve()}. Run: python scripts/download_model.py"
            )
        options = vision.GestureRecognizerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(path.resolve()),
                delegate=python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=config.detection_confidence,
            min_hand_presence_confidence=config.presence_confidence,
            min_tracking_confidence=config.tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._timestamps = VideoTimestamp()

    def __enter__(self):
        return self

    def recognize(self, bgr_frame):
        # OpenCV produces BGR; the task expects RGB. Preview and inference share
        # the same mirrored coordinates so the landmark overlay stays aligned.
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = self._recognizer.recognize_for_video(image, self._timestamps.next())
        height, width = bgr_frame.shape[:2]
        return observation_from_result(result, mirror=self._mirror, aspect_ratio=width / height), result.hand_landmarks

    def __exit__(self, *_):
        self._recognizer.close()
