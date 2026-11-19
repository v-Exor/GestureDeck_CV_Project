"""Time-based open-hand motion detection, independent of camera and keyboard APIs."""

from collections import deque
import math
from statistics import median

from gesturedeck.config import Config
from gesturedeck.mapping import Action


class SwipeDetector:
    def __init__(self, config: Config):
        self.config = config
        self.points = deque()
        self.status = "Show an open hand and move it sideways"

    def reset(self):
        self.points.clear()

    def interrupt(self, now):
        """A briefly curled/foreshortened finger is not actual tracking loss.

        Never add a sample during this grace period or count it as release.
        True hand loss, invalid coordinates, and tracking jumps still reset.
        """
        if self.points and now - self.points[-1][0] > self.config.swipe_tracking_grace_seconds:
            self.reset()

    def update(self, position: tuple[float, float] | None, now: float,
               hand_scale: float | None = None) -> Action | None:
        if position is None or len(position) != 2 or not all(
            math.isfinite(p) and 0 <= p <= 1 for p in position
        ):
            self.reset()
            self.status = "Hand position unavailable: keep the whole hand visible"
            return None
        if hand_scale is not None and (not math.isfinite(hand_scale)
                                      or hand_scale < self.config.swipe_min_hand_size):
            self.reset()
            self.status = "Hand too small to track reliably: move closer to the camera"
            return None
        x, y = position
        if self.points:
            last, px, py, _ = self.points[-1]
            if now - last > self.config.swipe_tracking_grace_seconds:
                self.reset()
            if math.hypot(x - px, y - py) > self.config.swipe_max_step:
                self.reset()  # Tracking jumps cannot become swipes.
                self.status = "Tracking jumped: start a new swipe"
        self.points.append((now, x, y, hand_scale))
        while self.points and now - self.points[0][0] > self.config.swipe_max_seconds:
            self.points.popleft()
        if len(self.points) < 3:
            self.status = "Hand tracked: begin a horizontal swipe"
            return None
        start, sx, _, _ = self.points[0]
        if now - start < self.config.swipe_min_seconds:
            return None
        dx = x - sx
        scales = [p[3] for p in self.points if p[3] is not None]
        required = (min(self.config.swipe_min_distance, max(self.config.swipe_distance_floor,
                    median(scales) * self.config.swipe_hand_distance)) if scales
                    else self.config.swipe_min_distance)
        self.status = f"Hand tracked: swipe travel {min(100, abs(dx) / required * 100):.0f}% / 100%; keep moving sideways"
        if abs(dx) < required:
            return None
        ys = [p[2] for p in self.points]
        samples = list(self.points)
        path = sum(abs(b[1] - a[1]) for a, b in zip(samples, samples[1:]))
        if max(ys) - min(ys) > min(self.config.swipe_max_vertical, abs(dx) * 0.7):
            self.status = "Swipe rejected: move sideways with less up/down motion"
            return None
        if not path or abs(dx) / path < self.config.swipe_min_consistency:
            self.status = "Swipe rejected: use one direction before lowering your hand"
            return None
        self.reset()
        self.status = "Swipe accepted: lower hand before another attempt"
        return Action.NEXT if dx < 0 else Action.PREVIOUS
