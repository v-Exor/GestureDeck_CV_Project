"""OpenCV rendering only; no decisions or keyboard side effects."""

import math
from types import SimpleNamespace

import cv2
import numpy as np

# Standard 21-point hand skeleton. Drawing directly avoids legacy mp.solutions.
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
)


def draw_landmarks(frame, hands) -> None:
    height, width = frame.shape[:2]
    for hand in hands:
        points = [(max(0, min(width - 1, int(p.x * width))),
                   max(0, min(height - 1, int(p.y * height)))) for p in hand]
        for start, end in HAND_CONNECTIONS:
            if end < len(points) and start < len(points):
                cv2.line(frame, points[start], points[end], (70, 220, 70), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 190, 255), -1)


def draw_recorded_landmarks(frame, landmarks) -> bool:
    """Draw this frame's saved image coordinates using the live preview style.

    Saved landmarks already match the recorded pixels, including mirroring.
    Never reuse a previous hand when detection is missing or coordinates invalid.
    The caller supplies a copy when an unannotated frame must be preserved.
    """
    if landmarks is None or len(landmarks) != 21:
        return False
    try:
        hand = [SimpleNamespace(x=float(point[0]), y=float(point[1])) for point in landmarks]
    except (TypeError, ValueError, IndexError):
        return False
    if not all(math.isfinite(point.x) and math.isfinite(point.y) for point in hand):
        return False
    draw_landmarks(frame, [hand])
    return True


def render_preview(frame, lines: list[str]):
    # A separate strip keeps text readable without covering the hand.
    panel_width = max(frame.shape[1], 760)
    panel = np.zeros((len(lines) * 26 + 16, panel_width, 3), dtype=np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (12, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (235, 235, 235), 1, cv2.LINE_AA)
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, panel_width - frame.shape[1],
                               cv2.BORDER_CONSTANT)
    return np.vstack((panel, frame))
