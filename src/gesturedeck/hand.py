"""Palm geometry from the hand model; face pixels are never inputs to these rules."""

import math


def palm_facing_score(hand, *, mirror=True, aspect_ratio=4 / 3):
    """Signed projected palm area, normalized for distance, for the RIGHT hand.

    Positive means palm toward camera in the mirrored presenter view. Negative
    means back toward camera. Near zero is edge-on/uncertain, never an endpoint.
    Do not multiply by the model's handedness: that label can flicker on a turn.
    This experimental convention is intentionally restricted to the right hand.
    """
    if len(hand) != 21 or not math.isfinite(aspect_ratio) or aspect_ratio <= 0:
        return None
    if not all(math.isfinite(p.x) and math.isfinite(p.y) for p in hand):
        return None
    points = [(p.x, p.y / aspect_ratio) for p in hand]
    wrist, index, middle, pinky = (points[i] for i in (0, 5, 9, 17))
    scale = max(math.dist(index, pinky), math.dist(wrist, middle))
    if scale < 1e-6:
        return None
    a = (index[0] - wrist[0], index[1] - wrist[1])
    b = (pinky[0] - wrist[0], pinky[1] - wrist[1])
    area = (a[0] * b[1] - a[1] * b[0]) / (scale * scale)
    return area if mirror else -area


def hand_geometry(hand, aspect_ratio=4 / 3):
    """Return center, hand size in frame-width units, and extended-finger count.

    Correct the image aspect ratio before measuring lengths/angles. Three of the
    four long fingers must be extended for a swipe. Thumb position is irrelevant.
    Invalid/incomplete landmarks fail closed; they cannot become an open hand.
    """
    if len(hand) != 21 or not all(math.isfinite(p.x) and math.isfinite(p.y) for p in hand):
        return None, None, 0
    points = [(p.x, p.y / aspect_ratio) for p in hand]
    palm = [hand[i] for i in (0, 5, 9, 13, 17)]
    center = (sum(p.x for p in palm) / 5, sum(p.y for p in palm) / 5)
    distance = lambda a, b: math.hypot(a[0] - b[0], a[1] - b[1])
    scale = max(distance(points[5], points[17]), distance(points[0], points[9]))
    if scale < 1e-6:
        return center, scale, 0
    extended = 0
    for base in (5, 9, 13, 17):
        mcp, pip, tip = points[base], points[base + 1], points[base + 3]
        a = (mcp[0] - pip[0], mcp[1] - pip[1])
        b = (tip[0] - pip[0], tip[1] - pip[1])
        product = math.hypot(*a) * math.hypot(*b)
        if (product > 1e-10 and (a[0] * b[0] + a[1] * b[1]) / product < -0.5
                and distance(tip, points[0]) > 1.05 * distance(pip, points[0])):
            extended += 1
    return center, scale, extended
