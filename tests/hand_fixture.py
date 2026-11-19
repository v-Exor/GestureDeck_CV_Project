"""Synthetic anatomical hand landmarks for motion-adapter regression tests."""
from types import SimpleNamespace

POINTS = [
    (.50,.85), (.42,.77), (.32,.68), (.23,.58), (.18,.45),
    (.37,.57), (.36,.40), (.35,.28), (.35,.18),
    (.50,.54), (.50,.34), (.50,.20), (.50,.10),
    (.62,.57), (.63,.39), (.64,.26), (.65,.17),
    (.73,.63), (.77,.49), (.79,.39), (.81,.32),
]


def model_result(x=.5, y=.5, scale=.3, label='Open_Palm', score=.9,
                 extended=(0,1,2,3), side='Right'):
    points = list(POINTS)
    for finger, base in enumerate((5,9,13,17)):
        if finger not in extended:
            px, py = points[base]
            points[base+1] = (px, py-.06)
            points[base+2] = (px+.025, py-.02)
            points[base+3] = (px+.03, py+.03)
    center_x = sum(POINTS[i][0] for i in (0,5,9,13,17))/5
    center_y = sum(POINTS[i][1] for i in (0,5,9,13,17))/5
    hand = [SimpleNamespace(x=x+(px-center_x)*scale, y=y+(py-center_y)*scale, z=0)
            for px,py in points]
    return SimpleNamespace(hand_landmarks=[hand],
        gestures=[[SimpleNamespace(category_name=label, score=score)]] if label else [],
        handedness=[[SimpleNamespace(category_name=side)]])
