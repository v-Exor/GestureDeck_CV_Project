import pytest
from gesturedeck.config import Config
from gesturedeck.mapping import Action, DEFAULT_MAPPING, action_for_gesture
from gesturedeck.swipe import SwipeDetector
from conftest import Rig


@pytest.mark.parametrize('label,expected', [
    ('Closed_Fist', Action.TOGGLE), ('Victory', Action.EXIT), ('Open_Palm', None),
    ('Thumb_Up', None), ('None', None), (None, None),
])
def test_pose_mapping(label, expected):
    assert action_for_gesture(label, DEFAULT_MAPPING) == expected


@pytest.mark.parametrize('direction,expected', [('next', Action.NEXT), ('previous', Action.PREVIOUS)])
def test_screen_view_swipes_start_active_and_trigger_once(rig, direction, expected):
    assert rig.processor.active
    assert rig.swipe(direction) == [expected]
    assert rig.hold('Open_Palm', 3) == []
    assert rig.swipe(direction) == []
    rig.release()
    assert rig.swipe(direction) == [expected]


def test_stationary_palm_and_jitter_never_navigate_or_pause(rig):
    assert rig.hold('Open_Palm', 4) == []
    for i in range(150):
        assert rig.sample('Open_Palm', position=(0.5 + (i % 3) * 0.01, 0.5)) is None
    assert rig.processor.active


def test_fist_toggles_once_per_release_and_does_not_send_navigation(rig):
    assert rig.hold('Closed_Fist', 0.8) == []
    assert rig.hold('Closed_Fist', 4) == [Action.TOGGLE]
    assert not rig.processor.active
    assert rig.swipe() == []
    rig.release()
    assert rig.hold('Closed_Fist') == [Action.TOGGLE]
    assert rig.processor.active
    rig.release()
    assert rig.swipe() == [Action.NEXT]


def test_exit_requires_two_seconds_and_pauses_even_when_already_paused(rig):
    assert rig.hold('Victory', 1.5) == []
    assert rig.hold('Victory', 3) == [Action.EXIT]
    assert not rig.processor.active
    assert rig.swipe() == []
    rig.release()
    assert rig.hold('Victory', 2.2) == [Action.EXIT]
    assert not rig.processor.active


@pytest.mark.parametrize('score', [0.59, 0.0, float('nan'), float('inf'), 1.1, -0.1])
def test_low_or_invalid_confidence_blocks_motion_and_control_poses(rig, score):
    assert rig.swipe(score=score) == []
    assert rig.hold('Closed_Fist', 2, score=score) == []
    assert rig.hold('Victory', 3, score=score) == []
    assert rig.processor.active


def test_confidence_drop_resets_hold_timer(rig):
    assert rig.hold('Closed_Fist', 0.8) == []
    rig.sample('Closed_Fist', score=0.1)
    assert rig.hold('Closed_Fist', 0.7) == []
    assert rig.hold('Closed_Fist', 0.5) == [Action.TOGGLE]


@pytest.mark.parametrize('interruption', ['no_hand', 'low_score', 'label', 'side'])
def test_swipe_cannot_bridge_interrupted_tracking(rig, interruption):
    for x in (0.8, 0.75, 0.7):
        assert rig.sample('Open_Palm', position=(x, 0.5), side='Right') is None
    if interruption == 'no_hand':
        rig.sample(hand=False)
    elif interruption == 'low_score':
        rig.sample('Open_Palm', score=0.1, position=(0.65, 0.5), side='Right')
    elif interruption == 'label':
        rig.sample('Thumb_Up', side='Right')
    else:
        rig.sample('Open_Palm', position=(0.65, 0.5), side='Left')
    for x in (0.6, 0.55, 0.5):
        assert rig.sample('Open_Palm', position=(x, 0.5), side='Right') is None


def test_only_sustained_hand_removal_releases_a_consumed_attempt(rig):
    assert rig.swipe() == [Action.NEXT]
    rig.sample(hand=False)
    assert rig.swipe('previous') == []
    assert rig.hold('Open_Palm', 1, score=0.1) == []
    assert rig.hold('Thumb_Up', 1) == []
    assert rig.swipe('previous') == []
    rig.release()
    assert rig.swipe('previous') == [Action.PREVIOUS]


def test_cooldown_rejection_is_consumed_not_queued():
    rig = Rig(Config(cooldown_seconds=4))
    assert rig.swipe() == [Action.NEXT]
    rig.release()
    assert rig.swipe('previous') == []
    assert rig.hold('Open_Palm', 5) == []
    assert rig.swipe('previous') == []
    rig.release()
    assert rig.swipe('previous') == [Action.PREVIOUS]


def test_manual_pause_and_stalls_discard_pending_input(rig):
    rig.hold('Closed_Fist', 0.5)
    rig.processor.pause()
    assert rig.hold('Closed_Fist', 3) == []
    rig.release()
    assert rig.hold('Closed_Fist') == [Action.TOGGLE]
    rig.release()
    assert rig.processor.active
    assert rig.sample('Open_Palm', position=(0.8, 0.5), delta=2) is None
    assert not rig.processor.active
    assert rig.swipe() == []
    rig.release()
    assert rig.hold('Closed_Fist') == [Action.TOGGLE]


def test_paused_swipes_are_not_saved_for_resume(rig):
    rig.processor.pause()
    rig.release()
    assert rig.swipe() == []
    rig.release()
    assert rig.hold('Closed_Fist') == [Action.TOGGLE]
    rig.release()
    assert rig.sample('Open_Palm', position=(0.2, 0.5)) is None


def test_non_increasing_time_rejected(rig):
    rig.sample()
    from gesturedeck.processor import Observation
    with pytest.raises(ValueError, match='strictly increasing'):
        rig.processor.update(Observation(), rig.now)


@pytest.mark.parametrize('points', [
    [(0.5, 0.2 + i * 0.05) for i in range(9)],  # Vertical
    [(0.8 - i * 0.04, 0.2 + i * 0.05) for i in range(9)],  # Diagonal
    [(0.8 - i * 0.03, 0.5) for i in range(15)],  # Too slow at 0.2 s per sample
])
def test_vertical_diagonal_and_slow_motion_ignored(points):
    swipe = SwipeDetector(Config())
    assert all(swipe.update(point, i * 0.2) is None for i, point in enumerate(points))


def test_single_tracking_jump_and_impossibly_fast_motion_ignored():
    swipe = SwipeDetector(Config())
    assert swipe.update((0.8, 0.5), 0) is None
    assert swipe.update((0.1, 0.5), 0.05) is None
    assert swipe.update((0.09, 0.5), 0.1) is None
    swipe.reset()
    assert all(swipe.update((0.8 - i * 0.07, 0.5), i * 0.01) is None for i in range(7))


@pytest.mark.parametrize('position', [None, (float('nan'), 0.5), (1.2, 0.5), (-0.1, 0.5)])
def test_invalid_position_cannot_bridge_a_swipe(position):
    swipe = SwipeDetector(Config())
    swipe.update((0.8, 0.5), 0)
    swipe.update((0.7, 0.5), 0.1)
    assert swipe.update(position, 0.2) is None
    assert swipe.update((0.5, 0.5), 0.3) is None
    assert swipe.update((0.4, 0.5), 0.4) is None
