import pytest
from gesturedeck.config import Config
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor
from gesturedeck.recognizer import observation_from_result
from hand_fixture import model_result


@pytest.mark.parametrize('labels', [
    [('Open_Palm', .75), ('Open_Palm', .40)] * 6,
    [('None', .4), ('Thumb_Up', .35), ('Open_Palm', .3)] * 4,
    [(None, 0)] * 12,
])
def test_open_landmarks_keep_motion_when_pose_classifier_fluctuates(labels):
    processor = GestureProcessor(Config())
    events = []
    for i, (label,score) in enumerate(labels):
        result = model_result(x=.2+i*.04, label=label, score=score)
        event = processor.update(observation_from_result(result), i*.05)
        if event is not None: events.append(event)
    assert events == [Action.PREVIOUS]


@pytest.mark.parametrize('scale,travel', [(.12,.13), (.30,.20), (.50,.32)])
def test_hand_relative_swipe_distance_at_different_camera_distances(scale,travel):
    processor = GestureProcessor(Config())
    events = []
    for i in range(12):
        result = model_result(x=.35+travel*i/11, scale=scale)
        event = processor.update(observation_from_result(result), i*.05)
        if event is not None: events.append(event)
    assert events == [Action.PREVIOUS]


@pytest.mark.parametrize('extended', [(), (0,), (0,1)])
def test_moving_closed_or_two_finger_hand_cannot_navigate_even_if_pose_label_wrong(extended):
    processor = GestureProcessor(Config())
    events = []
    for i in range(12):
        result = model_result(x=.2+i*.04, label='Open_Palm', extended=extended)
        event = processor.update(observation_from_result(result), i*.05)
        if event is not None: events.append(event)
    assert events == []


def test_bad_pose_label_on_geometrically_open_hand_cannot_pause_or_exit():
    processor = GestureProcessor(Config())
    for i in range(70):
        label = 'Closed_Fist' if i < 25 else 'Victory'
        observation = observation_from_result(model_result(label=label, score=.99))
        assert processor.update(observation, i*.05) is None
    assert processor.active


def test_face_or_body_metadata_does_not_change_hand_motion_decisions():
    processor = GestureProcessor(Config())
    events = []
    for i in range(12):
        result = model_result(x=.2+i*.04, score=.2)
        result.face_landmarks = ['irrelevant face data']
        result.pose_landmarks = ['irrelevant body data']
        event = processor.update(observation_from_result(result), i*.05)
        if event is not None: events.append(event)
    assert events == [Action.PREVIOUS]


def test_a_tiny_hand_is_rejected_with_a_specific_message():
    processor = GestureProcessor(Config())
    for i in range(12):
        result = model_result(x=.2+i*.04, scale=.02)
        assert processor.update(observation_from_result(result), i*.05) is None
    assert 'too small' in processor.status


def test_degenerate_landmarks_cannot_make_open_hand():
    result = model_result()
    for point in result.hand_landmarks[0]:
        point.x = point.y = .5
    observation = observation_from_result(result)
    assert not observation.open_hand


def test_geometry_and_mirrored_direction_survive_screen_aspect_ratio():
    for aspect in (4/3, 16/9, 1.0):
        result = model_result()
        observation = observation_from_result(result, aspect_ratio=aspect)
        assert observation.open_hand
        assert observation.extended_fingers == 4


def test_one_brief_geometry_flicker_does_not_discard_valid_motion():
    processor = GestureProcessor(Config())
    events = []
    for i in range(12):
        observation = observation_from_result(model_result(x=.2+i*.025,
            extended=(0,1) if i==3 else (0,1,2,3), label='None', score=.2))
        event = processor.update(observation, i*.05)
        if event is not None: events.append(event)
    assert events == [Action.PREVIOUS]


def test_true_hand_loss_and_invalid_landmarks_still_break_motion():
    from gesturedeck.processor import Observation
    for interrupted in ('missing', 'invalid'):
        processor = GestureProcessor(Config())
        for i, x in enumerate((.2,.23,.26)):
            assert processor.update(observation_from_result(model_result(x=x)), i*.05) is None
        if interrupted == 'missing':
            observation = Observation()
        else:
            result = model_result(x=.29)
            result.hand_landmarks[0][0].x = float('nan')
            observation = observation_from_result(result)
        assert processor.update(observation, .15) is None
        for i, x in enumerate((.32,.34,.36)):
            assert processor.update(observation_from_result(model_result(x=x)), .2+i*.05) is None


def test_real_geometry_handedness_label_flicker_does_not_break_continuous_motion():
    processor = GestureProcessor(Config())
    events = []
    for i in range(12):
        result = model_result(x=.2+i*.025, side='Left' if i%2 else 'Right')
        event = processor.update(observation_from_result(result), i*.05)
        if event is not None: events.append(event)
    assert events == [Action.PREVIOUS]
