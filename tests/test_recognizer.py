from types import SimpleNamespace

from gesturedeck.recognizer import VideoTimestamp, observation_from_result


def test_video_timestamps_increase_even_when_clock_does_not(monkeypatch):
    monkeypatch.setattr("gesturedeck.recognizer.perf_counter_ns", lambda: 100)
    timestamps = VideoTimestamp()
    assert [timestamps.next() for _ in range(5)] == [0, 1, 2, 3, 4]


def test_empty_result_means_no_hand():
    observation = observation_from_result(SimpleNamespace(hand_landmarks=[], gestures=[]))
    assert not observation.hand_present and observation.gesture is None


def test_hand_without_categories_is_ignored():
    observation = observation_from_result(SimpleNamespace(hand_landmarks=[[object()]], gestures=[[]]))
    assert observation.hand_present and observation.gesture is None


def test_result_uses_highest_score_and_exact_category():
    categories = [SimpleNamespace(category_name="Victory", score=0.9),
                  SimpleNamespace(category_name="None", score=0.1)]
    observation = observation_from_result(SimpleNamespace(hand_landmarks=[[object()]], gestures=[categories]))
    assert observation.hand_present
    assert observation.gesture == "Victory" and observation.confidence == 0.9


def test_palm_center_uses_stable_base_landmarks_and_mirror_direction():
    import pytest
    points = [SimpleNamespace(x=0.8, y=0.5) for _ in range(21)]
    for index in (0, 5, 9, 13, 17):
        points[index].x = 0.2
    result = SimpleNamespace(hand_landmarks=[points],
        gestures=[[SimpleNamespace(category_name='Open_Palm', score=0.9)]],
        handedness=[[SimpleNamespace(category_name='Right')]])
    mirrored = observation_from_result(result)
    unmirrored = observation_from_result(result, mirror=False)
    assert mirrored.position == pytest.approx((0.2, 0.5))
    assert unmirrored.position == pytest.approx((0.8, 0.5))
    assert mirrored.handedness == 'Right'


def test_invalid_palm_geometry_does_not_produce_motion_coordinates():
    points = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
    points[0].x = float('nan')
    result = SimpleNamespace(hand_landmarks=[points],
        gestures=[[SimpleNamespace(category_name='Open_Palm', score=0.9)]])
    assert observation_from_result(result).position is None


def test_real_result_adapter_and_processor_keep_directions_with_mirror_disabled():
    from gesturedeck.config import Config
    from gesturedeck.processor import GestureProcessor
    from gesturedeck.mapping import Action
    for mirror in (True, False):
        for target, xs in ((Action.NEXT, [0.8-i*0.05 for i in range(12)]),
                           (Action.PREVIOUS, [0.2+i*0.05 for i in range(12)])):
            processor = GestureProcessor(Config(mirror=mirror))
            events = []
            for i, x in enumerate(xs):
                from hand_fixture import model_result
                result = model_result(x=x)
                if not mirror:
                    for point in result.hand_landmarks[0]:
                        point.x = 1-point.x
                event = processor.update(observation_from_result(result, mirror=mirror), i*0.05)
                if event is not None:
                    events.append(event)
            assert events == [target]
