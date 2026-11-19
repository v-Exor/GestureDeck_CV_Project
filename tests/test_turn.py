from dataclasses import replace

import pytest

from gesturedeck.config import Config
from gesturedeck.hand import palm_facing_score
from gesturedeck.mapping import Action
from gesturedeck.processor import GestureProcessor, Observation
from gesturedeck.recognizer import observation_from_result
from hand_fixture import model_result


def observation(facing=0.5, **kwargs):
    return Observation(hand_present=True, open_hand=True, position=(0.5, 0.5),
                       hand_scale=0.15, extended_fingers=4, palm_facing=facing, **kwargs)


def hold(processor, obs, start, frames=9):
    return [action for i in range(frames)
            if (action := processor.update(obs, round(start + i * 0.05, 4))) is not None]


def test_complete_turns_work_both_ways_and_holding_never_repeats():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    assert hold(processor, observation(), 0, 20) == []
    assert hold(processor, observation(-0.5), 1, 24) == [Action.NEXT]
    assert hold(processor, observation(0.5), 2.2, 24) == [Action.PREVIOUS]
    assert hold(processor, observation(), 3.4, 20) == []


def test_showing_back_first_is_only_a_baseline():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    assert hold(processor, observation(-0.5), 0) == []
    assert hold(processor, observation(0.5), 0.5) == [Action.PREVIOUS]


@pytest.mark.parametrize("facing", [0, 0.1, -0.1, None, float("nan")])
def test_edge_on_or_invalid_orientation_does_not_navigate(facing):
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0)
    assert hold(processor, observation(facing), 0.5, 35) == []


def test_brief_wrong_side_and_actual_tracking_loss_do_not_turn():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0)
    assert hold(processor, observation(-0.5), 0.5, 2) == []
    assert hold(processor, observation(0), 0.6, 3) == []
    assert hold(processor, observation(), 0.8) == []
    processor.update(Observation(), 1.3)
    assert hold(processor, observation(-0.5), 1.4) == []


def test_handedness_flicker_does_not_reverse_palm_back_mapping():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(handedness="Right"), 0)
    assert hold(processor, observation(-0.5, handedness="Left"), 0.5) == [Action.NEXT]


def test_cooldown_consumes_turn_instead_of_sending_late():
    processor = GestureProcessor(Config(navigation_mode="palm_turn", turn_rearm_seconds=2, turn_rearm_policy="all_commands"))
    hold(processor, observation(), 0)
    assert hold(processor, observation(-0.5), 0.5) == [Action.NEXT]
    assert hold(processor, observation(), 1, 50) == []
    assert hold(processor, observation(-0.5), 3.5) == [Action.NEXT]


@pytest.mark.parametrize("label,hold_frames,action", [("Closed_Fist", 24, Action.TOGGLE), ("Victory", 45, Action.EXIT)])
def test_control_poses_keep_holds_and_pause_navigation(label, hold_frames, action):
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0)
    pose = replace(observation(-0.5, gesture=label, confidence=0.99), open_hand=False)
    assert hold(processor, pose, 0.5, hold_frames) == [action]
    assert not processor.active
    start = round(0.5 + hold_frames * 0.05, 4)
    assert hold(processor, observation(-0.5), start, 12) == []


def test_pause_resume_requires_release_and_new_turn_baseline():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    fist = replace(observation(gesture="Closed_Fist", confidence=0.99), open_hand=False)
    assert hold(processor, fist, 0, 25) == [Action.TOGGLE]
    hold(processor, Observation(), 1.3, 8)
    assert hold(processor, fist, 1.8, 25) == [Action.TOGGLE]
    hold(processor, Observation(), 3.1, 8)
    assert hold(processor, observation(-0.5), 3.6) == []
    assert hold(processor, observation(), 4.1) == [Action.PREVIOUS]


def test_tracking_jump_and_low_hand_size_reset_the_baseline():
    for bad in (replace(observation(-0.5), position=(0.9, 0.5)),
                replace(observation(-0.5), hand_scale=0.001)):
        processor = GestureProcessor(Config(navigation_mode="palm_turn"))
        hold(processor, observation(), 0)
        assert processor.update(bad, 0.5) is None
        assert hold(processor, observation(-0.5), 0.6) == []


def test_slow_uncertain_turn_times_out_without_a_command():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0)
    hold(processor, observation(0), 0.5, 60)
    assert hold(processor, observation(-0.5), 3.6) == []


def test_orientation_adapter_is_mirror_normalized_and_exports_landmarks():
    for mirror in (True, False):
        palm = model_result()
        if not mirror:
            for point in palm.hand_landmarks[0]:
                point.x = 1 - point.x
        adapted = observation_from_result(palm, mirror=mirror)
        assert adapted.palm_facing > 0.2
        assert len(adapted.landmarks) == 21
        for point in palm.hand_landmarks[0]:
            point.x = 1 - point.x
        assert observation_from_result(palm, mirror=mirror).palm_facing < -0.2
    assert palm_facing_score([]) is None


def test_config_and_cli_recording_validation():
    from gesturedeck.app import parse_args
    args = parse_args(["--gui", "--navigation", "palm_turn", "--record", "recordings"])
    assert args.navigation_mode == "palm_turn" and args.record_directory == "recordings"
    with pytest.raises(SystemExit):
        parse_args(["--record", "recordings"])
    for kwargs in ({"navigation_mode": "bad"}, {"turn_hold_seconds": 0},
                   {"turn_min_facing": 1}, {"record_directory": None}):
        with pytest.raises(ValueError):
            Config(**kwargs)


def test_open_geometry_with_wrong_control_label_cannot_pause_or_exit():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    assert hold(processor, observation(gesture="Closed_Fist", confidence=0.99), 0, 30) == []
    assert processor.active
    assert hold(processor, observation(gesture="Victory", confidence=0.99), 1.5, 50) == []
    assert processor.active


def test_camera_stall_pauses_turns_and_discards_starting_side():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0)
    assert processor.update(observation(-0.5), 1.2) is None
    assert not processor.active
    assert hold(processor, observation(-0.5), 1.3) == []


def test_fast_endpoint_confirms_in_three_good_frames_not_one_or_two():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert processor.update(observation(-0.5), 0.4) is None
    assert processor.update(observation(-0.5), 0.45) is None
    assert processor.update(observation(-0.5), 0.5) == Action.NEXT


@pytest.mark.parametrize("interruption", ["finger", "missing", "edge"])
def test_one_brief_interruption_keeps_endpoint_evidence(interruption):
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert processor.update(observation(-0.5), 0.4) is None
    assert processor.update(observation(-0.5), 0.45) is None
    weak = {"finger": replace(observation(-0.5), open_hand=False, extended_fingers=2),
            "missing": Observation(), "edge": observation(-0.05)}[interruption]
    assert processor.update(weak, 0.5) is None
    assert processor.update(observation(-0.5), 0.55) == Action.NEXT


def test_single_lost_frame_mid_turn_keeps_starting_side():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert processor.update(Observation(), 0.4) is None
    assert hold(processor, observation(-0.5), 0.45, 4) == [Action.NEXT]


def test_longer_loss_or_invalid_coordinates_discard_turn():
    for weak in (Observation(), replace(observation(-0.5), position=(float("nan"), 0.5))):
        processor = GestureProcessor(Config(navigation_mode="palm_turn"))
        hold(processor, observation(), 0, 8)
        processor.update(weak, 0.4)
        processor.update(weak, 0.5)
        assert hold(processor, observation(-0.5), 0.55, 8) == []


def test_repeated_two_frame_orientation_blips_do_not_accumulate_a_turn():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    for i in range(24):
        facing = 0.5 if i % 3 == 2 else -0.5
        assert processor.update(observation(facing), round(0.4 + i * 0.05, 4)) is None


def test_one_second_pause_consumes_return_motion_and_does_not_queue_commands():
    processor = GestureProcessor(Config(navigation_mode="palm_turn", turn_rearm_policy="all_commands"))
    hold(processor, observation(), 0, 8)
    assert hold(processor, observation(-0.5), 0.4, 8) == [Action.NEXT]  # Accepted at .5.
    assert hold(processor, observation(), 0.8, 22) == []  # Returning during pause stays consumed.
    assert processor.active
    assert hold(processor, observation(-0.5), 1.9, 8) == [Action.NEXT]


def test_turn_straddling_pause_deadline_can_complete_once_after_deadline():
    processor = GestureProcessor(Config(navigation_mode="palm_turn", turn_rearm_policy="all_commands"))
    hold(processor, observation(), 0, 8)
    hold(processor, observation(-0.5), 0.4, 20)  # Accepted at .5; blocked until 1.5.
    assert hold(processor, observation(), 1.4, 15) == [Action.PREVIOUS]


def test_lowering_hand_rearms_early_but_new_side_only_sets_baseline():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert hold(processor, observation(-0.5), 0.4, 4) == [Action.NEXT]
    hold(processor, Observation(), 0.6, 7)
    assert hold(processor, observation(-0.5), 0.95, 4) == []
    assert hold(processor, observation(), 1.15, 4) == [Action.PREVIOUS]  # Less than one second after next.


@pytest.mark.parametrize("kwargs", [
    {"turn_tracking_grace_seconds": 0}, {"turn_tracking_grace_seconds": 0.25},
    {"turn_rearm_seconds": float("nan")}, {"turn_rearm_seconds": -1},
])
def test_fast_turn_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_recorded_state_distinguishes_baseline_and_command_pause():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert processor.diagnostic_state(0.35)["starting_side"] == "palm"
    hold(processor, observation(-0.5), 0.4, 4)
    state = processor.diagnostic_state(0.55)
    assert state["starting_side"] == "back" and state["active"]
    assert state["rearm_remaining_seconds"] > 0.9


def test_confirmed_opposite_turns_can_alternate_immediately():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert hold(processor, observation(-0.5), 0.4, 4) == [Action.NEXT]
    assert hold(processor, observation(), 0.6, 4) == [Action.PREVIOUS]
    assert hold(processor, observation(-0.5), 0.8, 4) == [Action.NEXT]
    assert hold(processor, observation(-0.5), 1.0, 40) == []


def test_tracking_loss_does_not_bypass_same_direction_repeat_guard():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert hold(processor, observation(-0.5), 0.4, 4) == [Action.NEXT]
    hold(processor, Observation(), 0.6, 3)  # Lose baseline, but too short for deliberate release.
    assert hold(processor, observation(), 0.75, 4) == []
    assert hold(processor, observation(-0.5), 0.95, 8) == []


def test_deliberate_lowering_allows_a_fresh_same_direction_command_early():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, observation(), 0, 8)
    assert hold(processor, observation(-0.5), 0.4, 4) == [Action.NEXT]
    hold(processor, Observation(), 0.6, 7)
    assert hold(processor, observation(), 0.95, 4) == []
    assert hold(processor, observation(-0.5), 1.15, 4) == [Action.NEXT]


def test_brief_loss_allows_distance_travelled_during_missing_frames():
    processor = GestureProcessor(Config(navigation_mode="palm_turn"))
    hold(processor, replace(observation(), position=(0.3, 0.5)), 0, 8)
    assert processor.update(Observation(), 0.4) is None
    back = replace(observation(-0.5), position=(0.65, 0.5))
    assert hold(processor, back, 0.45, 4) == [Action.NEXT]


def test_short_loss_cannot_bridge_a_teleport_or_large_scale_change():
    for back in (replace(observation(-0.5), position=(0.95, 0.5)),
                 replace(observation(-0.5), hand_scale=0.5)):
        processor = GestureProcessor(Config(navigation_mode="palm_turn"))
        hold(processor, replace(observation(), position=(0.3, 0.5)), 0, 8)
        processor.update(Observation(), 0.4)
        assert hold(processor, back, 0.45, 4) == []


@pytest.mark.parametrize("kwargs", [
    {"turn_rearm_policy": "bad"}, {"turn_reacquire_speed": float("inf")},
    {"turn_reacquire_max_distance": 1.1}, {"turn_reacquire_max_distance": 0.1},
])
def test_reacquisition_and_policy_settings_are_validated(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_manually_reviewed_recorded_turns_survive_brief_tracking_loss():
    import json
    from pathlib import Path
    fixture = json.loads((Path(__file__).parent / "fixtures" / "standing_turns.json").read_text())
    for case in fixture["cases"]:
        processor = GestureProcessor(Config(navigation_mode="palm_turn"))
        actions = [action for row in case["rows"]
                   if (action := processor.update(Observation(**row["observation"]), row["time"])) is not None]
        assert [action.value for action in actions] == [case["expected"]], case["source_time"]
