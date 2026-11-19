from gesturedeck.final_evaluate import EndpointBaseline, score
from gesturedeck.processor import Observation


def test_baseline_establishes_first_side_and_missing_hand_resets():
    b=EndpointBaseline()
    palm=Observation(hand_present=True,open_hand=True,palm_facing=.7)
    back=Observation(hand_present=True,open_hand=True,palm_facing=-.7)
    assert b.update(palm) is None
    assert b.update(palm) is None
    assert b.update(back)=='next'
    assert b.update(palm)=='previous'
    assert b.update(Observation()) is None
    assert b.update(back) is None


def test_fast_next_attempt_is_not_consumed_by_previous_late_allowance():
    labels=[dict(trial_id='A',expected_action='next',start_seconds=1,end_seconds=1.5),
            dict(trial_id='B',expected_action='previous',start_seconds=1.6,end_seconds=2.0)]
    events=[dict(seconds=1.4,action='next'),dict(seconds=1.9,action='previous')]
    s=score(labels,events,[])
    assert s['correct']==2 and s['duplicates']==0 and s['unmatched_false_commands']==0


def test_wrong_first_event_cannot_be_rescued_and_extras_remain_false():
    label=dict(trial_id='A',expected_action='next',start_seconds=1,end_seconds=2)
    events=[dict(seconds=.5,action='next'),dict(seconds=1.2,action='previous'),dict(seconds=1.3,action='next')]
    s=score([label],events,[dict(start_seconds=0,end_seconds=.9)])
    assert s['correct']==0 and s['wrong_directions']==1 and s['duplicates']==1
    assert s['unmatched_false_commands']==1 and s['negative_commands']==1


def test_private_frozen_study_reproduces_report_without_native_inference():
    from pathlib import Path
    from gesturedeck.final_evaluate import evaluate
    study=Path(__file__).resolve().parents[1]/'recordings/milestone5_evaluation_inputs/study.json'
    if not study.is_file():
        import pytest
        pytest.skip('Private study is supplied in the teacher evaluation-data ZIP, outside the public repository.')
    result=evaluate(study)
    baseline=result['methods']['endpoint_baseline']
    final=result['methods']['temporal_final']
    assert baseline['attempts']==52 and baseline['correct']==17
    assert final['attempts']==52 and final['correct']==46 and final['missed']==6
    assert final['confusion_true_rows']==[[21,0,5],[0,25,1]]
    assert result['current_event_replay_exact'] and not result['raw_inference_performed']
    assert not result['held_out_generalization_claim']
    timing=result['benchmark']['stats']['temporal_analysis_ms']
    assert timing['frames']==1307
    assert round(timing['mean_ms'],2)==32.43 and round(timing['p95_ms'],2)==53.99
