import json
from dataclasses import asdict

import cv2
import numpy as np
import pytest

from gesturedeck.config import Config
from gesturedeck.evaluate import (FrameClock, method_configs, replay_observations,
    replay_time, replay_video, safe_output, score_controls, score_events, verify_inputs)
from gesturedeck.processor import Observation


def label(action='next', start=1., end=2., trial='T01'):
    return {'trial_id':trial,'expected_action':action,'start_seconds':start,'end_seconds':end}


def event(action='next', seconds=1.5, active=True):
    return {'action':action,'seconds':seconds,'active':active}


def test_first_wrong_direction_cannot_be_rescued_by_later_correct_event():
    result = score_events([label()], [event('previous',1.2),event('next',1.3)], [], 5)
    assert result['correct']==0 and result['wrong_direction']==1
    assert result['duplicates']==1 and result['predicted_commands']==2
    assert result['direction_confusion_true_rows']==[[0,1,0],[0,0,0]]


def test_duplicate_does_not_inflate_precision_and_extra_is_false():
    result = score_events([label()], [event(seconds=1.2),event(seconds=1.3),event(seconds=4)], [], 5)
    assert result['correct']==1 and result['duplicates']==1 and result['unmatched_false_commands']==1
    assert result['precision']==pytest.approx(1/3) and result['recall']==1


def test_tolerance_boundary_and_misses_are_explicit():
    result = score_events([label(),label('previous',4,5,'T02')],
                          [event(seconds=2.75),event(seconds=2.75001)],[],6)
    assert result['correct']==1 and result['missed']==1 and result['unmatched_false_commands']==1
    assert result['direction_confusion_true_rows']==[[1,0,0],[0,0,1]]


def test_setup_false_event_not_hidden_by_clean_short_negative_interval():
    result = score_events([label('previous',5,6)], [event(seconds=3)],
                          [{'start_seconds':7,'end_seconds':8.86}],9)
    assert result['missed']==1 and result['unmatched_false_commands']==1
    assert result['negative_commands']==0 and result['negative_seconds']==pytest.approx(1.86)


def test_no_predictions_has_undefined_precision_not_perfect_score():
    result = score_events([label()], [], [], 5)
    assert result['precision'] is None and result['recall']==0 and result['f1']==0


def test_one_event_cannot_match_two_overlapping_opportunities():
    result = score_events([label(),label(start=1.8,end=3,trial='T02')], [event(seconds=1.9)],[],4)
    assert result['correct']==1 and result['missed']==1


def test_controls_require_action_and_resulting_state():
    labels = [{'trial_id':'C1','start_seconds':1,'end_seconds':3,'expected_action':'toggle','expected_state':'paused'}]
    assert score_controls(labels,[event('toggle',2,False)])['correct']==1
    assert score_controls(labels,[event('toggle',2,True)])['correct']==0
    assert score_controls(labels,[event('exit',2,False)])['correct']==0


def test_replay_timestamp_zero_is_not_replaced_by_capture_time():
    assert replay_time({'elapsed_seconds':9,'decision_elapsed_seconds':0})==0
    assert replay_time({'elapsed_seconds':9,'decision_elapsed_seconds':None})==9


def test_clock_rounding_collisions_stay_monotonic_and_invalid_time_rejected():
    clock = FrameClock()
    values = []
    for t in (0,.0001,.001):
        clock.seconds=t
        values.append(clock.next())
    assert values==[0,1,2]
    clock.seconds=float('nan')
    with pytest.raises(ValueError):
        clock.next()


def test_comparison_only_changes_selected_navigation_method(tmp_path):
    c=Config(swipe_model_path='source.npz',swipe_trial=True,mirror=True)
    configs=method_configs({'config':asdict(c)},tmp_path)
    baseline,learned=asdict(configs['baseline']),asdict(configs['learned'])
    assert baseline.pop('swipe_model_path')==''
    assert baseline.pop('swipe_trial') is False
    assert learned.pop('swipe_model_path')==str(tmp_path/'swipe_model.npz')
    assert learned.pop('swipe_trial') is True
    assert baseline==learned and not baseline['record_directory']


def test_outputs_cannot_overwrite_frozen_recordings_or_escape_root(tmp_path):
    p=tmp_path/'recordings/new'
    assert safe_output(p,tmp_path)==p.resolve()
    p.mkdir(parents=True)
    for bad in (p,tmp_path/'recordings',tmp_path/'elsewhere'):
        with pytest.raises(ValueError):
            safe_output(bad,tmp_path)


def test_freeze_verification_rejects_missing_changed_and_traversal_inputs(tmp_path):
    for name in ('missing.bin','../outside.bin'):
        with pytest.raises((ValueError,FileNotFoundError)):
            verify_inputs({'inputs_sha256':{name:'bad'},'analysis_programs_sha256':{}},tmp_path)
    (tmp_path/'file').write_text('changed')
    with pytest.raises(ValueError):
        verify_inputs({'inputs_sha256':{'file':'bad'},'analysis_programs_sha256':{}},tmp_path)


def test_cached_comparison_detects_changed_landmark_inputs(tmp_path):
    from gesturedeck.evaluate import digest
    from gesturedeck.reproduce_comparison import verify_cache
    cache=tmp_path/'features.jsonl'
    cache.write_text('{"observation": {}}\n')
    registry={'files_sha256':{'features.jsonl':digest(cache)}}
    verify_cache(registry,tmp_path)
    cache.write_text('{"observation": {"changed": true}}\n')
    with pytest.raises(ValueError,match='changed or missing'):
        verify_cache(registry,tmp_path)


def test_cached_comparison_rejects_input_path_escape(tmp_path):
    from gesturedeck.reproduce_comparison import verify_cache
    with pytest.raises(ValueError):
        verify_cache({'files_sha256':{'../outside':'bad'}},tmp_path)


def test_raw_replay_reuses_actual_pixels_without_double_mirror_or_resize(tmp_path,monkeypatch):
    seen=[]
    class FakeRecognizer:
        def __init__(self,config):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def recognize(self,frame):
            seen.append((frame.shape,int(frame[100,30,2]),int(frame[100,290,0]),self._timestamps.next()))
            return Observation(),[]
    monkeypatch.setattr('gesturedeck.recognizer.GestureRecognizer',FakeRecognizer)
    monkeypatch.setattr(cv2,'imshow',lambda *_:pytest.fail('Evaluation must not open a window'))
    video=tmp_path/'webcam.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),20,(320,240))
    assert writer.isOpened()
    frame=np.zeros((240,320,3),np.uint8);frame[:,:160]=(0,0,250);frame[:,160:]=(250,0,0)
    for _ in range(3):writer.write(frame)
    writer.release()
    rows=[{'frame_index':i,'elapsed_seconds':.3+i*.06,'decision_elapsed_seconds':.33+i*.06} for i in range(3)]
    configs={'baseline':Config(audio_feedback=False),'learned':Config(audio_feedback=False)}
    events,timings,count=replay_video(tmp_path,rows,configs,tmp_path,warmup_frames=2)
    assert count==3 and events=={'baseline':[],'learned':[]}
    assert len(seen)==5 and all(s[0]==(240,320,3) for s in seen)
    assert all(s[1]>220 and s[2]>220 for s in seen[2:])
    assert all(b[3]>a[3] for a,b in zip(seen,seen[1:]))
    assert len(timings['baseline'])==len(timings['learned'])==3
    assert len((tmp_path/'raw_frame_predictions.jsonl').read_text().splitlines())==3
