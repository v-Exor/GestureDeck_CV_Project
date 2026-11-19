"""Paired offline comparison of frozen geometric and learned swipe methods.

Reads local files only. Never creates a webcam, controller, GUI, or training job.
Run python -m gesturedeck.evaluate --help for the reproducible study command.
"""
import argparse
import csv
from dataclasses import asdict, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import textwrap
from time import perf_counter_ns

import numpy as np

from gesturedeck.config import Config
from gesturedeck.processor import GestureProcessor, Observation

ROOT = Path(__file__).resolve().parents[2]
METHODS = ('baseline', 'learned')
NAV = ('next', 'previous')


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, columns=None):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_output(path, root=ROOT):
    path, recordings = Path(path).resolve(), (root / 'recordings').resolve()
    if path == recordings or not path.is_relative_to(recordings):
        raise ValueError('Evaluation output must be a new subfolder of this project recordings/.')
    if path.exists():
        raise ValueError(f'Output already exists; preserve it and choose another path: {path}')
    return path


def verify_inputs(study, root=ROOT):
    for group in ('inputs_sha256', 'analysis_programs_sha256'):
        for name, expected in study[group].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or digest(path) != expected:
                raise ValueError(f'Frozen input missing or changed: {name}')


def replay_time(row):
    value = row.get('decision_elapsed_seconds')
    return float(row['elapsed_seconds'] if value is None else value)


def method_configs(manifest, source, root=ROOT):
    known = {f.name for f in fields(Config)}
    options = {k:v for k,v in manifest['config'].items() if k in known}
    options.update(model_path=str(root / 'models/gesture_recognizer.task'),
                   swipe_model_path=str(source / 'swipe_model.npz'), swipe_trial=True,
                   audio_feedback=False, record_directory='', record_slideshow=False,
                   navigation_mode='swipe')
    learned = Config(**options)
    return {'baseline': replace(learned, swipe_model_path='', swipe_trial=False), 'learned': learned}


def score_events(labels, events, negative_intervals, duration, tolerance=.75):
    """Assign the first event chronologically; later events cannot rescue a wrong direction."""
    events = sorted([e for e in events if e['action'] in NAV], key=lambda e:e['seconds'])
    used, trials = set(), []
    matrix = [[0,0,0], [0,0,0]]  # true next/previous; predicted next/previous/missed
    for label in labels:
        start, end = float(label['start_seconds']), float(label['end_seconds'])
        indices = [i for i,e in enumerate(events) if i not in used and start <= e['seconds'] <= end+tolerance]
        used.update(indices)
        first = events[indices[0]]['action'] if indices else 'missed'
        expected = label['expected_action']
        verdict = 'missed' if not indices else 'correct' if first == expected else 'wrong_direction'
        matrix[NAV.index(expected)][('next','previous','missed').index(first)] += 1
        trials.append({'trial_id': label['trial_id'], 'expected_action': expected,
                       'start_seconds': start, 'end_seconds': end, 'observed_action': first,
                       'observed_actions': '|'.join(events[i]['action'] for i in indices),
                       'verdict': verdict, 'duplicates': max(0,len(indices)-1),
                       'event_seconds': events[indices[0]]['seconds'] if indices else None})
    extras = [e for i,e in enumerate(events) if i not in used]
    correct = sum(r['verdict']=='correct' for r in trials)
    per_class = {}
    for action in NAV:
        tp = sum(r['verdict']=='correct' and r['expected_action']==action for r in trials)
        actual = sum(r['expected_action']==action for r in trials)
        predicted = sum(e['action']==action for e in events)
        precision, recall = (tp/predicted if predicted else None), (tp/actual if actual else None)
        per_class[action] = {'true': actual, 'predicted': predicted, 'correct': tp,
                             'precision': precision, 'recall': recall,
                             'f1': 2*tp/(actual+predicted) if actual+predicted else None}
    negative_seconds = sum(float(n['end_seconds'])-float(n['start_seconds']) for n in negative_intervals)
    negative_commands = sum(any(float(n['start_seconds']) <= e['seconds'] <= float(n['end_seconds'])
                                for n in negative_intervals) for e in events)
    return {'attempts': len(labels), 'correct': correct, 'missed': sum(r['verdict']=='missed' for r in trials),
            'wrong_direction': sum(r['verdict']=='wrong_direction' for r in trials),
            'duplicates': sum(r['duplicates'] for r in trials), 'unmatched_false_commands': len(extras),
            'predicted_commands': len(events), 'precision': correct/len(events) if events else None,
            'recall': correct/len(labels) if labels else None,
            'f1': 2*correct/(len(labels)+len(events)) if len(labels)+len(events) else None,
            'per_class': per_class, 'macro_f1': float(np.mean([p['f1'] for p in per_class.values() if p['f1'] is not None])),
            'direction_confusion_true_rows': matrix,
            'confusion_rows': list(NAV), 'confusion_columns': ['next','previous','missed'],
            'negative_seconds': negative_seconds, 'negative_commands': negative_commands,
            'negative_false_commands_per_minute': negative_commands*60/negative_seconds if negative_seconds else None,
            'unmatched_commands_per_video_minute': len(extras)*60/duration if duration else None,
            'video_seconds': duration, 'trials': trials, 'unmatched_events': extras}


def score_controls(labels, events):
    used, trials = set(), []
    for label in labels:
        found = [i for i,e in enumerate(events) if i not in used
                 and float(label['start_seconds']) <= e['seconds'] <= float(label['end_seconds'])]
        used.update(found)
        good = len(found)==1 and events[found[0]]['action']==label['expected_action'] and (
            events[found[0]]['active'] == (label['expected_state']=='active'))
        trials.append({'trial_id':label['trial_id'], 'expected_action':label['expected_action'],
                       'expected_state':label['expected_state'], 'correct':good,
                       'observed_actions':'|'.join(events[i]['action'] for i in found)})
    return {'attempts':len(labels), 'correct':sum(t['correct'] for t in trials),
            'extra_commands':len(events)-len(used), 'trials':trials}


class FrameClock:
    """Adapter for the unchanged recognizer's timestamp provider; no wall-clock replay drift."""
    def __init__(self):
        self.seconds = 0.
        self.last = -1

    def next(self):
        if not math.isfinite(self.seconds) or self.seconds < 0:
            raise ValueError('Video timestamp must be finite and nonnegative')
        self.last = max(round(self.seconds*1000), self.last+1)
        return self.last


def event_for(action, index, row, processor):
    return {'frame_index':index, 'seconds':replay_time(row), 'capture_seconds':row['elapsed_seconds'],
            'action':action.value, 'active':processor.active, 'status':processor.status}


def replay_observations(rows, configs):
    processors = {m:GestureProcessor(c) for m,c in configs.items()}
    events = {m:[] for m in METHODS}
    for row in rows:
        if not row.get('inference_ran'):
            continue
        observation = Observation(**row['observation'])
        for method, processor in processors.items():
            action = processor.update(observation, replay_time(row))
            if action:
                events[method].append(event_for(action, row['frame_index'], row, processor))
    return events


def annotated_pair(frame, processors, last, capture_time):
    import cv2
    panels = []
    for method in METHODS:
        panel = np.full((410,480,3),245,np.uint8)
        panel[40:310] = cv2.resize(frame,(480,270))
        lines = [f'{method.upper()} | {"ACTIVE" if processors[method].active else "PAUSED"} | {capture_time:.2f}s',
                 f'Last event: {last[method]}']
        for i,line in enumerate(lines):
            y = 25 if i==0 else 330
            cv2.putText(panel,line,(8,y),cv2.FONT_HERSHEY_SIMPLEX,.44,(20,20,20),1,cv2.LINE_AA)
        for i,line in enumerate(textwrap.wrap(processors[method].status,58)[:3]):
            cv2.putText(panel,line,(8,350+i*18),cv2.FONT_HERSHEY_SIMPLEX,.4,(20,20,20),1,cv2.LINE_AA)
        panels.append(panel)
    return np.hstack(panels)


def replay_video(source, rows, configs, directory, warmup_frames=20, annotate=False):
    import cv2
    from gesturedeck.recognizer import GestureRecognizer
    processors = {m:GestureProcessor(c) for m,c in configs.items()}
    events, timings, last = ({m:[] for m in METHODS}, {m:[] for m in METHODS}, {m:'none' for m in METHODS})
    cap = cv2.VideoCapture(str(source/'webcam.avi'))
    if not cap.isOpened():
        raise ValueError(f'Cannot open recorded video: {source}')
    writer = None
    clock = FrameClock()
    frame_count = 0
    try:
        with GestureRecognizer(configs['learned']) as recognizer:
            # Reuse the production BGR-to-RGB/model/observation code unchanged.
            # Only the timestamp provider is replaced to follow the recorded stream.
            recognizer._timestamps = clock
            ok, first = cap.read()
            if not ok:
                raise ValueError('Empty recorded video')
            blank = np.zeros_like(first)
            for i in range(warmup_frames):
                clock.seconds = i/30
                recognizer.recognize(blank)
            offset = warmup_frames/30 + 1.
            cap.set(cv2.CAP_PROP_POS_FRAMES,0)
            if annotate:
                writer = cv2.VideoWriter(str(directory/'paired_annotated.avi'),cv2.VideoWriter_fourcc(*'MJPG'),20,(960,410))
                if not writer.isOpened():
                    raise RuntimeError('Could not create annotated video')
            with (directory/'raw_frame_predictions.jsonl').open('x',encoding='utf-8') as stream:
                for index,row in enumerate(rows):
                    start = perf_counter_ns()
                    ok, frame = cap.read()
                    decode_ns = perf_counter_ns()-start
                    if not ok:
                        raise ValueError(f'Video ended before JSONL frame {index}')
                    clock.seconds = offset+float(row['elapsed_seconds'])
                    start = perf_counter_ns()
                    observation, _ = recognizer.recognize(frame)
                    vision_ns = perf_counter_ns()-start
                    record = {'frame_index':index,'capture_seconds':row['elapsed_seconds'],
                              'decision_seconds':replay_time(row),'observation':asdict(observation),'methods':{}}
                    # Alternate evaluation order across frames to reduce order bias.
                    order = METHODS if index%2==0 else METHODS[::-1]
                    for method in order:
                        start = perf_counter_ns()
                        action = processors[method].update(observation,replay_time(row))
                        processor_ns = perf_counter_ns()-start
                        timings[method].append({'decode_ms':decode_ns/1e6,'shared_vision_ms':vision_ns/1e6,
                                                'processor_ms':processor_ns/1e6,
                                                'analysis_ms':(decode_ns+vision_ns+processor_ns)/1e6})
                        if action:
                            event = event_for(action,index,row,processors[method])
                            events[method].append(event)
                            last[method] = f'{action.value} @ {event["seconds"]:.2f}s'
                        record['methods'][method] = {'action':action.value if action else None,
                            'active':processors[method].active,'status':processors[method].status,
                            'diagnostics':processors[method].diagnostic_state(replay_time(row)),**timings[method][-1]}
                    stream.write(json.dumps(record,allow_nan=False)+'\n')
                    if writer:
                        writer.write(annotated_pair(frame,processors,last,row['elapsed_seconds']))
                    frame_count += 1
                    if index and index%500==0:
                        print(f'  raw video {index}/{len(rows)} frames',flush=True)
                if cap.read()[0]:
                    raise ValueError('Video has more frames than JSONL rows')
    finally:
        cap.release()
        if writer:
            writer.release()
    return events,timings,frame_count


def summarize_speed(timings):
    if not timings:
        return None
    values = np.array([t['analysis_ms'] for t in timings])
    return {'frames':len(values),'mean_analysis_ms':float(values.mean()),
            'median_analysis_ms':float(np.median(values)),'p95_analysis_ms':float(np.percentile(values,95)),
            'analysis_fps':1000/float(values.mean()),
            'mean_shared_vision_ms':float(np.mean([t['shared_vision_ms'] for t in timings])),
            'mean_processor_ms':float(np.mean([t['processor_ms'] for t in timings]))}


def aggregate(scores):
    keys = ('attempts','correct','missed','wrong_direction','duplicates','unmatched_false_commands',
            'predicted_commands','negative_seconds','negative_commands','video_seconds')
    total = {k:sum(s[k] for s in scores) for k in keys}
    total.update(precision=total['correct']/total['predicted_commands'] if total['predicted_commands'] else None,
                 recall=total['correct']/total['attempts'],
                 f1=2*total['correct']/(total['attempts']+total['predicted_commands']),
                 direction_confusion_true_rows=np.sum([s['direction_confusion_true_rows'] for s in scores],axis=0).tolist())
    per_class = {}
    for action in NAV:
        items = [s['per_class'][action] for s in scores]
        tp, actual, predicted = (sum(s[k] for s in items) for k in ('correct','true','predicted'))
        per_class[action] = {'correct':tp,'true':actual,'predicted':predicted,
                             'precision':tp/predicted if predicted else None,'recall':tp/actual,
                             'f1':2*tp/(actual+predicted)}
    total['per_class'] = per_class
    total['macro_f1'] = sum(v['f1'] for v in per_class.values())/2
    total['negative_false_commands_per_minute'] = total['negative_commands']*60/total['negative_seconds']
    total['unmatched_commands_per_video_minute'] = total['unmatched_false_commands']*60/total['video_seconds']
    return total


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'recordings/controlled_test_20261109_01/evaluation_inputs.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('observations','video','both'),default='both')
    parser.add_argument('--warmup-frames',type=int,default=20)
    parser.add_argument('--annotate',action='store_true',help='Save side-by-side annotated videos at nominal 20 FPS; JSONL times remain authoritative')
    args = parser.parse_args(argv)
    if args.warmup_frames < 1:
        parser.error('--warmup-frames must be positive')
    output = safe_output(args.output)
    study = json.loads(args.dataset.read_text(encoding='utf-8'))
    verify_inputs(study)
    output.mkdir(parents=True)
    all_scores = {mode:{m:[] for m in METHODS} for mode in ('observations','video')}
    all_timings = {m:[] for m in METHODS}
    comparison, trial_rows, control_rows, parity = [], [], [], []
    full_results = []
    for spec in study['blocks']:
        bid = spec['block_id']
        print(f'{bid} {spec["condition"]}',flush=True)
        source = (ROOT/spec['recording_directory']).resolve()
        manifest = json.loads((source/'manifest.json').read_text())
        rows = [json.loads(line) for line in (source/'frames.jsonl').read_text().splitlines()]
        if manifest['status']!='completed' or len(rows)!=manifest['frame_count']:
            raise ValueError('Incomplete recording')
        if [r['frame_index'] for r in rows]!=list(range(len(rows))) or any(replay_time(b)<=replay_time(a) for a,b in zip(rows,rows[1:])):
            raise ValueError('Frame ordering or timestamp mismatch')
        configs = method_configs(manifest,source)
        directory = output/bid
        directory.mkdir()
        is_control = bid=='C07'
        labels = read_csv(args.dataset.parent/bid/('reviewed_controls.csv' if is_control else 'reviewed_attempts.csv'))
        negatives = [] if is_control else json.loads((args.dataset.parent/bid/'reviewed_negative_intervals.json').read_text())
        modes = {}
        if args.mode in ('observations','both'):
            modes['observations'] = replay_observations(rows,configs)
            recorded = [{'frame_index':r['frame_index'],'action':r['predicted_action']} for r in rows if r.get('predicted_action')]
            reproduced = [{k:e[k] for k in ('frame_index','action')} for e in modes['observations']['learned']]
            parity.append({'block_id':bid,'exact_action_frame_parity':recorded==reproduced,
                           'recorded':recorded,'reproduced':reproduced})
            if recorded!=reproduced:
                raise ValueError(f'Learned observation replay differs from captured action frames in {bid}')
        if args.mode in ('video','both'):
            modes['video'],timings,count = replay_video(source,rows,configs,directory,args.warmup_frames,args.annotate)
            assert count==len(rows)
            # C07 controls are kept out of navigation speed and accuracy summaries.
            if not is_control:
                for m in METHODS:
                    all_timings[m].extend(timings[m])
        clip = {'block_id':bid,'condition':spec['condition'],'recording_directory':str(source.relative_to(ROOT)),
                'frames':len(rows),'configs':{m:asdict(c) for m,c in configs.items()},'results':{}}
        for mode,events_by_method in modes.items():
            clip['results'][mode] = {}
            for method,events in events_by_method.items():
                (directory/f'{mode}_{method}_events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
                score = score_controls(labels,events) if is_control else score_events(labels,events,negatives,
                    rows[-1]['elapsed_seconds']-rows[0]['elapsed_seconds'],study['navigation_post_attempt_tolerance_seconds'])
                clip['results'][mode][method] = score
                if is_control:
                    control_rows.append({'mode':mode,'method':method,**{k:v for k,v in score.items() if k!='trials'}})
                else:
                    all_scores[mode][method].append(score)
                    comparison.append({'block_id':bid,'condition':spec['condition'],'mode':mode,'method':method,
                        **{k:score[k] for k in ('attempts','correct','missed','wrong_direction','duplicates','unmatched_false_commands','predicted_commands','precision','recall','f1','macro_f1','negative_seconds','negative_commands')}})
                    for trial in score['trials']:
                        trial_rows.append({'block_id':bid,'condition':spec['condition'],'mode':mode,'method':method,**trial})
        (directory/'results.json').write_text(json.dumps(clip,indent=2),encoding='utf-8')
        full_results.append(clip)
    summary = {'created_utc':datetime.now(timezone.utc).isoformat(),'dataset':str(args.dataset.resolve()),
        'dataset_sha256':digest(args.dataset),'inputs_verified':True,'model_training_or_tuning':False,
        'camera_opened':False,'keys_sent':False,'matching_tolerance_seconds':study['navigation_post_attempt_tolerance_seconds'],
        'warmup_frames_per_video':args.warmup_frames,'runtime':{'python':platform.python_version(),'platform':platform.platform()},
        'primary_comparison':'video' if args.mode in ('video','both') else 'observations',
        'scores':{mode:{m:aggregate(s) for m,s in methods.items() if s} for mode,methods in all_scores.items() if any(methods.values())},
        'speed':{m:summarize_speed(t) for m,t in all_timings.items()},'controls':control_rows,'capture_parity':parity,
        'speed_scope':'Mean per-frame decode plus shared production MediaPipe/BGR-RGB inference plus method-specific processor time; shared inference measured once and attributed equally. Excludes model loading, rendering, video/JSON output writes, camera waiting, GUI, keyboard and physical slide latency. Warm-up blank frames precede each stream, excluded from timing.',
        'raw_replay_scope':'The two methods share fresh MediaPipe observations from every original mirrored raw frame, without another mirror/resize. Native VIDEO timestamps follow capture times; each processor follows original decision timestamps. Configured aspect/thresholds remain frozen, including the known actual/configured aspect mismatch.',
        'confusion_scope':'Rows are reviewed NEXT/PREVIOUS attempts; columns NEXT/PREVIOUS/missed. Extra and duplicate commands are reported separately; no true-negative count or overall classification accuracy is invented.',
        'limitations':['Single participant; count/condition deviations from Phase 1 remain.','C06 final no-hand period is only 1.86 seconds; setup false triggers are counted separately.','Recomputed raw landmarks may differ slightly from the original capture; exact saved-observation replay is the capture parity reference.','No test-set tuning or trained-model changes. Timings vary with machine load; accuracy counts reproduce for saved observations.']}
    (output/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    write_csv(output/'comparison_by_condition.csv',comparison)
    write_csv(output/'trial_log.csv',trial_rows)
    write_csv(output/'controls.csv',control_rows)
    (output/'capture_parity.json').write_text(json.dumps(parity,indent=2),encoding='utf-8')
    code_paths = [Path(__file__)]+list((ROOT/'src/gesturedeck').glob('*.py'))
    (output/'run_provenance.json').write_text(json.dumps({'evaluation_code_sha256':{str(p.relative_to(ROOT)):digest(p) for p in code_paths},
        'study_input_manifest':study,'command_mode':args.mode,'annotated':args.annotate},indent=2),encoding='utf-8')
    print(json.dumps(summary['scores'],indent=2),flush=True)
    print(f'Saved comparison: {output}',flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
