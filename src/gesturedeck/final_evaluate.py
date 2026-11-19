"""Reproduce the final palm-turn table from pinned, reduced saved observations.

No webcam, GUI, keys, native inference or training. This is a non-blind
functional replay, not a held-out generalization experiment or live benchmark.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

from gesturedeck.config import Config
from gesturedeck.processor import GestureProcessor, Observation

ROOT = Path(__file__).resolve().parents[2]
NAV = ('next', 'previous')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EndpointBaseline:
    """Single valid endpoint sample per side, no temporal confirmation.

    Uses the current detector's open-hand flag and 0.20 facing threshold.
    Actual missing-hand observations clear the baseline immediately. Unclear
    detected frames add no orientation evidence. No tuning is performed here.
    """
    def __init__(self, threshold=.20):
        self.threshold = threshold
        self.side = None

    def update(self, observation):
        if not observation.hand_present:
            self.side = None
            return None
        facing = observation.palm_facing
        if not observation.open_hand or facing is None or abs(facing) < self.threshold:
            return None
        side = 'palm' if facing > 0 else 'back'
        old, self.side = self.side, side
        if old is None or old == side:
            return None
        return 'next' if old == 'palm' else 'previous'


def score(labels, events, negatives, tolerance=.75):
    """First-event one-to-one matching; a later attempt owns its own interval.

    Clip the post-attempt allowance at the next attempt start. This prevents
    the next fast gesture from being consumed as the preceding one's duplicate.
    Wrong first events cannot be rescued by later correct events.
    """
    labels = sorted(labels, key=lambda r: float(r['start_seconds']))
    events = sorted(events, key=lambda r: r['seconds'])
    used, trials = set(), []
    for n, label in enumerate(labels):
        start, end = float(label['start_seconds']), float(label['end_seconds'])
        following = float(labels[n+1]['start_seconds']) if n+1 < len(labels) else float('inf')
        matches = [i for i, event in enumerate(events) if i not in used and
                   start <= event['seconds'] <= end+tolerance and event['seconds'] < following]
        used.update(matches)
        first = events[matches[0]]['action'] if matches else 'missed'
        expected = label['expected_action']
        trials.append(dict(trial_id=label['trial_id'], expected_action=expected,
            observed_action=first, verdict='missed' if not matches else 'correct' if first==expected else 'wrong_direction',
            duplicates=max(0,len(matches)-1)))
    correct=sum(t['verdict']=='correct' for t in trials)
    predicted=len(events);attempts=len(labels)
    negative_seconds=sum(float(r['end_seconds'])-float(r['start_seconds']) for r in negatives)
    negative_events=sum(any(float(r['start_seconds'])<=e['seconds']<=float(r['end_seconds']) for r in negatives) for e in events)
    matrix=[[0,0,0],[0,0,0]]
    classes={}
    for action in NAV:
        true=sum(t['expected_action']==action for t in trials)
        pred=sum(e['action']==action for e in events)
        tp=sum(t['expected_action']==action and t['verdict']=='correct' for t in trials)
        classes[action]=dict(attempts=true,predicted=pred,correct=tp,precision=tp/pred if pred else None,
            recall=tp/true if true else None,f1=2*tp/(true+pred) if true+pred else None)
    for t in trials:matrix[NAV.index(t['expected_action'])][('next','previous','missed').index(t['observed_action'])]+=1
    return dict(attempts=attempts,predicted=predicted,correct=correct,
        missed=sum(t['verdict']=='missed' for t in trials),wrong_directions=sum(t['verdict']=='wrong_direction' for t in trials),
        duplicates=sum(t['duplicates'] for t in trials),unmatched_false_commands=predicted-len(used),
        precision=correct/predicted if predicted else None,recall=correct/attempts if attempts else None,
        f1=2*correct/(attempts+predicted) if attempts+predicted else None,
        negative_seconds=negative_seconds,negative_commands=negative_events,
        negative_false_commands_per_minute=negative_events*60/negative_seconds if negative_seconds else None,
        per_class=classes,confusion_true_rows=matrix,confusion_rows=list(NAV),confusion_columns=['next','previous','missed'],trials=trials)


def aggregate(scores):
    keys=('attempts','predicted','correct','missed','wrong_directions','duplicates','unmatched_false_commands','negative_seconds','negative_commands')
    result={k:sum(s[k] for s in scores) for k in keys}
    a,p,c=result['attempts'],result['predicted'],result['correct']
    result.update(precision=c/p if p else None,recall=c/a if a else None,f1=2*c/(a+p) if a+p else None)
    result['negative_false_commands_per_minute']=result['negative_commands']*60/result['negative_seconds'] if result['negative_seconds'] else None
    result['per_class']={}
    for action in NAV:
        sums={k:sum(s['per_class'][action][k] for s in scores) for k in ('attempts','predicted','correct')}
        a,p,c=sums['attempts'],sums['predicted'],sums['correct']
        sums.update(precision=c/p if p else None,recall=c/a if a else None,f1=2*c/(a+p) if a+p else None)
        result['per_class'][action]=sums
    result['confusion_true_rows']=[[sum(s['confusion_true_rows'][i][j] for s in scores) for j in range(3)] for i in range(2)]
    return result


def evaluate(study_path):
    study_path=Path(study_path).resolve();study=json.loads(study_path.read_text())
    for name,expected in study['input_sha256'].items():
        target=(study_path.parent/name).resolve()
        if not target.is_relative_to(study_path.parent) or digest(target)!=expected:
            raise ValueError(f'Changed or missing study input: {name}')
    for name,expected in study['program_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Changed evaluation source: {name}')
    methods={'endpoint_baseline':[],'temporal_final':[]};blocks=[]
    for spec in study['blocks']:
        folder=study_path.parent/spec['id']
        config=json.loads((folder/'config.json').read_text())
        config.update(audio_feedback=False,record_directory='',record_slideshow=False)
        temporal=GestureProcessor(Config(**config));baseline=EndpointBaseline(config['turn_min_facing'])
        events={m:[] for m in methods}
        rows=[json.loads(s) for s in (folder/'observations.jsonl').read_text().splitlines()]
        for row in rows:
            if not row['inference_ran']:continue
            observation=Observation(**row['observation']);clock=row['decision_seconds']
            b=baseline.update(observation);f=temporal.update(observation,clock)
            for method,action in [('endpoint_baseline',b),('temporal_final',f.value if f else None)]:
                if action in NAV:events[method].append(dict(frame_index=row['frame_index'],seconds=clock,action=action))
        expected=json.loads((folder/'recorded_events.json').read_text())
        if events['temporal_final']!=expected:raise ValueError(f'Current processor no longer reproduces captured events: {spec["id"]}')
        with (folder/'labels.csv').open(newline='') as stream:labels=list(csv.DictReader(stream))
        negatives=json.loads((folder/'negatives.json').read_text())
        scores={m:score(labels,events[m],negatives,study['matching_tolerance_seconds']) for m in methods}
        for m in methods:methods[m].append(scores[m])
        blocks.append(dict(id=spec['id'],condition=spec['condition'],scores=scores,events=events))
    result=dict(schema_version=1,scope=study['scope'],matching_rule=study['matching_rule'],
        methods={m:aggregate(s) for m,s in methods.items()},blocks=blocks,
        current_event_replay_exact=True,raw_inference_performed=False,training_performed=False,
        held_out_generalization_claim=False)
    if 'benchmark' in study:
        metadata=json.loads((study_path.parent/study['benchmark']['metadata']).read_text())
        with (study_path.parent/study['benchmark']['timings']).open(newline='') as stream:
            samples=list(csv.DictReader(stream))
        stats={}
        for key in ('temporal_analysis_ms','baseline_analysis_ms'):
            values=np.array([float(r[key]) for r in samples])
            stats[key]=dict(frames=len(values),mean_ms=float(values.mean()),median_ms=float(np.median(values)),
                p95_ms=float(np.percentile(values,95)),throughput_fps=float(1000/values.mean()))
        result['benchmark']=dict(hardware=metadata['hardware'],scope=metadata['scope'],stats=stats,
            reproduction_scope='Statistics recomputed from pinned fresh-measurement samples; no new timing performed by this reproduction command.')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,default=ROOT/'recordings/milestone5_evaluation_inputs/study.json')
    parser.add_argument('--output',type=Path,default=ROOT/'recordings/milestone5_reproduced')
    args=parser.parse_args(argv);out=args.output.resolve()
    if not out.is_relative_to((ROOT/'recordings').resolve()) or out==(ROOT/'recordings').resolve():
        parser.error('Output must be a new subdirectory of this project recordings/.')
    if out.exists():parser.error('Output exists; choose a new folder to preserve prior results.')
    result=evaluate(args.study);out.mkdir(parents=True)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'results_table.csv').open('w',newline='') as stream:
        columns=['method','attempts','predicted','correct','missed','wrong_directions','duplicates','unmatched_false_commands','precision','recall','f1']
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for method,s in result['methods'].items():writer.writerow(dict(method=method,**{k:s[k] for k in columns[1:]}))
    print(json.dumps(dict(output=str(out),methods=result['methods'],benchmark=result.get('benchmark'),current_event_replay_exact=True),indent=2))
    return 0


if __name__=='__main__':raise SystemExit(main())
