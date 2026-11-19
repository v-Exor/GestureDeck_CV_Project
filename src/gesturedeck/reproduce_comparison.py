"""Recompute the Phase 2 table from pinned fresh-video features and saved timing samples.

Replays both actual processors, not saved prediction labels. No native vision model,
webcam, GUI, keys, training, or new speed measurement is used in this command.
"""
import argparse
import json
from pathlib import Path

from gesturedeck.evaluate import (METHODS, ROOT, aggregate, digest, method_configs, read_csv,
    replay_time, safe_output, score_controls, score_events, summarize_speed, write_csv)
from gesturedeck.processor import GestureProcessor, Observation


def verify_cache(registry, root=ROOT):
    for name, expected in registry['files_sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f'Comparison cache changed or missing: {name}')


def reproduce(run, output):
    run, output = Path(run).resolve(), safe_output(output)
    registry = json.loads((run/'reproduction_inputs.json').read_text())
    verify_cache(registry)
    study_path = ROOT/registry['study_manifest']
    study = json.loads(study_path.read_text())
    original = json.loads((run/'comparison.json').read_text())
    scores, timing = ({m:[] for m in METHODS},{m:[] for m in METHODS})
    by_condition, controls, trials, parity = [], [], [], []
    for spec in study['blocks']:
        bid = spec['block_id']
        source = ROOT/spec['recording_directory']
        manifest = json.loads((source/'manifest.json').read_text())
        configs = method_configs(manifest,source)
        processors = {m:GestureProcessor(c) for m,c in configs.items()}
        events = {m:[] for m in METHODS}
        count = 0
        with (run/bid/'raw_frame_predictions.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['frame_index'] != count:
                    raise ValueError('Cache frame order mismatch')
                observation = Observation(**row['observation'])
                for method,processor in processors.items():
                    action = processor.update(observation,float(row['decision_seconds']))
                    if action:
                        events[method].append({'frame_index':count,'seconds':row['decision_seconds'],
                            'capture_seconds':row['capture_seconds'],'action':action.value,
                            'active':processor.active,'status':processor.status})
                    if bid!='C07':
                        timing[method].append({k:row['methods'][method][k] for k in
                                             ('decode_ms','shared_vision_ms','processor_ms','analysis_ms')})
                count += 1
        if count != manifest['frame_count']:
            raise ValueError('Cache/recording frame count mismatch')
        control = bid=='C07'
        labels = read_csv(study_path.parent/bid/('reviewed_controls.csv' if control else 'reviewed_attempts.csv'))
        negatives = [] if control else json.loads((study_path.parent/bid/'reviewed_negative_intervals.json').read_text())
        for method in METHODS:
            saved_events = json.loads((run/bid/f'video_{method}_events.json').read_text())
            if events[method]!=saved_events:
                raise ValueError(f'Cached raw feature replay differs from original native-run events: {bid}/{method}')
            parity.append({'block_id':bid,'method':method,'exact_event_parity':True,'frames':count})
            if control:
                value=score_controls(labels,events[method])
                controls.append({'method':method,**{k:v for k,v in value.items() if k!='trials'}})
            else:
                clip = json.loads((run/bid/'results.json').read_text())
                expected=clip['results']['video'][method]
                value=score_events(labels,events[method],negatives,expected['video_seconds'],study['navigation_post_attempt_tolerance_seconds'])
                if value!=expected:
                    raise ValueError(f'Recomputed metrics differ: {bid}/{method}')
                scores[method].append(value)
                by_condition.append({'block_id':bid,'condition':spec['condition'],'method':method,
                    **{k:value[k] for k in ('attempts','correct','missed','wrong_direction','duplicates','unmatched_false_commands','predicted_commands','precision','recall','f1')}})
                trials.extend({'block_id':bid,'condition':spec['condition'],'method':method,**t} for t in value['trials'])
    total={m:aggregate(v) for m,v in scores.items()}
    speed={m:summarize_speed(v) for m,v in timing.items()}
    if total!=original['scores']['video'] or speed!=original['speed']:
        raise ValueError('Reproduced totals or saved timing summary differs')
    output.mkdir(parents=True)
    result={'status':'reproduced_exactly_from_verified_native_feature_cache','scores':total,'speed':speed,
        'controls':controls,'parity':parity,'source_run':str(run.relative_to(ROOT)),
        'new_benchmark_performed':False,
        'speed_scope':'Summary recomputed from the original native-run timing samples; not a new timing measurement.',
        'accuracy_scope':'Both unchanged processors rerun on verified shared fresh-video observations; event frames and metrics exactly match the original Phase 2 video run.'}
    (output/'reproduced_comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    write_csv(output/'comparison_by_condition.csv',by_condition)
    write_csv(output/'trial_log.csv',trials)
    write_csv(output/'controls.csv',controls)
    print(json.dumps({'status':result['status'],'scores':total},indent=2))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    reproduce(args.run,args.output)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
