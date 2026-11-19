# Final palm-turn reproduction

Use the final `v1.0` source snapshot and the environment installation in the root README. The application retains its internal 0.2.1 version label. The final tag identifies the coursework snapshot, not a claim of deployment readiness.

## One command

The public repository contains source and documentation. The reduced observations, frame/event traces and measured timing samples are provided privately in **GestureDeck_Evaluation_Data.zip** inside the teacher submission package. Extract that ZIP into the repository root before evaluation. It installs `recordings/milestone5_evaluation_inputs/` without changing source code. The public clone alone cannot reproduce the study without those inputs.

From the repository root after installation and extraction of the private data ZIP:

```powershell
.\.venv\Scripts\python.exe -m gesturedeck.final_evaluate --output recordings/milestone5_reproduced
```

Outputs must stay under a new project `recordings/` subdirectory. Existing results are not overwritten. The command verifies the study's SHA-256 input/source hashes, replays the real current processor and baseline, checks exact captured-event parity, and generates JSON and CSV results. No camera, GUI, key sending, training or native model inference occurs.

## Expected table

| Method | Attempts | Predicted | Correct | Missed | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Endpoint baseline | 52 | 17 | 17 | 35 | 1.0000 | 0.3269 | 0.4928 |
| Temporal final | 52 | 46 | 46 | 6 | 1.0000 | 0.8846 | 0.9388 |

Both methods have zero wrong-direction, duplicate and unmatched events in these inspected clips. Final confusion rows are true NEXT and PREVIOUS, columns predicted NEXT, PREVIOUS and missed: `[[21,0,5],[0,25,1]]`.

| Condition | Attempts | Baseline correct | Final correct |
| --- | ---: | ---: | ---: |
| Normal standing | 8 | 0 | 8 |
| Farther standing | 10 | 1 | 10 |
| Participant-described dim background | 8 | 7 | 8 |
| Fast turns, full clip | 26 | 9 | 20 |

The first four fast pairs contain five correct turns out of eight. The full recording contains thirteen observed cycles although the participant intended four. Additional movements and misses remain included in the full-sequence comparison. The observed stationary criterion changed to five seconds after capture; earlier actual durations are retained.

## Inputs and labels

The privately supplied `recordings/milestone5_evaluation_inputs/` folder contains four full scalar-observation sequences, recorded settings, reviewed approximate motion intervals, negative intervals and a separate captured-event parity list. There are 3,101 frame records and 52 balanced physical-turn labels from one participant on 19 November 2026. No current training or train/validation/test split exists. Labels were reviewed with predictions known, and all clips had been inspected during development. These are non-blind functional replay data, with no independent held-out or population-accuracy claim.

Camera pixels, audio, face images and complete hand-joint coordinate arrays are absent from the portable cache. The study preserves original video/log/label hashes. Original raw and annotated recordings remain local; the teacher package carries selected actual joint-overlay stills and excerpts. No external dataset is redistributed. No separate reuse licence is assigned to the team-derived observations.

## Comparator and matching

The endpoint baseline shares the current processor's open-hand flag, facing threshold of 0.20 and observations. One qualifying sample establishes each endpoint; an opposite side produces a navigation event. Actual missing-hand observations immediately reset its starting side. Unclear detected frames add no side evidence. It has no endpoint-duration confirmation or tracking-gap bridge. This is a minimal engineering comparator selected after the clips had already been inspected and frozen before running this comparison. It is not an optimized competing system.

Chronological matching assigns the first event within each labeled interval plus a 0.75-second post-attempt allowance, capped before the next attempt begins. Wrong first directions remain wrong. Later matched commands are duplicates. Unmatched events remain false. Negative rate uses 45.2689538 seconds of reviewed stationary/no-hand activity, with zero observed events. This duration does not validate the original ten-minute target.

## Fresh benchmark and its reproduction

Fresh native CPU inference was measured on decoded normal-standing and fast-turn raw clips on an AMD Ryzen 5 3400G, four cores/eight logical processors, approximately 16 GB RAM, Windows 11 Pro 10.0.26200, Python 3.11.3, OpenCV 5.0.0 and NumPy 2.4.6. Resolution is 640 by 480. Model initialization and the first twenty actual frames of each clip are excluded, leaving 1,307 measured samples.

| Method | Mean ms | Median ms | p95 ms | Offline analysis FPS |
| --- | ---: | ---: | ---: | ---: |
| Endpoint baseline | 32.41 | 28.21 | 53.98 | 30.85 |
| Temporal final | 32.43 | 28.23 | 53.99 | 30.83 |

Timing includes file decode, BGR-to-RGB conversion, fresh native inference and one processor. Both methods share each frame's inference. It excludes live camera, GUI, overlay drawing, encoding, snapshots, sound, keys and physical slide response. Throughput is 1,000/mean milliseconds. This is not full application FPS or end-to-end latency.

The command recomputes these statistics from the privately supplied pinned `benchmark/timings.csv` samples and verified metadata. It does not remeasure speed or regenerate native observations without source videos. A new native rerun may yield different observations/timing. The official model is available through `scripts/download_model.py` with its pinned checksum; see `models/README.md`.

## Remaining validation limits

The current sampled precision and per-direction recalls exceed the proposal's numeric thresholds. Its broader independent participant/session coverage, ten-minute negative period and median end-to-command latency target have not been established. Manual capture ends at fast turns. Tests and replay do not substitute for the team's live presentation or hardware rehearsal.
