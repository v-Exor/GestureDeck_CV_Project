# Milestone 4 demonstration results

The current manual scope contains 11 executed cases and ends with fast turns. Ten cases have Pass outcomes and the fast-turn case has a Fail outcome. Three hard-condition cases cover farther distance, participant-described dim background, and faster turning.

## Observed results

| Case | Observation | Outcome |
| --- | --- | --- |
| TC01 | Palm-to-back turns visibly advance slides | Pass |
| TC02 | Back-to-palm turns visibly reverse slides | Pass |
| TC03 | Five alternating pairs in the farther-standing clip yield ten correct commands and responses | Pass |
| TC04 | Stationary palm and back intervals last 5.36 and 5.43 seconds without commands or slide movement | Pass under revised five-second criterion |
| TC05 | Two fist-pause trials block navigation during paused hand motion | Pass |
| TC06 | Two resume trials allow a new starting side followed by visible navigation | Pass |
| TC07 | Victory exits the slideshow; participant confirms the editor stays open | Pass |
| TC08 | Approximately 9.7 seconds of active no-hand/body motion produce no navigation commands | Pass |
| TC10 | Farther standing: ten correct turns; the earlier standing clip retains eight correct turns | Pass |
| TC11 | Participant-described dim background: four pairs and eight matching slide changes | Pass for this functional trial |
| TC12 | First four physical fast-turn pairs: five recognized turns and three misses | Fail |

IDs stay stable so evidence references identify the same cases.

## Fast-turn failure

The participant intended four pairs. Video review identifies 13 palm/back cycles in the full sequence. The first four pairs remain separate from the additional observed movements.

| Scope | Physical turns | Recognized | Missed |
| --- | ---: | ---: | ---: |
| First four pairs | 8 | 5 | 3 |
| Entire sequence | 26 | 20 | 6 |

Five NEXT turns and one PREVIOUS turn are missed across the full clip. No wrong-direction, duplicate or unmatched navigation commands were identified. Each miss coincides with a valid-observation gap above the configured 0.14-second grace, after which the returning orientation becomes a fresh starting side.

Twenty commands report successful key delivery. Independent slide snapshots show 16 visible changes. Four PREVIOUS keys occur at the first-slide boundary and produce no visible change.

## Scope and limits

- These are inspected functional recordings with predictions known during review, not blind independent held-out evaluation.
- Distance and illumination were not measured. Lighting descriptions come from the participant.
- The stationary criterion changed from ten seconds to five seconds after capture. The report discloses how the observed shorter holds meet the revised criterion.
- The final nominal procedure is four pairs. Earlier actual counts, timing deviations and misses remain recorded.
- Feature replay reproduces recorded action frames and states. It is not fresh native-model inference or a speed benchmark.
- Raw videos and detailed session evidence remain in the development project's ignored `recordings/` folder and are not bundled in this source-only repository.
- At the Milestone 4 source snapshot, the final comparison and demo-computer benchmark remained unfinished. The later [Milestone 5 reproduction](FINAL_REPRODUCTION.md) adds the inspected feature-replay comparison and fresh offline CPU benchmark with their stated scope. Independent held-out generalization remains unestablished.

Joint-overlay videos and actual detected-joint stills support the classroom report and presentation. Missing tracking never receives invented or carried-forward joints.
