# Hand joints in recorded videos

Use the joint-overlay videos and matching annotated still frames in reports,
slides, demonstrations and submissions. Retain the clean originals for review
and evaluation. Internal ground-truth review may use raw footage, but selected
hand-evidence images presented with the report should show the actual detected
joints. Missing tracking must remain visible as missing; do not fabricate joints.

The retained Project Use app now saves two webcam videos whenever recording is
enabled through its existing controls. Recording still starts off. No additional
launcher or camera connection is needed.

| File | Purpose |
| --- | --- |
| webcam.avi | Original camera input before drawing, preserved for review and evaluation |
| webcam_landmarks.avi | Matching frames with the same yellow joints and green hand connections shown in the live preview |
| frames.jsonl | Original frame indexes, actual capture and decision times, hand landmarks and command/state data |

Both AVI files have one frame per JSONL row and use the same nominal 20 FPS
playback rate. Actual capture times in frames.jsonl are authoritative. The joint
video draws the saved image coordinates for that frame; it does not run another
model, reverse mirroring or reuse a hand from an earlier frame. Missing or
invalid hand landmarks leave that frame without a skeleton.

The recording manifest now uses schema 5 and identifies the raw video separately
from annotated_video, annotated_frame_count and annotated_hand_frames. Earlier
recordings are not rewritten. Their schema and original pixels remain valid.
Both writers close on Stop session, app close or an error. A joint-video encoder
failure is surfaced as a recording error rather than silently claiming success.

Stop any running session and close and reopen GestureDeck - Learned Trial to
load the updated recorder. Keep Project Use selected, enable Record webcam +
landmarks locally when needed, and use the usual Start session and Stop session
controls. The joint video appears in the same session folder as webcam.avi.

The extra video requires additional disk space and encoding work. Gesture logic,
model bytes and thresholds are unchanged, but future measured FPS may include
that additional recording cost. Do not present earlier timing measurements as a
benchmark of this recording revision.

## Existing recordings

An offline utility can produce a joint video from an already completed session:

```powershell
.\.venv\Scripts\python.exe -m gesturedeck.annotate_recording recordings/SESSION_FOLDER
```

Its default output is review/webcam_landmarks.avi beneath that session folder,
with provenance in review/webcam_landmarks.json. It refuses existing output
files. The original camera video, frame log, manifest and command data are not
modified. It opens no camera, sends no keys and performs no training or new
model inference. This review overlay visualizes model detections; it does not
supply human ground-truth labels or establish a gesture's correctness.
