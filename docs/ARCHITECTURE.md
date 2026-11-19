# GestureDeck architecture

## Current application

`GestureDeck-Learned Trial.pyw` anchors paths to the project folder and opens the Tkinter interface in dry run. Its arguments select right-hand palm/back navigation and an eight-second live countdown. Normal desktop use runs it with the local environment's `pythonw.exe`.

`desktop.py` owns the interface. `session.py` manages the background camera worker, model lifetime, latest preview snapshot, active state and cleanup. Tk updates stay on the main thread.

## Recognition and control

`camera.py` supplies normalized 640-by-480 frames. `recognizer.py` runs the official MediaPipe task on the CPU, returning landmarks and static-pose scores. `hand.py` derives geometry and finger extension.

`turn.py` determines palm/back orientation from projected landmarks and confirms endpoints over time. `processor.py` combines transitions with held fist/Victory controls, starting-side rules and repeat protection.

`focus.py` checks foreground slideshow focus. `controller.py` logs dry-run actions or sends guarded PowerPoint keys. `feedback.py` provides asynchronous Windows audio state feedback. Successful dispatch remains distinct from a visible slide response.

## Local evidence

`preview.py` draws detected joints. `recording.py` preserves raw frames and a matching same-frame joint video, with events and timestamps. `slideshow_recording.py` optionally captures the visible focused slideshow region. `annotate_recording.py` renders overlays from saved landmarks without new inference, webcam use or key delivery.

Recording is opt-in and stays beneath `recordings/`. JSONL capture/decision timestamps are authoritative; AVI playback uses nominal 20 FPS.

## Historical experiments

Historical swipe methods, training utilities and the existing swipe evaluator remain in the source. The retained interface does not expose them as everyday navigation choices. Their datasets, candidate weights, historical caches and recorded evidence are not bundled in this source-only repository. Current palm-turn navigation requires the official MediaPipe task and no custom temporal model.

## Verification boundary

Automated tests cover recognition rules, state changes, mocked camera/key delivery, interface behavior, recording and replay. A separate model check processes synthetic frames. Reviewed classroom trials document visible responses and misses. Independent current-mode accuracy and a current laptop benchmark require their own evidence.
