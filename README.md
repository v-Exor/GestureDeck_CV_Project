# GestureDeck

**A classroom computer vision prototype for controlling a Windows PowerPoint slideshow with an open right hand.**

GestureDeck combines a webcam, the pretrained MediaPipe Gesture Recognizer, and local geometric and temporal rules. Its white-and-orange desktop interface shows moving hand joints, recognition feedback, control state, and approximate live performance.

The current presentation mode is **Project Use**. The retained launcher is named **GestureDeck - Learned Trial** for compatibility, but its everyday interface uses right-hand palm/back turns. Historical swipe implementations remain in the source for research and evaluation.

## What to expect

- Open right-hand turns move one slide forward or backward.
- A held fist pauses or resumes gesture control.
- A held Victory pose exits the slideshow while leaving the presentation open.
- The preview shows actual detected joints, and optional local recording saves matching evidence.
- The app starts in dry run, with the session stopped, live keys unchecked, and recording off.

This is a guided classroom demonstration prototype. Fast turns can lose tracking and produce missed commands. Inspected trials do not establish general recognition accuracy or deployment reliability. The current palm-turn navigation does not use a custom trained temporal classifier.

## Requirements

| Component | Expected setup |
| --- | --- |
| Operating system | Windows x64 |
| Python | Python 3.11, as required by the project metadata |
| Presentation app | Desktop Microsoft PowerPoint for live slideshow control |
| Camera | A webcam accessible to Windows desktop apps |
| Compute | CPU inference; no GPU required by this implementation |
| Initial setup | Internet access for dependency installation and the explicit official-model download |

After setup, recognition runs locally. The app does not upload webcam footage or download models automatically at launch.

## Install from a fresh clone

Open PowerShell in the folder where you want the project:

```powershell
git clone https://github.com/v-Exor/GestureDeck_CV_Project.git
cd GestureDeck_CV_Project
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps -e .
.\.venv\Scripts\python.exe scripts\download_model.py
.\.venv\Scripts\python.exe -m pip check
```

The model downloader retrieves the official versioned MediaPipe task and verifies its pinned SHA-256. Model binaries and development environments are not committed. See [model provenance](models/README.md). If you already have the working development environment, keep using it rather than recreating it.

## Open Project Use

From the project root:

```powershell
& ".\.venv\Scripts\pythonw.exe" ".\GestureDeck-Learned Trial.pyw"
```

This opens the desktop app without a command window. If the **GestureDeck - Learned Trial** desktop shortcut already exists on your computer, use it. A Git clone does not include the original computer's desktop shortcut. Keep this as the single everyday launcher.

The retained launcher selects **Project Use**, an eight-second live countdown, and dry-run mode.

## First demonstration

1. Open a disposable PowerPoint deck and start its slideshow near the middle.
2. Open GestureDeck. First use **Recognition / dry run** to check the camera and joints without sending keys.
3. Select **PowerPoint / live keys** and check **Allow slide keys this session** when ready for real navigation.
4. Click **Start session**. Use **Alt+Tab** to focus the slideshow during the eight-second countdown.
5. Show your open right palm clearly to establish the initial side. This starting pose sends no navigation key.
6. Use the controls below. Keep the whole hand visible and show each flat endpoint clearly.
7. Return to GestureDeck and click **Stop session** when finished. Stop finalizes recordings and releases the camera.

Live keys require the focused PowerPoint slideshow. Its editor or another foreground app does not qualify. Clicking a slide may advance it, so Alt+Tab is the preferred way to restore slideshow focus.

## Gesture controls

| Input | Expected action |
| --- | --- |
| Open right hand, palm to back | NEXT slide |
| Open right hand, back to palm | PREVIOUS slide |
| First stable palm or back orientation | Establish the starting side, with no key |
| Hold either side stationary | No repeated navigation |
| Fist held for one second | Toggle pause or resume once |
| Victory held for two seconds | Exit the slideshow and leave control paused |

Lower the hand before reusing a held control pose. Confirmed opposite navigation turns can follow immediately. Turning back is a PREVIOUS action. To prepare another NEXT without reversing a slide, lower your hand and reset its orientation outside the camera view before showing the starting side again.

The pose-confidence display describes static-pose recognition. It is not a measured palm-turn accuracy percentage.

## Optional recording and moving joints

Recording is **off until enabled inside the app**. Check **Record webcam + landmarks locally** to capture evidence. Optionally check **Include PowerPoint slide snapshots**.

Each session stays beneath this project's `recordings/` folder:

| File | Contents |
| --- | --- |
| `webcam.avi` | Original webcam frames before drawing |
| `webcam_landmarks.avi` | Matching frames with actual detected joints and connections |
| `frames.jsonl` | Frame indexes, capture/decision times, landmarks, state and events |
| `commands.csv` | Recognized commands and delivery outcomes |
| `manifest.json` | Settings, dimensions, counts and completion status |
| `slides.jsonl` and `slides/` | Optional timed snapshots of the focused slideshow |

Use **Stop session** to finalize capture. Victory exits the slideshow but does not stop recording. Uncheck recording afterward. No microphone audio is captured. Recording does not train a model automatically.

Use joint-overlay videos and matching stills in demonstrations. Missing detections remain without a skeleton. Use JSONL capture timestamps for timing rather than nominal 20 FPS AVI playback. A successful key-delivery result alone does not prove a visible slide change.

Personal recordings, screenshots, datasets, generated reports and local model binaries are excluded from this source repository. See the [recording guide](docs/LANDMARK_RECORDINGS.md).

## Current demonstration results

The reviewed manual scope ends at fast turns: **11 cases, ten Pass outcomes and one fast-turn Fail outcome**, including three hard-condition cases.

| Inspected condition | Recognized physical turns |
| --- | ---: |
| Standing | 8 of 8 |
| Farther standing | 10 of 10 |
| Participant-described dim background | 8 of 8 |
| Fast turns, first four physical pairs | 5 of 8 |
| Fast turns, entire recorded sequence | 20 of 26 |

The full fast-turn sequence contains 13 visible cycles although four pairs were intended. It retains six missed turns. No wrong-direction or duplicate navigation commands were found in that inspected sequence. Twenty successful key deliveries produced 16 visible slide changes; four PREVIOUS deliveries occurred at the first-slide boundary.

These are reviewed functional observations, not independent held-out accuracy, a current speed benchmark, or a guarantee for other environments. See [results and limitations](docs/MILESTONE4_RESULTS.md).

## Verify the source

After installation, run from the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe scripts\check_model.py
```

Tests exercise logic, mocked camera/key delivery, UI state, recording and replay. The model smoke check runs CPU inference on synthetic frames. These checks do not replace real-hand demonstration or independently measure slide-control accuracy.

The existing `gesturedeck.evaluate` and `gesturedeck.reproduce_comparison` modules concern historical swipe studies. Their referenced private evidence and model snapshots are not bundled in a source-only clone. They are not a completed evaluator for the current palm-turn mode. A tag pins source bytes; it does not make unavailable evaluation assets reproducible.

## Project structure

```text
GestureDeck-Learned Trial.pyw   Retained desktop launcher
src/gesturedeck/
  desktop.py                  White-and-orange Project Use interface
  session.py                  Background capture and session lifecycle
  camera.py                   Webcam access and frame normalization
  recognizer.py               MediaPipe inference
  hand.py                     Hand geometry and extended fingers
  turn.py                     Palm/back transition rules
  processor.py                Navigation events and deliberate controls
  focus.py, controller.py     Guarded PowerPoint keys
  preview.py                  Hand landmark visualization
  recording.py                Raw and joint-overlay capture
  slideshow_recording.py      Optional visible slideshow snapshots
  annotate_recording.py       Offline overlay from saved landmarks
  swipe.py, learned_swipe.py  Historical swipe implementations
scripts/                      Model setup and research/replay utilities
tests/                        Automated tests and synthetic fixtures
docs/                         Current usage, architecture and evidence notes
models/README.md              Official model source and checksum
requirements-lock.txt         Pinned tested environment
pyproject.toml                Package metadata and entry point
```

See [architecture](docs/ARCHITECTURE.md) and [presentation mode](docs/PRESENTATION_MODE.md).

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing model at startup | Run `scripts\download_model.py` from the project root |
| No webcam image | Close competing camera apps, check Windows desktop-camera access and choose the correct camera index/backend |
| Gestures appear but slides do not move | Choose live mode, allow keys, wait for countdown and focus the actual slideshow |
| A turn is missed | Keep the complete open right hand visible and show stable flat endpoints |
| Control stays paused | Lower the hand, then hold a fist for one second |
| Capture continues after Victory | Return to GestureDeck and click Stop session |
| No audio feedback | Check Windows sound output and use the visible control state |

## Checkpoint status

This repository contains application source, tests, dependency pins and usage documentation. Current independent palm-turn evaluation, a demo-computer benchmark, the backup demonstration and complete checkpoint evidence packaging remain separate work. The 80% checkpoint label describes implementation progress, not recognition accuracy.

Developed for **CSST 106 Computer Vision**.
