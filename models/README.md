# Pretrained model

Historical temporal swipe implementations remain in `src/gesturedeck/` for
research. Their trained candidate weights and datasets are not bundled in this
source repository. The retained Project Use launcher uses right-hand palm/back
rules and the official MediaPipe bundle described below.

From the project root run `python scripts/download_model.py`. This explicitly
downloads the official MediaPipe Gesture Recognizer **float16 version 1** bundle
from Google's model storage, then checks a pinned SHA-256. The app never downloads
models automatically. The binary is ignored by Git.

- Official model documentation: https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer#models
- Versioned asset: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task
- SHA-256: `97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482`

The checksum was computed from the official HTTPS download during development;
it pins the tested model bytes, rather than claiming to be a publisher signature.
The documentation's `latest` link can change, so the script uses version 1.
Review the official model card and applicable terms on the model documentation
page before redistributing the model. No model training is performed here.
