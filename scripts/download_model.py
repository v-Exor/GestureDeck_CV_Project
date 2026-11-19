"""Explicit, verified model setup from official Google storage; stdlib only."""

import argparse
import hashlib
from pathlib import Path
import sys
import urllib.error
import urllib.request

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
    "gesture_recognizer/float16/1/gesture_recognizer.task"
)
# SHA-256 recorded from the official version-1 HTTPS asset during development.
# This pins the tested bytes; it is not a publisher-provided signature.
MODEL_SHA256 = "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482"


def download_model(destination: Path) -> None:
    if destination.exists():
        if hashlib.sha256(destination.read_bytes()).hexdigest() == MODEL_SHA256:
            print(f"Official model already present and verified: {destination}")
            return
        raise ValueError(f"{destination} exists with different bytes; move it aside before downloading.")
    print(f"Downloading official MediaPipe model v1:\n{MODEL_URL}")
    with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
        if response.geturl() != MODEL_URL:
            raise ValueError("Unexpected model URL redirect; download rejected.")
        content = response.read(20_000_001)
    if hashlib.sha256(content).hexdigest() != MODEL_SHA256:
        raise ValueError("Model checksum mismatch; no file written. Check the official source.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create avoids overwriting another file even if it appeared during download.
    with destination.open("xb") as stream:
        stream.write(content)
    print(f"Saved {len(content):,} bytes to {destination}; SHA-256 verified.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("models/gesture_recognizer.task"))
    args = parser.parse_args()
    try:
        download_model(args.output)
    except (OSError, ValueError, urllib.error.URLError) as exc:
        print(f"Model setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
