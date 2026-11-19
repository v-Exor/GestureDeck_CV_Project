"""Export hand joints from an existing recording's saved landmarks, without inference.

The raw recording and its manifest are never modified. Outputs stay beneath this
project's recordings/ directory and existing output files are never overwritten.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import cv2

from gesturedeck.preview import draw_recorded_landmarks

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_recording(recording, output=None, *, root=ROOT):
    recording = Path(recording).resolve()
    recording_root = (Path(root) / "recordings").resolve()
    if recording == recording_root or not recording.is_relative_to(recording_root):
        raise ValueError("Input must be a session beneath this project's recordings/ directory")
    manifest_path = recording / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Stop the recording and wait for completion before exporting its joints")
    sources = {}
    for key in ("video", "frames"):
        source = (recording / manifest[key]).resolve()
        if not source.is_relative_to(recording) or not source.is_file():
            raise ValueError("Recording source missing or outside the session directory")
        sources[key] = source
    output = Path(output).resolve() if output is not None else recording / "review/webcam_landmarks.avi"
    metadata_path = output.with_suffix(".json")
    if not output.is_relative_to(recording_root) or output.suffix.lower() != ".avi":
        raise ValueError("Output must be an AVI beneath this project's recordings/ directory")
    if output.exists() or metadata_path.exists():
        raise ValueError("Output already exists; preserve it and choose a new filename")
    fps = float(manifest["nominal_video_fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("Invalid nominal frame rate")
    input_hashes = {str(path.relative_to(recording)): digest(path)
                    for path in (manifest_path, *sources.values())}
    capture = cv2.VideoCapture(str(sources["video"]))
    writer = None
    count = hand_frames = 0
    try:
        if not capture.isOpened():
            raise ValueError("Could not open recorded webcam video")
        output.parent.mkdir(parents=True, exist_ok=True)
        with sources["frames"].open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["frame_index"] != count:
                    raise ValueError("Frame log indexes must be contiguous and zero-based")
                ok, frame = capture.read()
                if not ok:
                    raise ValueError("Raw video ended before its matching frame log")
                height, width = frame.shape[:2]
                if (width, height) != (manifest["width"], manifest["height"]):
                    raise ValueError("Video dimensions differ from the recording manifest")
                if writer is None:
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                    if not writer.isOpened():
                        raise RuntimeError("Could not open the hand-landmark video writer")
                hand_frames += int(draw_recorded_landmarks(frame, row["observation"].get("landmarks")))
                writer.write(frame)
                count += 1
        if count != manifest["frame_count"] or count == 0 or capture.read()[0]:
            raise ValueError("Raw video, frame log and manifest counts must agree")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if any(digest(recording / name) != expected for name, expected in input_hashes.items()):
        raise ValueError("Source recording changed during export")
    metadata = {
        "status": "completed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_recording": str(recording.relative_to(recording_root)),
        "source_sha256": input_hashes, "output": output.name, "output_sha256": digest(output),
        "frame_count": count, "hand_landmark_frames": hand_frames, "nominal_video_fps": fps,
        "landmarks_source": "Original frames.jsonl observation.landmarks; no new inference or interpolation",
        "timestamp_authority": "Original frames.jsonl elapsed_seconds, not AVI playback time",
        "raw_files_modified": False, "camera_opened": False, "keys_sent": False,
        "training_performed": False,
    }
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
    return {"video": str(output), "metadata": str(metadata_path), **metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(export_recording(args.recording, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
