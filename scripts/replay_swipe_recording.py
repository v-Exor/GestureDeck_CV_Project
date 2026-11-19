"""Replay recorded observations offline; never open a webcam or send slide keys."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from gesturedeck.config import Config
from gesturedeck.processor import GestureProcessor, Observation


def replay(directory, model=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    settings = {k: v for k, v in manifest["config"].items() if k in {f.name for f in fields(Config)}}
    settings.update(audio_feedback=False, record_directory="", record_slideshow=False)
    if settings.get("swipe_model_path"):
        settings["swipe_model_path"] = str((model or directory / "swipe_model.npz").resolve())
    processor = GestureProcessor(Config(**settings))
    events = []
    counts = {}
    for line in (directory / "frames.jsonl").open():
        row = json.loads(line)
        if not row.get("inference_ran"):
            continue
        observation = Observation(**row["observation"])
        now = row.get("decision_elapsed_seconds") or row["elapsed_seconds"]
        action = processor.update(observation, now)
        counts[processor.status] = counts.get(processor.status, 0) + 1
        if action:
            events.append({"frame_index": row["frame_index"], "seconds": now, "action": action.value})
    return {"events": events, "status_counts": counts, "keys_sent": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = replay(args.recording, args.model)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["events"], indent=2))
