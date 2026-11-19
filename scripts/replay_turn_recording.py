"""Replay recorded observations without a camera, keyboard events, or uploads."""

import argparse
from collections import Counter
from dataclasses import fields
import csv
import json
from pathlib import Path

from gesturedeck.config import Config
from gesturedeck.processor import GestureProcessor, Observation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--labels", type=Path, help="Human-reviewed CSV, never predicted labels")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--policy", choices=("same_direction", "all_commands"), default="same_direction")
    args = parser.parse_args()
    folder = args.recording.resolve()
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text(encoding="utf-8").splitlines()]
    known = {field.name for field in fields(Config)}
    config = {key: value for key, value in manifest["config"].items() if key in known}
    config.update(navigation_mode="palm_turn", record_directory="", record_slideshow=False,
                  turn_rearm_policy=args.policy)
    processor = GestureProcessor(Config(**config))
    events = []
    estimated = False
    for row in rows:
        if not row["inference_ran"]:
            continue
        decision = row.get("decision_elapsed_seconds")
        if decision is None:
            # Early recording schemas saved capture time and inference duration,
            # but not the exact processor timestamp. Disclose this approximation.
            decision = row["elapsed_seconds"] + (row.get("inference_ms") or 0) / 1000
            estimated = True
        action = processor.update(Observation(**row["observation"]), decision)
        if action is not None:
            events.append({"frame_index": row["frame_index"], "capture_seconds": row["elapsed_seconds"],
                           "decision_seconds": decision, "action": action.value})
    reviewed = []
    if args.labels:
        with args.labels.open(newline="", encoding="utf-8") as stream:
            for label in csv.DictReader(stream):
                expected = {"palm_to_back": "next", "back_to_palm": "previous"}.get(label["label"])
                if expected is None:
                    continue
                start, end = int(label["start_frame"]), int(label["end_frame"])
                matches = [event for event in events if start <= event["frame_index"] <= end
                           and event["action"] == expected]
                reviewed.append({**label, "recovered": bool(matches), "matched_events": matches})
    original = [row for row in rows if row["predicted_action"] in ("next", "previous")]
    preserved = sum(any(event["action"] == row["predicted_action"]
                        and abs(event["capture_seconds"] - row["elapsed_seconds"]) <= 0.4
                        for event in events) for row in original)
    summary = {
        "recording": str(folder), "frames": len(rows), "policy": args.policy,
        "events": dict(Counter(event["action"] for event in events)),
        "original_navigation_events": len(original), "original_events_preserved_within_400ms": preserved,
        "reviewed_turns": len(reviewed), "reviewed_turns_recovered": sum(case["recovered"] for case in reviewed),
        "decision_time_estimated": estimated, "independent_accuracy_estimate": False,
        "note": "Same-recording replay; not live accuracy or independently confirmed slide response.",
    }
    report = {"summary": summary, "reviewed_cases": reviewed, "events": events}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
