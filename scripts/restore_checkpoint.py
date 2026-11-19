"""Restore the pre-trial checkpoint, preserving current changes in a new archive."""

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4


CHECKPOINT = "backups/20261020-before-learned-trial"


def inside(root, relative):
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError(f"Path escapes project: {relative}")
    return path


def verify(root, snapshot):
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    if not snapshot.is_relative_to(root / "backups"):
        raise ValueError("Checkpoint must be inside this project's backups folder")
    manifest = json.loads((snapshot / "checkpoint.json").read_text(encoding="utf-8"))
    if not manifest.get("files"):
        raise ValueError("Empty checkpoint manifest")
    for relative, expected in manifest["files"].items():
        source = inside(snapshot, relative)
        target = inside(root, relative)
        if target.is_relative_to(root / "backups"):
            raise ValueError("Checkpoint cannot overwrite its recovery files")
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint verification failed: {relative}")
    return manifest


def restore(root, snapshot):
    root, snapshot = Path(root).resolve(), Path(snapshot).resolve()
    manifest = verify(root, snapshot)  # Verify ALL source files before writes.
    added = json.loads((snapshot / "trial_files.json").read_text(encoding="utf-8"))
    for relative in added:
        target = inside(root, relative)
        if relative in manifest["files"] or target.is_relative_to(root / "backups"):
            raise ValueError(f"Invalid trial-only file: {relative}")
    archive = root / "backups" / ("before-undo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    archive.mkdir(parents=True, exist_ok=False)
    changed = []
    for relative, expected in manifest["files"].items():
        target = inside(root, relative)
        if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
            continue
        if target.exists():
            saved = inside(archive, relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
        changed.append(relative)
    # Preserve every trial-only file before moving any of them out of the project.
    for relative in added:
        target = inside(root, relative)
        if target.is_file():
            saved = inside(archive, relative)
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, saved)
    for relative in changed:
        target = inside(root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(inside(snapshot, relative), target)
    for relative in added:
        target = inside(root, relative)
        if target.is_file():
            # Both resolved paths have been checked to remain in this workspace.
            moved = inside(archive, "trial-files/" + relative)
            moved.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(target), str(moved))
    (archive / "restore_log.json").write_text(json.dumps({"restored": changed, "trial_files": added}, indent=2), encoding="utf-8")
    (snapshot / "restored.json").write_text(json.dumps({"archive": str(archive)}, indent=2), encoding="utf-8")
    return archive, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the backup without restoring anything")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    snapshot = root / CHECKPOINT
    if args.check:
        print(f"Verified {len(verify(root, snapshot)['files'])} checkpoint files; no changes made.")
        return
    archive, changed = restore(root, snapshot)
    print(f"Restored {len(changed)} original files.")
    print(f"Preserved current edits and trial files in: {archive}")
    print("Close GestureDeck and restart it using your usual desktop shortcut.")


if __name__ == "__main__":
    main()
