"""Opt-in local webcam/data recorder, with optional slideshow-only snapshots."""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import shutil
from time import perf_counter
from uuid import uuid4

import cv2

from gesturedeck.preview import draw_recorded_landmarks


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_finite(item) for item in value]
    return value


class SessionRecorder:
    """One video frame per JSONL row. JSON timestamps are the timing authority.

    AVI has a nominal 20 FPS timebase; do not infer gesture timing from playback
    speed. webcam.avi preserves camera input before the landmark overlay;
    webcam_landmarks.avi adds the same frame's saved hand joints and connections.
    Predictions are deliberately separate from human-reviewed training labels.
    """

    def __init__(self, config, mode):
        self.config, self.mode = config, str(getattr(mode, "value", mode))
        self.path = Path(config.record_directory).resolve() / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        self.writer = None
        self.annotated_writer = None
        self.annotated_frame_count = 0
        self.annotated_hand_frames = 0
        self.stream = None
        self.command_stream = None
        self.command_writer = None
        self.command_count = 0
        self.frame_count = 0
        self.started = perf_counter()
        self.shape = None
        self.slideshow = None
        self.manifest = {
            "schema_version": 5, "status": "recording", "mode": self.mode,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "config": asdict(config), "video": "webcam.avi", "frames": "frames.jsonl",
            "annotations": "annotations.csv", "nominal_video_fps": 20,
            "commands": "commands.csv", "attempt_review": "attempts_review.csv",
            "annotated_video": "webcam_landmarks.avi",
            "annotated_pixels": "Copy of webcam.avi input with this frame's detected hand joints and connections; no extra inference or mirroring",
            "timestamp_authority": "frames.jsonl: elapsed_seconds (monotonic)",
            "decision_timestamp": "frames.jsonl: decision_elapsed_seconds (same clock, after inference/before key dispatch)",
            "pixels": "BGR webcam frames before drawing; mirror matches config",
            "landmarks": "model input normalized image x,y,z; positions may be mirror-normalized separately",
            "recording_scope": ("raw and hand-landmark webcam videos, inference/events, and timed PowerPoint slideshow snapshots; no audio or other apps"
                                if config.record_slideshow else "raw and hand-landmark webcam videos and inference/events only; no audio or desktop"),
            "prediction_labels_are_ground_truth": False,
            "human_review_required": True,
        }

    def __enter__(self):
        self.path.mkdir(parents=True, exist_ok=False)
        try:
            self.stream = (self.path / "frames.jsonl").open("x", encoding="utf-8")
            self.command_stream = (self.path / "commands.csv").open("x", newline="", encoding="utf-8")
            self.command_writer = csv.DictWriter(self.command_stream, fieldnames=[
                "frame_index", "capture_seconds", "decision_seconds", "action", "outcome", "active", "status", "last_action"])
            self.command_writer.writeheader()
            self.command_stream.flush()
            with (self.path / "attempts_review.csv").open("x", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["attempt_id", "start_frame", "end_frame", "intended_action",
                                            "observed_action", "dispatch_outcome", "slide_change", "verdict", "notes"])
            if self.config.swipe_model_path:
                source = Path(self.config.swipe_model_path)
                target = self.path / "swipe_model.npz"
                shutil.copy2(source, target)
                self.manifest["swipe_model_snapshot"] = {
                    "file": target.name, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                    "source": str(source.resolve()), "snapshot_time": "recording_start"}
            with (self.path / "annotations.csv").open("x", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(["start_frame", "end_frame", "label", "notes"])
            (self.path / "README.txt").write_text(
                "LOCAL GESTUREDECK TEST RECORDING\n"
                "webcam.avi contains unannotated model-input webcam frames; no audio.\n"
                "webcam_landmarks.avi shows the same frames with yellow hand joints and green connections.\n"
                "Both videos have one frame per frames.jsonl row, with the same index and nominal FPS.\n"
                "The overlay uses that frame's saved landmarks; no extra model inference or mirror is applied.\n"
                "No skeleton is drawn when complete, finite landmarks are unavailable.\n"
                "Video uses nominal 20 FPS. Actual capture times are in frames.jsonl.\n"
                "Frame numbers are zero-based; AVI frame N matches JSONL frame_index N.\n"
                "Frames include pose/landmarks, facing score, detector status, active state,\n"
                "candidate action and actual dispatch result. Null delivered means no dispatch.\n"
                "A delivered dry-run event is logged, not a physical key (check mode).\n"
                "commands.csv lists each command and whether it was dry-run logged, sent, or blocked.\n"
                "key_sender_succeeded is NOT proof of a visible slide change.\n"
                "attempts_review.csv is intentionally empty until the video is reviewed.\n"
                "Missed, wrong and duplicate gestures require reviewed intent labels.\n"
                "Predictions are NOT training labels. Review the video and fill annotations.csv.\n"
                "Suggested labels: swipe_left, swipe_right, stationary_palm, finger_wiggle,\n"
                "palm_to_back, back_to_palm, stationary_back,\n"
                "fist, victory, other, no_hand. Use inclusive frame ranges.\n"
                "Keep whole recording sessions in one train/validation/test split.\n"
                "Stop session or close GestureDeck to finalize the AVI and manifest.\n",
                encoding="utf-8")
            self._manifest()
            if self.config.record_slideshow:
                from gesturedeck.slideshow_recording import SlideshowRecorder
                self.slideshow = SlideshowRecorder(self.path, self.started)
                self.slideshow.start()
                with (self.path / "README.txt").open("a", encoding="utf-8") as stream:
                    stream.write("\nSLIDESHOW SNAPSHOTS ENABLED: slides.jsonl links timed PowerPoint-only images.\n"
                                 "Snapshots and webcam frames share elapsed_seconds. Other apps are not captured.\n"
                                 "Inspect the images to verify slide response; key delivery alone is not proof.\n")
        except Exception:
            if self.slideshow is not None:
                self.slideshow.close()
            if self.stream is not None:
                self.stream.close()
            if self.command_stream is not None:
                self.command_stream.close()
            raise
        return self

    def _manifest(self):
        self.manifest["frame_count"] = self.frame_count
        self.manifest["annotated_frame_count"] = self.annotated_frame_count
        self.manifest["annotated_hand_frames"] = self.annotated_hand_frames
        self.manifest["command_count"] = self.command_count
        self.manifest["elapsed_seconds"] = perf_counter() - self.started
        if self.slideshow is not None:
            self.manifest["slideshow_capture"] = self.slideshow.summary()
        temporary = self.path / "manifest.tmp"
        temporary.write_text(json.dumps(_finite(self.manifest), indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.path / "manifest.json")

    def write(self, frame, observation, *, captured_at, active, status, phase,
              action=None, delivered=None, inference_ms=None, inference_ran=False,
              last_action="", fps=0, navigation_state=None, decision_at=None):
        if self.writer is None:
            self.shape = frame.shape
            height, width = frame.shape[:2]
            self.writer = cv2.VideoWriter(str(self.path / "webcam.avi"),
                                         cv2.VideoWriter_fourcc(*"MJPG"), 20, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError("Recording failed: could not open the local MJPEG video writer")
            self.annotated_writer = cv2.VideoWriter(str(self.path / "webcam_landmarks.avi"),
                                                  cv2.VideoWriter_fourcc(*"MJPG"), 20, (width, height))
            if not self.annotated_writer.isOpened():
                raise RuntimeError("Recording failed: could not open the hand-landmark video writer")
            self.manifest["width"], self.manifest["height"] = width, height
        if frame.shape != self.shape:
            raise RuntimeError("Recording stopped: camera resolution changed mid-session")
        row = {
            "frame_index": self.frame_count,
            "elapsed_seconds": captured_at - self.started,
            "decision_elapsed_seconds": decision_at - self.started if decision_at is not None else None,
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "observation": asdict(observation), "inference_ran": inference_ran,
            "active": active, "status": status, "phase": phase,
            "predicted_action": getattr(action, "value", action), "delivered": delivered,
            "last_action": last_action, "inference_ms": inference_ms, "fps": fps,
            "ground_truth": None,
            "navigation_state": navigation_state,
        }
        self.writer.write(frame)
        annotated = frame.copy()
        self.annotated_hand_frames += int(draw_recorded_landmarks(annotated, observation.landmarks))
        self.annotated_writer.write(annotated)
        self.annotated_frame_count += 1
        self.stream.write(json.dumps(_finite(row), allow_nan=False) + "\n")
        self.stream.flush()
        if action is not None:
            label = getattr(action, "value", action)
            outcome = ("control_state_changed" if label == "toggle" else
                       "blocked" if delivered is False else
                       "not_dispatched" if delivered is not True else
                       "dry_run_logged" if "dry" in self.mode.lower() else "key_sender_succeeded")
            self.command_writer.writerow({"frame_index": self.frame_count,
                                          "capture_seconds": captured_at - self.started,
                                          "decision_seconds": decision_at - self.started if decision_at is not None else "",
                                          "action": label, "outcome": outcome, "active": active,
                                          "status": status, "last_action": last_action})
            self.command_stream.flush()
            self.command_count += 1
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self._manifest()

    def __exit__(self, exc_type, exc, _):
        if self.slideshow is not None:
            self.slideshow.close()
        if self.writer is not None:
            self.writer.release()
        if self.annotated_writer is not None:
            self.annotated_writer.release()
        if self.stream is not None:
            self.stream.close()
        if self.command_stream is not None:
            self.command_stream.close()
        self.manifest["status"] = "error" if exc_type else "completed"
        self.manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        if exc is not None:
            self.manifest["error"] = str(exc)
        self._manifest()
