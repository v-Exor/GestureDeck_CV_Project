"""Validated defaults plus optional TOML configuration (Python 3.11 stdlib)."""

from dataclasses import dataclass, field, fields
import math
from pathlib import Path
import tomllib

from gesturedeck.mapping import Action, CANNED_GESTURES, DEFAULT_MAPPING


@dataclass
class Config:
    model_path: str = "models/gesture_recognizer.task"
    camera_index: int = 0
    camera_backend: str = "dshow"
    width: int = 640
    height: int = 480
    mirror: bool = True
    confidence_threshold: float = 0.60
    pause_hold_seconds: float = 1.0
    exit_hold_seconds: float = 2.0
    release_seconds: float = 0.25
    cooldown_seconds: float = 0.8
    max_frame_gap_seconds: float = 0.5
    detection_confidence: float = 0.5
    presence_confidence: float = 0.5
    tracking_confidence: float = 0.5
    countdown_seconds: int = 5
    audio_feedback: bool = True
    swipe_min_distance: float = 0.22
    swipe_min_seconds: float = 0.12
    swipe_max_seconds: float = 0.90
    swipe_max_vertical: float = 0.15
    swipe_min_consistency: float = 0.80
    swipe_max_step: float = 0.18
    swipe_distance_floor: float = 0.08
    swipe_hand_distance: float = 1.30
    swipe_min_hand_size: float = 0.02
    swipe_tracking_grace_seconds: float = 0.12
    diagnostics: bool = False
    navigation_mode: str = "swipe"
    swipe_model_path: str = ""
    swipe_trial: bool = False
    turn_hold_seconds: float = 0.10
    turn_min_facing: float = 0.20
    turn_max_seconds: float = 2.5
    turn_tracking_grace_seconds: float = 0.14
    turn_rearm_seconds: float = 1.0
    turn_rearm_policy: str = "same_direction"
    turn_reacquire_speed: float = 4.5
    turn_reacquire_max_distance: float = 0.40
    record_directory: str = ""
    record_slideshow: bool = False
    mapping: dict[str, Action] = field(default_factory=lambda: DEFAULT_MAPPING.copy())

    def __post_init__(self) -> None:
        for name in ("confidence_threshold", "detection_confidence",
                     "presence_confidence", "tracking_confidence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a number between 0 and 1")
        for name in ("pause_hold_seconds", "exit_hold_seconds", "release_seconds",
                     "cooldown_seconds", "max_frame_gap_seconds", "swipe_min_distance",
                     "swipe_min_seconds", "swipe_max_seconds", "swipe_max_vertical",
                     "swipe_min_consistency", "swipe_max_step", "swipe_distance_floor",
                     "swipe_hand_distance", "swipe_min_hand_size", "swipe_tracking_grace_seconds",
                     "turn_hold_seconds", "turn_min_facing", "turn_max_seconds",
                     "turn_tracking_grace_seconds", "turn_rearm_seconds",
                     "turn_reacquire_speed", "turn_reacquire_max_distance"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be a finite positive number")
        for name, minimum in (("camera_index", 0), ("width", 320),
                              ("height", 240), ("countdown_seconds", 3)):
            value = getattr(self, name)
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.camera_backend not in ("auto", "dshow", "msmf"):
            raise ValueError("camera_backend must be auto, dshow, or msmf")
        for name in ("mirror", "audio_feedback", "diagnostics", "record_slideshow", "swipe_trial"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be true or false")
        for name in ("swipe_min_distance", "swipe_max_vertical", "swipe_min_consistency", "swipe_max_step"):
            if getattr(self, name) > 1:
                raise ValueError(f"{name} must be <= 1")
        if self.swipe_min_seconds >= self.swipe_max_seconds:
            raise ValueError("swipe_min_seconds must be less than swipe_max_seconds")
        if self.swipe_distance_floor > self.swipe_min_distance:
            raise ValueError("swipe_distance_floor must be <= swipe_min_distance")
        if self.swipe_tracking_grace_seconds >= self.max_frame_gap_seconds:
            raise ValueError("swipe_tracking_grace_seconds must be less than max_frame_gap_seconds")
        if not isinstance(self.model_path, str) or not self.model_path.strip():
            raise ValueError("model_path must be a nonempty path string")
        if self.navigation_mode not in ("swipe", "palm_turn"):
            raise ValueError("navigation_mode must be swipe or palm_turn")
        if not isinstance(self.swipe_model_path, str):
            raise ValueError("swipe_model_path must be a path string (empty uses the original swipe rules)")
        if self.swipe_trial and not self.swipe_model_path:
            raise ValueError("swipe_trial requires a swipe_model_path")
        if self.turn_rearm_policy not in ("same_direction", "all_commands"):
            raise ValueError("turn_rearm_policy must be same_direction or all_commands")
        if self.turn_min_facing >= 1 or self.turn_hold_seconds >= self.turn_max_seconds:
            raise ValueError("Invalid palm-turn facing threshold or timing")
        if self.turn_tracking_grace_seconds >= min(self.release_seconds, self.max_frame_gap_seconds):
            raise ValueError("turn_tracking_grace_seconds must be less than release_seconds and max_frame_gap_seconds")
        if not self.swipe_max_step <= self.turn_reacquire_max_distance <= 1:
            raise ValueError("turn_reacquire_max_distance must be between swipe_max_step and 1")
        if not isinstance(self.record_directory, str):
            raise ValueError("record_directory must be a path string")
        if not isinstance(self.mapping, dict) or not self.mapping:
            raise ValueError("mapping must be a nonempty gesture-to-action table")
        if any(label not in CANNED_GESTURES for label in self.mapping):
            raise ValueError("mapping keys must be supported pretrained gesture names")
        self.mapping = {label: Action(action) for label, action in self.mapping.items()}
        if self.mapping != DEFAULT_MAPPING:
            raise ValueError("Swipe controls require Closed_Fist = toggle and Victory = exit; open palm is reserved for movement")


def load_config(path: Path | None = None, **overrides) -> Config:
    data = {}
    if path is not None:
        with path.open("rb") as stream:
            data = tomllib.load(stream)
    unknown = set(data) - {item.name for item in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration options: {', '.join(sorted(unknown))}")
    data.update({key: value for key, value in overrides.items() if value is not None})
    # Relative model paths are relative to the working directory, like CLI paths.
    return Config(**data)
