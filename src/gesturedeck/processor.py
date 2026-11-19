"""Swipe navigation, deliberate control poses, and one event per released attempt."""

from dataclasses import dataclass
import math

from gesturedeck.config import Config
from gesturedeck.mapping import Action, action_for_gesture
from gesturedeck.swipe import SwipeDetector
from gesturedeck.turn import PalmTurnDetector
from gesturedeck.validation import StableGesture


@dataclass(frozen=True)
class Observation:
    gesture: str | None = None
    confidence: float = 0.0
    hand_present: bool = False
    position: tuple[float, float] | None = None
    handedness: str | None = None
    open_hand: bool | None = None
    hand_scale: float | None = None
    extended_fingers: int = 0
    palm_facing: float | None = None
    landmarks: tuple = ()
    world_landmarks: tuple = ()


class GestureProcessor:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.active = True
        self.status = ("Active: turn your open RIGHT hand; fist pauses" if config.navigation_mode == "palm_turn"
                       else "Active: swipe an open hand; fist pauses")
        self._stable = StableGesture()
        self._learned_swipe = bool(config.swipe_model_path) and config.navigation_mode == "swipe"
        if self._learned_swipe:
            from gesturedeck.learned_swipe import LearnedSwipeDetector
            self._swipe = LearnedSwipeDetector(config)
        else:
            self._swipe = SwipeDetector(config)
        self._turn = PalmTurnDetector(config)
        self._needs_release = False
        self._neutral_since: float | None = None
        self._last_command = float("-inf")
        self._last_frame: float | None = None
        self._handedness: str | None = None
        self._turn_locked_until = float("-inf")
        self._last_turn_action = None

    def pause(self) -> None:
        self.active = False
        self._reset_pending()
        self._needs_release = True
        self._neutral_since = None
        self.status = "Paused: lower hand, then hold a fist to resume"

    def _reset_pending(self):
        self._stable.reset()
        self._swipe.reset()
        self._turn.reset()

    def diagnostic_state(self, now):
        """Per-frame state for reviewing missed turns; never a training label."""
        return {
            "navigation_mode": self.config.navigation_mode,
            "swipe_engine": "learned" if self._learned_swipe else "original_rules",
            "swipe_diagnostics": ({**self._swipe.diagnostic_state(now),
                                   "current_observation_scored": self._last_frame is not None and self._swipe.last_evaluation == self._last_frame}
                                  if self._learned_swipe else
                                  {"status": self._swipe.status, "tracking_samples": len(self._swipe.points)}),
            "starting_side": self._turn.side,
            "candidate_side": self._turn.candidate,
            "candidate_samples": self._turn.samples,
            "candidate_seconds": (now - self._turn.candidate_since
                                  if self._turn.candidate_since is not None else None),
            "candidate_started_while_allowed": self._turn.candidate_allowed,
            "tracking_interrupted": self._turn.tracking_interrupted,
            "rearm_remaining_seconds": max(0, self._turn_locked_until - now),
            "rearm_policy": self.config.turn_rearm_policy,
            "last_turn_action": self._last_turn_action.value if self._last_turn_action else None,
            "requires_hand_release": self._needs_release,
            "hand_absent_seconds": now - self._neutral_since if self._neutral_since is not None else None,
            "active": self.active,
        }

    def update(self, observation: Observation, now: float) -> Action | None:
        if not math.isfinite(now) or (self._last_frame is not None and now <= self._last_frame):
            raise ValueError("Observation times must be finite and strictly increasing")
        if self._last_frame is not None and now - self._last_frame > self.config.max_frame_gap_seconds:
            self.pause()
            self.status = "Frame gap: paused; lower hand, then hold a fist to resume"
        self._last_frame = now
        if observation.handedness != self._handedness:
            self._stable.reset()
            # A handedness label can flicker during rotation. Actual motion
            # continuity is checked geometrically by the swipe detector.
            if observation.open_hand is None or not observation.hand_present:
                self._swipe.reset()
        self._handedness = observation.handedness

        score_ok = (math.isfinite(observation.confidence)
                    and 0 <= observation.confidence <= 1
                    and observation.confidence >= self.config.confidence_threshold)
        if not observation.hand_present:
            self._stable.reset()
            if self._learned_swipe and self.active and not self._needs_release:
                self._swipe.tracking_lost(now)
            else:
                self._swipe.reset()
            if self.config.navigation_mode == "palm_turn" and self.active and not self._needs_release:
                self._turn.interrupt(now)
                self.status = self._turn.status
            else:
                self._turn.reset()
            if self._neutral_since is None:
                self._neutral_since = now
            if now - self._neutral_since >= self.config.release_seconds:
                self._needs_release = False
                self._turn.reset()
                self._turn_locked_until = float("-inf")  # Deliberate hand lowering rearms early.
                self._last_turn_action = None
                self.status = (f"Active: ready for {self.config.navigation_mode}" if self.active
                               else "Paused: hold a fist to resume")
            return None
        self._neutral_since = None
        if self._needs_release:
            self.status = "Lower hand to reset; " + ("ACTIVE" if self.active else "PAUSED")
            return None
        if observation.open_hand is not None and (
            observation.position is None or not all(
                math.isfinite(value) and 0 <= value <= 1 for value in observation.position)):
            self._reset_pending()
            self.status = "Hand landmarks unavailable: keep the whole hand visible"
            return None

        pose_action = action_for_gesture(observation.gesture, self.config.mapping)
        motion_open = (observation.open_hand is True or (
            observation.open_hand is None and score_ok and observation.gesture == "Open_Palm"))
        if self._learned_swipe:
            # The trained motion classifier consumes landmarks regardless of
            # finger count. Deliberate fist/Victory controls take precedence.
            motion_open = not (pose_action is not None and score_ok)
        if self.config.navigation_mode == "palm_turn" and not (
                pose_action is not None and score_ok and not motion_open):
            self._stable.reset()
            if not self.active:
                self._turn.reset()
                self.status = "Paused: hold a fist to resume"
                return None
            expected = (Action.NEXT if self._turn.side == "palm" else
                        Action.PREVIOUS if self._turn.side == "back" else None)
            allowed = (now >= self._turn_locked_until or (
                self.config.turn_rearm_policy == "same_direction" and expected != self._last_turn_action))
            action = self._turn.update(observation.position, observation.hand_scale,
                                      observation.open_hand, observation.palm_facing, now,
                                      allow_command=allowed)
            self.status = self._turn.status
            if action is not None:
                self._turn_locked_until = now + self.config.turn_rearm_seconds
                self._last_command = now
                self._last_turn_action = action
            elif not allowed:
                self.status = f"Repeat command pause: {self._turn_locked_until - now:.1f}s; opposite turn allowed" if (
                    self.config.turn_rearm_policy == "same_direction") else (
                    f"Command pause: {self._turn_locked_until - now:.1f}s; or lower hand to reset")
            return action
        if self.config.navigation_mode == "palm_turn":
            self._turn.reset()  # Fist/Victory must not become a navigation turn.

        if motion_open:
            self._stable.reset()
            if not self.active:
                self._swipe.reset()
                self.status = "Paused: hold a fist to resume"
                return None
            action = (self._swipe.observe(observation, now) if self._learned_swipe else
                      self._swipe.update(observation.position, now, observation.hand_scale))
            self.status = self._swipe.status
            if action is None:
                return None
            self._needs_release = True
            if now - self._last_command < self.config.cooldown_seconds:
                self.status = "Ignored: cooldown; lower hand and try again"
                return None
            self._last_command = now
            self.status = f"Accepted: {action.value}; lower hand"
            return action

        if observation.open_hand is not None:
            self._swipe.interrupt(now)
        else:
            self._swipe.reset()
        pose_action = action_for_gesture(observation.gesture, self.config.mapping)
        if pose_action is not None and score_ok:
            hold = self.config.pause_hold_seconds if pose_action == Action.TOGGLE else self.config.exit_hold_seconds
            if not self._stable.update(observation.gesture, now, hold):
                self.status = (f"Hold fist {hold:g}s to {'pause' if self.active else 'resume'}"
                               if pose_action == Action.TOGGLE else f"Hold Victory {hold:g}s to exit slideshow")
                return None
            self._needs_release = True
            self._swipe.reset()
            if pose_action == Action.TOGGLE:
                self.active = not self.active
                self.status = ("Active" if self.active else "Paused") + ": lower hand"
            else:
                self.active = False
                self.status = "Exit slideshow requested; control PAUSED; lower hand"
            return pose_action

        self._stable.reset()
        if not self.active:
            self.status = "Paused: hold a fist to resume"
        elif observation.position is None:
            self.status = "Hand landmarks unavailable: keep the whole hand visible"
        elif observation.open_hand is False:
            self.status = "Hand tracked: extend at least three fingers for a swipe"
        else:
            self.status = "Low pose confidence: held controls ignored"
        return None
