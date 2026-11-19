"""Experimental right-hand palm/back transition detector; no camera or UI APIs."""

import math

from gesturedeck.mapping import Action


class PalmTurnDetector:
    def __init__(self, config):
        self.config = config
        self.status = "Right hand only: hold palm or back facing the camera"
        self.reset()

    def reset(self):
        self.side = None
        self._reset_candidate()
        self.last_endpoint = None
        self.last_position = None
        self.last_scale = None
        self.last_observed = None
        self.tracking_interrupted = False

    def _reset_candidate(self):
        self.candidate = None
        self.candidate_since = None
        self.candidate_last = None
        self.candidate_allowed = False
        self.samples = 0

    def interrupt(self, now):
        """Bridge only short tracking loss, without adding any orientation evidence."""
        if self.last_observed is not None and now - self.last_observed > self.config.turn_tracking_grace_seconds:
            self.reset()
            self.status = "Tracking lost: establish a new starting side"
        else:
            self.tracking_interrupted = True
            self.status = "Tracking briefly interrupted: keep the right hand visible"

    def update(self, position, scale, open_hand, facing, now, *, allow_command=True):
        if (position is None or len(position) != 2
                or not all(math.isfinite(v) and 0 <= v <= 1 for v in position)
                or scale is None or not math.isfinite(scale)
                or scale < self.config.swipe_min_hand_size):
            self.reset()
            self.status = "Turn reset: keep the whole right hand visible and close enough"
            return None
        if self.last_observed is not None and now - self.last_observed > self.config.turn_tracking_grace_seconds:
            self.reset()
        allowed_step = self.config.swipe_max_step
        if self.tracking_interrupted and self.last_observed is not None:
            # The hand can travel during missing frames. A one-frame distance
            # limit applied to that whole interval rejected real recorded turns.
            # Expand only after actual short loss; never allow an unbounded jump.
            allowed_step = min(self.config.turn_reacquire_max_distance,
                               max(allowed_step, (now - self.last_observed) * self.config.turn_reacquire_speed))
        if self.last_position is not None and (
                math.dist(position, self.last_position) > allowed_step
                or not 0.5 <= scale / self.last_scale <= 2):
            self.reset()
            self.status = "Turn reset: tracking jumped; establish a new starting side"
            return None
        self.last_position, self.last_scale = position, scale
        self.last_observed = now
        self.tracking_interrupted = False
        if self.side is not None and now - self.last_endpoint > self.config.turn_max_seconds:
            self.reset()
            self.last_position, self.last_scale, self.last_observed = position, scale, now
        side = ("palm" if facing > 0 else "back") if (
            open_hand and facing is not None and math.isfinite(facing)
            and abs(facing) >= self.config.turn_min_facing) else None
        if side is None:
            # A brief curled finger or low-area frame may interrupt an endpoint.
            # It never contributes a sample. Opposite orientation or a longer
            # gap discards the candidate so jitter cannot accumulate a turn.
            contradicts = (self.candidate is not None and facing is not None and math.isfinite(facing)
                           and abs(facing) >= self.config.turn_min_facing
                           and (facing > 0) != (self.candidate == "palm"))
            if contradicts or (self.candidate_last is not None
                              and now - self.candidate_last > self.config.turn_tracking_grace_seconds):
                self._reset_candidate()
            self.status = "Turning / edge-on: finish with fingers open and a flat hand"
            return None
        if side == self.side:
            self.last_endpoint = now
            self._reset_candidate()
            self.status = ("PALM ready: turn to BACK for next" if side == "palm"
                           else "BACK ready: turn to PALM for previous")
            return None
        if (side != self.candidate or self.candidate_last is None
                or now - self.candidate_last > self.config.turn_tracking_grace_seconds):
            self.candidate, self.candidate_since, self.samples = side, now, 1
            self.candidate_allowed = allow_command
        else:
            self.samples += 1
        self.candidate_last = now
        self.status = f"Hold {side.upper()} facing camera briefly"
        if self.samples < 3 or now - self.candidate_since + 1e-9 < self.config.turn_hold_seconds:
            return None
        previous = self.side
        # Completion, rather than the first candidate frame, decides eligibility.
        # A turn completed DURING the pause still updates the baseline silently,
        # so holding it cannot queue a delayed key. A turn completing AFTER the
        # deadline may fire even if it began just before the deadline.
        permitted = allow_command
        self.side, self.last_endpoint = side, now
        self._reset_candidate()
        if previous is None:
            self.status = f"{side.upper()} starting side established; now turn your hand"
            return None
        if not permitted:
            self.status = f"{side.upper()} reset during command pause; turn again when ready"
            return None
        self.status = f"Turn accepted: {previous.upper()} to {side.upper()}"
        return Action.NEXT if previous == "palm" else Action.PREVIOUS
