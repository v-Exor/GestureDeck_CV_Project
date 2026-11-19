"""Temporal validation independent of frame rate, webcam, and MediaPipe."""


class StableGesture:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.candidate: str | None = None
        self.since = 0.0
        self.accepted = False

    def update(self, gesture: str | None, now: float, hold_seconds: float) -> bool:
        if gesture is None:
            self.reset()
            return False
        if gesture != self.candidate:
            self.candidate = gesture
            self.since = now
            self.accepted = False
        if not self.accepted and now - self.since >= hold_seconds:
            self.accepted = True
            return True
        return False
