import math
import pytest
from gesturedeck.config import Config
from gesturedeck.processor import GestureProcessor, Observation


class Rig:
    """Deterministic 20 Hz observations; no sleeping or hardware."""
    def __init__(self, config=None):
        self.processor = GestureProcessor(config or Config(audio_feedback=False))
        self.now = 0.0

    def sample(self, gesture=None, *, score=0.99, hand=True, position=None, side=None, delta=0.05):
        self.now += delta
        return self.processor.update(Observation(gesture, score, hand, position, side), self.now)

    def hold(self, gesture, seconds=1.15, *, score=0.99, hand=True):
        events = []
        for _ in range(math.ceil(seconds / 0.05) + 1):
            event = self.sample(gesture, score=score, hand=hand, position=(0.5, 0.5))
            if event is not None:
                events.append(event)
        return events

    def release(self):
        return self.hold(None, 0.35, hand=False)

    def swipe(self, direction='next', *, score=0.99, side=None):
        xs = [0.8 - i * 0.05 for i in range(13)]
        if direction == 'previous':
            xs.reverse()
        events = []
        for x in xs:
            event = self.sample('Open_Palm', score=score, position=(x, 0.5), side=side)
            if event is not None:
                events.append(event)
        return events


@pytest.fixture
def rig():
    return Rig()
