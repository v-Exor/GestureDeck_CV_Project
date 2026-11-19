"""Short Windows state tones on a worker; audio never blocks recognition."""

import logging
import os
from queue import Empty, Full, Queue
from threading import Thread

LOGGER = logging.getLogger(__name__)
TONES = {
    "active": ((660, 90), (990, 120)),
    "paused": ((990, 90), (440, 150)),
    "exit": ((660, 100), (330, 220)),
    "blocked": ((330, 120),),
}


class AudioFeedback:
    def __init__(self, enabled=True):
        self.enabled = enabled and os.name == "nt"
        self.queue = Queue(maxsize=1)
        self.thread = None

    def __enter__(self):
        if self.enabled:
            self.thread = Thread(target=self._run, name="GestureDeck-audio", daemon=True)
            self.thread.start()
        return self

    def notify(self, state):
        if self.enabled:
            self._replace(state)

    def _replace(self, state):
        try:
            self.queue.put_nowait(state)
        except Full:
            try:
                self.queue.get_nowait()
            except Empty:
                pass
            self.queue.put_nowait(state)

    def _run(self):
        try:
            import winsound
            while (state := self.queue.get()) is not None:
                for frequency, milliseconds in TONES.get(state, ()):
                    winsound.Beep(frequency, milliseconds)
        except Exception:
            LOGGER.warning("Audio feedback unavailable; use the on-screen status", exc_info=True)

    def __exit__(self, *_):
        if self.thread is not None:
            self._replace(None)
            self.thread.join(timeout=0.5)
