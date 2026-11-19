from threading import Event
from types import SimpleNamespace
import sys
import gesturedeck.feedback as feedback


def test_state_tones_run_on_worker_and_stop_cleanly(monkeypatch):
    active_played, paused_played = Event(), Event()
    tones = []
    def beep(frequency, duration):
        tones.append((frequency, duration))
        if frequency == 990: active_played.set()
        if frequency == 440: paused_played.set()
    monkeypatch.setattr(feedback, 'os', SimpleNamespace(name='nt'))
    monkeypatch.setitem(sys.modules, 'winsound', SimpleNamespace(Beep=beep))
    with feedback.AudioFeedback(True) as audio:
        audio.notify('active')
        assert active_played.wait(1)
        audio.notify('paused')
        assert paused_played.wait(1)
    assert not audio.thread.is_alive()
    assert tones == list(feedback.TONES['active']) + list(feedback.TONES['paused'])


def test_disabled_feedback_starts_no_thread():
    with feedback.AudioFeedback(False) as audio:
        audio.notify('active')
    assert audio.thread is None
