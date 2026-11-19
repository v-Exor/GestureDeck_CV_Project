from pathlib import Path
import runpy
import sys

from gesturedeck.session import Mode


def test_retained_launcher_opens_palm_only_desktop_with_idle_safe_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr('gesturedeck.desktop.run_desktop', lambda config, mode: calls.append((config, mode)) or 0)
    monkeypatch.setattr('logging.basicConfig', lambda **_: None)
    monkeypatch.setattr(sys, 'path', sys.path.copy())
    monkeypatch.chdir(tmp_path)
    launcher = Path(__file__).resolve().parents[1] / 'GestureDeck-Learned Trial.pyw'
    module = runpy.run_path(str(launcher), run_name='launcher_test')
    assert module['main']() == 0
    config, mode = calls[0]
    assert mode == Mode.DRY_RUN
    assert config.navigation_mode == 'palm_turn'
    assert not config.swipe_model_path and not config.swipe_trial
    assert not config.record_directory and not config.record_slideshow
    assert config.countdown_seconds == 8
    assert (config.width, config.height) == (640, 480)
