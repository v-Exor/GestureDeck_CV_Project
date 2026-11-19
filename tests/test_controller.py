import builtins

import pytest

from gesturedeck.controller import ControllerError, SlideController
from gesturedeck.mapping import Action


def test_dry_run_never_sends_keys_or_imports_pyautogui(monkeypatch, caplog):
    original = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name != "pyautogui", "Dry run must not import keyboard automation"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    calls = []
    controller = SlideController(press=calls.append, focused=lambda: True)
    with caplog.at_level("INFO"):
        controller.execute(Action.NEXT)
        controller.execute(Action.PREVIOUS)
        controller.execute(Action.TOGGLE)
    assert calls == []
    assert "DRY RUN" in controller.last_action
    assert "right" in caplog.text and "left" in caplog.text


def test_live_routes_only_navigation_with_fake_sender():
    calls = []
    controller = SlideController(dry_run=False, press=calls.append, focused=lambda: True)
    for action in (Action.NEXT, Action.PREVIOUS, Action.EXIT, Action.TOGGLE):
        controller.execute(action)
    assert calls == ["right", "left", "esc"]


def test_failsafe_errors_stop_control_without_retry():
    calls = []

    def fail(key):
        calls.append(key)
        raise RuntimeError("fail-safe triggered")

    controller = SlideController(dry_run=False, press=fail, focused=lambda: True)
    with pytest.raises(ControllerError, match="fail-safe"):
        controller.execute(Action.NEXT)
    assert calls == ["right"]


def test_real_sender_enables_builtin_failsafe_without_sending_keys(monkeypatch):
    import sys
    from types import SimpleNamespace

    calls = []
    fake = SimpleNamespace(FAILSAFE=False, press=calls.append, focused=lambda: True)
    monkeypatch.setitem(sys.modules, "pyautogui", fake)
    SlideController(dry_run=False)
    assert fake.FAILSAFE is True
    assert calls == []


def test_live_focus_guard_blocks_all_keys_in_other_windows():
    calls = []
    controller = SlideController(dry_run=False, press=calls.append, focused=lambda: False)
    for action in (Action.NEXT, Action.PREVIOUS, Action.EXIT):
        assert controller.execute(action) is False
    assert calls == []
    assert 'Blocked' in controller.last_action


def test_dry_run_does_not_consult_foreground_window():
    controller = SlideController(focused=lambda: pytest.fail('Dry run inspected foreground'))
    assert controller.execute(Action.EXIT) is True
    assert controller.last_action == 'DRY RUN: exit (esc)'
