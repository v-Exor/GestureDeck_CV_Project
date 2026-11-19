import tkinter as tk
from types import SimpleNamespace

import pytest

from gesturedeck.config import Config
from gesturedeck.desktop import DesktopApp
from gesturedeck.processor import Observation
from gesturedeck.session import Mode, Session, Snapshot


@pytest.fixture(scope="module")
def tcl_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()  # Real widgets, but no camera, keys, or visible test window.
    yield root
    root.destroy()


@pytest.fixture
def desktop(tcl_root):
    window = tk.Toplevel(tcl_root)
    window.withdraw()
    app = DesktopApp(window, Config(audio_feedback=False), Mode.DEMO)
    yield app
    window.destroy()


def test_desktop_is_idle_stopped_and_demo_is_explicit(desktop):
    assert desktop.session is None
    assert desktop.control.get() == "STOPPED"
    assert not desktop.allow_keys.get()
    assert desktop.demo_panel.winfo_manager() == "pack"
    assert desktop.start_button.cget("state") == "normal"


def test_live_gui_needs_opt_in_even_with_live_startup_mode(desktop, monkeypatch):
    messages = []
    monkeypatch.setattr("gesturedeck.desktop.messagebox.showerror", lambda *args, **_: messages.append(args))
    monkeypatch.setattr(Session, "start", lambda _: None)
    desktop.mode.set(Mode.LIVE.value)
    desktop._mode_changed()
    desktop.start()
    assert desktop.session is None
    assert messages and "Allow slide keys" in messages[0][1]
    desktop.allow_keys.set(True)
    desktop.start()
    assert desktop.session.mode == Mode.LIVE
    assert desktop.start_button.cget("state") == "disabled"


def test_demo_controls_and_manual_pause_reach_session(desktop, monkeypatch):
    monkeypatch.setattr(Session, "start", lambda _: None)
    desktop.start()
    desktop.pose.set("Victory")
    desktop._demo_pose()
    assert desktop.session._demo_observation.gesture == "Victory"
    desktop.pause()
    assert desktop.session.pause_requested.is_set()
    assert desktop.control.get() == "PAUSED"


def test_snapshot_updates_ui_and_stop_resets_opt_in(desktop, monkeypatch):
    monkeypatch.setattr(Session, "start", lambda _: None)
    desktop.start()
    desktop.session._publish(Snapshot(observation=Observation("Victory", 0.9, True),
                                      active=True, fps=20, last_action="DRY RUN: next (right)"))
    desktop._poll()
    assert desktop.gesture.get() == "Victory"
    assert desktop.control.get() == "ACTIVE"
    assert desktop.confidence.get() == "90%"
    assert desktop.latency.get() == "Simulated frames"
    desktop.allow_keys.set(True)
    desktop.stop()
    desktop.session.thread = SimpleNamespace(is_alive=lambda: False, join=lambda **_: None)
    desktop._poll()
    assert desktop.session is None
    assert desktop.control.get() == "STOPPED"
    assert not desktop.allow_keys.get()
    assert desktop.start_button.cget("state") == "normal"


def test_demo_swipe_button_feeds_motion_instead_of_static_pose(desktop, monkeypatch):
    monkeypatch.setattr(Session, 'start', lambda _: None)
    desktop.start()
    desktop.pose.set('Swipe left')
    desktop._demo_pose()
    points = [o.position for o in desktop.session._demo_sequence if o.position]
    assert points[0][0] > points[-1][0]
    desktop.pose.set('Swipe right')
    desktop._demo_pose()
    points = [o.position for o in desktop.session._demo_sequence if o.position]
    assert points[0][0] < points[-1][0]


def test_countdown_never_displays_active(desktop, monkeypatch):
    monkeypatch.setattr(Session, 'start', lambda _: None)
    desktop.start()
    desktop.session._publish(Snapshot(active=False, phase='Focus slideshow: 3s'))
    desktop._poll()
    assert desktop.control.get() == 'COUNTDOWN'


def test_palm_turn_guide_opens_with_current_reverse_turn_policy(desktop, monkeypatch):
    messages = []
    monkeypatch.setattr(desktop, '_help_window', lambda *args: messages.append(args))
    desktop.navigation.set('Project Use')
    desktop.show_guide()
    assert 'opposite turns work immediately' in messages[0][2]


def test_learned_trial_and_original_are_explicit_session_choices(desktop, monkeypatch):
    monkeypatch.setattr(Session, 'start', lambda _: None)
    desktop.mode.set(Mode.DRY_RUN.value)
    desktop.navigation.set('Learned swipe trial')
    desktop.start()
    assert desktop.session.config.swipe_trial
    assert desktop.session.config.swipe_model_path == 'models/swipe_motion_trial.npz'
    assert 'LEARNED SWIPE TRIAL' in desktop.root.title()
    desktop.session = None
    desktop.navigation.set('Sideways swipe')
    desktop.start()
    assert not desktop.session.config.swipe_trial
    assert desktop.session.config.swipe_model_path == ''
    assert 'ORIGINAL SWIPE RULES' in desktop.root.title()


def test_slideshow_recording_checkbox_requires_recording_and_reaches_session(desktop, monkeypatch):
    monkeypatch.setattr(Session, 'start', lambda _: None)
    assert desktop.slide_record_check.cget('state') == 'disabled'
    desktop.record_enabled.set(True)
    desktop._recording_changed()
    assert desktop.slide_record_check.cget('state') == 'normal'
    desktop.record_slides.set(True)
    desktop.mode.set(Mode.DRY_RUN.value)
    desktop.start()
    assert desktop.session.config.record_directory == 'recordings'
    assert desktop.session.config.record_slideshow
    assert desktop.slide_record_check.cget('state') == 'disabled'
    desktop.session = None
    desktop.record_enabled.set(False)
    desktop._recording_changed()
    assert not desktop.record_slides.get()


def test_presentation_launcher_exposes_only_palm_turn_and_starts_without_swipe_model(tcl_root, monkeypatch):
    monkeypatch.setattr(Session, 'start', lambda _: None)
    window = tk.Toplevel(tcl_root)
    window.withdraw()
    try:
        app = DesktopApp(window, Config(navigation_mode='palm_turn', audio_feedback=False))
        assert app.navigation.get() == 'Project Use'
        assert tuple(app.navigation_box['values']) == ('Project Use',)
        assert Mode.DEMO.value not in app.mode_box['values']
        assert 'Project Use' in window.title()
        assert not app.record_enabled.get() and not app.allow_keys.get()
        app.start()
        assert app.session.mode == Mode.DRY_RUN
        assert app.session.config.navigation_mode == 'palm_turn'
        assert not app.session.config.swipe_model_path
        assert not app.session.config.swipe_trial
        assert (app.session.config.width, app.session.config.height) == (640, 480)
    finally:
        window.destroy()


def test_presentation_launcher_rejects_an_alternate_navigation_value(tcl_root, monkeypatch):
    messages = []
    monkeypatch.setattr('gesturedeck.desktop.messagebox.showerror', lambda *args, **_: messages.append(args))
    monkeypatch.setattr(Session, 'start', lambda _: pytest.fail('Invalid mode must not start a camera'))
    window = tk.Toplevel(tcl_root)
    window.withdraw()
    try:
        app = DesktopApp(window, Config(navigation_mode='palm_turn', audio_feedback=False))
        app.navigation.set('Learned swipe trial')
        app.start()
        assert app.session is None and messages
    finally:
        window.destroy()
