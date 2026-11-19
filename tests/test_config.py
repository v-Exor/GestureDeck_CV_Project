import pytest

from gesturedeck.app import parse_args
from gesturedeck.config import Config, load_config
from gesturedeck.mapping import Action


def test_cli_dry_defaults_defaults():
    args = parse_args([])
    assert not args.live
    assert not args.camera_only


def test_desktop_demo_cannot_enable_live_keys():
    assert parse_args(["--gui"]).gui
    assert parse_args(["--demo"]).demo
    for option in ("--live", "--camera-only"):
        with pytest.raises(SystemExit):
            parse_args(["--demo", option])


def test_live_cannot_combine_with_camera_only_or_dry_run():
    for other in ("--dry-run", "--camera-only"):
        with pytest.raises(SystemExit):
            parse_args(["--live", other])


@pytest.mark.parametrize("kwargs", [
    {"confidence_threshold": 1.2}, {"pause_hold_seconds": 0},
    {"release_seconds": -1}, {"cooldown_seconds": float("nan")},
    {"exit_hold_seconds": float("inf")}, {"camera_index": -1},
    {"countdown_seconds": 0}, {"mirror": "false"},
    {"mapping": {"Unknown": "next"}}, {"mapping": {"Victory": "next"}},
    {"camera_backend": "invalid"}, {"confidence_threshold": True},
    {"swipe_min_seconds": 1.0}, {"swipe_min_consistency": 1.1},
    {"audio_feedback": "true"}, {"swipe_max_step": float("nan")},
])
def test_invalid_settings_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_example_config_and_cli_override():
    from pathlib import Path
    config = load_config(Path(__file__).resolve().parents[1] / "config.example.toml", camera_index=2)
    assert config.camera_index == 2
    assert config.mapping["Victory"] == Action.EXIT
    assert config.mapping["Closed_Fist"] == Action.TOGGLE


def test_unknown_setting_caught(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text("typo = 0.4", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)
