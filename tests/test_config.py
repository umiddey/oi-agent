"""Tests for the OpenCode-only configuration contract."""

import shutil

import pytest
from typer.testing import CliRunner

from oi_agent.config import Config, WatchTarget, load_config, save_config


def _binary() -> str:
    """Return a portable executable accepted by config validation."""
    return shutil.which("true") or "/bin/true"


def _valid_config() -> Config:
    """Build a valid isolated test configuration."""
    cfg = Config(opencode_binary=_binary(), opencode_model="provider/model")
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/srv/repo")]
    return cfg


def test_roundtrip_is_opencode_only(tmp_path):
    """New settings round-trip and contain no provider credentials."""
    cfg = _valid_config()
    cfg.personality = 'Warm "expert"\nUse bullets.'
    cfg.opencode_steps = 50
    cfg.opencode_timeout_seconds = 1200
    path = tmp_path / "config.toml"

    save_config(cfg, path)
    back = load_config(path)

    assert back.opencode_model == "provider/model"
    assert back.opencode_steps == 50
    assert back.opencode_timeout_seconds == 1200
    assert back.personality == cfg.personality
    assert back.watches[0].channel_id == 111
    assert "api_key" not in path.read_text()
    assert "base_url" not in path.read_text()


def test_legacy_harness_keys_fail_with_migration_message(tmp_path):
    """Old provider/tool-loop settings fail instead of being ignored."""
    path = tmp_path / "legacy.toml"
    path.write_text(
        '[agent]\nbase_url = "https://example.invalid"\n'
        '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n'
        'search_aliases = { bug = ["defect"] }\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rerun `oi init`"):
        load_config(path)
def test_missing_watch_and_missing_required_field_rejected(tmp_path):
    """A daemon config must contain at least one complete watch target."""
    missing = tmp_path / "missing.toml"
    missing.write_text(
        "[[watch]]\nchannel_id = 1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="repo_path"):
        load_config(missing)

    empty = tmp_path / "empty.toml"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="watch"):
        load_config(empty)


def test_invalid_model_binary_and_limits_rejected(tmp_path):
    """Provider/model shape, executable, steps, and timeout are bounded."""
    cfg = _valid_config()
    cfg.opencode_model = "bare-model"
    with pytest.raises(ValueError, match="provider/model"):
        save_config(cfg, tmp_path / "bad-model.toml")

    cfg = _valid_config()
    cfg.opencode_binary = str(tmp_path / "missing-opencode")
    with pytest.raises(ValueError, match="not executable"):
        save_config(cfg, tmp_path / "bad-binary.toml")

    for field, value in (("opencode_steps", 51),
                         ("opencode_timeout_seconds", 86_401)):
        cfg = _valid_config()
        setattr(cfg, field, value)
        with pytest.raises(ValueError, match=field):
            save_config(cfg, tmp_path / f"bad-{field}.toml")


def test_atomic_save_preserves_previous_bytes_on_validation_failure(tmp_path):
    """Validation happens before the atomic replacement and never clobbers."""
    path = tmp_path / "config.toml"
    save_config(_valid_config(), path)
    before = path.read_bytes()

    invalid = _valid_config()
    invalid.max_reply_chars = 1
    with pytest.raises(ValueError):
        save_config(invalid, path)

    assert path.read_bytes() == before
    assert not (tmp_path / "config.toml.tmp").exists()


def test_single_message_limit_and_target_resolution(tmp_path):
    """Retained Discord delivery constraints still validate in the new schema."""
    cfg = _valid_config()
    cfg.reply_delivery = "single_message"
    cfg.max_reply_chars = 2000
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    assert load_config(path).target_for_channel(111).repo_path == "/srv/repo"

    cfg.max_reply_chars = 2001
    with pytest.raises(ValueError, match="single_message"):
        save_config(cfg, tmp_path / "too-large.toml")


def test_environment_mode_roundtrip_for_both_values(tmp_path):
    """workstation and server both survive a validated TOML round-trip."""
    for mode in ("workstation", "server"):
        cfg = _valid_config()
        cfg.environment_mode = mode
        path = tmp_path / f"config-{mode}.toml"
        save_config(cfg, path)
        assert load_config(path).environment_mode == mode


def test_environment_mode_invalid_value_rejected(tmp_path):
    """Only workstation | server are valid; anything else fails validation."""
    cfg = _valid_config()
    cfg.environment_mode = "hybrid"
    with pytest.raises(ValueError, match="environment_mode"):
        save_config(cfg, tmp_path / "bad-mode.toml")


def test_environment_mode_legacy_auto_variants_get_migration_hint(tmp_path):
    """'auto' and case/whitespace variants fail with the exact repair command."""
    for raw_value in ("auto", "AUTO", " auto ", "Auto"):
        path = tmp_path / "legacy.toml"
        path.write_text(
            f'environment_mode = "{raw_value}"\n'
            '[[watch]]\nchannel_id = 111\nrepo_path = "/srv/repo"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="oi config set environment_mode"):
            load_config(path)


def test_config_set_environment_mode_cli_roundtrip(tmp_path):
    """oi config set accepts environment_mode; show displays it; invalid rejected."""
    from oi_agent.cli import app

    runner = CliRunner()
    path = tmp_path / "config.toml"
    save_config(_valid_config(), path)

    result = runner.invoke(
        app, ["config", "set", "environment_mode", "server", "--config", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert load_config(path).environment_mode == "server"

    shown = runner.invoke(app, ["config", "show", "--config", str(path)])
    assert shown.exit_code == 0, shown.output
    assert "environment_mode = server" in shown.output

    bad = runner.invoke(
        app, ["config", "set", "environment_mode", "hybrid", "--config", str(path)]
    )
    assert bad.exit_code == 1
    assert load_config(path).environment_mode == "server"


def test_config_set_environment_mode_repairs_legacy_auto(tmp_path):
    """The documented repair command works on a legacy 'auto' config."""
    from oi_agent.cli import app

    runner = CliRunner()
    path = tmp_path / "config.toml"
    save_config(_valid_config(), path)
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            'environment_mode = "workstation"', 'environment_mode = "auto"'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="oi config set environment_mode"):
        load_config(path)

    result = runner.invoke(
        app, ["config", "set", "environment_mode", "server", "--config", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert load_config(path).environment_mode == "server"
