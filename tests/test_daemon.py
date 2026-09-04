"""Tests for systemd unit hardening and safe daemon preflight."""

import stat

import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.daemon import (
    DaemonError,
    DaemonIdentity,
    doctor,
    ensure_opencode_for_user,
    ensure_user,
    install,
    render_unit,
    stream_logs,
)
from oi_agent.store import Store


def _config(tmp_path):
    """Build a valid daemon config for unit rendering."""
    cfg = Config(opencode_binary="/bin/true", opencode_model="provider/model")
    cfg.watches = [WatchTarget(111, str(tmp_path / "repo"))]
    return cfg


def test_render_unit_contains_required_sandbox_and_exact_paths(tmp_path):
    """Generated unit runs unprivileged with isolated writable roots."""
    identity = DaemonIdentity("oi", tmp_path / "home", 1001, 1001)
    unit = render_unit(_config(tmp_path), identity)
    assert "User=oi" in unit
    assert "Group=oi" in unit
    assert "ProtectSystem=strict" in unit
    assert "ProtectHome=read-only" in unit
    assert str(identity.home / ".local/state/oi") in unit
    assert "ProtectControlGroups=true" in unit
    assert str((tmp_path / "repo").resolve()) in unit
    assert "OPENCODE_DISABLE_PROJECT_CONFIG=1" in unit
    assert "ReadWritePaths=/home/oi" not in unit


def test_install_dry_run_does_not_require_root_or_mutate(tmp_path):
    """Dry-run can render a missing service account without host mutation."""
    unit = install(
        _config(tmp_path),
        user="oi-test-missing",
        dry_run=True,
        service_dir=tmp_path / "systemd",
    )
    assert "User=oi-test-missing" in unit
    assert not (tmp_path / "systemd").exists()


def test_real_install_requires_root(tmp_path, monkeypatch):
    """Non-root installation fails before user or systemd mutation."""
    monkeypatch.setattr("oi_agent.daemon.os.geteuid", lambda: 1000)
    with pytest.raises(DaemonError, match="requires root"):
        install(_config(tmp_path), service_dir=tmp_path / "systemd")


def test_ensure_user_and_opencode_directories_are_0700(tmp_path, monkeypatch):
    """User directories are created with private 0700 permissions and chowned."""
    home = tmp_path / "oi_home"
    home.mkdir()
    identity = DaemonIdentity("oi-test", home, 1001, 1001)

    chowned = []
    monkeypatch.setattr("oi_agent.daemon.shutil.chown", lambda path, u, g: chowned.append((str(path), u, g)))
    monkeypatch.setattr("oi_agent.opencode.paths.os.chown", lambda path, u, g: chowned.append((str(path), u, g)))

    ensure_user(identity, dry_run=False)

    for path in (home / ".config/oi", home / ".local/state/oi", home / ".local/bin"):
        assert path.exists()
        assert stat.S_IMODE(path.stat().st_mode) == 0o700

    cfg = _config(tmp_path)
    (home / ".local/bin/opencode").write_text("#!/bin/sh\n", encoding="utf-8")
    (home / ".local/bin/opencode").chmod(0o755)
    auth_dir = home / ".local/share/oi/opencode/data/opencode"
    auth_dir.mkdir(parents=True)
    (auth_dir / "auth.json").write_text("{}", encoding="utf-8")

    ensure_opencode_for_user(cfg, identity, dry_run=False)
    assert (home / ".local/share/oi/opencode/home").exists()
    assert stat.S_IMODE((home / ".local/share/oi/opencode/home").stat().st_mode) == 0o700
    assert any("opencode" in item[0] for item in chowned)


def test_stream_logs_runs_journalctl_interactive(monkeypatch):
    """Stream logs runs without capturing output or timing out."""
    called = []
    monkeypatch.setattr(
        "oi_agent.daemon.subprocess.run",
        lambda cmd, check=False: called.append(cmd),
    )
    stream_logs("oi")
    assert called == [["journalctl", "-u", "oi.service", "-f"]]


def test_doctor_validates_user_and_permissions(tmp_path, monkeypatch):
    """Doctor checks repo readability, state writability, and service user root status."""
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    repo.mkdir()
    cfg = Config(
        opencode_binary="/bin/true",
        opencode_model="provider/model",
        db_path=str(tmp_path / "state" / "state.db"),
    )
    cfg.watches = [WatchTarget(1, str(repo))]
    (tmp_path / "state").mkdir()

    version_mock = SimpleNamespace(binary="/bin/true", version=(1, 18, 25), raw="1.18.25")
    monkeypatch.setattr("oi_agent.daemon.check_version", lambda *args: version_mock)
    monkeypatch.setattr("oi_agent.daemon.authenticated", lambda *args: True)
    monkeypatch.setattr("oi_agent.daemon.list_models", lambda *args: ["provider/model"])
    monkeypatch.setenv("OI_DISCORD_TOKEN", "token")

    monkeypatch.setattr("oi_agent.daemon._identity", lambda user, *args: DaemonIdentity(user, tmp_path, 0, 0))
    failures = doctor(cfg, user="root")
    assert any("must not be root" in f for f in failures)


def test_doctor_detects_stale_leases_and_dead_jobs(tmp_path, monkeypatch):
    """Doctor identifies stale leases and dead jobs in the state store.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        None: Asserts doctor returns failures for unhealthy queue states.
    """
    from types import SimpleNamespace

    repo = tmp_path / "repo"
    repo.mkdir()
    state_file = tmp_path / "state" / "state.db"
    (tmp_path / "state").mkdir()

    cfg = Config(
        opencode_binary="/bin/true",
        opencode_model="provider/model",
        db_path=str(state_file),
    )
    cfg.watches = [WatchTarget(1, str(repo))]

    version_mock = SimpleNamespace(binary="/bin/true", version=(1, 18, 25), raw="1.18.25")
    monkeypatch.setattr("oi_agent.daemon.check_version", lambda *args: version_mock)
    monkeypatch.setattr("oi_agent.daemon.authenticated", lambda *args: True)
    monkeypatch.setattr("oi_agent.daemon.list_models", lambda *args: ["provider/model"])
    monkeypatch.setenv("OI_DISCORD_TOKEN", "token")

    # Clean store: no failures
    store = Store(state_file)
    store.close()
    failures = doctor(cfg)
    assert failures == []

    # Insert a stale lease (lease_until in the past) and a dead job
    store = Store(state_file)
    store.admit_mention_job(5001, 10, 20, repo)
    store.claim_next_conversation_jobs(lease_duration_seconds=-10)
    store.admit_mention_job(5002, 10, 20, repo)
    store.fail_jobs([5002], "fatal", is_permanent=True)
    store.close()

    failures = doctor(cfg)
    assert any("stale lease" in f for f in failures)
    assert any("dead job" in f for f in failures)


def test_install_user_mode_renders_user_service(tmp_path):
    """User-mode installation renders a unit without User=/Group= and with default.target."""
    unit = install(_config(tmp_path), dry_run=True, user_mode=True)
    assert "WantedBy=default.target" in unit
    assert "User=" not in unit
    assert "Group=" not in unit
    assert "ExecStart=" in unit
