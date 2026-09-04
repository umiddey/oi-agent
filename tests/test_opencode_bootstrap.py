"""Tests for OpenCode discovery, version, auth, and model bootstrap."""

from types import SimpleNamespace

import pytest

from oi_agent.opencode import bootstrap
from oi_agent.opencode.paths import resolve_opencode_paths


def test_installed_binary_version_is_verified():
    """The local compatible executable reports a parsed semantic version."""
    version = bootstrap.check_version("opencode", resolve_opencode_paths())
    assert version.binary.endswith(("opencode", "opencode.exe"))
    assert len(version.version) == 3
    assert version.version >= bootstrap.MIN_OPENCODE_VERSION


def test_old_binary_version_rejected(monkeypatch):
    """Versions older than 1.18.0 fail compatible check."""
    monkeypatch.setattr(
        bootstrap.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="opencode 1.10.0\n"
        ),
    )
    with pytest.raises(bootstrap.OpenCodeBootstrapError, match="too old"):
        bootstrap.check_version("opencode", resolve_opencode_paths())


def test_missing_binary_has_actionable_diagnostic(tmp_path):
    """Missing executables identify the official installation documentation."""
    with pytest.raises(bootstrap.OpenCodeBootstrapError, match="opencode.ai/docs"):
        bootstrap.resolve_binary(str(tmp_path / "missing"))


def test_model_listing_returns_exact_ids_without_raw_output(monkeypatch):
    """Model parsing extracts provider/model IDs and deduplicates them."""
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0,
            stdout="Provider\nopenai/gpt-4o-mini\nopenai/gpt-4o-mini\n",
        ),
    )
    assert bootstrap.list_models("opencode", resolve_opencode_paths()) == [
        "openai/gpt-4o-mini"
    ]


def test_authentication_status_never_requires_credentials_in_oi(monkeypatch):
    """Auth probing is boolean and does not expose credential contents."""
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="openai  authenticated\n"
        ),
    )
    assert bootstrap.authenticated("opencode", resolve_opencode_paths()) is True


def test_paths_are_isolated_and_shared(tmp_path):
    """The resolver sets private HOME/XDG roots and suppresses inherited config."""
    paths = resolve_opencode_paths(tmp_path)
    env = paths.environment({
        "HOME": "/operator",
        "XDG_CONFIG_HOME": "/personal",
        "OPENCODE_CONFIG": "/personal/opencode.json",
        "OPENCODE_CONFIG_DIR": "/personal/opencode",
    })
    assert env["HOME"].startswith(str(tmp_path))
    assert env["XDG_CONFIG_HOME"].startswith(str(tmp_path))
    assert "OPENCODE_CONFIG" not in env
    assert "OPENCODE_CONFIG_DIR" not in env
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"


def test_environment_strictly_excludes_parent_secrets(monkeypatch, tmp_path):
    """Parent process secrets are never inherited by the OpenCode subprocess."""
    monkeypatch.setenv("OI_DISCORD_TOKEN", "discord-token-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-secret")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "anthropic-token-secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("DATABASE_PASSWORD", "db-pass")
    monkeypatch.setenv("MY_PRIVATE_KEY", "key-data")
    monkeypatch.setenv("CUSTOM_CREDENTIAL", "cred-data")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setenv("LANG", "en_US.UTF-8")

    paths = resolve_opencode_paths(tmp_path)
    env = paths.environment()  # base=None -> reads os.environ through allowlist

    assert "OI_DISCORD_TOKEN" not in env
    assert "OPENAI_API_KEY" not in env
    assert "ANTHROPIC_AUTH_TOKEN" not in env
    assert "AWS_SECRET_ACCESS_KEY" not in env
    assert "DATABASE_PASSWORD" not in env
    assert "MY_PRIVATE_KEY" not in env
    assert "CUSTOM_CREDENTIAL" not in env
    assert env["PATH"] == "/usr/bin:/bin"
    assert env["LANG"] == "en_US.UTF-8"
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"


def test_install_command_selection(monkeypatch):
    """Install command selects official package managers in order."""
    monkeypatch.setattr(bootstrap.shutil, "which", lambda cmd: "/bin/" + cmd if cmd == "npm" else None)
    assert bootstrap.install_command() == ["npm", "install", "--global", "opencode-ai"]

    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda cmd: "/bin/" + cmd if cmd == "brew" else None)
    assert bootstrap.install_command() == ["brew", "install", "anomalyco/tap/opencode"]

    monkeypatch.setattr(bootstrap.platform, "system", lambda: "Linux")
    monkeypatch.setattr(bootstrap.shutil, "which", lambda cmd: "/bin/" + cmd if cmd == "mise" else None)
    assert bootstrap.install_command() == ["mise", "use", "--global", "npm:opencode-ai"]
