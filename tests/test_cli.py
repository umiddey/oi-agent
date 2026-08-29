"""CLI tests using typer's test runner (no network, no discord)."""

from typer.testing import CliRunner
import pytest


from oi_agent.cli import app

runner = CliRunner()


def test_config_set_and_show(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)

    r = runner.invoke(app, ["config", "watch-add",
                            "--channel-id", "111",
                            "--repo-path", str(base),
                            "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    assert "watching 111" in r.output

    r = runner.invoke(app, ["config", "set", "agent.model", "Qwen3.8-27B",
                            "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["config", "set", "personality",
                            "warm expert", "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output

    r = runner.invoke(app, ["config", "set", "max_response_tokens",
                            "8000", "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    r = runner.invoke(app, ["config", "show", "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output
    assert "Qwen3.8-27B" in r.output
    assert "warm expert" in r.output
    assert "max_response_tokens = 8000" in r.output


def test_config_set_rejects_unknown_key(tmp_path):
    cfg_path = tmp_path / "config.toml"
    r = runner.invoke(app, ["config", "set", "bogus.key", "x",
                            "--config", str(cfg_path)])
    assert r.exit_code == 1


def test_config_set_rejects_dotted_scalar(tmp_path):
    """A dotted suffix on a scalar key (typo) must be rejected, not ignored."""
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    runner.invoke(app, ["config", "watch-add", "--channel-id", "111",
                        "--repo-path", str(base), "--config", str(cfg_path)])
    r = runner.invoke(app, ["config", "set", "personality.typo", "x",
                            "--config", str(cfg_path)])
    assert r.exit_code == 1


def test_watch_remove(tmp_path):
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    for channel_id in ("111", "222"):
        runner.invoke(app, [
            "config", "watch-add", "--channel-id", channel_id,
            "--repo-path", str(base), "--config", str(cfg_path),
        ])

    result = runner.invoke(app, [
        "config", "watch-remove", "111", "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output

    result = runner.invoke(app, [
        "config", "watch-remove", "111", "--config", str(cfg_path),
    ])
    assert result.exit_code == 1


def test_watch_remove_rejects_final_target(tmp_path):
    """Removing the last watch cannot create an unloadable config."""
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    runner.invoke(app, [
        "config", "watch-add", "--channel-id", "111",
        "--repo-path", str(base), "--config", str(cfg_path),
    ])
    original = cfg_path.read_bytes()

    result = runner.invoke(app, [
        "config", "watch-remove", "111", "--config", str(cfg_path),
    ])

    assert result.exit_code == 1
    assert "cannot remove watch target" in result.output
    assert cfg_path.read_bytes() == original


def test_config_set_below_minimum_rejected(tmp_path):
    """Semantic floors are enforced, not just types."""
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    runner.invoke(app, ["config", "watch-add",
                        "--channel-id", "111", "--repo-path", str(base),
                        "--config", str(cfg_path)])
    good = cfg_path.read_bytes()

    r = runner.invoke(app, ["config", "set", "max_reply_chars", "10",
                            "--config", str(cfg_path)])

    assert r.exit_code != 0
    assert cfg_path.read_bytes() == good
    runner.invoke(app, ["config", "watch-add",
                        "--channel-id", "111", "--repo-path", str(base),
                        "--config", str(cfg_path)])
    good = cfg_path.read_bytes()

    r = runner.invoke(app, ["config", "set", "max_response_tokens",
                            "banana", "--config", str(cfg_path)])

    assert r.exit_code != 0
    assert "invalid" in r.output.lower()
    assert cfg_path.read_bytes() == good  # byte-for-byte unchanged




def test_prompt_int_reprompts_on_garbage(monkeypatch):
    """Wizard numeric prompts re-ask instead of crashing mid-init."""
    from oi_agent import cli

    replies = iter(["banana", "4000"])
    monkeypatch.setattr(cli.typer, "prompt", lambda *a, **k: next(replies))
    assert cli._prompt_int("n", 1) == 4000




def test_run_flags_are_honest_about_logging():
    """The old --stream-logs claimed answer logging it did not do.

    Now: --stream only enables SSE transport; the sensitive answer logging
    lives behind an explicitly named --log-answers.
    """
    # Rich wraps help text across terminal lines, so assert on short tokens
    # that survive wrapping rather than full sentences.
    r = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
    assert r.exit_code == 0
    assert "--stream-logs" not in r.output
    assert "--stream" in r.output
    assert "transport only" in r.output
    assert "--log-answers" in r.output
    assert "PRIVACY-SENSITIVE" in r.output



def test_config_set_and_show_max_tool_iterations(tmp_path):
    """The knob is settable via config set and visible in config show."""
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    runner.invoke(app, ["config", "watch-add", "--channel-id", "111",
                        "--repo-path", str(base), "--config", str(cfg_path)])

    r = runner.invoke(app, ["config", "set", "max_tool_iterations", "7",
                            "--config", str(cfg_path)])
    assert r.exit_code == 0, r.output

    assert r.exit_code == 0, r.output
    assert "max_tool_iterations = 7" in r.output




@pytest.mark.parametrize("bad_value", ["0", "51", "banana"])
def test_config_set_rejects_bad_max_tool_iterations(tmp_path, bad_value):
    """Out-of-range and non-int values exit nonzero; file stays intact."""
    cfg_path = tmp_path / "config.toml"
    base = tmp_path / "repo"
    (base / ".git").mkdir(parents=True)
    runner.invoke(app, ["config", "watch-add", "--channel-id", "111",
                        "--repo-path", str(base), "--config", str(cfg_path)])
    good = cfg_path.read_bytes()

    r = runner.invoke(app, ["config", "set", "max_tool_iterations",
                            bad_value, "--config", str(cfg_path)])

    assert r.exit_code != 0
    assert "invalid" in r.output.lower()
    assert cfg_path.read_bytes() == good
