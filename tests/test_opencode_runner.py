"""Behavioral tests for the async OpenCode runner boundary."""

import asyncio
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.opencode.runner import MAX_PROMPT_BYTES, OpenCodeRunner, _bounded_text


def _repo(tmp_path):
    """Create a minimal committed git fixture."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "marker.txt").write_text("UNIQUE_MARKER\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "marker.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo


def _script(tmp_path, behavior="success"):
    """Create a fake OpenCode executable recording only safe invocation metadata."""
    trace = tmp_path / "trace.json"
    body = f'''#!{sys.executable}
import json, os, sys, time
trace = {str(trace)!r}
with open(trace, "w", encoding="utf-8") as handle:
    json.dump({{"argv": sys.argv[1:], "stdin": sys.stdin.read(),
               "pid": os.getpid(),
               "config": os.environ.get("OPENCODE_CONFIG_CONTENT", ""),
               "disable": os.environ.get("OPENCODE_DISABLE_PROJECT_CONFIG"),
               "env_home": os.environ.get("HOME", ""),
               "xdg_config": os.environ.get("XDG_CONFIG_HOME", ""),
               "xdg_data": os.environ.get("XDG_DATA_HOME", "")}}, handle)
if {behavior!r} == "malformed":
    print("not json")
elif {behavior!r} == "unknown":
    print(json.dumps({{"type": "error", "sessionID": "old", "error": "unknown session"}}))
elif {behavior!r} == "cli_missing_session":
    sys.stderr.write("Session not found: ses_12345\\n")
    sys.exit(1)
elif {behavior!r} == "empty":
    sys.exit(0)
elif {behavior!r} == "large_line":
    print("x" * (200 * 1024))
elif {behavior!r} == "large_stderr":
    sys.stderr.write("e" * (64 * 1024))
    print(json.dumps({{"type": "text", "sessionID": "session-123",
                      "part": {{"text": "with stderr"}}}}))
elif {behavior!r} == "timeout":
    time.sleep(10)
else:
    print(json.dumps({{"type": "text", "sessionID": "session-123",
                      "part": {{"text": "UNIQUE_MARKER"}}}}))
    print(json.dumps({{"type": "text", "sessionID": "session-123",
                      "part": {{"text": " final answer"}}}}))
'''
    path = tmp_path / f"fake-opencode-{behavior}"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path, trace


@pytest.mark.asyncio
async def test_runner_uses_argv_stdin_isolated_env_and_jsonl_text(tmp_path, monkeypatch):
    """The runner sends a bounded prompt and extracts only completed text parts."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_steps=30,
        opencode_timeout_seconds=5,
    )
    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "What changed?", "user1: hi"
    )

    assert reply.ok is True
    assert reply.session_id == "session-123"
    assert "UNIQUE_MARKER" in reply.text
    assert "audited at " in reply.text
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["argv"] == [
        "run", "--format", "json", "--model", "provider/model",
        "--agent", "oi", "--dir", str(repo.resolve()),
    ]
    assert "What changed?" in trace["stdin"]
    assert trace["disable"] == "1"
    policy = json.loads(trace["config"])["agent"]["oi"]["permission"]
    assert policy["*"] == "deny"
    assert "--auto" not in trace["argv"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("behavior", "error_class"), [
    ("malformed", "malformed_jsonl"),
    ("unknown", "unknown_session"),
    ("cli_missing_session", "unknown_session"),
    ("empty", "empty_response"),
    ("large_line", "json_line_limit"),
])
async def test_runner_rejects_malformed_and_error_events(tmp_path, monkeypatch,
                                                          behavior, error_class):
    """Malformed, error, and limit events become honest failures with bounded classes."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, behavior)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == error_class
    assert "not going to guess" in reply.text


@pytest.mark.asyncio
async def test_runner_large_stderr_does_not_deadlock(tmp_path, monkeypatch):
    """Large stderr output is bounded without blocking subprocess execution."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "large_stderr")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "question", ""
    )

    assert reply.ok is True
    assert "with stderr" in reply.text


@pytest.mark.asyncio
async def test_runner_timeout_returns_failure(tmp_path, monkeypatch):
    """Timeout terminates the process group and never returns partial text."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=1,
    )

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "timeout"


@pytest.mark.asyncio
async def test_runner_subprocess_spawn_oserror_returns_process_error(tmp_path, monkeypatch):
    """When process creation raises OSError, the runner returns process_error without unbound locals."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "success")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    async def raise_oserror(*args, **kwargs):
        raise OSError("Permission denied / cannot fork")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", raise_oserror)

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "process_error"
    assert "not going to guess" in reply.text

def test_reply_cap_preserves_audit_footer():
    """Long model text is bounded without dropping provenance."""
    text = _bounded_text("x" * 1000, "abc1234", 200)
    assert len(text) <= 200
    assert text.endswith("-# audited at abc1234")


@pytest.mark.asyncio
async def test_runner_cancellation_terminates_process_group(tmp_path, monkeypatch):
    """Cancelling a run reaps its OpenCode child instead of orphaning it."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, trace_path = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=30,
    )
    task = asyncio.create_task(OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", "question", ""
    ))
    for _ in range(100):
        if trace_path.exists():
            break
        await asyncio.sleep(0.01)
    assert trace_path.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = json.loads(trace_path.read_text(encoding="utf-8"))["pid"]
    assert not os.path.exists(f"/proc/{pid}")



@pytest.mark.asyncio
async def test_run_raw_prompt_uses_eval_agent_with_denied_permissions(tmp_path, monkeypatch):
    """run_raw_prompt invokes OpenCode with oi_eval agent and all tools denied."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, trace_path = _script(tmp_path)
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_steps=1,
        opencode_timeout_seconds=5,
    )
    runner = OpenCodeRunner(cfg)
    res = await runner.run_raw_prompt("Analyze this JSON", system_prompt="Custom prompt")
    assert res is not None
    assert "UNIQUE_MARKER" in res
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert "--agent" in trace["argv"]
    assert "oi_eval" in trace["argv"]
    policy = json.loads(trace["config"])["agent"]["oi_eval"]["permission"]
    assert policy["*"] == "deny"


def _isolated_roots(operator_home: Path):
    """Expected isolated HOME/XDG roots under the operator HOME."""
    root = Path(operator_home).expanduser().resolve() / ".local" / "share" / "oi" / "opencode"
    return root / "home", root / "config", root / "data"


@pytest.mark.asyncio
async def test_run_raw_prompt_subprocess_isolation(tmp_path, monkeypatch):
    """run_raw_prompt isolates HOME/XDG, disables project config, and grants no tools."""
    operator_home = tmp_path / "operator"
    monkeypatch.setenv("HOME", str(operator_home))
    binary, trace_path = _script(tmp_path)
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=5,
    )
    res = await OpenCodeRunner(cfg).run_raw_prompt("Isolation probe")
    assert res is not None

    iso_home, iso_config, iso_data = _isolated_roots(operator_home)
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert Path(trace["env_home"]) == iso_home
    assert Path(trace["xdg_config"]) == iso_config
    assert Path(trace["xdg_data"]) == iso_data
    assert trace["disable"] == "1"
    # The eval agent has no repository binding at all.
    assert "--dir" not in trace["argv"]
    agent_def = json.loads(trace["config"])["agent"]["oi_eval"]
    assert agent_def["permission"] == {"*": "deny"}
    assert agent_def["steps"] == 1


@pytest.mark.asyncio
async def test_run_raw_prompt_timeout_reaps_process(tmp_path, monkeypatch):
    """Timeout in run_raw_prompt terminates the subprocess group (no orphan)."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, trace_path = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=1,
    )
    res = await OpenCodeRunner(cfg).run_raw_prompt("slow analysis", timeout=1)
    assert res is None
    pid = json.loads(trace_path.read_text(encoding="utf-8"))["pid"]
    assert not os.path.exists(f"/proc/{pid}")


@pytest.mark.asyncio
async def test_run_raw_prompt_cancellation_reaps_process(tmp_path, monkeypatch):
    """B9: cancelling run_raw_prompt reaps its subprocess instead of orphaning it."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, trace_path = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=30,
    )
    task = asyncio.create_task(OpenCodeRunner(cfg).run_raw_prompt("slow analysis"))
    for _ in range(100):
        if trace_path.exists():
            break
        await asyncio.sleep(0.01)
    assert trace_path.exists()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pid = json.loads(trace_path.read_text(encoding="utf-8"))["pid"]
    assert not os.path.exists(f"/proc/{pid}")


@pytest.mark.asyncio
async def test_run_raw_prompt_stdin_bound_never_spawns(tmp_path, monkeypatch):
    """B10: an over-limit raw prompt fails closed before any subprocess spawns."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")
    oversized = "x" * (MAX_PROMPT_BYTES + 1)

    with pytest.raises(ValueError):
        await OpenCodeRunner(cfg).run_raw_prompt(oversized)

    assert not trace_path.exists()  # no process was ever spawned


@pytest.mark.asyncio
async def test_runner_stdin_bound_returns_failure_without_spawn(tmp_path, monkeypatch):
    """B10: an over-limit audit prompt yields a bounded failure class, no spawn."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")
    oversized_question = "q" * (MAX_PROMPT_BYTES + 1)

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "user1", oversized_question, ""
    )

    assert reply.ok is False
    assert reply.error_class == "prompt_too_large"
    assert "not going to guess" in reply.text
    assert not trace_path.exists()  # no process was ever spawned


@pytest.mark.asyncio
async def test_runner_and_raw_prompt_do_not_log_prompt_or_response(tmp_path, monkeypatch, caplog):
    """Neither the audit path nor run_raw_prompt logs prompt text or model output."""
    import logging

    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path)
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=5,
    )
    runner = OpenCodeRunner(cfg)

    with caplog.at_level(logging.DEBUG, logger="oi_agent"):
        reply = await runner.run(
            WatchTarget(1, str(repo)), "user1", "SECRET_QUESTION_TEXT", ""
        )
        assert reply.ok is True
        raw = await runner.run_raw_prompt("SECRET_RAW_PROMPT_TEXT")
        assert raw is not None

    joined = "\n".join(r.getMessage() for r in caplog.records)
    for sensitive in ("SECRET_QUESTION_TEXT", "SECRET_RAW_PROMPT_TEXT", "UNIQUE_MARKER"):
        assert sensitive not in joined

def test_all_python_sources_compatible_with_python_311():
    """All package files parse cleanly without 3.12+ syntax (like backslashes in f-string expressions)."""
    import ast

    root = Path(__file__).parent.parent / "src" / "oi_agent"
    for py_file in root.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, str(py_file))
        # In Python <=3.11, backslashes inside FormattedValue expressions raise SyntaxError.
        for node in ast.walk(tree):
            if isinstance(node, ast.FormattedValue):
                segment = ast.get_source_segment(source, node.value)
                assert segment is None or "\\" not in segment
