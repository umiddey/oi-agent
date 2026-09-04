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
from oi_agent.opencode.paths import resolve_opencode_paths
from oi_agent.opencode.prompt import FINAL_OUTPUT_MARKER
from oi_agent.opencode.provenance import worktree_fingerprint
from oi_agent.opencode.review_target import provider_ref_template
from oi_agent.opencode.runner import (
    MAX_PROMPT_BYTES,
    OpenCodeRunner,
    _bounded_text,
    _error_class,
    _select_final_text,
)

SESSION = "session-123"


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


def _part(part_type, msg, part_id, **extra):
    """One pinned-contract part: shared identity fields plus part payload."""
    part = {"id": part_id, "messageID": msg, "sessionID": SESSION, "type": part_type}
    part.update(extra)
    return part


def _event(event_type, part, **extra):
    """One pinned-contract JSONL event envelope."""
    event = {"type": event_type, "timestamp": 0, "sessionID": SESSION, "part": part}
    event.update(extra)
    return event


def _tool_step(msg, narration_part_id, narration):
    """One tool-loop step: start, narration text, tool call, tool-calls finish."""
    return [
        _event("step_start", _part("step-start", msg, msg + "-s")),
        _event("text", _part("text", msg, narration_part_id, text=narration)),
        _event("tool_use", _part("tool", msg, msg + "-t", tool="read",
                                 callID="call-" + msg,
                                 state={"status": "completed",
                                        "input": {}, "output": ""})),
        _event("step_finish", _part("step-finish", msg, msg + "-f",
                                    reason="tool-calls")),
    ]


def _final_step(msg, parts, reason="stop", finish=True, marked=False):
    """Final assistant message: start, text parts, then its completion marker."""
    stream = [_event("step_start", _part("step-start", msg, msg + "-s"))]
    first_part_id = parts[0][0] if parts else None
    for part_id, text in parts:
        if marked and part_id == first_part_id:
            text = f"{FINAL_OUTPUT_MARKER}\n{text}"
        stream.append(_event("text", _part("text", msg, part_id, text=text)))
    if finish:
        stream.append(_event("step_finish", _part("step-finish", msg, msg + "-f",
                                                  reason=reason)))
    return stream


def _pinned_stream(final_parts, final_reason="stop", finish_final=True):
    """Canonical narration -> tool -> narration -> marked final answer stream."""
    stream = _tool_step("msg_narration_a", "tx1", "Checking the repository now.")
    stream += _tool_step("msg_narration_b", "tx2", "Narrowing the answer down.")
    return stream + _final_step("msg_final", final_parts,
                                reason=final_reason, finish=finish_final,
                                marked=True)


_ERROR_EVENT = {"type": "error", "timestamp": 1, "sessionID": SESSION,
                "error": {"name": "ProviderAuthError", "data": {"message": "x"}}}

# The exact P1-2 auditor repro: a dict payload whose ``name`` is an arbitrary
# multi-line hostile string that must never become the error class.
_LEAK_NAME = "LEAK_MARKER\nraw-detail"
_LEAK_NAME_ERROR_EVENT = {"type": "error", "timestamp": 1, "sessionID": SESSION,
                          "error": {"name": _LEAK_NAME}}

# A long, hostile raw-string error payload: fake credentials, path-like
# fragments, and filler that must never survive classification.
_HOSTILE_PAYLOAD = (
    "ProviderError: token ghp_FAKE_TOKEN_VALUE leaked at "
    "/home/operator/.secret/config: " + "x" * 4096
)
_HOSTILE_ERROR_EVENT = {"type": "error", "timestamp": 1, "sessionID": SESSION,
                        "error": _HOSTILE_PAYLOAD}

_STREAMS = {
    "success": _pinned_stream([("p9", "UNIQUE_MARKER")]),
    "raw": (
        _tool_step("msg_narration_a", "tx1", "Checking the repository now.")
        + _tool_step("msg_narration_b", "tx2", "Narrowing the answer down.")
        + _final_step("msg_final", [("p9", "UNIQUE_MARKER")])
    ),
    "multipart": _pinned_stream([
        ("p9", "ALPHA"), ("p10", "BETA"), ("p11", "GAMMA"), ("p9", "ALPHA"),
    ]),
    "reasoning": [
        _event("step_start", _part("step-start", "msg_final", "s9")),
        _event("reasoning", _part("reasoning", "msg_final", "r1",
                                  text="REASONING_MARKER")),
        _event("text", _part("text", "msg_final", "p9",
                             text=f"{FINAL_OUTPUT_MARKER}\nUNIQUE_MARKER")),
        _event("step_finish", _part("step-finish", "msg_final", "f9",
                                    reason="stop")),
    ],
    "no_completion": _pinned_stream([("p9", "UNIQUE_MARKER")], finish_final=False),
    "midloop_end": _pinned_stream([("p9", "UNIQUE_MARKER")],
                                  final_reason="tool-calls"),
    "ambiguous": _pinned_stream([("p9", "FIRST_FINAL")]) + _final_step(
        "msg_extra", [("p20", "SECOND_FINAL")]),
    "incomplete_reason": _pinned_stream([("p9", "UNIQUE_MARKER")],
                                        final_reason="length"),
    "missing_id": [
        _event("step_start", _part("step-start", "msg_final", "s9")),
        {"type": "text", "timestamp": 0, "sessionID": SESSION,
         "part": {"id": "p9", "type": "text", "text": "UNIQUE_MARKER"}},
        _event("step_finish", _part("step-finish", "msg_final", "f9",
                                    reason="stop")),
    ],
    "conflicting_part": _pinned_stream([("p9", "FIRST"), ("p9", "SECOND")]),
    "unknown_event": _pinned_stream([("p9", "UNIQUE_MARKER")]) + [
        {"type": "teleport", "timestamp": 1, "sessionID": SESSION},
    ],
    "error_after_text": _pinned_stream([("p9", "UNIQUE_MARKER")]) + [_ERROR_EVENT],
    "hostile_string_error": _pinned_stream([("p9", "UNIQUE_MARKER")]) + [
        _HOSTILE_ERROR_EVENT,
    ],
    "hostile_name_error": _pinned_stream([("p9", "UNIQUE_MARKER")]) + [
        _LEAK_NAME_ERROR_EVENT,
    ],
    "unframed_planning": [
        _event("step_start", _part("step-start", "msg_final", "s9")),
        _event(
            "text",
            _part(
                "text",
                "msg_final",
                "p9",
                text="The latest question is essentially the same as before. "
                "I've already gathered the evidence. Let me be punchier this time.",
            ),
        ),
        _event(
            "step_finish",
            _part("step-finish", "msg_final", "f9", reason="stop"),
        ),
    ],
    "post_finish_text": _pinned_stream([("p9", "UNIQUE_MARKER")]) + [
        _event("text", _part("text", "msg_final", "late", text="SMUGGLED")),
    ],
    "large_stderr": _pinned_stream([("p9", "with stderr")]),
}


def _script(tmp_path, behavior="success"):
    """Create a fake OpenCode executable emitting pinned-contract JSONL streams."""
    trace = tmp_path / "trace.json"
    stream = _STREAMS.get(behavior, [])
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
behavior = {behavior!r}
if behavior == "large_stderr":
    sys.stderr.write("e" * (64 * 1024))
if behavior == "malformed":
    print("not json")
elif behavior == "unknown":
    print(json.dumps({{"type": "error", "sessionID": "old", "error": "unknown session"}}))
elif behavior == "cli_missing_session":
    sys.stderr.write("Session not found: ses_12345\\n")
    sys.exit(1)
elif behavior == "empty":
    sys.exit(0)
elif behavior == "large_line":
    print("x" * (200 * 1024))
elif behavior == "timeout":
    time.sleep(10)
else:
    for event in {stream!r}:
        print(json.dumps(event))
'''
    path = tmp_path / f"fake-opencode-{behavior}"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path, trace


@pytest.mark.asyncio
async def test_runner_uses_argv_stdin_isolated_env_and_final_text(tmp_path, monkeypatch):
    """The runner sends a bounded prompt and posts only the final completed message."""
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
        WatchTarget(1, str(repo)), "alice", "What changed?", "alice: hi"
    )

    assert reply.ok is True
    assert reply.session_id == "session-123"
    assert reply.text.startswith("UNIQUE_MARKER")
    assert "Checking the repository now." not in reply.text
    assert "Narrowing the answer down." not in reply.text
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
async def test_runner_multipart_final_message_assembled_in_order(tmp_path, monkeypatch):
    """Duplicate and interleaved parts of one final message join deterministically."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "multipart")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is True
    assert reply.text.startswith("ALPHABETAGAMMA")


@pytest.mark.asyncio
async def test_runner_reasoning_parts_never_reach_reply(tmp_path, monkeypatch):
    """Reasoning parts inside the final step are discarded, not posted."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "reasoning")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is True
    assert reply.text.startswith("UNIQUE_MARKER")
    assert "REASONING_MARKER" not in reply.text

@pytest.mark.asyncio
async def test_unframed_planning_is_never_delivered(tmp_path, monkeypatch):
    """A stop-completed planning block fails closed instead of being posted."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "unframed_planning")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "protocol_missing_final_marker"
    assert "The latest question is essentially" not in reply.text


@pytest.mark.asyncio
@pytest.mark.parametrize(("behavior", "error_class"), [
    ("malformed", "malformed_jsonl"),
    ("unknown", "unknown_session"),
    ("cli_missing_session", "unknown_session"),
    ("empty", "empty_response"),
    ("large_line", "json_line_limit"),
    ("no_completion", "protocol_no_final_message"),
    ("midloop_end", "protocol_no_final_message"),
    ("ambiguous", "protocol_ambiguous_final"),
    ("incomplete_reason", "protocol_incomplete_final"),
    ("missing_id", "protocol_missing_message_id"),
    ("conflicting_part", "protocol_conflicting_part"),
    ("unknown_event", "protocol_unknown_event"),
    ("error_after_text", "ProviderAuthError"),
    ("post_finish_text", "protocol_malformed_part"),
])
async def test_runner_rejects_malformed_and_error_events(tmp_path, monkeypatch,
                                                          behavior, error_class):
    """Malformed, error, and limit events become honest failures with bounded classes."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, behavior)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == error_class
    assert "not going to guess" in reply.text
    assert "UNIQUE_MARKER" not in reply.text
    assert "FIRST_FINAL" not in reply.text
    assert "SECOND_FINAL" not in reply.text


@pytest.mark.asyncio
async def test_runner_large_stderr_does_not_deadlock(tmp_path, monkeypatch):
    """Large stderr output is bounded without blocking subprocess execution."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "large_stderr")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
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
        WatchTarget(1, str(repo)), "alice", "question", ""
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
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "process_error"
    assert "not going to guess" in reply.text

@pytest.mark.asyncio
async def test_run_raw_prompt_returns_only_final_message_text(tmp_path, monkeypatch):
    """run_raw_prompt returns exactly the final completed message, nothing else."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, _ = _script(tmp_path, "raw")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=5,
    )
    res = await OpenCodeRunner(cfg).run_raw_prompt("Analyze this JSON")
    assert res == "UNIQUE_MARKER"


@pytest.mark.asyncio
async def test_run_raw_prompt_protocol_failure_returns_none(tmp_path, monkeypatch):
    """A stream without a completed final message returns None like other failures."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    binary, _ = _script(tmp_path, "no_completion")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=5,
    )
    res = await OpenCodeRunner(cfg).run_raw_prompt("Analyze this JSON")
    assert res is None


def _arrives_first(events):
    """Selection result tuple for a synthetic event list."""
    return _select_final_text(events)


def test_select_final_text_preserves_arrival_order_and_dedupes():
    """Out-of-order arrival is deterministic; exact duplicate parts dedupe."""
    events = [
        _event("step_start", _part("step-start", "m1", "s1")),
        _event("text", _part("text", "m1", "b2", text="BETA")),
        _event("text", _part("text", "m1", "a1", text="ALPHA")),
        _event("text", _part("text", "m1", "a1", text="ALPHA")),
        _event("step_finish", _part("step-finish", "m1", "f1", reason="stop")),
    ]
    text, session_id, error_class, text_parts = _arrives_first(events)
    assert text == "BETAALPHA"
    assert session_id == SESSION
    assert error_class is None
    assert text_parts == 2


def test_select_final_text_fail_closed_classes():
    """Structurally invalid streams return bounded classes without text."""
    base = [_event("step_start", _part("step-start", "m1", "s1"))]
    cases = [
        (base + [{"type": "teleport", "timestamp": 1, "sessionID": SESSION}],
         "protocol_unknown_event"),
        (base + [{"type": "text", "timestamp": 1, "sessionID": SESSION, "part": "nope"}],
         "protocol_malformed_part"),
        (base + [{"type": "text", "timestamp": 1, "sessionID": SESSION,
                  "part": {"id": "x", "type": "text", "text": "orphan"}}],
         "protocol_missing_message_id"),
        (base + [_event("text", _part("text", "m1", "a1", text="ONE")),
                 _event("text", _part("text", "m1", "a1", text="TWO"))],
         "protocol_conflicting_part"),
        (base + [_event("step_finish", _part("step-finish", "m1", "f1"))],
         "protocol_malformed_part"),
        ([_event("text", _part("text", "m1", "a1", text="UNFINISHED"))],
         "protocol_no_final_message"),
        (base + [_event("step_finish", _part("step-finish", "m1", "f1",
                                             reason="content-filter"))],
         "protocol_incomplete_final"),
    ]
    for events, expected in cases:
        text, _, error_class, _ = _select_final_text(events)
        assert text == ""
        assert error_class == expected


def test_select_final_text_synthetic_and_ignored_parts_are_not_answers():
    """Synthetic/ignored text flags never qualify a message as the final answer."""
    events = [
        _event("step_start", _part("step-start", "m1", "s1")),
        _event("text", _part("text", "m1", "a1", text="system notice",
                             synthetic=True)),
        _event("text", _part("text", "m1", "a2", text="queued echo",
                             ignored=True)),
        _event("step_finish", _part("step-finish", "m1", "f1", reason="stop")),
    ]
    text, _, error_class, text_parts = _select_final_text(events)
    assert text == ""
    assert error_class == "protocol_no_final_message"
    assert text_parts == 0


def test_select_final_text_rejects_text_after_finish_for_same_message():
    """Text streamed after a finish for the same messageID fails closed."""
    events = _final_step("m1", [("a1", "FINAL")]) + [
        _event("text", _part("text", "m1", "late", text="SMUGGLED_TEXT")),
    ]
    text, _, error_class, _ = _select_final_text(events)
    assert text == ""
    assert error_class == "protocol_malformed_part"


def test_select_final_text_new_message_after_finish_is_not_malformed():
    """A new messageID after a finish follows the normal superseded rules."""
    superseded = _final_step("m1", [("a1", "FIRST_FINAL")]) + _final_step(
        "m2", [("a2", "SECOND_FINAL")]
    )
    text, _, error_class, _ = _select_final_text(superseded)
    assert text == ""
    assert error_class == "protocol_ambiguous_final"

    # A follow-on message that never completes does not disturb the
    # already-completed final answer.
    follow_on = _final_step("m1", [("a1", "FINAL")]) + [
        _event("step_start", _part("step-start", "m2", "s2")),
        _event("text", _part("text", "m2", "a2", text="UNFINISHED_TAIL")),
    ]
    text, _, error_class, text_parts = _select_final_text(follow_on)
    assert text == "FINAL"
    assert error_class is None
    assert text_parts == 2


def test_select_final_text_in_stream_error_takes_precedence():
    """A stream error class is surfaced so the caller refuses any final text."""
    events = _final_step("m1", [("a1", "FINAL")]) + [_ERROR_EVENT]
    _, _, error_class, _ = _select_final_text(events)
    assert error_class == "ProviderAuthError"


def test_error_class_string_payloads_are_bounded():
    """Raw-string error payloads contribute only the bounded classes."""
    assert _error_class({"error": "Session not found: ses_deadbeef"}) == "unknown_session"
    assert _error_class({"error": "UNKNOWN_SESSION token replay"}) == "unknown_session"
    assert _error_class({"error": _HOSTILE_PAYLOAD}) == "opencode_error"
    assert _error_class({"error": ""}) == "opencode_error"
    assert _error_class({"data": 42}) == "opencode_error"
    # Dict payloads keep their short enum-like names.
    assert _error_class({"error": {"name": "ProviderAuthError"}}) == "ProviderAuthError"
    assert _error_class({"error": {"type": "rate_limit"}}) == "rate_limit"


def test_error_class_repro_leaking_dict_name_collapses():
    """P1-2 repro: a multi-line dict name never becomes the error class."""
    assert _error_class(_LEAK_NAME_ERROR_EVENT) == "opencode_error"
    # In-stream precedence still holds: the leaked name cannot ride along
    # with a completed final answer.
    text, _, error_class, _ = _select_final_text(
        _final_step("m1", [("a1", "FINAL")]) + [_LEAK_NAME_ERROR_EVENT]
    )
    assert error_class == "opencode_error"
    assert "LEAK_MARKER" not in (error_class or "")
    assert "raw-detail" not in (error_class or "")


def test_error_class_hostile_dict_candidates_collapse():
    """Hostile dict name/type shapes fall through to the bounded constant."""
    hostile_names = [
        _LEAK_NAME,
        "A" * 200,                    # valid charset, over the 64-char bound
        "../etc/passwd",              # path syntax
        "http://evil.example/leak",   # URL syntax
        "has spaces",
        "tab\tseparator",
        "bell\x07control",
        {"deeply": "nested"},         # non-string candidates never pass
        42,
        None,
        "",
    ]
    for name in hostile_names:
        event = {"type": "error", "error": {"name": name}}
        assert _error_class(event) == "opencode_error", repr(name)
    # A hostile type alone collapses too.
    assert _error_class({"error": {"type": "http://evil.example"}}) == "opencode_error"


def test_error_class_valid_candidates_fall_through_hostile_ones():
    """Validation falls through name to type; valid enum-like names still win."""
    # Hostile name falls through to a valid type.
    assert _error_class({"error": {"name": _LEAK_NAME, "type": "rate_limit"}}) == "rate_limit"
    assert _error_class({"error": {"name": "../evil", "type": "SessionNotFound"}}) == "SessionNotFound"
    # A valid name wins even when the type is hostile.
    assert _error_class({"error": {"name": "ProviderAuthError", "type": _LEAK_NAME}}) == "ProviderAuthError"
    # Positive controls: enum-like names propagate; punctuation in the
    # allowlist charset is fine.
    assert _error_class(_ERROR_EVENT) == "ProviderAuthError"
    assert _error_class({"error": {"name": "Provider.RateLimit-429"}}) == "Provider.RateLimit-429"


def test_error_class_name_length_boundary_at_64():
    """A 64-char valid-charset name passes; 65 chars collapse (bounded cap)."""
    assert _error_class({"error": {"name": "A" * 64}}) == "A" * 64
    assert _error_class({"error": {"name": "A" * 65}}) == "opencode_error"


@pytest.mark.asyncio
async def test_runner_hostile_string_error_payload_is_fully_bounded(tmp_path, monkeypatch):
    """A long hostile string error never leaks into the class or the reply."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "hostile_string_error")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "opencode_error"
    assert "not going to guess" in reply.text
    for fragment in ("ghp_FAKE_TOKEN_VALUE", ".secret", "ProviderError",
                     "x" * 256, "/home/operator"):
        assert fragment not in reply.text
        assert fragment not in (reply.error_class or "")


@pytest.mark.asyncio
async def test_runner_leaking_dict_error_name_never_reaches_reply(tmp_path, monkeypatch):
    """P1-2 repro end to end: a hostile dict error name stays out of the reply."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    repo = _repo(tmp_path)
    binary, _ = _script(tmp_path, "hostile_name_error")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(repo)), "alice", "question", ""
    )

    assert reply.ok is False
    assert reply.error_class == "opencode_error"
    assert "not going to guess" in reply.text
    for fragment in ("LEAK_MARKER", "raw-detail"):
        assert fragment not in reply.text
        assert fragment not in (reply.error_class or "")
    assert "UNIQUE_MARKER" not in reply.text

def test_reply_cap_drops_metadata():
    """Long model text is bounded without appending provenance metadata."""
    text = _bounded_text("x" * 1000, 200)
    assert len(text) <= 200
    assert text.endswith("[reply truncated]")


def test_bounded_text_prefers_paragraph_boundary():
    """Paragraph boundaries win even when a later sentence end would fit."""
    para1 = "First paragraph stands alone."
    second = "Second para. "
    text = para1 + "\n\n" + second + "y" * 400
    cap = len(para1) + len("\n\n[reply truncated]")
    out = _bounded_text(text, cap)
    assert out.startswith(para1)
    assert "Second para" not in out
    assert "[reply truncated]" in out
    assert len(out) <= cap


def test_bounded_text_falls_back_to_sentence_boundary():
    """Without paragraph boundaries the last fitting sentence end is used."""
    body = "One sentence ends here. Another ends there. " + "z" * 400
    out = _bounded_text(body, 120)
    assert out.startswith("One sentence ends here. Another ends there.")
    assert "zzz" not in out
    assert "[reply truncated]" in out
    assert len(out) <= 120


def test_bounded_text_char_fallback_without_metadata():
    """A boundary-free text is cut at the cap with only a truncation marker."""
    out = _bounded_text("q" * 500, 200)
    assert "\n\n[reply truncated]" in out
    assert out.endswith("[reply truncated]")
    assert len(out) <= 200
    assert out.startswith("qqq")


def test_bounded_text_within_cap_keeps_no_marker():
    """Text already inside the cap is returned unchanged."""
    out = _bounded_text("short answer", 200)
    assert out == "short answer"
    assert "[reply truncated]" not in out


def test_bounded_text_degenerate_cap_truncates_body():
    """A cap smaller than the truncation marker returns the bounded prefix."""
    out = _bounded_text("BODY_TEXT" * 100, 10)
    assert out == "BODY_TEXTB"
    assert len(out) == 10

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
        WatchTarget(1, str(repo)), "alice", "question", ""
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
        WatchTarget(1, str(repo)), "alice", oversized_question, ""
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
            WatchTarget(1, str(repo)), "alice", "SECRET_QUESTION_TEXT", ""
        )
        assert reply.ok is True
        raw = await runner.run_raw_prompt("SECRET_RAW_PROMPT_TEXT")
        assert raw is not None

    joined = "\n".join(r.getMessage() for r in caplog.records)
    for sensitive in ("SECRET_QUESTION_TEXT", "SECRET_RAW_PROMPT_TEXT", "UNIQUE_MARKER"):
        assert sensitive not in joined

# ==============================================================================
# MR/PR review integration (plan Phase 5): controller snapshots, provenance,
# session isolation, honest failures, and cleanup.
# ==============================================================================

def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _bare_with_mr(
    tmp_path: Path,
    name: str = "remote",
    *,
    provider: str = "gitlab",
    number: int = 188,
) -> tuple[Path, Path, str, str]:
    """Create a local bare remote and watched clone with synthetic provider MR refs."""
    remote = tmp_path / f"{name}.git"
    _git("init", "-q", "--bare", str(remote))
    clone = tmp_path / f"clone-{name}"
    clone.mkdir(parents=True)
    _git("init", "-q", "-b", "main", str(clone))
    _git("-C", str(clone), "config", "user.email", "t@example.com")
    _git("-C", str(clone), "config", "user.name", "Test")
    _git("-C", str(clone), "remote", "add", "origin", str(remote))
    (clone / "base.txt").write_text("base\n", encoding="utf-8")
    _git("-C", str(clone), "add", ".")
    _git("-C", str(clone), "commit", "-qm", "base")
    base_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "-b", "feature")
    (clone / "base.txt").write_text("head\n", encoding="utf-8")
    (clone / "head.txt").write_text("head\n", encoding="utf-8")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "head")
    head_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge result", "feature")
    _git("-C", str(clone), "push", "-q", "origin", "main", "feature")
    _git(
        "-C", str(remote), "update-ref",
        provider_ref_template(provider, number, "head"), head_sha,
    )
    merge_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git(
        "-C", str(remote), "update-ref",
        provider_ref_template(provider, number, "merge"), merge_sha,
    )
    return remote, clone, base_sha, head_sha


def _reviews_glob() -> list[Path]:
    reviews = resolve_opencode_paths().state_home / "reviews"
    if not reviews.exists():
        return []
    return list(reviews.glob("oi-review-*"))


@pytest.mark.asyncio
async def test_runner_mr_review_audits_snapshots_with_mr_provenance(tmp_path, monkeypatch):
    """An explicit MR reference audits controller snapshots, never the checkout."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    binary, trace_path = _script(tmp_path)
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=5,
    )
    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", "alice: please review"
    )

    assert reply.ok is True
    assert reply.session_id is None  # review sessions are never resumed or stored
    assert reply.sha == f"MR !188 base {base_sha} head {head_sha}"
    assert reply.text.startswith("UNIQUE_MARKER")
    assert "Checking the repository now." not in reply.text  # narration dropped

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    argv = trace["argv"]
    assert "--session" not in argv
    head_dir = Path(argv[argv.index("--dir") + 1])
    assert head_dir != clone.resolve()  # a fresh snapshot dir, NOT the watched repo
    assert head_dir.name == "head"      # head snapshot is the primary workspace
    base_dir = head_dir.parent / "base"
    policy = json.loads(trace["config"])["agent"]["oi"]["permission"]
    assert policy["*"] == "deny"
    assert policy["bash"] == "deny"
    assert policy["edit"] == "deny"
    assert policy["webfetch"] == "deny"
    ext = policy["external_directory"]
    assert ext["*"] == "deny"
    assert ext[str(base_dir)] == "allow"
    assert ext[f"{base_dir}/**"] == "allow"
    stdin = trace["stdin"]
    assert reply.text == "UNIQUE_MARKER"
    assert base_sha in stdin and head_sha in stdin
    assert "MR !188" in stdin
    assert "M base.txt" in stdin and "A head.txt" in stdin
    assert str(base_dir) in stdin and str(head_dir) in stdin
    assert str(clone) not in stdin  # the watched checkout is not the audit subject

    # Snapshots removed after the run; watched worktree untouched.
    assert not head_dir.exists()
    assert not base_dir.exists()
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint


@pytest.mark.asyncio
async def test_runner_pr_review_reports_github_provenance(tmp_path, monkeypatch):
    """GitHub pull requests use the PR #n provenance form end to end."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path, provider="github")
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "PR 188", ""
    )

    assert reply.ok is True
    assert reply.session_id is None
    assert reply.sha == f"PR #188 base {base_sha} head {head_sha}"
    stdin = json.loads(trace_path.read_text(encoding="utf-8"))["stdin"]
    assert "PR #188" in stdin


@pytest.mark.asyncio
async def test_runner_ambiguous_mr_ref_fails_without_auditing(tmp_path, monkeypatch):
    """A bare MR number matching two configured repos fails honestly; no fallback."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote1, clone1, _b1, _h1 = _bare_with_mr(tmp_path, name="one")
    _remote2, clone2, _b2, _h2 = _bare_with_mr(tmp_path, name="two")
    fp1 = worktree_fingerprint(clone1)
    fp2 = worktree_fingerprint(clone2)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, repos=[str(clone1), str(clone2)]), "alice", "MR 188", ""
    )

    assert reply.ok is False
    assert reply.error_class == "review_ref_ambiguous"
    assert reply.session_id is None
    assert "repository" in reply.text or "URL" in reply.text
    assert not trace_path.exists()  # the fake OpenCode binary was never invoked
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone1) == fp1  # no checkout was audited or touched
    assert worktree_fingerprint(clone2) == fp2


@pytest.mark.asyncio
async def test_runner_unavailable_mr_ref_fails_honestly(tmp_path, monkeypatch):
    """A bare MR number in no configured repository fails without a fallback audit."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 999", ""
    )

    assert reply.ok is False
    assert reply.error_class == "review_ref_unavailable"
    assert reply.session_id is None
    assert "repository" in reply.text or "URL" in reply.text
    assert not trace_path.exists()
    assert _reviews_glob() == []


@pytest.mark.asyncio
async def test_runner_invalid_mr_ref_fails_without_spawn(tmp_path, monkeypatch):
    """Malformed review references fail closed before any Git fetch or subprocess."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 0", ""
    )

    assert reply.ok is False
    assert reply.error_class == "review_ref_invalid"
    assert not trace_path.exists()
    assert _reviews_glob() == []


@pytest.mark.asyncio
async def test_runner_snapshot_failure_returns_bounded_class(tmp_path, monkeypatch):
    """Snapshot verification failures become honest review failures with cleanup."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(remote), "update-ref", "-d", "refs/merge-requests/188/merge")
    fingerprint = worktree_fingerprint(clone)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", ""
    )

    assert reply.ok is False
    assert reply.error_class == "review_base_unavailable"
    assert reply.session_id is None
    assert not trace_path.exists()
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint


@pytest.mark.asyncio
async def test_runner_review_prompt_bound_fails_closed(tmp_path, monkeypatch):
    """An over-limit review prompt fails closed after materialization, no spawn."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", "x" * (MAX_PROMPT_BYTES + 1)
    )

    assert reply.ok is False
    assert reply.error_class == "prompt_too_large"
    assert reply.sha == f"MR !188 base {base_sha} head {head_sha}"
    assert not trace_path.exists()
    assert _reviews_glob() == []

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


def _bare_with_defect(
    tmp_path: Path, *, provider: str = "gitlab", number: int = 188
) -> tuple[Path, Path, str, str]:
    """Create a bare remote whose proposed head introduces a defect base lacks."""
    remote = tmp_path / "remote.git"
    _git("init", "-q", "--bare", str(remote))
    clone = tmp_path / "clone-defect"
    clone.mkdir(parents=True)
    _git("init", "-q", "-b", "main", str(clone))
    _git("-C", str(clone), "config", "user.email", "t@example.com")
    _git("-C", str(clone), "config", "user.name", "Test")
    _git("-C", str(clone), "remote", "add", "origin", str(remote))
    (clone / "defect_check.py").write_text("THRESHOLD = 42\n", encoding="utf-8")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "base")
    base_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "-b", "feature")
    (clone / "defect_check.py").write_text("THRESHOLD = 999999\n", encoding="utf-8")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "head")
    head_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge result", "feature")
    _git("-C", str(clone), "push", "-q", "origin", "main", "feature")
    _git("-C", str(remote), "update-ref",
         provider_ref_template(provider, number, "head"), head_sha)
    merge_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(remote), "update-ref",
         provider_ref_template(provider, number, "merge"), merge_sha)
    return remote, clone, base_sha, head_sha


def _reading_script(tmp_path: Path, filename: str) -> Path:
    """Fake OpenCode that reads filename from its --dir and the prompt-authorized base dir.

    Proves real read access to both controller snapshots: the head file comes
    from the primary ``--dir`` workspace and the base file from the auxiliary
    root exactly as named in the prompt's BASE DIRECTORY line. The final
    message carries both file contents so the test can see the defect only
    where it exists.
    """
    body = f'''#!{sys.executable}
import json, sys
head_dir = sys.argv[sys.argv.index("--dir") + 1]
stdin = sys.stdin.read()
base_dir = ""
for line in stdin.splitlines():
    if line.startswith("BASE DIRECTORY"):
        base_dir = line.split(":", 1)[1].strip()
try:
    with open(head_dir + "/{filename}", encoding="utf-8") as handle:
        head_value = handle.read().strip()
except OSError:
    head_value = "MISSING"
try:
    with open(base_dir + "/{filename}", encoding="utf-8") as handle:
        base_value = handle.read().strip()
except OSError:
    base_value = "MISSING"
final = {{"id": "rf", "messageID": "msg_final", "sessionID": "session-123",
        "type": "text", "text": "[[OI_FINAL_ANSWER]]\\nHEAD_VALUE=" + head_value + " BASE_VALUE=" + base_value}}
print(json.dumps({{"type": "step_start", "timestamp": 0, "sessionID": "session-123",
                  "part": {{"id": "rs", "messageID": "msg_final",
                           "sessionID": "session-123", "type": "step-start"}}}}))
print(json.dumps({{"type": "text", "timestamp": 0, "sessionID": "session-123", "part": final}}))
print(json.dumps({{"type": "step_finish", "timestamp": 0, "sessionID": "session-123",
                  "part": {{"id": "rfin", "messageID": "msg_final",
                           "sessionID": "session-123", "type": "step-finish",
                           "reason": "stop"}}}}))
'''
    path = tmp_path / "fake-opencode-reader"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


@pytest.mark.asyncio
async def test_runner_mr_review_reads_defect_from_head_and_base_snapshots(
    tmp_path, monkeypatch,
):
    """The agent really reads both snapshots: the defect exists only in head."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_with_defect(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    binary = _reading_script(tmp_path, "defect_check.py")
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", ""
    )

    assert reply.ok is True
    # The head snapshot exposes the defective constant; the base the correct one.
    assert "HEAD_VALUE=THRESHOLD = 999999" in reply.text
    assert "BASE_VALUE=THRESHOLD = 42" in reply.text
    assert "MISSING" not in reply.text
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint


@pytest.mark.asyncio
async def test_runner_mr_review_permission_matrix_is_read_only(tmp_path, monkeypatch):
    """The generated review policy denies everything except scoped read tools."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    binary, trace_path = _script(tmp_path)
    cfg = Config(opencode_binary=str(binary), opencode_model="provider/model")

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", ""
    )
    assert reply.ok is True

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    policy = json.loads(trace["config"])["agent"]["oi"]["permission"]
    head_dir = Path(trace["argv"][trace["argv"].index("--dir") + 1])
    base_dir = head_dir.parent / "base"

    assert policy["*"] == "deny"
    for tool in ("bash", "webfetch", "websearch", "edit", "write", "patch",
                 "task", "skill", "question", "mcp", "grep", "todolist"):
        assert policy[tool] == "deny", tool
    assert policy["lsp"] == "allow"
    secret_patterns = (".git/**", ".env", "**/.env", "**/*secret*", "**/*.pem")
    for tool in ("read", "glob", "list"):
        rules = policy[tool]
        assert rules["*"] == "allow"
        for pattern in secret_patterns:
            assert rules[pattern] == "deny"                      # workspace (head) root
            assert rules[f"{base_dir}/{pattern}"] == "deny"      # base snapshot root
    ext = policy["external_directory"]
    assert isinstance(ext, dict)
    # ONLY the base snapshot directory (plus its subtree) is externally authorized.
    assert set(ext) == {"*", str(base_dir), f"{base_dir}/**"}
    assert ext["*"] == "deny"
    assert ext[str(base_dir)] == "allow"
    assert ext[f"{base_dir}/**"] == "allow"


@pytest.mark.asyncio
async def test_runner_mr_review_timeout_cleans_snapshots_and_fails_honestly(
    tmp_path, monkeypatch,
):
    """A mid-review timeout removes snapshots, reaps the process, and fails honestly."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    binary, trace_path = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=1,
    )

    reply = await OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", ""
    )

    assert reply.ok is False
    assert reply.error_class == "timeout"
    assert reply.sha == f"MR !188 base {base_sha} head {head_sha}"
    assert "not going to guess" in reply.text
    assert "UNIQUE_MARKER" not in reply.text
    pid = json.loads(trace_path.read_text(encoding="utf-8"))["pid"]
    assert not os.path.exists(f"/proc/{pid}")
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint


@pytest.mark.asyncio
async def test_runner_mr_review_cancellation_cleans_snapshots(tmp_path, monkeypatch):
    """Cancelling a review run removes snapshots and reaps the subprocess."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    binary, trace_path = _script(tmp_path, "timeout")
    cfg = Config(
        opencode_binary=str(binary),
        opencode_model="provider/model",
        opencode_timeout_seconds=30,
    )
    task = asyncio.create_task(OpenCodeRunner(cfg).run(
        WatchTarget(1, str(clone)), "alice", "MR 188", ""
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
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint
