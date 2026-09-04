"""Async OpenCode subprocess runner with bounded output and process cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..store import Store

from ..agent.reply import Reply
from ..config import Config, WatchTarget
from .bootstrap import resolve_binary
from .paths import resolve_opencode_paths
from .policy import build_agent_config
from .prompt import FINAL_OUTPUT_MARKER, build_prompt
from .provenance import audit_sha, git_pull, mutation_warning, worktree_fingerprint
from .review_snapshots import ReviewSnapshot, ReviewSnapshotError, materialized_review
from .review_target import (
    PROVIDER_GITHUB,
    PROVIDER_GITLAB,
    ReviewRefError,
    ReviewTarget,
    resolve_review_target,
)

logger = logging.getLogger(__name__)

MAX_JSON_LINE_BYTES = 128 * 1024
MAX_STDOUT_BYTES = 2 * 1024 * 1024
MAX_STDERR_BYTES = 32 * 1024
# Bounded stdin (plan Phase 8 item 6): prompts above this size fail closed
# before any subprocess is spawned.
MAX_PROMPT_BYTES = 256 * 1024

_repo_locks: dict[str, asyncio.Lock] = {}
_repo_locks_guard = asyncio.Lock()


def scope_id_for_target(target: WatchTarget) -> str:
    """Return the ONE canonical memory scope key for a conversation target.

    Guild-wide watches scope memory to the guild id; channel watches scope it
    to the channel id. The prompt fetch in ``_run_locked`` and the reflection
    enqueue in the Discord client MUST both use this rule (finding B1) so a
    stored profile is always readable by the next prompt for the same scope.
    """
    return str(target.guild_id if target.is_guild_watch else target.channel_id)


class _ProtocolError(RuntimeError):
    """Internal bounded JSONL protocol failure."""

    def __init__(self, error_class: str) -> None:
        super().__init__(error_class)
        self.error_class = error_class


async def _repo_lock(repo_path: str) -> asyncio.Lock:
    """Return the process-wide async lock for one canonical repository."""
    key = str(Path(repo_path).expanduser().resolve())
    async with _repo_locks_guard:
        lock = _repo_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _repo_locks[key] = lock
        return lock


async def _acquire_repo_locks(repo_paths: list[str]) -> AsyncExitStack:
    """Acquire async locks for all involved repositories in canonical sorted order.

    Sorting paths lexicographically guarantees deadlock-free lock acquisition
    across concurrent multi-repository audits.

    Args:
        repo_paths: List of repository paths to lock.

    Returns:
        AsyncExitStack: Managed context holding all acquired repository locks.
    """
    stack = AsyncExitStack()
    canonical_paths = sorted({str(Path(p).expanduser().resolve()) for p in repo_paths if p})
    for path_str in canonical_paths:
        lock = await _repo_lock(path_str)
        await stack.enter_async_context(lock)
    return stack


async def _consume_jsonl(reader: asyncio.StreamReader) -> tuple[list[dict[str, Any]], int]:
    """Consume bounded JSONL without retaining raw output or evidence."""
    events: list[dict[str, Any]] = []
    total = 0
    pending = b""
    while True:
        chunk = await reader.read(8192)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_STDOUT_BYTES:
            raise _ProtocolError("stdout_limit")
        pending += chunk
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue
            if len(line) > MAX_JSON_LINE_BYTES:
                raise _ProtocolError("json_line_limit")
            try:
                event = json.loads(line.decode("utf-8"))
            except Exception as exc:  # noqa: BLE001 - protocol boundary
                raise _ProtocolError("malformed_jsonl") from exc
            if not isinstance(event, dict):
                raise _ProtocolError("malformed_jsonl")
            events.append(event)
    if pending.strip():
        if len(pending) > MAX_JSON_LINE_BYTES:
            raise _ProtocolError("json_line_limit")
        try:
            event = json.loads(pending.decode("utf-8"))
            if isinstance(event, dict):
                events.append(event)
        except Exception as exc:  # noqa: BLE001
            raise _ProtocolError("malformed_jsonl") from exc
    return events, total


async def _consume_stderr(reader: asyncio.StreamReader) -> tuple[int, str]:
    """Drain stderr to prevent pipe blockage, returning (byte_count, sample_text)."""
    sample = b""
    total = 0
    while True:
        chunk = await reader.read(4096)
        if not chunk:
            break
        total += len(chunk)
        if len(sample) < MAX_STDERR_BYTES:
            sample += chunk[:MAX_STDERR_BYTES - len(sample)]
    return total, sample.decode("utf-8", errors="replace").strip()


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    """Terminate the OpenCode process group, escalating to SIGKILL."""
    if process.returncode is not None:
        return
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        else:
            process.terminate()
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("[opencode] SIGTERM delivery failed: %s", exc)
    try:
        await asyncio.wait_for(process.wait(), timeout=3.0)
    except asyncio.TimeoutError:
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except (ProcessLookupError, OSError):
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=2.0)
        except asyncio.TimeoutError:
            pass


# Dict-derived error classes are embedded verbatim in Discord-bound failure
# text (finding P1-2), so each candidate must be a bounded enum-like
# identifier: no whitespace, newlines, control characters, or path/URL syntax.
_ERROR_CLASS_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _error_class(event: dict[str, Any]) -> str:
    """Classify an error event without returning its potentially sensitive body.

    Dict payloads contribute only their short enum-like ``name``/``type``:
    a candidate is used only when it is a string fully matching
    ``[A-Za-z0-9_.-]{1,64}``; otherwise the next candidate is tried and, when
    none validates, the payload collapses to ``opencode_error``.
    Plain-string payloads are unbounded harness diagnostics: the only signal
    kept is the ``session`` substring (``unknown_session``); everything else
    collapses to the bounded constant ``opencode_error`` so no raw harness
    text can reach a reply.
    """
    err = event.get("error") or event.get("data")
    if isinstance(err, dict):
        for candidate in (err.get("name"), err.get("type")):
            if isinstance(candidate, str) and _ERROR_CLASS_PATTERN.fullmatch(candidate):
                return candidate
        return "opencode_error"
    if isinstance(err, str):
        if "session" in err.lower():
            return "unknown_session"
        return "opencode_error"
    return "opencode_error"


# Pinned OpenCode ``run --format json`` event contract (verified against a
# controlled local v1.18.x run and the public part schema):
#
#   {"type": "step_start",  "timestamp": int, "sessionID": "ses_...",
#    "part": {"id": "prt_...", "messageID": "msg_...", "type": "step-start"}}
#   {"type": "text",        ...
#    "part": {"id": "prt_...", "messageID": "msg_...", "type": "text",
#             "text": str, "time": {"start": int, "end": int | missing},
#             "synthetic"?: bool, "ignored"?: bool}}
#   {"type": "tool_use",    ...
#    "part": {"id": "prt_...", "messageID": "msg_...", "type": "tool",
#             "tool": str, "callID": str,
#             "state": {"status": "pending" | "running" | "completed" | "error",
#                       "input": {...}, "output": str}}}
#   {"type": "step_finish", ...
#    "part": {"id": "prt_...", "messageID": "msg_...", "type": "step-finish",
#             "reason": str, "tokens": {...}, "cost": number}}
#   {"type": "error", "timestamp": int, "sessionID": "ses_...",
#    "error": {"name": str, "data": {...}}}          (no ``part``)
#
# ``reason`` is a free-form string; observed values are ``tool-calls`` for
# steps that ended to run tools (their text is narration) and ``stop`` for the
# successful terminal answer. Non-successful terminal endings (``length``,
# ``content-filter``, ``error``, ...) are possible per the public schema. A
# resumed ``--session`` stream contains only the new turn, so each run yields
# at most one ``stop``-completed assistant message.
_KNOWN_EVENT_TYPES = frozenset({
    "step_start", "step_finish", "text", "tool_use", "reasoning", "error",
})
_ANSWER_COMPLETED_REASON = "stop"
_TOOL_CALLS_REASON = "tool-calls"


def _select_final_text(
    events: list[dict[str, Any]],
) -> tuple[str, str | None, str | None, int]:
    """Select exactly the final completed assistant message from parsed events.

    Supported event state machine (see the pinned contract above):
    ``step_start(msg)`` opens an assistant message; ``reasoning``,
    ``tool_use``, and ``text(msg)`` parts belong to the open message; and
    ``step_finish(msg, reason)`` closes it. Once a message is closed, any
    further text part for the same ``messageID`` is a protocol violation and
    fails closed with ``protocol_malformed_part``; a new ``messageID`` after
    a finish is a legitimate superseding message. Text parts are grouped by
    their ``messageID`` and kept in stream arrival order. A message is a
    final-answer candidate only when its ``step_finish`` reason is ``stop``
    and it owns at least one visible (non-``synthetic``, non-``ignored``)
    text part.
    Reasoning parts, tool input/output, tool-loop narration, and superseded or
    incomplete messages are discarded. Exactly one candidate is returned;
    zero, multiple ambiguous, or structurally invalid streams yield a bounded
    ``protocol_*`` error class instead of any text (fail closed).

    Args:
        events: Parsed JSONL events in stream order.

    Returns:
        Tuple of (final text or "", session id, bounded error class or None,
        count of visible text parts seen). The error class is set both for
        in-stream ``error`` events and for protocol violations; no retained or
        discarded text content is ever included.
    """
    session_id: str | None = None
    error_class: str | None = None
    text_parts = 0
    # messageID -> {"by_id": {part id: text}, "order": [part ids],
    #               "reason": str, "finished": bool}
    messages: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    def fail(protocol_class: str) -> tuple[str, str | None, str | None, int]:
        return "", session_id, error_class or protocol_class, text_parts

    for event in events:
        session_id = event.get("sessionID") or session_id
        event_type = event.get("type")
        if event_type not in _KNOWN_EVENT_TYPES:
            return fail("protocol_unknown_event")
        if event_type == "error":
            error_class = error_class or _error_class(event)
            continue
        part = event.get("part")
        if not isinstance(part, dict):
            return fail("protocol_malformed_part")
        message_id = part.get("messageID")
        if not isinstance(message_id, str) or not message_id:
            return fail("protocol_missing_message_id")
        message = messages.get(message_id)
        if message is None:
            message = {"by_id": {}, "order": [], "reason": None, "finished": False}
            messages[message_id] = message
            order.append(message_id)
        if event_type == "step_finish":
            reason = part.get("reason")
            if not isinstance(reason, str) or not reason:
                return fail("protocol_malformed_part")
            message["reason"] = reason
            message["finished"] = True
        elif event_type == "text":
            if message["finished"]:
                # The message was closed by its step_finish; any further text
                # for the same messageID is a malformed/hacked stream and must
                # never contribute partial text to a reply (fail closed).
                return fail("protocol_malformed_part")
            if part.get("synthetic") or part.get("ignored"):
                continue
            text = part.get("text")
            part_id = part.get("id")
            if not isinstance(text, str) or not isinstance(part_id, str) or not part_id:
                return fail("protocol_malformed_part")
            existing = message["by_id"].get(part_id)
            if existing is not None:
                if existing != text:
                    return fail("protocol_conflicting_part")
                continue
            message["by_id"][part_id] = text
            message["order"].append(part_id)
            text_parts += 1
        # step_start, reasoning, and tool_use parts only advance lifecycle.

    candidates = [
        message_id for message_id in order
        if messages[message_id]["reason"] == _ANSWER_COMPLETED_REASON
        and messages[message_id]["order"]
    ]
    if len(candidates) > 1:
        return fail("protocol_ambiguous_final")
    if len(candidates) == 1:
        final = messages[candidates[0]]
        text = "".join(final["by_id"][pid] for pid in final["order"]).strip()
        return text, session_id, error_class, text_parts
    for message_id in order:
        reason = messages[message_id]["reason"]
        if reason not in (None, _ANSWER_COMPLETED_REASON, _TOOL_CALLS_REASON):
            return fail("protocol_incomplete_final")
    return fail("protocol_no_final_message")

def _extract_marked_answer(text: str) -> tuple[str, str | None]:
    """Return text after the last explicit final-output marker."""
    if FINAL_OUTPUT_MARKER not in text:
        return "", "protocol_missing_final_marker"
    answer = text.rsplit(FINAL_OUTPUT_MARKER, 1)[1].strip()
    if not answer:
        return "", "protocol_empty_final_answer"
    return answer, None


def _failure(sha: str, error_class: str) -> Reply:
    """Build the honest failure reply without exposing subprocess diagnostics."""
    return Reply(
        text=f"I couldn't verify this against the code ({error_class}) and I'm not going to guess.",
        sha=sha,
        ok=False,
        session_id=None,
        error_class=error_class,
    )


# Review-reference failures the user can fix by naming the repository or the
# full MR/PR URL; the reply must say so instead of a bare class.
_REVIEW_HINT_CLASSES = frozenset({"review_ref_ambiguous", "review_ref_unavailable"})


def _review_failure(error_class: str, sha: str = "none") -> Reply:
    """Build an honest merge-request failure reply without fetch diagnostics.

    Never falls back to auditing the watched checkout, never includes remote
    URLs, stderr, or snapshot paths, and never carries a resumable session.
    """
    if error_class in _REVIEW_HINT_CLASSES:
        text = (
            f"I couldn't resolve that merge request ({error_class}) and I won't guess. "
            "Name the repository or include the full MR/PR URL so I review the right one."
        )
    else:
        text = (
            f"I couldn't verify that merge request ({error_class}) "
            "and I'm not going to guess."
        )
    return Reply(text=text, sha=sha, ok=False, session_id=None, error_class=error_class)


def _review_label(provider: str, number: int) -> str:
    """Return the provider-conventional review label (``MR !188`` / ``PR #188``)."""
    if provider == PROVIDER_GITHUB:
        return f"PR #{number}"
    if provider == PROVIDER_GITLAB:
        return f"MR !{number}"
    return f"review {number}"


def _review_prompt_section(snapshot: ReviewSnapshot) -> str:
    """Render the bounded review context section of the user prompt.

    Contains only validated provider/number metadata, verified full SHAs,
    directory roles, and the bounded changed-path manifest already collected
    by the trusted controller. No remote URLs, credentials, or patch bodies.
    """
    label = _review_label(snapshot.provider, snapshot.number)
    lines = [
        f"  {entry.status} {entry.path}"
        + (f" -> {entry.new_path}" if entry.new_path else "")
        for entry in snapshot.changed_paths
    ]
    manifest = "\n".join(lines) if lines else "  (no changed paths)"
    if snapshot.manifest_truncated:
        manifest += "\n  (changed-path list truncated; additional paths changed)"
    return (
        f"MERGE REQUEST REVIEW: {label}\n"
        f"VERIFIED BASE SHA: {snapshot.base_sha}\n"
        f"VERIFIED HEAD SHA: {snapshot.head_sha}\n"
        f"BASE DIRECTORY (verified merge target, before state): {snapshot.base_dir}\n"
        f"HEAD DIRECTORY (proposed changes, after state): {snapshot.head_dir}\n"
        f"CHANGED PATHS (base -> head, status path):\n{manifest}\n"
        "REVIEW INSTRUCTION: Audit the proposed head changes against the verified "
        "base. Cite concrete file paths from both directories and separate "
        "verified findings from unknowns."
    )


TRUNCATION_MARKER = "\n\n[reply truncated]"


def _truncate_cut(text: str, limit: int) -> int:
    """Return the best cut index at or below ``limit``.

    Boundary preference: paragraph (``\\n\\n``), then sentence (``.``/``!``/
    ``?`` followed by whitespace), then any whitespace; ``limit`` itself only
    when the text contains no boundary at all.

    Args:
        text: Final assistant text that needs truncation.
        limit: Maximum allowed cut index (and the char-level fallback).

    Returns:
        Cut index in ``[0, limit]``.
    """
    paragraph = text.rfind("\n\n", 0, limit)
    if paragraph != -1:
        return paragraph
    for index in range(min(limit - 1, len(text) - 2), -1, -1):
        if text[index] in ".!?" and text[index + 1].isspace():
            return index + 1
    for index in range(min(limit, len(text)) - 1, -1, -1):
        if text[index].isspace():
            return index
    return limit


def _bounded_text(text: str, sha: str, cap: int) -> str:
    """Apply the reply cap while preserving the provenance footer.

    ``cap`` covers the whole reply including the footer. When truncation is
    necessary the text is cut at the cleanest boundary that fits, and an
    explicit truncation marker is inserted before the footer so provenance
    always stays intact and mid-word cuts only happen when no boundary exists.

    Args:
        text: Selected final assistant text.
        sha: Audited repository SHA for the footer.
        cap: Total reply character cap.

    Returns:
        Bounded reply text with footer, within ``cap`` when possible. Only
        when ``cap`` itself is smaller than the footer — unreachable via
        validated config (``max_reply_chars >= 200``) — is the footer
        truncated to ``cap`` as a documented last resort.
    """
    footer = f"\n\n-# audited at {sha}"
    if len(text) + len(footer) <= cap:
        return f"{text}{footer}".strip()
    body_cap = cap - len(footer) - len(TRUNCATION_MARKER)
    if body_cap <= 0:
        # Degenerate cap: drop the body first and keep the provenance footer
        # intact whenever the cap can hold it; only a cap below the footer
        # itself (unreachable via validated config) may truncate the footer.
        bounded = footer.strip()
        if cap < len(bounded):
            bounded = bounded[:cap]
        return bounded
    cut = _truncate_cut(text, body_cap)
    return f"{text[:cut].rstrip()}{TRUNCATION_MARKER}{footer}".strip()


class OpenCodeRunner:
    """Run the configured native OI OpenCode agent for one conversation."""

    def __init__(self, cfg: Config, store: Store | None = None) -> None:
        """Create a runner using the same config and isolated XDG resolver."""
        self._cfg = cfg
        self._store = store

    async def run(
        self,
        target: WatchTarget,
        author: str,
        question: str,
        excerpt: str,
        session_id: str | None = None,
        member_id: str | None = None,
    ) -> Reply:
        """Refresh, snapshot, execute, and provenance-stamp one or more repository audits.

        When the latest question carries an explicit MR/PR reference, the run
        becomes a review of controller-fetched base/head snapshots instead of
        the watched checkout. Unresolvable or ambiguous references fail
        honestly; they never fall back to auditing the current branch.
        Args:
            target: Watched repository/guild binding.
            author: Discord display name.
            question: Latest user question.
            excerpt: Bounded Discord conversation excerpt.
            session_id: Optional OpenCode session to resume (ignored for
                MR/PR reviews, which never share branch-audit context).
            member_id: Optional author member id; when known, memory rendering
                is restricted to current participants (plan Phase 3.6).

        Returns:
            Bounded Reply. OpenCode/process failures have ``ok=False``.
        """
        repo_paths = target.repo_paths
        primary_root = Path(target.primary_repo).expanduser().resolve()
        additional_roots = [
            Path(p).expanduser().resolve()
            for p in repo_paths
            if str(Path(p).expanduser().resolve()) != str(primary_root)
        ]

        async with await _acquire_repo_locks(repo_paths):
            try:
                # Network-probing resolution runs off the event loop; it only
                # probes configured repositories for fixed provider refs.
                review_target = await asyncio.to_thread(
                    resolve_review_target, question, repo_paths
                )
            except ReviewRefError as exc:
                logger.warning(
                    "[opencode] review refused class=%s", exc.failure_class
                )
                return _review_failure(exc.failure_class)
            if review_target is not None:
                return await self._run_review_locked(
                    review_target, author, question, excerpt
                )
            return await self._run_locked(
                primary_root,
                additional_roots,
                target,
                author,
                question,
                excerpt,
                session_id,
                member_id,
            )
    async def run_raw_prompt(
        self,
        prompt: str,
        *,
        system_prompt: str = "",
        timeout: int = 60,
    ) -> str | None:
        """Execute a raw model query without repository tool access.

        Raises:
            ValueError: When the prompt exceeds the bounded stdin limit; no
                subprocess is spawned in that case (finding B10).
        """
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            logger.warning(
                "[opencode] raw prompt rejected class=prompt_too_large bytes=%d",
                len(prompt_bytes),
            )
            raise ValueError("prompt exceeds the bounded stdin limit")
        paths = resolve_opencode_paths()
        paths.ensure()
        try:
            binary = resolve_binary(self._cfg.opencode_binary)
        except Exception:  # noqa: BLE001 - config/bootstrap boundary
            return None

        agent_def = {
            "agent": {
                "oi_eval": {
                    "description": "Evaluation agent",
                    "mode": "primary",
                    "model": self._cfg.opencode_model,
                    "prompt": system_prompt or "You are a helpful JSON analysis agent. Output valid JSON.",
                    "steps": 1,
                    "permission": {"*": "deny"},
                }
            }
        }
        config_content = json.dumps(agent_def, separators=(",", ":"))
        argv = [binary, "run", "--format", "json", "--model", self._cfg.opencode_model, "--agent", "oi_eval"]
        environment = paths.environment()
        environment["OPENCODE_CONFIG_CONTENT"] = config_content

        process = None
        stdout_task = None
        stderr_task = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                start_new_session=(os.name == "posix"),
            )
            assert process.stdin is not None
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            assert process.stdout is not None and process.stderr is not None
            stdout_task = asyncio.create_task(_consume_jsonl(process.stdout))
            stderr_task = asyncio.create_task(_consume_stderr(process.stderr))
            await asyncio.wait_for(
                asyncio.gather(process.wait(), stdout_task, stderr_task),
                timeout=timeout,
            )
            events, _ = stdout_task.result()
            text, _, selection_error, _ = _select_final_text(events)
            if selection_error is None and text:
                return text
            return None
        except asyncio.CancelledError:
            # Cancellation must not orphan the start_new_session process group.
            if process is not None:
                await _terminate_process(process)
            raise
        except Exception as exc:  # noqa: BLE001 - subprocess/model boundary
            logger.warning(
                "[opencode] run_raw_prompt failed: %s", type(exc).__name__
            )
            if process is not None:
                await _terminate_process(process)
            return None
        finally:
            for task in (stdout_task, stderr_task):
                if task is not None and not task.done():
                    task.cancel()


    async def _run_locked(
        self,
        root: Path,
        additional_roots: list[Path],
        target: WatchTarget,
        author: str,
        question: str,
        excerpt: str,
        session_id: str | None,
        member_id: str | None = None,
    ) -> Reply:
        """Execute one audit while holding repository serialization locks."""
        all_roots = [root, *additional_roots]
        freshness: list[str] = []
        repo_states: list[dict[str, Any]] = []

        for r in all_roots:
            pulled = await asyncio.to_thread(git_pull, str(r))
            r_sha = audit_sha(r)
            fp_before = worktree_fingerprint(r)
            repo_states.append({
                "root": r,
                "name": r.name,
                "sha_before": r_sha,
                "fp_before": fp_before,
                "pulled": pulled,
            })
            if not pulled:
                freshness.append(f"git pull failed for {r.name} — code may be behind remote")
            if r_sha.endswith("-dirty"):
                freshness.append(f"worktree {r.name} has uncommitted changes")

        initial_shas = [
            f"{st['name']}@{st['sha_before']}" if len(all_roots) > 1 else st["sha_before"]
            for st in repo_states
        ]
        combined_sha_prompt = ", ".join(initial_shas)

        freshness_warning = (
            f"FRESHNESS WARNING: {'; '.join(freshness)}.\n" if freshness else ""
        )
        prompt = (
            f"From: {author}\n\n"
            f"AUDITED SHA: {combined_sha_prompt}\n"
            f"{freshness_warning}"
            f"THREAD CONTEXT:\n{excerpt or '(start of conversation)'}\n\n"
            f"LATEST QUESTION:\n{question}"
        )

        try:
            binary = resolve_binary(self._cfg.opencode_binary)
        except Exception:  # noqa: BLE001 - config/bootstrap boundary
            return _failure(combined_sha_prompt, "binary_unavailable")
        # Bounded stdin (finding B10): fail closed before spawning anything.
        prompt_bytes = prompt.encode("utf-8")
        if len(prompt_bytes) > MAX_PROMPT_BYTES:
            logger.warning(
                "[opencode] prompt rejected class=prompt_too_large bytes=%d",
                len(prompt_bytes),
            )
            return _failure(combined_sha_prompt, "prompt_too_large")
        snapshot = None
        scope_id = scope_id_for_target(target)
        if self._store is not None and scope_id:
            try:
                snapshot = self._store.memory_snapshot(
                    "discord",
                    scope_id,
                    participant_member_ids=(
                        [str(member_id)] if member_id else None
                    ),
                    max_profiles=10,
                )
            except Exception as exc:  # noqa: BLE001 - memory is best-effort
                logger.debug(
                    "[runner] could not fetch memory snapshot: %s",
                    type(exc).__name__,
                )

        prompt_str = build_prompt(
            self._cfg.personality,
            primary_repo=str(root),
            additional_repos=[str(r) for r in additional_roots],
            environment_mode=self._cfg.resolved_environment_mode,
            memory=snapshot,
        )

        events, stderr_bytes, stderr_sample, returncode, error_class, duration_ms = (
            await self._execute_agent(
                binary=binary,
                root=root,
                additional_roots=additional_roots,
                prompt_str=prompt_str,
                prompt=prompt,
                session_id=session_id,
            )
        )
        text, returned_session, event_error, text_parts = _select_final_text(events)
        error_class = error_class or event_error
        if error_class is None:
            text, marker_error = _extract_marked_answer(text)
            error_class = marker_error
        if error_class is None and not text:
            error_class = "empty_response"

        final_sha_list = []
        for st in repo_states:
            r = st["root"]
            r_name = st["name"]
            r_sha = st["sha_before"]
            warning = mutation_warning(r, st["fp_before"])
            if warning:
                if not r_sha.endswith("-dirty"):
                    r_sha = f"{r_sha}-dirty"
                logger.warning("[opencode] %s in %s", warning, r_name)
            final_sha_list.append(f"{r_name}@{r_sha}" if len(all_roots) > 1 else r_sha)

        final_combined_sha = ", ".join(final_sha_list)

        logger.info(
            "[opencode] events=%d text_parts=%d stderr_bytes=%d duration_ms=%d "
            "exit=%s model=%s session=%s error=%s",
            len(events), text_parts, stderr_bytes, duration_ms,
            returncode if returncode is not None else "none",
            self._cfg.opencode_model,
            (returned_session or "")[:8] or "none",
            error_class or "none",
        )
        if error_class:
            return _failure(final_combined_sha, error_class)
        return Reply(
            text=_bounded_text(text, final_combined_sha, self._cfg.max_reply_chars),
            sha=final_combined_sha,
            ok=True,
            session_id=returned_session,
        )

    async def _execute_agent(
        self,
        *,
        binary: str,
        root: Path,
        additional_roots: list[Path],
        prompt_str: str,
        prompt: str,
        session_id: str | None,
    ) -> tuple[list[dict[str, Any]], int, str, int | None, str | None, int]:
        """Spawn one bounded OpenCode audit subprocess and collect telemetry.

        The agent config grants only the existing fail-closed read policy for
        ``root`` plus explicitly authorized ``additional_roots``; all other
        tools remain denied (policy.py is the single source of that map).

        Args:
            binary: Resolved OpenCode executable path.
            root: Primary workspace passed as ``--dir``.
            additional_roots: Auxiliary roots authorized read-only.
            prompt_str: Embedded system prompt.
            prompt: Bounded user prompt written to stdin.
            session_id: Optional OpenCode session to resume.

        Returns:
            ``(events, stderr_bytes, stderr_sample, returncode, error_class,
            duration_ms)``. ``error_class`` covers spawn, timeout, protocol,
            and exit failures; final-text selection stays with the caller.
        """
        paths = resolve_opencode_paths()
        paths.ensure()
        config_content = json.dumps(
            build_agent_config(
                self._cfg.opencode_model,
                self._cfg.opencode_steps,
                root,
                prompt_str,
                additional_repo_roots=additional_roots,
            ),
            separators=(",", ":"),
        )
        argv = [binary, "run", "--format", "json", "--model",
                self._cfg.opencode_model, "--agent", "oi", "--dir", str(root.resolve())]
        if session_id:
            argv.extend(["--session", session_id])
        environment = paths.environment()
        environment["OPENCODE_CONFIG_CONTENT"] = config_content
        started = time.monotonic()
        process: asyncio.subprocess.Process | None = None
        stdout_task: asyncio.Task | None = None
        stderr_task: asyncio.Task | None = None
        events: list[dict[str, Any]] = []
        stderr_bytes = 0
        stderr_sample = ""
        error_class: str | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                start_new_session=(os.name == "posix"),
            )
            assert process.stdin is not None
            process.stdin.write(prompt.encode("utf-8"))
            await process.stdin.drain()
            process.stdin.close()
            assert process.stdout is not None and process.stderr is not None
            stdout_task = asyncio.create_task(_consume_jsonl(process.stdout))
            stderr_task = asyncio.create_task(_consume_stderr(process.stderr))
            await asyncio.wait_for(
                asyncio.gather(process.wait(), stdout_task, stderr_task),
                timeout=self._cfg.opencode_timeout_seconds,
            )
            events, _ = stdout_task.result()
            stderr_bytes, stderr_sample = stderr_task.result()
        except asyncio.TimeoutError:
            error_class = "timeout"
            if process is not None:
                await _terminate_process(process)
        except asyncio.CancelledError:
            if process is not None:
                await _terminate_process(process)
            for task in (stdout_task, stderr_task):
                if task is not None and not task.done():
                    task.cancel()
            raise
        except _ProtocolError as exc:
            error_class = exc.error_class
            if process is not None:
                await _terminate_process(process)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            logger.warning("[opencode] process boundary failed: %s", type(exc).__name__)
            if process is not None:
                await _terminate_process(process)
        finally:
            for task in (stdout_task, stderr_task):
                if task is not None and not task.done():
                    task.cancel()
        if process is None:
            error_class = error_class or "process_error"
        elif process.returncode not in (0, None):
            stderr_lower = stderr_sample.lower()
            if "session" in stderr_lower and any(
                w in stderr_lower for w in ("not found", "unknown", "deleted", "invalid")
            ):
                error_class = "unknown_session"
            else:
                error_class = error_class or "nonzero_exit"
        if not events:
            error_class = error_class or "empty_response"
        duration_ms = int((time.monotonic() - started) * 1000)
        returncode = process.returncode if process is not None else None
        return events, stderr_bytes, stderr_sample, returncode, error_class, duration_ms

    async def _run_review_locked(
        self,
        review_target: ReviewTarget,
        author: str,
        question: str,
        excerpt: str,
    ) -> Reply:
        """Execute one MR/PR audit against controller-owned base/head snapshots.

        The watched checkout is neither refreshed nor audited: the review
        subject is the immutable snapshot pair the trusted controller fetched
        read-only from the configured origin. Only the head snapshot (primary
        ``--dir``) and the base snapshot (explicit auxiliary root) are
        authorized; other configured repositories are excluded from review
        sessions. No OpenCode session is resumed and none is returned, so
        review runs cannot reuse or store conversation branch-audit context.

        Args:
            review_target: Reference resolved against configured repositories.
            author: Discord display name.
            question: Latest user question.
            excerpt: Bounded Discord conversation excerpt.

        Returns:
            Bounded Reply whose provenance is the MR form
            ``MR !188 base <sha> head <sha>`` (or the GitHub ``PR #188`` form).
        """
        provider = review_target.provider
        number = review_target.number
        try:
            # Controller-side fetch/archive/diff is synchronous networked Git;
            # run it off the event loop while holding the repository locks.
            review = await asyncio.to_thread(
                materialized_review, review_target.repo_path, provider, number
            )
        except ReviewSnapshotError as exc:
            logger.warning(
                "[opencode] review provider=%s number=%d class=%s",
                provider, number, exc.failure_class,
            )
            return _review_failure(exc.failure_class)
        try:
            snapshot = review.snapshot
            provenance = (
                f"{_review_label(provider, number)} "
                f"base {snapshot.base_sha} head {snapshot.head_sha}"
            )
            prompt = (
                f"From: {author}\n\n"
                f"{_review_prompt_section(snapshot)}\n"
                f"AUDITED SHA: {provenance}\n"
                f"THREAD CONTEXT:\n{excerpt or '(start of conversation)'}\n\n"
                f"LATEST QUESTION:\n{question}"
            )
            try:
                binary = resolve_binary(self._cfg.opencode_binary)
            except Exception:  # noqa: BLE001 - config/bootstrap boundary
                return _review_failure("binary_unavailable", sha=provenance)
            # Bounded stdin: fail closed before spawning anything.
            prompt_bytes = prompt.encode("utf-8")
            if len(prompt_bytes) > MAX_PROMPT_BYTES:
                logger.warning(
                    "[opencode] review prompt rejected class=prompt_too_large bytes=%d",
                    len(prompt_bytes),
                )
                return _review_failure("prompt_too_large", sha=provenance)
            # Memory is intentionally omitted: reviews are isolated one-shot
            # audits and the snapshot section must keep the prompt bounded.
            prompt_str = build_prompt(
                self._cfg.personality,
                primary_repo=str(snapshot.head_dir),
                additional_repos=[str(snapshot.base_dir)],
                environment_mode=self._cfg.resolved_environment_mode,
            )
            events, stderr_bytes, stderr_sample, returncode, error_class, duration_ms = (
                await self._execute_agent(
                    binary=binary,
                    root=snapshot.head_dir,
                    additional_roots=[snapshot.base_dir],
                    prompt_str=prompt_str,
                    prompt=prompt,
                    session_id=None,
                )
            )
            text, _, event_error, text_parts = _select_final_text(events)
            error_class = error_class or event_error
            if error_class is None:
                text, marker_error = _extract_marked_answer(text)
                error_class = marker_error
            if error_class is None and not text:
                error_class = "empty_response"
            logger.info(
                "[opencode] review provider=%s number=%d events=%d text_parts=%d "
                "stderr_bytes=%d duration_ms=%d exit=%s model=%s error=%s",
                provider, number, len(events), text_parts, stderr_bytes,
                duration_ms, returncode if returncode is not None else "none",
                self._cfg.opencode_model, error_class or "none",
            )
            if error_class:
                return _review_failure(error_class, sha=provenance)
            # Provenance flows through the same bounded-text footer so the
            # exact base/head SHAs survive truncation; no session is stored.
            return Reply(
                text=_bounded_text(text, provenance, self._cfg.max_reply_chars),
                sha=provenance,
                ok=True,
                session_id=None,
            )
        finally:
            # Guaranteed cleanup on success, failure, timeout, and cancellation.
            review.close()
