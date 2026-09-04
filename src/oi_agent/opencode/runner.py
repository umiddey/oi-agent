"""Async OpenCode subprocess runner with bounded output and process cleanup."""

from __future__ import annotations

import asyncio
import json
import logging
import os
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
from .prompt import build_prompt
from .provenance import audit_sha, git_pull, mutation_warning, worktree_fingerprint

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


def _error_class(event: dict[str, Any]) -> str:
    """Classify an error event without returning its potentially sensitive body."""
    err = event.get("error") or event.get("data")
    if isinstance(err, dict):
        return str(err.get("name") or err.get("type") or "opencode_error")
    if isinstance(err, str):
        if "session" in err.lower():
            return "unknown_session"
        return err
    return "opencode_error"


def _extract_text(events: list[dict[str, Any]]) -> tuple[str, str | None, str | None, int]:
    """Extract completed text parts and bounded metadata from parsed events."""
    texts: list[str] = []
    session_id: str | None = None
    error_class: str | None = None
    text_parts = 0
    for event in events:
        session_id = event.get("sessionID") or session_id
        event_type = event.get("type")
        if event_type == "text":
            part = (event.get("part") or {}).get("text")
            if isinstance(part, str):
                texts.append(part)
                text_parts += 1
        elif event_type == "error":
            error_class = error_class or _error_class(event)
    return "".join(texts).strip(), session_id, error_class, text_parts


def _failure(sha: str, error_class: str) -> Reply:
    """Build the honest failure reply without exposing subprocess diagnostics."""
    return Reply(
        text=f"I couldn't verify this against the code ({error_class}) and I'm not going to guess.",
        sha=sha,
        ok=False,
        session_id=None,
        error_class=error_class,
    )


def _bounded_text(text: str, sha: str, cap: int) -> str:
    """Apply the reply cap while preserving the provenance footer."""
    footer = f"\n\n-# audited at {sha}"
    if len(text) + len(footer) <= cap:
        return f"{text}{footer}".strip()
    body_cap = max(0, cap - len(footer) - 1)
    return f"{text[:body_cap].rstrip()}{footer}".strip()


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

        Args:
            target: Watched repository/guild binding.
            author: Discord display name.
            question: Latest user question.
            excerpt: Bounded Discord conversation excerpt.
            session_id: Optional OpenCode session to resume.
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
            text, _, _, _ = _extract_text(events)
            return text
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
        paths = resolve_opencode_paths()
        paths.ensure()
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
        text, returned_session, event_error, text_parts = _extract_text(events)
        error_class = error_class or event_error
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

        duration_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "[opencode] events=%d text_parts=%d stderr_bytes=%d duration_ms=%d "
            "exit=%s model=%s session=%s error=%s",
            len(events), text_parts, stderr_bytes, duration_ms,
            process.returncode if process is not None else "none",
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
