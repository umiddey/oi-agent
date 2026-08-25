"""Tests for the response pipeline's seeding, provenance, and agentic loop."""

import subprocess
import threading
from pathlib import Path

import pytest

from oi_agent.agent import responder
from oi_agent.config import WatchTarget
from oi_agent.manifest.generator import manifest_git_sha, refresh_repo_manifest


def _init_repo(tmp_path) -> str:
    """Create a minimal committed git repo fixture and return its path.

    Args:
        tmp_path: Pytest temporary directory.

    Returns:
        Path string of the initialized repository.
    """
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    (repo / "src" / "app.py").write_text("value = 1\n")
    for args in (["init", "-q"], ["config", "user.email", "t@t"],
                 ["config", "user.name", "t"], ["add", "."],
                 ["commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True,
                       capture_output=True)
    return str(repo)


class _Cfg:
    """Minimal LLM config stand-in with a non-empty base_url."""

    base_url = "http://x/v1"


@pytest.mark.asyncio
async def test_mutation_during_audit_is_flagged(tmp_path, monkeypatch):
    """A worktree edit during the tool loop must taint the SHA.

    Reproduces the reviewer's finding: bytes changed after SHA capture
    reached the model while the reply still claimed a clean commit. The
    post-loop fingerprint recheck must catch it and stamp -dirty.
    """
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(responder, "git_pull", lambda _p: True)
    target = WatchTarget(channel_id=1, repo_path=repo)

    # Unit level: seeding captures a fingerprint; an edit afterwards is
    # detected by the post-loop recheck with its stable wording.
    seed = responder._build_context_sync(target, "")
    (Path(repo) / "src" / "app.py").write_text("value = 999\n")
    warning = responder._mutation_warning(repo, seed.fingerprint)
    assert "changed during the audit" in warning

    # E2E: a fake loop that performs one seeded read and mutates mid-audit
    # must produce a -dirty stamped reply.
    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        execute("grep_repo", {"patterns": ["value"]})
        (Path(repo) / "src" / "app.py").write_text("value = 424242\n")
        return "answer", "prov/model"

    monkeypatch.setattr(responder, "chat_agentic", fake_loop)
    reply = await responder.respond(
        _Cfg(), _Cfg(), WatchTarget(channel_id=2, repo_path=repo),
        "alice", "q", "")

    assert reply.sha.endswith("-dirty")
    assert reply.text.endswith(f"-# audited at {reply.sha}")

@pytest.mark.asyncio
async def test_build_context_runs_off_the_event_loop(monkeypatch):
    """Blocking evidence work must be offloaded so the gateway stays alive.

    The sync body records the running thread; it must differ from the loop's
    thread, proving the event loop is never blocked by git/repo/sqlite work.
    """
    loop_thread = threading.get_ident()
    seen = {}

    def fake_sync(target, thread_excerpt, memory_block=""):
        seen["thread"] = threading.get_ident()
        return responder._SeedContext(context="CONTEXT", sha="abc123",
                                      fingerprint="fp")

    monkeypatch.setattr(responder, "_build_context_sync", fake_sync)
    target = WatchTarget(channel_id=1, repo_path="/repo")

    seed = await responder.build_context(target, "excerpt")

    assert (seed.context, seed.sha, seed.fingerprint) == \
        ("CONTEXT", "abc123", "fp")
    assert seen["thread"] != loop_thread


@pytest.mark.asyncio
async def test_reply_provenance_is_snapshot_sha_never_post_lock_reread(
        monkeypatch):
    """respond() must reuse the SHA captured under the repo lock.

    Regression: respond() used to re-read the manifest after build_context
    released the lock, so a concurrent pull between the two reads produced
    evidence stamped with a DIFFERENT commit than it was built from. Also
    guards the chat_agentic seam: tools and iteration cap reach the loop.
    """
    monkeypatch.setattr(responder, "_mutation_warning", lambda *_a: "")

    async def fake_build_context(*args, **kwargs):
        # Snapshot taken at oldsha; by return time a rival audit has already
        # advanced the manifest to newsha. respond() must NOT notice.
        monkeypatch.setattr(responder.tools, "audit_sha",
                            lambda _p: "newsha456")
        return responder._SeedContext("CONTEXT\nAUDITED SHA: oldsha123",
                                      "oldsha123", "fp")

    captured = {}

    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        captured.update(system=system, tools=tools,
                        max_iterations=max_iterations,
                        max_tokens=max_tokens)
        return "plain answer", "provider/model"

    monkeypatch.setattr(responder, "build_context", fake_build_context)
    monkeypatch.setattr(responder, "chat_agentic", fake_loop)

    reply = await responder.respond(
        _Cfg(), _Cfg(), WatchTarget(channel_id=1, repo_path="/repo"),
        "alice", "question", "")

    assert reply.sha == "oldsha123"
    assert "oldsha123" in captured["system"]
    assert "newsha456" not in captured["system"]
    assert reply.text.endswith("-# audited at oldsha123")
    assert "newsha456" not in reply.text
    # The model-driven seam: OPENAI-shaped schemas and the default cap.
    assert captured["tools"] is responder.tools.TOOL_SCHEMAS_OPENAI
    assert captured["max_iterations"] == 4


@pytest.mark.asyncio
async def test_configured_budget_reaches_chat(monkeypatch):
    """respond() forwards budgets to chat_agentic untouched.

    Guards the reasoning-model fix (max_response_tokens -> max_tokens) AND
    the resolved per-watch tool-loop depth (max_tool_iterations ->
    max_iterations).
    """
    captured = {}

    async def fake_build_context(*args, **kwargs):
        return responder._SeedContext("CONTEXT", "abc123", "fp")

    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        captured["max_tokens"] = max_tokens
        captured["max_iterations"] = max_iterations
        return "answer\n\n-# audited at abc123", "provider/model"

    monkeypatch.setattr(responder, "build_context", fake_build_context)
    monkeypatch.setattr(responder, "chat_agentic", fake_loop)

    reply = await responder.respond(
        _Cfg(), _Cfg(), WatchTarget(channel_id=1, repo_path="/repo"),
        "alice", "question", "excerpt",
        max_response_tokens=8_192, max_tool_iterations=3,
    )

    assert captured["max_tokens"] == 8_192
    assert captured["max_iterations"] == 3
    assert "answer" in reply.text


@pytest.mark.asyncio
async def test_iteration_cap_yields_stamped_reply_without_hanging(
        tmp_path, monkeypatch):
    """A cap of one round still terminates with honest provenance.

    The fake loop honors the contract: with max_iterations <= 1 it answers
    immediately from the seeded context instead of requesting more rounds.
    """
    repo = _init_repo(tmp_path)
    monkeypatch.setattr(responder, "git_pull", lambda _p: True)

    seen = {}

    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        seen["max_iterations"] = max_iterations
        if max_iterations <= 1:
            return "final answer from seed only", "provider/model"
        raise AssertionError("loop should have stopped at the cap")

    monkeypatch.setattr(responder, "chat_agentic", fake_loop)
    reply = await responder.respond(
        _Cfg(), _Cfg(), WatchTarget(channel_id=1, repo_path=repo),
        "alice", "what does value do?", "", max_tool_iterations=1)

    assert seen["max_iterations"] == 1
    expected_sha = responder.tools.audit_sha(repo)
    assert reply.sha == expected_sha
    assert f"-# audited at {expected_sha}" in reply.text


@pytest.mark.asyncio
async def test_model_driven_tools_ground_the_answer(tmp_path, monkeypatch):
    """E2E: retrieval is model-driven through the jailed tool executor.

    Round one greps via grep_repo, round two reads via read_file — both run
    through tools.execute_tool against the REAL fixture repo — and the final
    answer carries the evidence plus the correct audited sha.
    """
    repo = Path(_init_repo(tmp_path))
    (repo / "src" / "app.py").write_text(
        "value = 1\n\ndef ground_marker():\n    return 42\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True,
                   capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "marker"],
                   check=True, capture_output=True)
    monkeypatch.setattr(responder, "git_pull", lambda _p: True)

    seen = {}

    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        seen["tools"] = tools
        seen["max_iterations"] = max_iterations
        hits = execute("grep_repo", {"patterns": ["ground_marker"]})
        seen["grep_hit"] = hits
        source = execute("read_file", {"path": "src/app.py"})
        seen["read"] = source
        return f"ground_marker lives in src/app.py:\n{hits.splitlines()[0]}", \
            "provider/model"

    monkeypatch.setattr(responder, "chat_agentic", fake_loop)
    target = WatchTarget(channel_id=1, repo_path=str(repo))

    reply = await responder.respond(_Cfg(), _Cfg(), target, "alice",
                                    "where is ground_marker?", "",
                                    max_tool_iterations=5)

    # The executor ran the REAL jailed tools against the fixture repo.
    assert seen["tools"] == responder.tools.TOOL_SCHEMAS_OPENAI
    assert seen["max_iterations"] == 5
    assert "src/app.py" in seen["grep_hit"]
    assert "ground_marker" in seen["grep_hit"]
    assert "return 42" in seen["read"]
    assert "ground_marker" in reply.text
    expected_sha = responder.tools.audit_sha(str(repo))
    assert f"-# audited at {expected_sha}" in reply.text


@pytest.mark.asyncio
async def test_dirty_provenance_reaches_context_and_reply(tmp_path,
                                                          monkeypatch):
    """E2E criterion: uncommitted code is audited AND labeled -dirty.

    The original regression only checked ManifestResult.git_sha; the parse
    step (`\\w+` regex) stripped `-dirty`, so replies were stamped with the
    clean commit while evidence saw uncommitted bytes. This test walks the
    observable path: real generator -> manifest_git_sha -> build_context ->
    respond().
    """
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)

    def _run_git(*args: str) -> None:
        """Run a git command in the fixture repository."""
        subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, check=True)

    _run_git("init", "-q")
    (repo / "app" / "service.py").write_text("x = 1\n", encoding="utf-8")
    _run_git("add", ".")
    _run_git("commit", "-qm", "init")
    # Uncommitted edit: symbol exists only in the worktree.
    (repo / "app" / "service.py").write_text(
        "x = 1\n\ndef uncommitted_symbol():\n    return 42\n",
        encoding="utf-8")
    # Real generator must rebuild entries from the dirty worktree and keep
    # the -dirty token through every later read of the header.
    result = refresh_repo_manifest(repo)
    assert result.git_sha.endswith("-dirty")
    assert "uncommitted_symbol" in result.manifest_path.read_text()
    audit_sha = manifest_git_sha(repo)
    assert audit_sha.endswith("-dirty")

    monkeypatch.setattr(responder, "git_pull", lambda _p: True)
    captured = {}

    async def fake_loop(configs, system, user, *, tools, execute,
                        max_iterations, max_tokens, temperature):
        captured["system"] = system
        captured["user"] = user
        return "answer text", "provider/model"

    monkeypatch.setattr(responder, "chat_agentic", fake_loop)

    class _Cfg:
        base_url = "http://x/v1"

    target = WatchTarget(channel_id=1, repo_path=str(repo))
    reply = await responder.respond(_Cfg(), _Cfg(), target, "alice",
                                    "what does uncommitted_symbol do?", "")

    context = captured["user"]
    assert f"AUDITED SHA: {audit_sha}" in context
    assert "FRESHNESS WARNING" in context
    assert "uncommitted changes" in context
    # The reply can never be stamped with the bare clean commit.
    clean = audit_sha.removesuffix("-dirty")
    assert reply.sha.endswith("-dirty")
    assert reply.sha == audit_sha
    assert f"audited at {clean}\n" not in reply.text.replace(
        f"audited at {audit_sha}", "")


def test_same_repo_audits_serialize_never_overlap(tmp_path, monkeypatch):
    """Criterion: concurrent audits of ONE clone never interleave snapshots."""
    import time as time_mod

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(responder, "git_pull", lambda _p: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", lambda _p: None)
    monkeypatch.setattr(responder.tools, "manifest_summary", lambda _p: "")

    guard = threading.Lock()
    state = {"inside": 0, "max_inside": 0}

    def slow_audit_sha(root):
        # Stands in for any lock-held repo read during seeding.
        with guard:
            state["inside"] += 1
            state["max_inside"] = max(state["max_inside"], state["inside"])
        time_mod.sleep(0.03)
        with guard:
            state["inside"] -= 1
        return "abc123"

    monkeypatch.setattr(responder.tools, "audit_sha", slow_audit_sha)

    target = WatchTarget(channel_id=1, repo_path=str(repo))
    threads = [
        threading.Thread(
            target=responder._build_context_sync,
            args=(target, f"excerpt {i}"))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert state["max_inside"] == 1


def test_different_repos_audit_in_parallel(tmp_path, monkeypatch):
    """The per-repo lock must not become a global lock across clones."""
    repos = [tmp_path / "a", tmp_path / "b"]
    for r in repos:
        r.mkdir()
    monkeypatch.setattr(responder, "git_pull", lambda _p: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", lambda _p: None)
    monkeypatch.setattr(responder.tools, "manifest_summary", lambda _p: "")

    # Both audits must be able to sit inside their critical section at the
    # same time; if the registry were global this barrier would break.
    barrier = threading.Barrier(2, timeout=5)

    def barrier_audit_sha(root):
        barrier.wait()
        return "abc123"

    monkeypatch.setattr(responder.tools, "audit_sha", barrier_audit_sha)

    results: list[responder._SeedContext] = []

    def run(target):
        results.append(responder._build_context_sync(target, ""))

    threads = [
        threading.Thread(target=run,
                         args=(WatchTarget(channel_id=i, repo_path=str(r)),))
        for i, r in enumerate(repos, 1)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(results) == 2


# --- LLM-driven summary compaction (update_memory_llm) --------------------

from oi_agent.config import LLMConfig  # noqa: E402
from oi_agent.store import Store  # noqa: E402


def _cfgs():
    """Return a minimal agent+fallback provider chain for compaction calls."""
    return [LLMConfig("http://p1", "m1"), LLMConfig("", "")]


@pytest.mark.asyncio
async def test_update_memory_llm_skips_llm_under_cap(tmp_path, monkeypatch):
    """A within-cap summary is written without ever calling the LLM."""
    store = Store(tmp_path / "s.db")
    called = []

    async def spy_chat(*a, **k):
        called.append(a)
        return "SHOULD NOT BE USED", "mock"

    monkeypatch.setattr(responder, "chat", spy_chat)

    await responder.update_memory_llm(
        store, 1, "q1", "short answer", "abc1234", configs=_cfgs())

    assert called == []                      # LLM untouched under cap
    _, exchanges = store.get_memory(1)
    assert len(exchanges) == 1


@pytest.mark.asyncio
async def test_update_memory_llm_rewrites_when_over_cap(tmp_path, monkeypatch):
    """An over-cap summary is replaced by the LLM rewrite and bounded."""
    store = Store(tmp_path / "s.db")
    # Pre-seed an already-huge summary so the next eviction overflows the cap.
    store.set_memory(1, "- old fact\n" * 400, [])
    seen_prompt = {}

    async def fake_chat(configs, system, user, *, max_tokens, temperature):
        seen_prompt["user"] = user
        seen_prompt["system"] = system
        return "- compact durable fact one\n- compact durable fact two", "mock"

    monkeypatch.setattr(responder, "chat", fake_chat)

    # Two writes push the first exchange out of the kept window -> eviction.
    await responder.update_memory_llm(
        store, 1, "q1", "decision: must migrate the schema", "abc1234",
        configs=_cfgs())
    await responder.update_memory_llm(
        store, 1, "q2", "another answer", "def5678", configs=_cfgs())

    summary, _ = store.get_memory(1)
    assert "compact durable fact one" in summary
    assert len(summary) <= responder.MEMORY_SUMMARY_CAP
    # Read-then-update: the model saw the existing over-cap digest.
    assert "old fact" in seen_prompt["user"]


@pytest.mark.asyncio
async def test_update_memory_llm_falls_back_on_llm_failure(
        tmp_path, monkeypatch):
    """When the compaction LLM call raises, deterministic trim still writes."""
    store = Store(tmp_path / "s.db")
    store.set_memory(1, "- old fact\n" * 400, [])

    async def boom_chat(*a, **k):
        raise RuntimeError("provider down")

    monkeypatch.setattr(responder, "chat", boom_chat)

    await responder.update_memory_llm(
        store, 1, "q1", "some answer", "abc1234", configs=_cfgs())
    await responder.update_memory_llm(
        store, 1, "q2", "another", "def5678", configs=_cfgs())

    summary, exchanges = store.get_memory(1)
    # Fell back to the deterministic trim: bounded and marked, never lost.
    assert len(summary) <= responder.MEMORY_SUMMARY_CAP
    assert "older memory trimmed" in summary
    assert len(exchanges) == 2
