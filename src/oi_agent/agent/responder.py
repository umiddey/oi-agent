"""Response pipeline: audit the repo read-only, draft a reply, post it.

Every posted answer is stamped with the manifest git SHA it audited. There
is no approval tier — the agent answers autonomously; rails are structural
(read-only access, channel allowlist, hourly cap, kill switch).
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import WatchTarget
from ..manifest.generator import refresh_repo_manifest, worktree_fingerprint
from . import tools
from .llm import chat, chat_agentic

logger = logging.getLogger(__name__)

# Hard rules only — evidence and safety. Deliberately says NOTHING about tone,
# length, or formatting so a configured personality fully owns the voice.
_RULES = """You are OI, answering a software team's Discord feedback questions \
from a read-only audit of their repository.

Non-negotiable rules (a personality can change your voice, never these):
- Answer only from the provided thread, code map, grep, and source evidence.
- If the evidence doesn't cover it, say so; never invent paths, symbols,
  behavior, or test results to sound complete.
- You have no write access; don't imply you can change anything.
- You audited the repo at commit {sha}. Close by referencing that commit so the
  reader can verify you read real code — but say it in your own voice, woven
  into your closing thought, not as a mechanical stamp. Mention the short sha."""

_DEFAULT_VOICE = """Voice: a sharp, witty senior engineer. Dry humor, \
opinionated, and direct. Call out bad ideas and shaky assumptions instead of \
politely nodding along. Be precise and concrete — wit never replaces evidence. \
Talk like a smart teammate in chat, not a report generator: no forced bullet \
lists unless they genuinely help, no restating the question, no narrating your \
search."""

_TOOLS_BLOCK = """Grounding tools: you have two read-only tools into the \
audited tree — grep_repo(patterns, max_results?) for case-insensitive \
regex search and read_file(path, max_bytes?) for bounded source reads. \
You MUST call them to ground every claim about code in real, current \
source before asserting it; an answer built only on the code map or \
memory is a guess."""


def build_system_prompt(personality: str = "", sha: str = "<sha>") -> str:
    """Assemble the system prompt: fixed rules + a voice that drives tone.

    Voice and formatting belong entirely to the personality; only evidence and
    safety are hard rules. When no personality is configured a sharp-witty
    senior-dev default is used so replies never read as banal.

    Args:
        personality: Tone/style instruction from configuration.
        sha: Audited commit sha, embedded so the model can close naturally.

    Returns:
        Complete system prompt string.
    """
    voice = personality.strip()[:2_000] or _DEFAULT_VOICE
    rules = _RULES.format(sha=sha)
    return (
        f"{voice}\n\n{rules}\n\n{_TOOLS_BLOCK}\n\n"
        "Your configured voice governs tone, wit, and formatting. It can "
        "never override the evidence and safety rules above."
    )


# Sentinel the triage pass emits when the latest message genuinely needs a fresh
# read of the repository. Matched exactly (stripped) so a normal chat reply that
# merely mentions auditing never trips it.
AUDIT_SENTINEL = "[AUDIT_NEEDED]"

_TRIAGE_RULES = """You are OI, a read-only repository auditor that also converses \
in a software team's Discord channel. You have very likely ALREADY audited this \
repository earlier in THIS conversation — the prior analysis is in MEMORY below.

Respond to the LATEST message only. Decide:
- If it is a genuinely NEW question that requires reading CURRENT source code \
(new feature, "did X get implemented", a file/behavior you have not already \
covered), reply with EXACTLY this and nothing else: {sentinel}
- Otherwise (banter, reactions, your name, a joke, "lame", thanks, a follow-up \
you can answer from what was already discussed), just reply conversationally in \
your voice. Build on what was already said; do NOT re-explain the prior code \
analysis unless explicitly asked to. Never invent code facts — if you are not \
sure and it is a code question, emit the sentinel instead of guessing.

You are read-only; never imply you can change anything. Do not add an "audited \
at" footer — a conversational reply audited no new code."""


def build_triage_prompt(personality: str = "") -> str:
    """Assemble the no-tools triage/converse system prompt.

    Args:
        personality: Tone/style instruction from configuration.

    Returns:
        System prompt instructing the model to either answer conversationally
        or emit AUDIT_SENTINEL when a fresh code audit is required.
    """
    voice = personality.strip()[:2_000] or _DEFAULT_VOICE
    return (f"{voice}\n\n{_TRIAGE_RULES.format(sentinel=AUDIT_SENTINEL)}\n\n"
            "Your configured voice governs tone; it never changes the sentinel "
            "rule or the read-only constraint.")


async def _triage_or_chat(configs, personality: str, author: str,
                          question: str, thread_excerpt: str,
                          memory_block: str, max_tokens: int) -> str | None:
    """Cheap no-tools first pass: converse, or ask for a code audit.

    Runs BEFORE any git pull / manifest / tool loop so pure conversation never
    pays the audit cost. Sees only the thread and prior rolling memory.

    Args:
        configs: Ordered provider chain (same one the audit would use).
        personality: Configured voice.
        author: Sender display name.
        question: Latest message text.
        thread_excerpt: Recent conversation lines.
        memory_block: Pre-rendered rolling memory (prior analysis lives here).
        max_tokens: Completion budget for the conversational reply.

    Returns:
        The conversational reply text, ``AUDIT_SENTINEL`` when a fresh audit is
        needed, or None when the provider call failed (caller then audits
        rather than dropping the message).
    """
    user = f"From: {author}\n\n"
    if memory_block:
        user += f"MEMORY (your earlier analysis in this conversation):\n{memory_block}\n\n"
    user += (f"RECENT CONVERSATION:\n{thread_excerpt or '(start)'}\n\n"
             f"LATEST MESSAGE:\n{question}")
    try:
        text, _ = await chat(
            configs, build_triage_prompt(personality), user,
            max_tokens=max_tokens, temperature=0.3,
        )
    except Exception as exc:  # noqa: BLE001 - provider boundary
        logger.warning("[responder] triage failed, will audit: %s", exc)
        return None
    return text.strip() if isinstance(text, str) else None




@dataclass
class Reply:
    """A fully drafted reply ready for posting.

    Attributes:
        text: The reply body (Discord markdown).
        sha: Manifest SHA audited ('none' when unavailable).
        ok: False for honest-failure fallbacks (repo/LLM failure) that are
            posted but must NOT be folded into rolling memory as a real
            exchange. True for genuine answers.
    """

    text: str
    sha: str
    ok: bool = True


def git_pull(repo_path: str) -> bool:
    """Pull the watched clone so audits see fresh code (read-only remote).

    Uses --autostash so pre-existing local modifications (including the
    .gitignore entry OI appends) don't block the pull.

    Args:
        repo_path: Local clone path; remote creds must be read-scoped.

    Returns:
        True on success, False on any failure (audit proceeds stale).
    """
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "-c", "pull.rebase=false",
             "pull", "--ff-only", "--autostash"],
            capture_output=True, text=True, check=False, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("[responder] pull failed: %s", exc)
        return False
    if proc.returncode != 0:
        logger.warning("[responder] pull failed: %s", proc.stderr.strip())
        return False
    return True

MEMORY_EXCHANGES_KEPT = 2
MEMORY_SUMMARY_CAP = 2000

# Salience patterns for deterministic (no-LLM) memory compaction: facts worth
# keeping when an exact exchange falls out of the kept window.
_MEMORY_SALIENT = re.compile(
    r"\b(decis\w*|must|should|blocked|blocker|open question|constraint|"
    r"requires|broken|bug|fails|failure|error|crash|deploy\w*|migration|"
    r"migrate|release|root cause)\b", re.IGNORECASE)
_MEMORY_PATH = re.compile(
    r"(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+"
    r"|\b[A-Za-z0-9_-]+\.(?:py|ts|tsx|js|jsx|md|toml|yml|yaml|json|sql|sh)\b")
_MEMORY_SHA = re.compile(r"\b[0-9a-f]{7,40}(?:-dirty)?\b")
_MEMORY_SYMBOL = re.compile(r"`([^`]{1,80})`")


def _compact_exchange(question: str, answer: str) -> str:
    """Distill one dropped exchange into one bounded salient-fact line.

    Preserves decisions, unresolved questions, constraints, defects,
    referenced paths, backtick-quoted symbols, and audit SHAs. Exchanges with
    no durable fact return an empty string so social chatter is not persisted.

    Args:
        question: The question that was asked.
        answer: The answer that was posted.

    Returns:
        One compact line no longer than 500 characters, or an empty string
        when the exchange contains no salient information.
    """
    question_text = " ".join(question.split())
    parts: list[str] = []
    if "?" in question_text or _MEMORY_SALIENT.search(question_text):
        parts.append("q: " + question_text[:200])

    key_sentences: list[str] = []
    key_length = 0
    for raw_sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        sentence = " ".join(raw_sentence.split())
        if not sentence or not (
                _MEMORY_SALIENT.search(sentence) or "?" in sentence):
            continue
        if len(sentence) > 300:
            sentence = sentence[:150].rstrip() + " … " + sentence[-145:].lstrip()
        proposed_length = key_length + len(sentence) + (
            2 if key_sentences else 0)
        if proposed_length > 400:
            break
        key_sentences.append(sentence)
        key_length = proposed_length
    if key_sentences:
        parts.append("; ".join(key_sentences))

    evidence_text = f"{question_text} {answer}"
    paths = list(dict.fromkeys(_MEMORY_PATH.findall(evidence_text)))[:6]
    if paths:
        parts.append("paths: " + ", ".join(paths))
    symbols = list(dict.fromkeys(_MEMORY_SYMBOL.findall(answer)))[:6]
    if symbols:
        parts.append("symbols: " + ", ".join(symbols))
    shas = list(dict.fromkeys(_MEMORY_SHA.findall(answer)))[:4]
    if shas:
        parts.append("shas: " + ", ".join(shas))
    if not parts:
        return ""

    compacted: list[str] = []
    length = 0
    for part in parts:
        proposed_length = length + len(part) + (3 if compacted else 0)
        if proposed_length > 500:
            continue
        compacted.append(part)
        length = proposed_length
    return " | ".join(compacted)


def render_memory(summary: str, exchanges: list[dict]) -> str:
    """Render the rolling memory into a compact prompt block.

    Args:
        summary: Long-term digest text.
        exchanges: Recent exchange dicts ({q, a, sha, ts}).

    Returns:
        Prompt block string (empty when no memory at all).
    """
    if not summary and not exchanges:
        return ""
    parts: list[str] = []
    if summary:
        parts.append(f"EARLIER DISCUSSIONS DIGEST:\n{summary}")
    if exchanges:
        lines = []
        for e in exchanges:
            lines.append(f"Q: {e.get('q', '')[:200]}\n"
                         f"A: {e.get('a', '')[:400]}")
        parts.append("RECENT EXCHANGES (newest last):\n" + "\n---\n".join(lines))
    return "\n\n".join(parts)


def _evict_into_summary(summary: str, exchanges: list[dict]) -> str:
    """Compact exchanges beyond the kept window into salient summary lines.

    Args:
        summary: Existing digest text (mutated copy returned).
        exchanges: Exchange list; oldest are popped in place until the kept
            window remains.

    Returns:
        The summary with newly-evicted facts appended.
    """
    while len(exchanges) > MEMORY_EXCHANGES_KEPT:
        dropped = exchanges.pop(0)
        fact = _compact_exchange(dropped["q"], dropped["a"])
        if fact:
            summary += "\n- " + fact
    return summary


def _deterministic_trim(summary: str) -> str:
    """Bound an over-cap summary by dropping whole oldest lines, never slicing.

    Args:
        summary: Digest text that may exceed MEMORY_SUMMARY_CAP.

    Returns:
        A digest within MEMORY_SUMMARY_CAP.
    """
    if len(summary) <= MEMORY_SUMMARY_CAP:
        return summary
    # Semantic-unit trimming first: drop WHOLE oldest summary lines (each line
    # is one compacted fact), never slice through a fact.
    marker = "(older memory trimmed)\n"
    lines = summary.splitlines()
    while len(lines) > 1 and (
            len(marker) + sum(len(ln) + 1 for ln in lines) - 1
            > MEMORY_SUMMARY_CAP):
        lines.pop(0)
    summary = marker + "\n".join(lines)
    if len(summary) > MEMORY_SUMMARY_CAP:
        # Fallback for one oversized semantic unit only.
        summary = marker + summary[-MEMORY_SUMMARY_CAP // 2:]
    return summary


def update_memory(store, channel_id: int, question: str,
                  reply_text: str, sha: str) -> None:
    """Fold a finished exchange into the rolling memory (deterministic).

    Keeps the last MEMORY_EXCHANGES_KEPT exchanges verbatim; older ones are
    compacted into salient-fact summary lines (decisions, open questions,
    constraints, paths, symbols, SHAs) so memory stays bounded without
    losing the facts that matter. This is the pure no-LLM path; the async
    ``update_memory_llm`` wraps it with an LLM re-summarization on overflow.

    Args:
        store: State store instance.
        channel_id: Watch target channel id.
        question: Question that was answered.
        reply_text: Reply that was posted.
        sha: Audit SHA stamped on the reply.
    """
    summary, exchanges = store.get_memory(channel_id)
    exchanges.append({"q": question, "a": reply_text, "sha": sha,
                      "ts": int(time.time())})
    summary = _evict_into_summary(summary, exchanges)
    summary = _deterministic_trim(summary)
    store.set_memory(channel_id, summary, exchanges)


_RESUMMARIZE_SYSTEM = (
    "You compress an assistant's rolling memory of a software team's Discord "
    "conversation. You are given the current memory digest, which has grown too "
    "long. Rewrite it TIGHTER while preserving every durable fact: decisions, "
    "open questions, blockers, constraints, file paths, code symbols, and audit "
    "SHAs. Drop social chatter, pleasantries, and any line that carries no fact "
    "(e.g. a bare commit SHA with nothing attached). Merge duplicates. Output "
    "ONLY the rewritten digest as concise '- ' bullet lines, nothing else."
)


async def _llm_resummarize(configs, summary: str,
                           target_chars: int) -> str | None:
    """Re-summarize an over-cap digest with the answering LLM chain.

    Args:
        configs: Ordered provider chain (same one the answer used).
        summary: The current over-cap digest to compress.
        target_chars: Soft length budget passed to the model.

    Returns:
        The rewritten digest bounded to the cap, or None on any provider
        failure or empty output so the caller can fall back deterministically.
    """
    user = (
        f"Rewrite this memory digest to under about {target_chars} characters, "
        f"keeping only durable facts:\n\n{summary}"
    )
    try:
        text, _ = await chat(
            configs, _RESUMMARIZE_SYSTEM, user,
            max_tokens=1024, temperature=0.1,
        )
    except Exception as exc:  # noqa: BLE001 - provider boundary
        logger.warning("[responder] llm resummarize failed: %s", exc)
        return None
    text = text.strip() if isinstance(text, str) else ""
    if not text:
        return None
    return _deterministic_trim(text)


async def update_memory_llm(store, channel_id: int, question: str,
                            reply_text: str, sha: str, configs) -> None:
    """Fold an exchange into memory, using the LLM to compress on overflow.

    Identical to ``update_memory`` except that when the digest exceeds the cap
    the answering model is asked to read and rewrite it (preserving facts,
    dropping chatter). Any LLM failure falls back to the deterministic trim, so
    memory is never lost or left unwritten.

    Args:
        store: State store instance.
        channel_id: Conversation memory key (thread/channel id).
        question: Question that was answered.
        reply_text: Reply that was posted.
        sha: Audit SHA stamped on the reply.
        configs: Ordered provider chain for the compression call.
    """
    summary, exchanges = store.get_memory(channel_id)
    exchanges.append({"q": question, "a": reply_text, "sha": sha,
                      "ts": int(time.time())})
    summary = _evict_into_summary(summary, exchanges)
    if len(summary) > MEMORY_SUMMARY_CAP:
        rewritten = await _llm_resummarize(
            [c for c in configs if getattr(c, "base_url", "")],
            summary, MEMORY_SUMMARY_CAP)
        summary = rewritten if rewritten is not None \
            else _deterministic_trim(summary)
    store.set_memory(channel_id, summary, exchanges)





async def build_context(target: WatchTarget, thread_excerpt: str,
                        memory_block: str = "") -> "_SeedContext":
    """Seed the evidence snapshot off the event loop.

    All the work here is blocking (git pull subprocess, manifest rebuild,
    sqlite). Running it inline on discord.py's loop would stall gateway
    heartbeats and zombie the connection, so it is offloaded to a worker
    thread. Retrieval itself is model-driven: grep_repo/read_file run
    inside chat_agentic's bounded loop, not here.

    Args:
        target: Watch target owning the repo binding.
        thread_excerpt: Recent channel/thread messages for framing.
        memory_block: Pre-rendered rolling memory (may be empty).

    Returns:
        The seeded snapshot; callers must use ``seed.sha`` verbatim for
        provenance and never re-read the manifest.
    """
    return await asyncio.to_thread(
        _build_context_sync, target, thread_excerpt, memory_block)


_repo_locks: dict[str, threading.Lock] = {}
_repo_locks_guard = threading.Lock()


def _repo_lock(repo_path: str) -> threading.Lock:
    """Return the serialization lock for one watched repository.

    Every repository-touching step of an audit (git pull, manifest refresh,
    SHA capture, grep, source reads) must hold this lock so two concurrent
    messages auditing the same clone cannot interleave and produce evidence
    that mixes two snapshots. Different repositories use different locks and
    stay fully parallel.

    Args:
        repo_path: Watched clone path (canonicalized to a stable key).

    Returns:
        The threading.Lock for this repository.
    """
    key = str(Path(repo_path).resolve())
    with _repo_locks_guard:
        lock = _repo_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _repo_locks[key] = lock
        return lock

def _build_context_sync(target: WatchTarget, thread_excerpt: str,
                        memory_block: str = "") -> "_SeedContext":
    """Synchronous snapshot seeding; must run in a worker thread.

    The seed is one critical section per repository: pull -> manifest
    refresh -> SHA capture -> code map all observe a single snapshot.
    Later model-driven tool reads re-acquire the same lock per call. The
    captured SHA is returned WITH the context so callers never re-read
    mutable manifest state after the lock is released.

    Args:
        target: Watch target owning the repo binding.
        thread_excerpt: Recent channel/thread messages for framing.
        memory_block: Pre-rendered rolling memory (may be empty).

    Returns:
        The seeded snapshot (context prefix, audited SHA, worktree
        fingerprint at capture time).
    """
    with _repo_lock(target.repo_path):
        return _seed_context(target, thread_excerpt, memory_block)


@dataclass
class _SeedContext:
    """One consistent pre-loop evidence snapshot.

    Attributes:
        context: Prompt-ready context prefix (sha, freshness, thread,
            memory, code map).
        sha: Manifest SHA captured under the repo lock ('-dirty' included).
        fingerprint: Worktree content fingerprint at capture time; compared
            after the tool loop to detect mid-audit mutations.
    """

    context: str
    sha: str
    fingerprint: str


def _seed_context(target: WatchTarget, thread_excerpt: str,
                  memory_block: str = "") -> _SeedContext:
    """Capture the pre-loop snapshot (caller holds the repo lock).

    Retrieval itself is deliberately NOT done here: the model drives
    grep_repo/read_file through the agentic loop instead. Seeding only fixes
    the audited commit, surfaces freshness caveats, and frames the
    conversation.

    Args:
        target: Watch target owning the repo binding.
        thread_excerpt: Recent channel/thread messages for framing.
        memory_block: Pre-rendered rolling memory (may be empty).

    Returns:
        The seeded snapshot.
    """
    pulled = git_pull(target.repo_path)
    refresh_repo_manifest(Path(target.repo_path))
    sha = tools.audit_sha(target.repo_path)
    # Content fingerprint of the live worktree, captured with the SHA.
    # Compared after the tool loop to detect ANY mutation during the audit —
    # the manifest SHA alone cannot (it only moves on a rebuild).
    fingerprint = worktree_fingerprint(Path(target.repo_path))
    # Surface freshness so the model can caveat instead of silently auditing
    # stale or uncommitted code. `-dirty` in the sha comes from _head_sha.
    freshness = []
    if not pulled:
        freshness.append("git pull failed — code may be behind the remote")
    if sha.endswith("-dirty"):
        freshness.append("worktree has uncommitted changes")

    context = f"AUDITED SHA: {sha}\n"
    if freshness:
        context += ("FRESHNESS WARNING: " + "; ".join(freshness)
                    + ". Note this caveat in your answer.\n")
    context += (
        "\n"
        f"THREAD CONTEXT:\n{thread_excerpt or '(start of conversation)'}\n\n"
    )
    if memory_block:
        context += f"MEMORY:\n{memory_block}\n\n"
    context += f"CODE MAP:\n{tools.manifest_summary(target.repo_path)}\n"
    return _SeedContext(context=context, sha=sha, fingerprint=fingerprint)


# Wording kept stable: tests assert this exact freshness phrase.
MIDAUDIT_MUTATION_WARNING = "repository changed during the audit"


def _mutation_warning(repo_path: str, fingerprint_before: str) -> str:
    """Check whether the worktree changed since the snapshot was seeded.

    Runs AFTER the agentic loop ends (its last tool read is then done), so
    any mutation an external process slipped in mid-audit still taints the
    provenance instead of the reply claiming a clean snapshot.

    Args:
        repo_path: Watched clone path.
        fingerprint_before: Worktree content fingerprint from seed time.

    Returns:
        The freshness warning text on any change, else an empty string.
    """
    if worktree_fingerprint(Path(repo_path)) != fingerprint_before:
        return MIDAUDIT_MUTATION_WARNING
    return ""





async def respond(agent_cfg, fallback_cfg, target: WatchTarget, author: str,
                  question: str, thread_excerpt: str,
                  store=None, personality: str = "",
                  max_response_tokens: int = 4_000,
                  max_tool_iterations: int = 4,
                  memory_key: int | None = None) -> Reply:
    """Seed the audit snapshot and draft a full reply via the tool loop.

    Args:
        agent_cfg: Primary strong-model config.
        fallback_cfg: Fallback provider config.
        target: Watch target with the repo binding.
        author: Sender display name.
        question: Message text to answer.
        thread_excerpt: Recent conversation lines.
        store: Optional state store enabling rolling memory.
        personality: Optional user-configured tone/style.
        max_response_tokens: LLM output-token budget; must be large enough
            that reasoning models can finish hidden reasoning and still
            emit content.
        max_tool_iterations: Resolved agentic tool-loop depth cap (from
            effective_max_tool_iterations at the call site).
        memory_key: Conversation key for rolling memory — the thread id when
            the message is in a thread, else the channel id. Defaults to
            ``target.channel_id`` so each thread keeps memory separate from its
            parent channel and sibling threads. Caps/allowlist stay keyed on
            the parent target elsewhere; only recall is per-conversation.

    Returns:
        Drafted Reply (never None; failures produce an honest error text).
    """
    if memory_key is None:
        memory_key = target.channel_id
    memory_block = ""
    if store is not None:
        try:
            summary, exchanges = store.get_memory(memory_key)
            memory_block = render_memory(summary, exchanges)
        except Exception as exc:  # noqa: BLE001 - state boundary
            logger.warning("[responder] memory unavailable: %s", exc)

    # Conversational triage FIRST (no pull, no tools): the bot has very likely
    # already audited this repo earlier in the thread, so banter/reactions must
    # not re-run the whole audit and repeat the analysis. Only a genuinely new
    # code question falls through to the audit pipeline. A provider failure
    # (None) also falls through, so a message is never silently dropped.
    configs = [c for c in (agent_cfg, fallback_cfg) if c.base_url]
    triage = await _triage_or_chat(
        configs, personality, author, question, thread_excerpt,
        memory_block, max_tokens=max_response_tokens)
    if triage is not None and triage != AUDIT_SENTINEL and triage.strip():
        # Pure conversation: answered from thread + memory, audited no new code.
        return Reply(text=triage, sha="none", ok=True)

    try:
        seed = await build_context(target, thread_excerpt,
                                   memory_block=memory_block)
    except Exception as exc:  # noqa: BLE001 - repo/tool boundary
        logger.exception("[responder] context build failed")
        logger.error("[responder] context build failed: %s", exc)
        # Nothing was audited, so there is no SHA to claim.
        return Reply(
            text="Couldn't inspect the repository just now; "
                 "not going to guess.\n\n-# audited at none",
            sha="none",
            ok=False,
        )
    # sha came from inside the repo lock with the evidence snapshot; it must
    # be the ONLY provenance used from here on (prompt, footer, Reply.sha,
    # memory).
    sha = seed.sha
    user_prompt = f"From: {author}\n\n{thread_excerpt}\n\nQuestion:\n{question}"

    def execute(name: str, arguments: dict) -> str:
        """Run one model-requested tool read against the jailed repo view.

        Acquires the per-repo audit lock so tool reads stay serialized with
        the seed and any other concurrent audit of the same clone.

        Args:
            name: Tool name requested by the model ("grep_repo" or
                "read_file").
            arguments: JSON-style tool arguments from the model.

        Returns:
            Tool output string, or an "ERROR: ..." description on failure.
        """
        with _repo_lock(target.repo_path):
            return tools.execute_tool(target.repo_path, name, arguments)

    try:
        raw, _ = await chat_agentic(
            configs,
            build_system_prompt(personality, sha),
            f"{user_prompt}\n\n---\nCONTEXT:\n{seed.context}",
            tools=tools.TOOL_SCHEMAS_OPENAI,
            execute=execute,
            max_iterations=max(1, int(max_tool_iterations)),
            max_tokens=max_response_tokens,
            temperature=0.2,
        )
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("provider returned non-text content")
    except Exception as exc:  # noqa: BLE001 - provider boundary
        logger.error("[responder] llm failed: %s", exc)
        return Reply(text=f"Couldn't complete the audit just now; "
                          f"not going to guess.\n\n-# audited at {sha}",
                     sha=sha, ok=False)
    # The loop ended, so its last possible tool read is done: compare the
    # live worktree against the seed fingerprint NOW — a mid-audit mutation
    # taints the provenance instead of passing as a clean snapshot.
    warning = await asyncio.to_thread(_mutation_warning, target.repo_path,
                                      seed.fingerprint)
    if warning:
        if not sha.endswith("-dirty"):
            sha = f"{sha}-dirty"
        logger.warning("[responder] %s", warning)
    # Fallback only: if the model never referenced the commit, append a quiet
    # footer so the audit trail is never lost. Match on the short sha, since
    # the model may weave it in without the literal words "audited at".
    short = sha[:9]
    if short not in raw and sha not in raw:
        raw += f"\n\n-# audited at {sha}"
    reply = Reply(text=raw.strip(), sha=sha)
    # Memory is deliberately NOT updated here: the reply has not been accepted
    # by Discord yet. The watcher records it after a confirmed send, so blocked
    # or failed posts are never remembered as completed conversations.
    return reply