"""Discord gateway watcher: durable admission, OpenCode execution, and posting.

Safety invariants:
- OpenCode is the ONLY model harness.
- No storage of source message text, thread evidence, prompts, tool outputs, reasoning, or credentials in SQLite.
- Bounded public-intended generated reply outbox is stored ONLY until Discord delivery
  confirmation, after which body is scrubbed (body = NULL).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import discord

from ..config import Config, WatchTarget
from ..opencode.runner import OpenCodeRunner, scope_id_for_target
from ..poster import Poster, split_message
from ..reflection.engine import ReflectionEngine
from ..store import Store
from .gate import Action, evaluate

logger = logging.getLogger(__name__)

EXCERPT_LIMIT = 40
BURST_QUIET_SECONDS = 8.0
MAX_CONCURRENT_AUDITS_PER_CHANNEL = 2
LEASE_DURATION_SECONDS = 60
LEASE_RENEW_INTERVAL_SECONDS = 20
RETRY_DELAY_SECONDS = 10
MAX_ATTEMPTS = 5
# Documented bound for the in-memory reflection idempotence FIFO (finding B8).
MAX_TRACKED_REFLECTION_DELIVERIES = 500


def _message_evidence_line(message: discord.Message) -> str:
    """Render message text and attachment metadata for model context.

    Attachment URLs and embed descriptions are retained so multimodal
    context reaches the model even when image parsing is disabled.

    Args:
        message (discord.Message): Incoming Discord message.

    Returns:
        str: Context line for the conversation history.
    """
    body = (message.content or "").strip()
    evidence: list[str] = []
    for att in getattr(message, "attachments", []):
        fn = getattr(att, "filename", "unnamed")
        url = getattr(att, "url", "")
        evidence.append(f"[attachment: {fn} ({url})]" if url else f"[attachment: {fn}]")
    for emb in getattr(message, "embeds", []):
        title = getattr(emb, "title", None)
        desc = getattr(emb, "description", None)
        if title and desc:
            evidence.append(f"[embed: {title} - {desc}]")
        elif title or desc:
            evidence.append(f"[embed: {title or desc}]")
    if evidence:
        body = f"{body} {' '.join(evidence)}".strip()
    return (body or "(attachment/embed)")[:600]


async def _thread_excerpt(channel: discord.abc.Messageable, limit: int = EXCERPT_LIMIT) -> str:
    """Fetch bounded conversation excerpt with attachments and embeds.

    Args:
        channel (discord.abc.Messageable): Discord channel or thread.
        limit (int): Maximum messages to retrieve.

    Returns:
        str: Chronological conversation history string.
    """
    lines: list[str] = []
    try:
        if hasattr(channel, "history"):
            history_iter = channel.history(limit=limit, oldest_first=False)
            if hasattr(history_iter, "__aiter__"):
                raw_messages: list[discord.Message] = []
                async for m in history_iter:
                    raw_messages.append(m)
                for msg in reversed(raw_messages):
                    author = getattr(msg.author, "display_name", "unknown")
                    line = _message_evidence_line(msg)
                    lines.append(f"[{msg.id}] {author}: {line}")
            elif asyncio.iscoroutine(history_iter):
                raw_messages = await history_iter
                for msg in reversed(raw_messages):
                    author = getattr(msg.author, "display_name", "unknown")
                    line = _message_evidence_line(msg)
                    lines.append(f"[{msg.id}] {author}: {line}")
        elif hasattr(channel, "_history"):
            raw_messages = list(getattr(channel, "_history", []))[-limit:]
            for msg in raw_messages:
                author = getattr(msg.author, "display_name", "unknown")
                line = _message_evidence_line(msg)
                lines.append(f"[{msg.id}] {author}: {line}")
    except Exception:
        logger.warning(
            "[watcher] failed fetching thread excerpt for %s",
            getattr(channel, "id", "unknown"),
            exc_info=True,
        )
    return "\n".join(lines)


def _is_permanent_error(error_class: str | None, status_code: int | None) -> bool:
    """Classify Discord/OpenCode failures as permanent vs retryable.

    Args:
        error_class (str | None): Machine-readable failure category.
        status_code (int | None): HTTP status code if applicable.

    Returns:
        bool: True for permanent failures, False for transient failures.
    """
    if status_code in (401, 403, 404):
        return True
    if error_class in (
        "channel_not_allowed",
        "message_not_found",
        "missing_chunk_body",
        "forbidden",
        "not_found",
    ):
        return True
    return False


class OIWatcher(discord.Client):
    """Discord gateway client with durable queue, quiet-period debounce, and Poster delivery."""

    def __init__(self, cfg: Config, store: Store, token: str = "") -> None:
        """Initialize watcher with intents, stores, and runners.

        Args:
            cfg (Config): Daemon configuration.
            store (Store): Persistent SQLite store.
            token (str): Bot authorization token.
        """
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True
        super().__init__(intents=intents)
        self._cfg = cfg
        self._store = store
        self._token = token
        self._runner = OpenCodeRunner(cfg, store)
        self._reflection_engine = ReflectionEngine(self._runner, store)
        self._channel_sems: dict[int, asyncio.Semaphore] = {}
        self._conversation_last_event: dict[int, float] = {}
        self._channel_cache: dict[int, discord.abc.Messageable] = {}
        self._worker_task: asyncio.Task | None = None
        self._stopping = False
        self._work_event = asyncio.Event()
        self._active_conversation_tasks: dict[int, asyncio.Task] = {}
        self._reflection_queue: asyncio.Queue[tuple[str, str, str, str, str, str, str]] = asyncio.Queue(maxsize=32)
        self._reflection_worker_task: asyncio.Task | None = None
        # Bounded FIFO of delivery batch ids already enqueued for reflection
        # (process-local idempotence only; never persisted). Oldest entries are
        # evicted deterministically at MAX_TRACKED_REFLECTION_DELIVERIES.
        self._processed_reflection_deliveries: OrderedDict[str, None] = OrderedDict()
        self._reflection_dropped_count: int = 0
    def _channel_semaphore(self, channel_id: int) -> asyncio.Semaphore:
        """Return (creating on first use) the per-channel work semaphore.

        Args:
            channel_id (int): Channel snowflake id.

        Returns:
            asyncio.Semaphore: Bounded concurrency semaphore for the channel.
        """
        sem = self._channel_sems.get(channel_id)
        if sem is None:
            sem = asyncio.Semaphore(MAX_CONCURRENT_AUDITS_PER_CHANNEL)
            self._channel_sems[channel_id] = sem
        return sem

    def _repo_lock(self, repo_path: str | Path) -> asyncio.Lock:
        """Return (creating on first use) the per-repo execution lock.

        Args:
            repo_path (str | Path): Path to the audited repository.

        Returns:
            asyncio.Lock: The lock serializing model runs for that repository.
        """
        key = str(Path(repo_path).expanduser().resolve())
        lock = self._repo_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._repo_locks[key] = lock
        return lock

    def _resolve_target(
        self, channel_id: int, channel: discord.abc.Messageable | None = None
    ) -> tuple[WatchTarget | None, int | None]:
        """Return (WatchTarget, owner id), checking direct channel and guild-wide watches.

        For threads, always resolves against the parent channel for guild ignore/allow
        filters and sets the parent channel as the owner channel id for cap tracking.

        Args:
            channel_id (int): Channel or thread snowflake id.
            channel (discord.abc.Messageable | None): Optional Discord channel object.

        Returns:
            tuple[WatchTarget | None, int | None]: Tuple of (WatchTarget or None, owner channel id or None).
        """
        if channel is not None:
            self._channel_cache[channel_id] = channel
        ch = channel or self._channel_cache.get(channel_id) or self.get_channel(channel_id)

        guild_id: int | None = None
        parent_id: int | None = None
        if ch is not None:
            guild_id = getattr(ch, "guild_id", None)
            if guild_id is None and hasattr(ch, "guild"):
                guild_id = getattr(getattr(ch, "guild", None), "id", None)
            parent_id = getattr(ch, "parent_id", None)

        # 1. Direct channel-level watch on the exact channel/thread ID
        direct = self._cfg.target_for_channel(channel_id)
        if direct and not direct.is_guild_watch:
            owner_id = direct.channel_id if direct.channel_id > 0 else channel_id
            return direct, owner_id

        # 2. If this is a thread, check direct channel-level watch on parent channel
        if parent_id is not None:
            parent_direct = self._cfg.target_for_channel(parent_id)
            if parent_direct and not parent_direct.is_guild_watch:
                owner_id = parent_direct.channel_id if parent_direct.channel_id > 0 else parent_id
                return parent_direct, owner_id

        # 3. Guild-wide watch target matching
        if guild_id is not None:
            # For threads, evaluate the parent channel against guild allow/ignore filters
            effective_channel_id = parent_id if parent_id is not None else channel_id
            owner_id = effective_channel_id

            guild_target = self._cfg.target_for_channel(effective_channel_id, guild_id=guild_id)
            if guild_target:
                return guild_target, owner_id

        return None, None

    async def on_ready(self) -> None:
        """Reconcile all watch targets, active threads, and archived threads upon startup.

        Returns:
            None: No return value.
        """
        logger.info(
            "[watcher] ready; watching %s",
            [t.channel_id or t.guild_id for t in self._cfg.watches],
        )
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._durable_worker_loop())
        if self._reflection_worker_task is None or self._reflection_worker_task.done():
            self._reflection_worker_task = asyncio.create_task(self._reflection_worker_loop())
        try:
            await self._reconcile_targets()
        except Exception:
            logger.exception("[watcher] startup reconciliation failed")

    async def _reconcile_targets(self) -> None:
        """Discover and reconcile all direct targets, active threads, and archived threads.

        Returns:
            None: No return value.
        """
        for target in self._cfg.watches:
            if target.is_guild_watch:
                # Guild-wide watch target reconciliation
                guild = None
                for g in getattr(self, "_test_guilds", getattr(self, "guilds", [])):
                    if g.id == target.guild_id:
                        guild = g
                        break
                if guild is None:
                    continue

                channels_to_check: list[Any] = []
                if hasattr(guild, "text_channels") and guild.text_channels:
                    channels_to_check.extend(guild.text_channels)
                if hasattr(guild, "forum_channels") and guild.forum_channels:
                    channels_to_check.extend(guild.forum_channels)
                if not channels_to_check and hasattr(guild, "channels"):
                    channels_to_check.extend([
                        c for c in guild.channels
                        if getattr(c, "type", None) in (0, 5, 15, "text", "news", "forum", None)
                    ])
                for ch in channels_to_check:
                    if target.matches_channel(ch.id, target.guild_id):
                        self._channel_cache[ch.id] = ch
                        await self._reconcile_scope(ch, target, ch.id)
                        if hasattr(ch, "archived_threads"):
                            try:
                                archived = ch.archived_threads(limit=50)
                                if hasattr(archived, "__aiter__"):
                                    async for thread in archived:
                                        self._channel_cache[thread.id] = thread
                                        await self._reconcile_scope(thread, target, ch.id)
                                elif asyncio.iscoroutine(archived):
                                    for thread in await archived:
                                        self._channel_cache[thread.id] = thread
                                        await self._reconcile_scope(thread, target, ch.id)
                                elif isinstance(archived, list):
                                    for thread in archived:
                                        self._channel_cache[thread.id] = thread
                                        await self._reconcile_scope(thread, target, ch.id)
                            except Exception:
                                pass
            else:
                # Direct channel target
                owner_id = target.channel_id
                ch = self._channel_cache.get(owner_id) or self.get_channel(owner_id)
                if ch is None:
                    try:
                        ch = await self.fetch_channel(owner_id)
                        if ch is not None:
                            self._channel_cache[owner_id] = ch
                    except Exception:
                        logger.warning("[watcher] cannot access watch channel %s", owner_id)
                        continue

                # Reconcile parent watch channel
                await self._reconcile_scope(ch, target, owner_id)

                # Reconcile archived threads if available
                if hasattr(ch, "archived_threads"):
                    try:
                        archived = ch.archived_threads(limit=50)
                        if hasattr(archived, "__aiter__"):
                            async for thread in archived:
                                self._channel_cache[thread.id] = thread
                                await self._reconcile_scope(thread, target, owner_id)
                        elif asyncio.iscoroutine(archived):
                            thread_list = await archived
                            for thread in thread_list:
                                self._channel_cache[thread.id] = thread
                                await self._reconcile_scope(thread, target, owner_id)
                        elif isinstance(archived, list):
                            for thread in archived:
                                self._channel_cache[thread.id] = thread
                                await self._reconcile_scope(thread, target, owner_id)
                    except Exception:
                        logger.warning(
                            "[watcher] failed fetching archived threads for %s",
                            owner_id,
                            exc_info=True,
                        )

        # Reconcile active threads from all guilds
        for guild in getattr(self, "_test_guilds", getattr(self, "guilds", [])):
            threads: list[Any] = []
            if hasattr(guild, "active_threads"):
                try:
                    res = guild.active_threads() if callable(guild.active_threads) else guild.active_threads
                    if asyncio.iscoroutine(res):
                        res = await res
                    if isinstance(res, list):
                        threads.extend(res)
                except Exception:
                    logger.warning("[watcher] failed fetching active threads", exc_info=True)
            elif hasattr(guild, "threads"):
                threads.extend(guild.threads)

            for thread in threads:
                self._channel_cache[thread.id] = thread
                parent_id = getattr(thread, "parent_id", None)
                if parent_id is not None:
                    target, owner_id = self._resolve_target(thread.id, thread)
                    if target is not None and owner_id is not None:
                        await self._reconcile_scope(thread, target, owner_id)

    async def _reconcile_scope(
        self,
        channel: discord.abc.Messageable,
        target: WatchTarget,
        owner_channel_id: int,
    ) -> None:
        """Fetch history and reconcile missed messages for one scope.

        Args:
            channel (discord.abc.Messageable): Discord channel or thread.
            target (WatchTarget): Target configuration for repository and rules.
            owner_channel_id (int): Owning channel snowflake id for caps.

        Returns:
            None: No return value.
        """
        scope_id = str(channel.id)
        self._channel_cache[channel.id] = channel
        last_seen = self._store.get_scope_cursor(scope_id)
        installed_at = self._store.get_installed_at()
        admitted_any = False
        try:
            if last_seen is not None:
                history_iter = channel.history(
                    limit=50,
                    after=discord.Object(id=last_seen),
                    oldest_first=True,
                )
            else:
                history_iter = channel.history(
                    limit=50,
                    oldest_first=False,
                )

            messages: list[discord.Message] = []
            if hasattr(history_iter, "__aiter__"):
                async for msg in history_iter:
                    messages.append(msg)
            elif asyncio.iscoroutine(history_iter):
                messages = await history_iter
            elif isinstance(history_iter, list):
                messages = history_iter

            if last_seen is None and messages:
                messages = sorted(messages, key=lambda m: getattr(m, "id", 0))
            for msg in messages:
                self._store.update_scope_cursor(scope_id, owner_channel_id, msg.id)
                if hasattr(msg, "created_at") and msg.created_at:
                    if msg.created_at.timestamp() < installed_at:
                        continue
                bot_user_id = target.bot_user_id or (self.user.id if self.user else 0)
                mention_ids = [m.id for m in getattr(msg, "mentions", [])]
                decision = evaluate(
                    content=msg.content or "",
                    mention_ids=mention_ids,
                    bot_user_id=bot_user_id,
                )
                if decision.action == Action.RESPOND:
                    if self._store.admit_mention_job(
                        source_message_id=msg.id,
                        conversation_id=channel.id,
                        owner_channel_id=owner_channel_id,
                        repo_path=target.primary_repo,
                    ):
                        admitted_any = True
        except Exception:
            logger.warning("[watcher] error reconciling scope %s", scope_id, exc_info=True)

        if admitted_any:
            self._work_event.set()

    async def on_message(self, message: discord.Message) -> None:
        """Admit explicit mention jobs and advance scope cursors from incoming messages.

        Args:
            message (discord.Message): Incoming Discord message event.

        Returns:
            None: No return value.
        """
        target, owner = self._resolve_target(message.channel.id, message.channel)
        if target is None or owner is None:
            return
        installed_at = self._store.get_installed_at()
        if hasattr(message, "created_at") and message.created_at:
            if message.created_at.timestamp() < installed_at:
                logger.info(
                    "[watcher] ignore message %s created before install cutoff (%s < %s)",
                    message.id,
                    message.created_at.timestamp(),
                    installed_at,
                )
                return


        bot_user_id = target.bot_user_id or (self.user.id if self.user else 0)
        mention_ids = [m.id for m in getattr(message, "mentions", [])]
        decision = evaluate(
            content=message.content or "",
            mention_ids=mention_ids,
            bot_user_id=bot_user_id,
        )
        logger.info(
            "[watcher] %s %s in %s",
            decision.action.value,
            decision.reason,
            message.channel.id,
        )
        self._store.update_scope_cursor(str(message.channel.id), owner, message.id)

        if decision.action == Action.RESPOND:
            self._conversation_last_event[message.channel.id] = time.time()
            if self._store.admit_mention_job(
                source_message_id=message.id,
                conversation_id=message.channel.id,
                owner_channel_id=owner,
                repo_path=target.primary_repo,
            ):
                if self._worker_task is None or self._worker_task.done():
                    self._worker_task = asyncio.create_task(self._durable_worker_loop())
                self._work_event.set()
        elif decision.action == Action.IGNORE:
            # Chatter during burst extends quiet period
            self._conversation_last_event[message.channel.id] = time.time()
            self._work_event.set()

    async def _durable_worker_loop(self) -> None:
        """Continuously process pending outbox batches and queued conversation jobs.

        Returns:
            None: No return value.
        """
        while not self._stopping:
            try:
                # Clean up finished conversation tasks
                self._active_conversation_tasks = {
                    cid: t for cid, t in self._active_conversation_tasks.items()
                    if not t.done()
                }

                # 1. First priority: Check pending or unconfirmed outbox batches
                outbox_batch = self._store.claim_pending_outbox_batch(
                    lease_duration_seconds=LEASE_DURATION_SECONDS
                )
                if outbox_batch is not None:
                    await self._process_outbox_batch(outbox_batch)
                    continue

                # 2. Second priority: Claim next eligible conversation jobs
                claim_result = self._store.claim_next_conversation_jobs(
                    lease_duration_seconds=LEASE_DURATION_SECONDS,
                    exclude_conversations=list(self._active_conversation_tasks.keys()),
                )
                if claim_result is not None:
                    conv_id, jobs = claim_result
                    task = asyncio.create_task(
                        self._process_conversation_jobs(conv_id, jobs)
                    )
                    self._active_conversation_tasks[conv_id] = task
                    continue

                # No immediate work; wait for event or idle timeout
                timeout = 0.5 if self._active_conversation_tasks else 5.0
                try:
                    await asyncio.wait_for(self._work_event.wait(), timeout=timeout)
                    self._work_event.clear()
                except asyncio.TimeoutError:
                    pass
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("[watcher] unexpected error in durable worker loop")
                await asyncio.sleep(1.0)

    async def _process_outbox_batch(self, outbox_batch: dict[str, Any]) -> None:
        """Deliver unconfirmed chunks for a claimed outbox batch.

        Args:
            outbox_batch (dict[str, Any]): Claimed outbox batch metadata and chunks.

        Returns:
            None: No return value.
        """
        batch_id = outbox_batch["batch_id"]
        conv_id = outbox_batch["conversation_id"]
        owner_id = outbox_batch["owner_channel_id"]
        repo_path = outbox_batch.get("repo_path", "")
        candidate_session_id = outbox_batch.get("candidate_session_id")
        source_message_ids = outbox_batch.get("source_message_ids", [])
        chunks = outbox_batch.get("chunks", [])
        reply_to_id = source_message_ids[-1] if source_message_ids else None
        if self._store.is_paused():
            self._store.defer_batch(batch_id, error_class="paused", delay_seconds=15)
            return

        target, _ = self._resolve_target(conv_id)
        if target and self._store.posts_in_last_hour(owner_id) >= target.post_hourly_cap:
            self._store.defer_batch(
                batch_id, error_class="hourly_cap_exceeded", delay_seconds=15
            )
            return

        poster = Poster(self._cfg, self._store, self._token)
        for chunk_info in chunks:
            if chunk_info.get("discord_message_id") is not None:
                continue
            chunk_body = chunk_info.get("body")
            if chunk_body is None:
                logger.error("[watcher] chunk body is missing for batch %s", batch_id)
                self._store.fail_batch(
                    batch_id,
                    error_class="missing_chunk_body",
                    is_permanent=True,
                )
                return

            chunk_idx = chunk_info["chunk_index"]
            nonce = chunk_info["nonce"]
            result = await poster.deliver_chunk(
                channel_id=conv_id,
                target_channel_id=owner_id,
                chunk=chunk_body,
                nonce=nonce,
                reply_to_message_id=reply_to_id,
            )
            if result.ok:
                self._store.record_chunk_delivery(
                    batch_id, chunk_idx, result.discord_message_id or 0
                )
            else:
                is_perm = _is_permanent_error(result.error_class, result.status_code)
                err_cls = result.error_class or "delivery_error"
                retry_delay = int(result.retry_after) if result.retry_after else RETRY_DELAY_SECONDS
                self._store.fail_batch(
                    batch_id,
                    error_class=err_cls,
                    is_permanent=is_perm,
                    retry_delay_seconds=retry_delay,
                    max_attempts=MAX_ATTEMPTS,
                )
                return

        # Complete batch and scrub outbound body text
        self._store.complete_delivery_batch(
            batch_id,
            repo_path=repo_path,
            candidate_session_id=candidate_session_id,
        )

    async def _process_conversation_jobs(
        self,
        conversation_id: int,
        jobs: list[dict[str, Any]],
    ) -> None:
        """Process claimed mention jobs for one conversation after debounce.

        Args:
            conversation_id (int): Channel or thread snowflake id.
            jobs (list[dict[str, Any]]): Claimed jobs in chronological order.

        Returns:
            None: No return value.
        """
        source_message_ids = [j["source_message_id"] for j in jobs]
        latest_msg_id = source_message_ids[-1]
        owner_channel_id = jobs[0]["owner_channel_id"]
        repo_path = jobs[0]["repo_path"]

        try:
            # 1. Quiet period debounce
            while True:
                last_event = self._conversation_last_event.get(conversation_id, 0.0)
                elapsed = time.time() - last_event
                remaining = BURST_QUIET_SECONDS - elapsed
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, 1.0))

            # Claim any additional pending jobs that arrived during debounce
            more_jobs = self._store.claim_more_pending_jobs(
                conversation_id, lease_duration_seconds=LEASE_DURATION_SECONDS
            )
            if more_jobs:
                jobs.extend(more_jobs)
                source_message_ids = [j["source_message_id"] for j in jobs]
                latest_msg_id = source_message_ids[-1]

            # 2. Check kill switch and hourly cap
            if self._store.is_paused():
                self._store.defer_jobs(
                    source_message_ids,
                    error_class="paused",
                    delay_seconds=15,
                )
                return
            # 3. Resolve Discord channel / thread
            ch = self._channel_cache.get(conversation_id) or self.get_channel(conversation_id)
            if ch is None:
                try:
                    ch = await self.fetch_channel(conversation_id)
                    if ch is not None:
                        self._channel_cache[conversation_id] = ch
                except Exception:
                    pass

            if ch is None:
                self._store.fail_jobs(
                    source_message_ids,
                    error_class="channel_not_found",
                    is_permanent=True,
                )
                return

            target, _ = self._resolve_target(conversation_id, ch)
            if target is None:
                self._store.fail_jobs(
                    source_message_ids,
                    error_class="target_unresolved",
                    is_permanent=True,
                )
                return

            if self._store.posts_in_last_hour(owner_channel_id) >= target.post_hourly_cap:
                self._store.defer_jobs(
                    source_message_ids,
                    error_class="hourly_cap_exceeded",
                    delay_seconds=30,
                )
                return

            # 4. Fetch source triggering message
            trigger_message = None
            if hasattr(ch, "fetch_message"):
                try:
                    trigger_message = await ch.fetch_message(latest_msg_id)
                except Exception:
                    pass
            elif hasattr(ch, "_history"):
                for m in getattr(ch, "_history", []):
                    if getattr(m, "id", None) == latest_msg_id:
                        trigger_message = m
                        break

            if trigger_message is None:
                self._store.fail_jobs(
                    source_message_ids,
                    error_class="message_not_found",
                    is_permanent=True,
                )
                return

            author_name = getattr(trigger_message.author, "display_name", "unknown")
            author_id = getattr(trigger_message.author, "id", 0)
            question = trigger_message.content or ""
            excerpt = await _thread_excerpt(ch)
            sem = self._channel_semaphore(owner_channel_id)

            async with sem:
                # Background lease renewal task
                async def _renew_leases_task() -> None:
                    while True:
                        await asyncio.sleep(LEASE_RENEW_INTERVAL_SECONDS)
                        self._store.renew_job_leases(
                            source_message_ids,
                            lease_duration_seconds=LEASE_DURATION_SECONDS,
                        )

                renew_handle = asyncio.create_task(_renew_leases_task())
                # ONE canonical memory scope per conversation target: the same
                # rule the runner uses when fetching prompt memory (finding B1).
                memory_scope_id = scope_id_for_target(target)
                # Participant-scoped memory (plan Phase 3.6): pass the author's
                # member id so only current participants reach the prompt.
                participant_id = str(author_id) if author_id else None
                try:
                    session_id = self._store.get_opencode_session(conversation_id, repo_path)
                    reply = await self._runner.run(
                        target,
                        author_name,
                        question,
                        excerpt,
                        session_id=session_id,
                        member_id=participant_id,
                    )
                    if session_id and not reply.ok and reply.error_class == "unknown_session":
                        self._store.clear_opencode_session(conversation_id, repo_path)
                        reply = await self._runner.run(
                            target,
                            author_name,
                            question,
                            excerpt,
                            session_id=None,
                            member_id=participant_id,
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.exception("[watcher] OpenCode runner failed: %s", exc)
                    self._store.fail_jobs(
                        source_message_ids,
                        error_class="runner_crash",
                        is_permanent=False,
                        retry_delay_seconds=RETRY_DELAY_SECONDS,
                    )
                    return
                finally:
                    renew_handle.cancel()
                    try:
                        await renew_handle
                    except asyncio.CancelledError:
                        pass

                reply_cap = self._cfg.max_reply_chars if self._cfg else 2000
                chunks = split_message(reply.text, reply_cap) if reply.text else ["(no response generated)"]
                batch_id = uuid.uuid4().hex
                candidate_session_id = reply.session_id if reply.ok else None

                # Commit outbox atomically
                self._store.commit_outbox(
                    batch_id=batch_id,
                    conversation_id=conversation_id,
                    owner_channel_id=owner_channel_id,
                    repo_path=repo_path,
                    source_message_ids=source_message_ids,
                    chunks=chunks,
                    candidate_session_id=candidate_session_id,
                )

                # Attempt immediate delivery
                poster = Poster(self._cfg, self._store, self._token)
                delivery_failed = False
                for idx, chunk in enumerate(chunks):
                    nonce = f"{batch_id[:16]}-{idx}"
                    res = await poster.deliver_chunk(
                        channel_id=conversation_id,
                        target_channel_id=owner_channel_id,
                        chunk=chunk,
                        nonce=nonce,
                        reply_to_message_id=latest_msg_id,
                    )
                    if res.ok:
                        self._store.record_chunk_delivery(
                            batch_id, idx, res.discord_message_id or 0
                        )
                    else:
                        is_perm = _is_permanent_error(res.error_class, res.status_code)
                        err_cls = res.error_class or "delivery_error"
                        retry_delay = int(res.retry_after) if res.retry_after else RETRY_DELAY_SECONDS
                        self._store.fail_batch(
                            batch_id,
                            error_class=err_cls,
                            is_permanent=is_perm,
                            retry_delay_seconds=retry_delay,
                            max_attempts=MAX_ATTEMPTS,
                        )
                        delivery_failed = True
                        break

                if not delivery_failed:
                    self._store.complete_delivery_batch(
                        batch_id,
                        repo_path=repo_path,
                        candidate_session_id=candidate_session_id,
                    )
                    self._enqueue_reflection(
                        batch_id=batch_id,
                        platform="discord",
                        scope_id=memory_scope_id,
                        author_id=str(author_id),
                        author_name=author_name,
                        question=question,
                        reply_text=reply.text,
                        thread_excerpt=excerpt,
                    )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("[watcher] unexpected error processing conversation %s", conversation_id)
            self._store.fail_jobs(
                source_message_ids,
                error_class="conversation_processing_error",
                is_permanent=False,
                retry_delay_seconds=RETRY_DELAY_SECONDS,
            )
        finally:
            self._work_event.set()
    async def _reflection_worker_loop(self) -> None:
        """Process bounded reflection observations sequentially without blocking delivery.

        On graceful shutdown the worker keeps DRAINING until the queue is empty
        (finding B7); ``close()`` bounds the drain with a fixed timeout and
        cancels the worker as the fallback so shutdown always terminates.
        """
        while True:
            try:
                if self._stopping:
                    # Intake has stopped: drain what remains, then exit.
                    item = self._reflection_queue.get_nowait()
                else:
                    item = await asyncio.wait_for(
                        self._reflection_queue.get(), timeout=1.0
                    )
            except asyncio.QueueEmpty:
                break
            except (asyncio.TimeoutError, TimeoutError):
                continue
            except asyncio.CancelledError:
                break

            platform, scope_id, author_id, author_name, question, reply_text, excerpt = item
            try:
                await self._reflection_engine.reflect_after_delivery(
                    platform=platform,
                    scope_id=scope_id,
                    author_id=author_id,
                    author_name=author_name,
                    question=question,
                    reply=reply_text,
                    thread_excerpt=excerpt,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("[watcher] reflection worker error: %s", type(exc).__name__)
            finally:
                self._reflection_queue.task_done()

    def _enqueue_reflection(
        self,
        batch_id: str,
        platform: str,
        scope_id: str,
        author_id: str,
        author_name: str,
        question: str,
        reply_text: str,
        thread_excerpt: str,
    ) -> None:
        """Enqueue a reflection observation idempotently into the bounded in-memory queue."""
        if self._stopping:
            return
        if batch_id in self._processed_reflection_deliveries:
            return
        self._processed_reflection_deliveries[batch_id] = None
        # Bounded FIFO: evict the OLDEST tracked delivery deterministically.
        while len(self._processed_reflection_deliveries) > MAX_TRACKED_REFLECTION_DELIVERIES:
            self._processed_reflection_deliveries.popitem(last=False)

        item = (platform, scope_id, author_id, author_name, question, reply_text, thread_excerpt)
        try:
            self._reflection_queue.put_nowait(item)
        except asyncio.QueueFull:
            # Overload policy: drop oldest observation to accommodate newest
            try:
                self._reflection_queue.get_nowait()
                self._reflection_queue.task_done()
                self._reflection_queue.put_nowait(item)
            except Exception:
                pass
            self._reflection_dropped_count += 1
            logger.warning(
                "[watcher] reflection queue full; dropped oldest (total dropped=%d)",
                self._reflection_dropped_count,
            )
    async def close(self) -> None:
        """Gracefully stop background worker tasks and close gateway connection.

        Returns:
            None: No return value.
        """
        self._stopping = True
        self._work_event.set()
        if self._worker_task is not None and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        for t in list(self._active_conversation_tasks.values()):
            if not t.done():
                t.cancel()
        if self._reflection_worker_task is not None and not self._reflection_worker_task.done():
            # Drain briefly up to 2 seconds
            try:
                await asyncio.wait_for(self._reflection_queue.join(), timeout=2.0)
            except (asyncio.TimeoutError, TimeoutError, Exception):
                pass
            self._reflection_worker_task.cancel()
            try:
                await self._reflection_worker_task
            except asyncio.CancelledError:
                pass
        try:
            await super().close()
        except Exception:
            pass
