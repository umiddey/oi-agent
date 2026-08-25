"""Discord gateway watcher: one connection serving all configured channels.

Routes each incoming message through gate -> responder -> poster. Threads under
a watched channel inherit its binding automatically.
"""

from __future__ import annotations

import asyncio
import logging

import discord

from ..agent import responder
from ..config import Config, effective_max_tool_iterations
from ..poster import Poster
from ..store import Store
from .gate import Action, evaluate

logger = logging.getLogger(__name__)

EXCERPT_LIMIT = 40
MAX_CONCURRENT_AUDITS_PER_CHANNEL = 2


def _message_evidence_line(message) -> str:
    """Render message text and attachment metadata for model context.

    Args:
        message: Discord message-like object.

    Returns:
        One bounded evidence line containing text, attachment names, and URLs.
    """
    body = (message.content or "").replace("\n", " ").strip()
    attachments = []
    for attachment in getattr(message, "attachments", []):
        name = getattr(attachment, "filename", "attachment")
        url = getattr(attachment, "url", "")
        content_type = getattr(attachment, "content_type", "")
        label = f"{name}{f' ({content_type})' if content_type else ''}"
        attachments.append(f"{label}: {url}" if url else label)
    for embed in getattr(message, "embeds", []):
        url = getattr(embed, "url", "")
        title = getattr(embed, "title", "") or "embed"
        attachments.append(f"{title}: {url}" if url else title)
    if attachments:
        body = f"{body} [attachments: {'; '.join(attachments)}]".strip()
    return (body or "(attachment/embed)")[:600]


async def _thread_excerpt(
    channel: discord.abc.Messageable | None,
) -> str:
    """Collect recent channel history and preserve attachment evidence.

    Args:
        channel: Discord channel/thread, or ``None`` when it is not cached.

    Returns:
        Newline-joined ``author: evidence`` lines, oldest first.
    """
    if channel is None:
        return ""
    try:
        msgs = [m async for m in channel.history(limit=EXCERPT_LIMIT)]
    except Exception:  # noqa: BLE001 - Discord cache/transport boundary
        return ""
    lines = []
    for message in reversed(msgs):
        if message.author.bot and (message.content or "").startswith("-#"):
            continue
        lines.append(
            f"{message.author.display_name}: {_message_evidence_line(message)}"
        )
    return "\n".join(lines)


class OIWatcher(discord.Client):
    """Gateway client implementing the full message pipeline."""

    def __init__(self, cfg: Config, store: Store) -> None:
        """Initialize intents and collaborators.

        Args:
            cfg: Daemon config with all watch targets.
            store: Shared state store.
        """
        intents = discord.Intents.none()
        intents.guilds = True
        intents.messages = True
        intents.message_content = True
        super().__init__(intents=intents)
        self._cfg = cfg
        self._store = store
        self._token: str | None = None
        # Bound concurrent audits per channel so a burst of messages cannot all
        # pass the cap read and launch parallel audit+LLM work. The atomic
        # posting reservation is still the authoritative outbound gate; this
        # caps inbound work under untrusted traffic.
        self._channel_sems: dict[int, asyncio.Semaphore] = {}

    def _channel_semaphore(self, channel_id: int) -> asyncio.Semaphore:
        """Return (creating on first use) the per-channel work semaphore.

        Args:
            channel_id: Owning watch-target channel id.

        Returns:
            The asyncio.Semaphore bounding concurrent audits for that channel.
        """
        sem = self._channel_sems.get(channel_id)
        if sem is None:
            sem = asyncio.Semaphore(MAX_CONCURRENT_AUDITS_PER_CHANNEL)
            self._channel_sems[channel_id] = sem
        return sem

    async def on_ready(self) -> None:
        """Log startup. Discord itself is the canonical history; the store
        keeps only bounded posting state and rolling memory."""
        logger.info("[watcher] ready; watching %s",
                    [t.channel_id for t in self._cfg.watches])

    def _resolve_target(self, channel_id: int, channel=None):
        """Return (WatchTarget, owner id), using the event channel if supplied.

        Args:
            channel_id: Channel or thread id for the incoming message.
            channel: Discord channel object from the event, when available.

        Returns:
            Tuple of (WatchTarget|None, owning watch channel id|None).
        """
        direct = self._cfg.target_for_channel(channel_id)
        if direct:
            return direct, direct.channel_id
        # The event already carries the thread object; use it before cache.
        ch = channel or self.get_channel(channel_id)
        parent_id = getattr(ch, "parent_id", None)
        if parent_id is not None:
            target = self._cfg.target_for_channel(parent_id)
            if target:
                return target, target.channel_id
        return None, None

    async def _handle_reply_path(self, target, author_name: str,
                                 question: str,
                                 reply_to_channel_id: int,
                                 owner_channel_id: int,
                                 reply_channel=None) -> None:
        """Run the audit pipeline and post the draft through safety rails.

        Args:
            target: Resolved watch target.
            author_name: Sender display name.
            question: Message text.
            reply_to_channel_id: Channel/thread to post into.
            owner_channel_id: Owning watch-target id (cap/allowlist key).
            reply_channel: Event channel object used for uncached threads.
        """
        # Admission control: bail before the expensive audit + LLM calls if the
        # channel's hourly cap is already spent or posting is paused. The
        # atomic reservation in Poster.send remains the authoritative gate; this
        # only stops an exhausted channel from burning audit/LLM work.
        if self._store.is_paused():
            return
        if self._store.posts_in_last_hour(owner_channel_id) \
                >= target.post_hourly_cap:
            logger.info("[watcher] cap spent; skipping audit for %s",
                        owner_channel_id)
            return
        # Bound concurrent audits per channel: if the channel is already at its
        # in-flight limit, drop this message rather than pile on parallel
        # audit+LLM work under a burst.
        sem = self._channel_semaphore(owner_channel_id)
        if sem.locked():
            logger.info("[watcher] channel %s at audit concurrency limit; "
                        "dropping message", owner_channel_id)
            return
        async with sem:
            excerpt = await _thread_excerpt(
                reply_channel or self.get_channel(reply_to_channel_id))
            try:
                reply = await responder.respond(
                    self._cfg.agent, self._cfg.fallback, target, author_name,
                    question, excerpt, store=self._store,
                    personality=self._cfg.personality,
                    max_response_tokens=self._cfg.max_response_tokens,
                    max_tool_iterations=effective_max_tool_iterations(
                        self._cfg, target),
                    # Recall is per-conversation: a thread keeps its own memory
                    # separate from the parent channel and sibling threads.
                    memory_key=reply_to_channel_id,
                )
            except Exception:  # noqa: BLE001 - event boundary
                logger.exception("[watcher] responder crashed")
                return
            try:
                poster = Poster(self._cfg, self._store, self._token or "")
                sent = await poster.send(reply_to_channel_id, owner_channel_id,
                                         reply.text)
            except Exception:  # noqa: BLE001 - network/event boundary
                logger.exception("[watcher] poster crashed")
                return
            if not sent:
                logger.warning("[watcher] post blocked or failed in %s",
                               reply_to_channel_id)
                return
            # Memory is recorded only after Discord accepted the reply, so
            # blocked or failed posts are never remembered as completed.
            # Honest-failure fallbacks (repo/LLM failure) post but are not real
            # exchanges, so they must not pollute the rolling memory with
            # fact-less "couldn't complete the audit" noise.
            if not reply.ok:
                return
            # Keyed on the conversation (thread/channel), matching the read in
            # respond(), so thread recall never mixes with the parent channel.
            try:
                await responder.update_memory_llm(
                    self._store, reply_to_channel_id, question,
                    reply.text, reply.sha,
                    configs=[self._cfg.agent, self._cfg.fallback])
            except Exception:  # noqa: BLE001 - state boundary
                logger.warning("[watcher] memory update failed", exc_info=True)

    async def on_message(self, message: discord.Message) -> None:
        """Main pipeline entry point for every gateway message event.

        Args:
            message: Incoming Discord message.
        """
        # Never react to bots (own user included): bot-to-bot mention loops
        # would otherwise trigger autonomous LLM replies and channel spam.
        if message.author.bot:
            return
        target, owner = self._resolve_target(
            message.channel.id, message.channel)
        if target is None:
            return
        mention_ids = [u.id for u in message.mentions]
        decision = evaluate(
            content=message.content or "",
            mention_ids=mention_ids,
            author_id=message.author.id,
            bot_user_id=target.bot_user_id,
            founder_ids=target.founder_ids,
            claim_keywords=target.claim_keywords,
        )
        logger.info("[watcher] %s %s in %s", decision.action.value,
                    decision.reason, message.channel.id)
        if decision.action == Action.IGNORE:
            return

        # Every gated message goes directly to the main evidence responder.
        await self._handle_reply_path(
            target, message.author.display_name, message.content or "",
            reply_to_channel_id=message.channel.id,
            owner_channel_id=owner,
            reply_channel=message.channel,
        )

    def run_with_token(self, token: str) -> None:
        """Start the gateway loop with an explicit token.

        Args:
            token: Discord bot token.
        """
        self._token = token
        super().run(token)


def run_daemon(cfg: Config, store: Store, token: str) -> None:
    """Blocking entry point that runs the watcher until interrupted.

    Args:
        cfg: Loaded daemon config.
        store: State store instance.
        token: Discord bot token.
    """
    OIWatcher(cfg, store).run_with_token(token)
