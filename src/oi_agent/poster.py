"""Poster: the ONLY component allowed to send messages to Discord.

Enforces the channel allowlist (watch targets + threads under them), the
per-channel hourly cap, and the global kill switch. Everything else in the
daemon is silent by construction.
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from .config import Config, DISCORD_MAX_MESSAGE_CHARS, REPLY_DELIVERY_MODES
from .store import Store

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)


def split_message(content: str, max_chars: int) -> list[str]:
    """Split a configured reply cap into Discord-safe readable chunks.

    Args:
        content: Full reply text.
        max_chars: Total configured reply cap.

    Returns:
        Non-empty chunks, each no longer than Discord's 2,000-character limit.
    """
    remaining = content[:max(0, max_chars)].strip()
    chunks: list[str] = []
    while remaining:
        if len(remaining) <= DISCORD_MAX_MESSAGE_CHARS:
            chunks.append(remaining)
            break
        cut = remaining.rfind("\n", 0, DISCORD_MAX_MESSAGE_CHARS + 1)
        if cut < DISCORD_MAX_MESSAGE_CHARS // 2:
            cut = remaining.rfind(" ", 0, DISCORD_MAX_MESSAGE_CHARS + 1)
        if cut <= 0:
            cut = DISCORD_MAX_MESSAGE_CHARS
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:].lstrip()
    return chunks




class Poster:
    """Discord REST sender with safety rails."""

    def __init__(self, cfg: Config, store: Store, token: str | None) -> None:
        """Create a poster bound to the daemon config and state store.

        Args:
            cfg: Full daemon config (allowlist source).
            store: State store for caps and pause flag.
            token: Bot token, or ``None`` before the gateway starts.
        """
        self._cfg = cfg
        self._store = store
        self._headers = {"Authorization": f"Bot {token or ''}"}


    async def send(self, channel_id: int, target_channel_id: int,
                   content: str) -> bool:
        """Post one Discord-safe reply through the safety rails.

        One reply consumes exactly one unit of the hourly cap. The configured
        reply cap limits the total logical answer; ``chunked`` delivery sends
        each Discord-safe chunk, while ``single_message`` requires one chunk.
        Mentions in generated text are suppressed: Discord parses no
        @everyone/@here/role/user mentions from our output.

        Args:
            channel_id: Destination channel/thread id.
            target_channel_id: Owning watch-target channel id.
            content: Full reply body.

        Returns:
            True only when the reply is sent successfully.
        """
        delivery = getattr(self._cfg, "reply_delivery", None)
        if delivery not in REPLY_DELIVERY_MODES:
            logger.error("[poster] invalid reply_delivery: %r", delivery)
            return False
        reply_cap = self._cfg.max_reply_chars
        if (isinstance(reply_cap, bool) or not isinstance(reply_cap, int)
                or reply_cap < 200):
            logger.error("[poster] invalid max_reply_chars: %r", reply_cap)
            return False
        if (delivery == "single_message"
                and reply_cap > DISCORD_MAX_MESSAGE_CHARS):
            logger.error(
                "[poster] single_message requires max_reply_chars <= 2000 "
                "(got %s)", reply_cap)
            return False
        chunks = split_message(content, reply_cap)
        if delivery == "single_message" and len(chunks) != 1:
            logger.error("[poster] single_message produced %d chunks",
                         len(chunks))
            return False
        if not chunks:
            return False
        # Reserve the whole reply atomically before the first network call so
        # concurrent events cannot both squeeze past the cap (TOCTOU).
        target = self._cfg.target_for_channel(target_channel_id)
        if target is None:
            logger.error("[poster] target not watchlisted: %s",
                         target_channel_id)
            return False
        if not self._store.reserve_post(target.channel_id,
                                        target.post_hourly_cap):
            logger.warning("[poster] hourly cap reached for %s",
                           target.channel_id)
            return False
        # One client for the whole reply: chunks reuse the connection pool and
        # TLS session instead of reconnecting per chunk.
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for index, chunk in enumerate(chunks, 1):
                # Re-check pause immediately before each request so `oi pause`
                # takes effect mid-reply, not just between replies.
                if self._store.is_paused():
                    logger.warning("[poster] paused before chunk %d/%d",
                                   index, len(chunks))
                    return False
                resp = await self._post_chunk(client, channel_id, chunk)
                if resp is None:
                    return False
                if resp.status_code not in (200, 201):
                    logger.error("[poster] discord refused chunk %d/%d: %s %s",
                                 index, len(chunks), resp.status_code,
                                 resp.text[:200])
                    return False
        return True

    async def _post_chunk(self, client, channel_id: int, chunk: str):
        """POST one chunk, honoring Discord 429 rate limits with a bounded retry.

        A 429 is a transient rate limit, not a permanent failure: Discord tells
        us how long to wait via ``retry_after``. Retrying a few times prevents a
        single burst from dropping replies.

        Args:
            client: Shared httpx client for the whole reply.
            channel_id: Destination channel/thread id.
            chunk: Message text to send.

        Returns:
            The final httpx response, or None on a network error.
        """
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        body = {"content": chunk, "allowed_mentions": {"parse": []}}
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=self._headers, json=body)
            except Exception as exc:  # noqa: BLE001 - Discord network boundary
                logger.error("[poster] discord request failed: %s", exc)
                return None
            if resp.status_code != 429:
                return resp
            try:
                retry_after = float(resp.headers.get("retry-after", "1"))
            except ValueError:
                retry_after = 1.0
            retry_after = min(retry_after, 10.0)
            logger.warning("[poster] 429 rate limited; retry in %.2fs "
                           "(attempt %d)", retry_after, attempt + 1)
            await asyncio.sleep(retry_after)
        return resp
