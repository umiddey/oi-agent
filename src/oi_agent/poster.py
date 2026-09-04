"""Poster: the ONLY component allowed to send messages to Discord.

Enforces the channel allowlist (watch targets + threads under them), the
per-channel hourly cap, and the global kill switch. Everything else in the
daemon is silent by construction.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .config import DISCORD_MAX_MESSAGE_CHARS, REPLY_DELIVERY_MODES, Config
from .store import Store

logger = logging.getLogger(__name__)

_TIMEOUT = httpx.Timeout(30.0)


@dataclass
class DeliveryResult:
    """Outcome of attempting to deliver a single outbound chunk to Discord.

    Attributes:
        ok (bool): True if Discord confirmed message creation.
        discord_message_id (int | None): Snowflake message id from Discord response.
        error_class (str | None): Bounded machine-readable failure category.
        status_code (int | None): HTTP status code from Discord response.
        retry_after (float | None): Rate limit backoff duration in seconds if 429.
    """

    ok: bool
    discord_message_id: int | None = None
    error_class: str | None = None
    status_code: int | None = None
    retry_after: float | None = None


def split_message(content: str, max_chars: int) -> list[str]:
    """Split a configured reply cap into Discord-safe readable chunks.

    Args:
        content (str): Full reply text.
        max_chars (int): Total configured reply cap.

    Returns:
        list[str]: Non-empty chunks, each no longer than Discord's 2,000-character limit.
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


def _is_first_chunk(nonce: str) -> bool:
    """Determine if the nonce represents the first chunk in a batch.

    0-indexed chunk nonces (e.g. ``batch_id-0``) and unindexed nonces represent
    the initial chunk that reserves a slot against the hourly cap.

    Args:
        nonce (str): Unique delivery nonce for the chunk.

    Returns:
        bool: True if this is the first chunk (suffix 0 or unindexed).
    """
    if "-" in nonce:
        suffix = nonce.rsplit("-", 1)[-1]
        if suffix.isdigit():
            return int(suffix) == 0
    return True


class Poster:
    """Discord REST sender with safety rails."""

    def __init__(self, cfg: Config, store: Store, token: str | None) -> None:
        """Create a poster bound to the daemon config and state store.

        Args:
            cfg (Config): Full daemon config (allowlist source).
            store (Store): State store for caps and pause flag.
            token (str | None): Bot token, or None before the gateway starts.

        Returns:
            None: Initializer does not return a value.
        """
        self._cfg = cfg
        self._store = store
        self._headers = {"Authorization": f"Bot {token or ''}"}

    async def deliver_chunk(
        self,
        channel_id: int,
        target_channel_id: int,
        chunk: str,
        nonce: str,
        reply_to_message_id: int | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> DeliveryResult:
        """Deliver one outbound chunk to Discord through safety rails.

        Args:
            channel_id (int): Destination channel/thread snowflake id.
            target_channel_id (int): Owning watch-target channel snowflake id.
            chunk (str): Message text chunk to send.
            nonce (str): Unique idempotency nonce for this chunk.
            reply_to_message_id (int | None): Optional source message snowflake id to reply to.
            client (httpx.AsyncClient | None): Optional shared async HTTP client.

        Returns:
            DeliveryResult: Outcome with delivery status and metadata or classified error.
        """
        target = self._cfg.target_for_channel(target_channel_id)
        if target is None:
            logger.error("[poster] target not watchlisted: %s", target_channel_id)
            return DeliveryResult(ok=False, error_class="target_not_found")

        if self._store.is_paused():
            logger.warning("[poster] posting paused")
            return DeliveryResult(ok=False, error_class="paused")

        if _is_first_chunk(nonce):
            cap_key = target.guild_id if target.is_guild_watch else target.channel_id
            if not self._store.reserve_post(cap_key, target.post_hourly_cap):
                logger.warning("[poster] hourly cap reached for %s", cap_key)
                return DeliveryResult(ok=False, error_class="hourly_cap_exceeded")

        url = f"https://discord.com/api/v10/channels/{channel_id}/messages"
        payload: dict[str, Any] = {
            "content": chunk,
            "nonce": nonce,
            "enforce_nonce": True,
            "allowed_mentions": {"parse": []},
        }
        if reply_to_message_id is not None:
            payload["message_reference"] = {"message_id": reply_to_message_id}

        if client is not None:
            reconciled = await self._reconcile_unconfirmed_post(
                client, channel_id, nonce, chunk, reply_to_message_id
            )
            if reconciled is not None:
                return DeliveryResult(ok=True, discord_message_id=reconciled, status_code=200)
            return await self._execute_http(
                client, url, payload, channel_id, nonce, chunk, reply_to_message_id
            )

        async with httpx.AsyncClient(timeout=_TIMEOUT) as owned_client:
            reconciled = await self._reconcile_unconfirmed_post(
                owned_client, channel_id, nonce, chunk, reply_to_message_id
            )
            if reconciled is not None:
                return DeliveryResult(ok=True, discord_message_id=reconciled, status_code=200)
            return await self._execute_http(
                owned_client, url, payload, channel_id, nonce, chunk, reply_to_message_id
            )

    async def _reconcile_unconfirmed_post(
        self,
        client: httpx.AsyncClient,
        channel_id: int,
        nonce: str,
        chunk: str,
        reply_to_message_id: int | None = None,
    ) -> int | None:
        """Query recent channel history to check if an unconfirmed chunk was created.

        Guards against duplicate message creation when a Discord POST request
        succeeds on the server but the client connection drops before receiving
        the HTTP response, or when a retry happens after Discord's ephemeral
        nonce cache expires.

        Args:
            client (httpx.AsyncClient): HTTP client.
            channel_id (int): Destination channel snowflake id.
            nonce (str): Expected delivery nonce.
            chunk (str): Expected message text.
            reply_to_message_id (int | None): Expected reply-to message snowflake id.

        Returns:
            int | None: Snowflake id of existing message if reconciled, else None.
        """
        url = f"https://discord.com/api/v10/channels/{channel_id}/messages?limit=10"
        try:
            resp = await client.get(url, headers=self._headers)
            if resp.status_code == 200 and callable(getattr(resp, "json", None)):
                messages = resp.json()
                if isinstance(messages, list):
                    for msg in messages:
                        if not isinstance(msg, dict):
                            continue
                        msg_nonce = str(msg.get("nonce", ""))
                        if msg_nonce and msg_nonce == str(nonce):
                            msg_id = int(msg["id"])
                            logger.info(
                                "[poster] reconciled unconfirmed chunk via nonce %s -> msg %s",
                                nonce,
                                msg_id,
                            )
                            return msg_id
                        if msg.get("content") == chunk:
                            if reply_to_message_id is not None:
                                ref = msg.get("message_reference")
                                if isinstance(ref, dict) and str(ref.get("message_id")) == str(reply_to_message_id):
                                    msg_id = int(msg["id"])
                                    logger.info(
                                        "[poster] reconciled unconfirmed chunk via content/ref -> msg %s",
                                        msg_id,
                                    )
                                    return msg_id
                            else:
                                msg_id = int(msg["id"])
                                logger.info(
                                    "[poster] reconciled unconfirmed chunk via content -> msg %s",
                                    msg_id,
                                )
                                return msg_id
        except Exception as exc:  # noqa: BLE001
            logger.warning("[poster] channel history reconciliation failed: %s", exc)
        return None

    async def _execute_http(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        channel_id: int,
        nonce: str,
        chunk: str,
        reply_to_message_id: int | None = None,
    ) -> DeliveryResult:
        """Execute Discord message POST with retry, error classification, and history reconciliation.

        Args:
            client (httpx.AsyncClient): HTTP client for sending request.
            url (str): Target Discord API endpoint URL.
            payload (dict[str, Any]): JSON payload to transmit.
            channel_id (int): Destination channel snowflake id.
            nonce (str): Expected delivery nonce.
            chunk (str): Expected message text.
            reply_to_message_id (int | None): Expected reply-to message snowflake id.

        Returns:
            DeliveryResult: Outcome with message ID or error classification.
        """
        for attempt in range(3):
            try:
                resp = await client.post(url, headers=self._headers, json=payload)
            except Exception as exc:  # noqa: BLE001 - Discord network boundary
                logger.error("[poster] discord request failed: %s", exc)
                reconciled_id = await self._reconcile_unconfirmed_post(
                    client, channel_id, nonce, chunk, reply_to_message_id
                )
                if reconciled_id is not None:
                    return DeliveryResult(
                        ok=True,
                        discord_message_id=reconciled_id,
                        status_code=200,
                    )
                return DeliveryResult(ok=False, error_class="network_error")
            if resp.status_code in (200, 201):
                msg_id = None
                try:
                    if callable(getattr(resp, "json", None)):
                        data = resp.json()
                        if isinstance(data, dict) and "id" in data:
                            msg_id = int(data["id"])
                except Exception:
                    pass
                return DeliveryResult(
                    ok=True,
                    discord_message_id=msg_id,
                    status_code=resp.status_code,
                )

            if resp.status_code == 429:
                retry_after = 1.0
                try:
                    headers = getattr(resp, "headers", {})
                    content_type = ""
                    if hasattr(headers, "get"):
                        content_type = headers.get("content-type", "")
                    if content_type.startswith("application/json") and callable(
                        getattr(resp, "json", None)
                    ):
                        data = resp.json()
                        if isinstance(data, dict) and "retry_after" in data:
                            retry_after = float(data["retry_after"])
                    elif hasattr(headers, "get") and headers.get("retry-after") is not None:
                        retry_after = float(headers.get("retry-after"))
                except Exception:
                    retry_after = 1.0

                retry_after = min(max(retry_after, 0.01), 10.0)
                logger.warning(
                    "[poster] 429 rate limited; retry in %.2fs (attempt %d)",
                    retry_after,
                    attempt + 1,
                )
                if attempt < 2:
                    await asyncio.sleep(retry_after)
                    continue
                return DeliveryResult(
                    ok=False,
                    error_class="rate_limited",
                    status_code=429,
                    retry_after=retry_after,
                )

            error_class = "server_error"
            if resp.status_code == 403:
                error_class = "permission_denied"
            elif resp.status_code == 404:
                error_class = "channel_not_found"
            elif resp.status_code >= 500:
                reconciled_id = await self._reconcile_unconfirmed_post(
                    client, channel_id, nonce, chunk, reply_to_message_id
                )
                if reconciled_id is not None:
                    return DeliveryResult(
                        ok=True,
                        discord_message_id=reconciled_id,
                        status_code=200,
                    )
                error_class = "server_error"
            else:
                error_class = f"http_{resp.status_code}"

            logger.error(
                "[poster] discord refused chunk: %s %s",
                resp.status_code,
                getattr(resp, "text", "")[:200],
            )
            return DeliveryResult(
                ok=False,
                error_class=error_class,
                status_code=resp.status_code,
            )

        return DeliveryResult(
            ok=False,
            error_class="rate_limited",
            status_code=429,
        )

    async def send(
        self,
        channel_id: int,
        target_channel_id: int,
        content: str,
    ) -> bool:
        """Post one Discord-safe reply through the safety rails.

        Args:
            channel_id (int): Destination channel/thread id.
            target_channel_id (int): Owning watch-target channel id.
            content (str): Full reply body.

        Returns:
            bool: True only when all chunks of the reply are sent successfully.
        """
        delivery = getattr(self._cfg, "reply_delivery", None)
        if delivery not in REPLY_DELIVERY_MODES:
            logger.error("[poster] invalid reply_delivery: %r", delivery)
            return False
        reply_cap = self._cfg.max_reply_chars
        if (
            isinstance(reply_cap, bool)
            or not isinstance(reply_cap, int)
            or reply_cap < 200
        ):
            logger.error("[poster] invalid max_reply_chars: %r", reply_cap)
            return False
        if (
            delivery == "single_message"
            and reply_cap > DISCORD_MAX_MESSAGE_CHARS
        ):
            logger.error(
                "[poster] single_message requires max_reply_chars <= 2000 "
                "(got %s)",
                reply_cap,
            )
            return False
        chunks = split_message(content, reply_cap)
        if delivery == "single_message" and len(chunks) != 1:
            logger.error(
                "[poster] single_message produced %d chunks", len(chunks)
            )
            return False
        if not chunks:
            return False

        batch_id = uuid.uuid4().hex[:16]
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            for index, chunk in enumerate(chunks):
                nonce = f"{batch_id}-{index}"
                result = await self.deliver_chunk(
                    channel_id=channel_id,
                    target_channel_id=target_channel_id,
                    chunk=chunk,
                    nonce=nonce,
                    client=client,
                )
                if not result.ok:
                    return False
        return True
