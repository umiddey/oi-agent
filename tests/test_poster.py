"""Tests for configurable Discord-safe reply chunking."""

from types import SimpleNamespace

import httpx
import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.poster import Poster, _is_first_chunk, split_message
from oi_agent.store import Store


def test_split_message_respects_discord_limit_and_total_cap():
    """Configured long replies split without exceeding 2,000 characters."""
    content = ("word " * 1_800).strip()

    chunks = split_message(content, 10_000)

    assert len(chunks) > 1
    assert all(len(chunk) <= 2_000 for chunk in chunks)
    assert "".join(chunks).replace(" ", "") == content.replace(" ", "")


def test_split_message_applies_total_cap():
    """The configured total cap truncates before Discord chunking."""
    chunks = split_message("x" * 10_000, 4_500)

    assert sum(len(chunk) for chunk in chunks) == 4_500
    assert all(len(chunk) <= 2_000 for chunk in chunks)


@pytest.mark.asyncio
async def test_poster_emits_chunked_reply_and_one_cap_unit(tmp_path, monkeypatch):
    """Chunked delivery sends the full logical cap in Discord-safe chunks."""
    cfg = Config(max_reply_chars=4_500)
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=1)
    ]
    store = Store(tmp_path / "state.db")
    calls = []

    class FakeClient:
        """Async HTTP client returning successful Discord responses."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter the fake client."""
            return self

        async def __aexit__(self, *args):
            """Leave the fake client."""
            return False

        async def post(self, url, headers, json):
            """Capture one Discord message request."""
            calls.append((json["content"], json.get("allowed_mentions")))
            return SimpleNamespace(
                status_code=200, text="", json=lambda: {"id": "1001"}
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")
    content = "x" * 4_500

    assert await poster.send(111, 111, content) is True
    assert [len(content) for content, _ in calls] == [2_000, 2_000, 500]
    assert all(mentions == {"parse": []} for _, mentions in calls)
    assert "".join(content for content, _ in calls) == content
    # One reply = one unit of the hourly cap, and cap 1 is now exhausted.
    assert store.posts_in_last_hour(111) == 1
    assert await poster.send(111, 111, "next reply") is False


@pytest.mark.asyncio
async def test_poster_rejects_direct_single_message_cap(
    tmp_path, monkeypatch, caplog
):
    """Directly constructed strict configs cannot bypass the cap invariant."""
    cfg = Config(max_reply_chars=4_500, reply_delivery="single_message")
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo")]
    store = Store(tmp_path / "state.db")

    class FakeClient:
        """Async HTTP client that would expose an unexpected send."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter the fake client."""
            return self

        async def __aexit__(self, *args):
            """Leave the fake client."""
            return False

        async def post(self, url, headers, json):
            """Fail if invalid configuration reaches the network."""
            raise AssertionError("invalid config must not send")

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    assert await poster.send(111, 111, "reply") is False
    assert "single_message" in caplog.text
    assert store.posts_in_last_hour(111) == 0


def test_reserve_post_is_atomic_against_the_cap(tmp_path):
    """Regression: two reply attempts cannot both pass a cap of one.

    The old check-then-record flow let both through (reproduced in review);
    reserve_post must reject the second reservation.
    """
    store = Store(tmp_path / "state.db")

    assert store.reserve_post(111, cap=1) is True
    assert store.reserve_post(111, cap=1) is False
    assert store.posts_in_last_hour(111) == 1


def test_reserve_post_atomic_across_concurrent_connections(tmp_path):
    """Two independent DB connections racing a cap of 1: exactly one wins.

    This is the real concurrency test — separate Store connections (as two
    daemon events would use) synchronized on a barrier. BEGIN IMMEDIATE must
    serialize the check+insert so only one reservation succeeds.
    """
    import threading

    db = tmp_path / "state.db"
    Store(db).close()  # ensure schema exists before the racers open it

    barrier = threading.Barrier(8)
    results: list[bool] = []
    guard = threading.Lock()

    def racer():
        store = Store(db)
        barrier.wait()  # release all racers at once
        ok = store.reserve_post(111, cap=1)
        with guard:
            results.append(ok)
        store.close()

    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert sum(results) == 1, f"expected exactly one winner, got {results}"
    assert Store(db).posts_in_last_hour(111) == 1


@pytest.mark.asyncio
async def test_deliver_chunk_success_with_nonce_and_reply_reference(
    tmp_path, monkeypatch
):
    """deliver_chunk formats payload with nonce and message reference on 200/201."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")
    captured = {}

    class FakeClient:
        """Async HTTP client recording request arguments."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, json):
            """Capture post payload and return created message ID."""
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return SimpleNamespace(
                status_code=201,
                text='{"id": "9876543210"}',
                json=lambda: {"id": "9876543210"},
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "test_token")

    result = await poster.deliver_chunk(
        channel_id=222,
        target_channel_id=111,
        chunk="chunk payload",
        nonce="batch123-0",
        reply_to_message_id=444,
    )

    assert result.ok is True
    assert result.discord_message_id == 9876543210
    assert result.status_code == 201
    assert captured["url"] == "https://discord.com/api/v10/channels/222/messages"
    assert captured["headers"]["Authorization"] == "Bot test_token"
    assert captured["json"] == {
        "content": "chunk payload",
        "nonce": "batch123-0",
        "enforce_nonce": True,
        "allowed_mentions": {"parse": []},
        "message_reference": {"message_id": 444},
    }
    assert store.posts_in_last_hour(111) == 1


@pytest.mark.asyncio
async def test_deliver_chunk_rate_limit_429_retry_success(
    tmp_path, monkeypatch
):
    """deliver_chunk backs off and retries on 429 before succeeding."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")
    calls = 0

    class FakeClient:
        """Async HTTP client returning 429 once then 200."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, json):
            """Return 429 on first call, 200 on second."""
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(
                    status_code=429,
                    headers={"content-type": "application/json"},
                    text='{"retry_after": 0.01}',
                    json=lambda: {"retry_after": 0.01},
                )
            return SimpleNamespace(
                status_code=200,
                headers={"content-type": "application/json"},
                text='{"id": "55555"}',
                json=lambda: {"id": "55555"},
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-0",
    )

    assert result.ok is True
    assert result.discord_message_id == 55555
    assert calls == 2


@pytest.mark.asyncio
async def test_deliver_chunk_rate_limit_429_exhausted(tmp_path, monkeypatch):
    """deliver_chunk returns rate_limited DeliveryResult when retries exhaust."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")
    calls = 0

    class FakeClient:
        """Async HTTP client always returning 429."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, json):
            """Always return 429."""
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                status_code=429,
                headers={"retry-after": "0.01"},
                text='{"retry_after": 0.01}',
                json=lambda: {"retry_after": 0.01},
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-0",
    )

    assert result.ok is False
    assert result.error_class == "rate_limited"
    assert result.status_code == 429
    assert result.retry_after == pytest.approx(0.01, rel=1e-2)
    assert calls == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "expected_error_class"),
    [
        (403, "permission_denied"),
        (404, "channel_not_found"),
        (500, "server_error"),
        (502, "server_error"),
        (503, "server_error"),
    ],
)
async def test_deliver_chunk_error_classification(
    tmp_path, monkeypatch, status_code, expected_error_class
):
    """deliver_chunk classifies HTTP failure status codes into bounded error categories."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")

    class FakeClient:
        """Async HTTP client returning parameterized error status."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, json):
            """Return configured status code."""
            return SimpleNamespace(
                status_code=status_code,
                headers={},
                text=f"Error {status_code}",
                json=lambda: {},
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-0",
    )

    assert result.ok is False
    assert result.error_class == expected_error_class
    assert result.status_code == status_code


@pytest.mark.asyncio
async def test_deliver_chunk_network_error(tmp_path, monkeypatch):
    """deliver_chunk returns network_error when client raises a connection exception."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")

    class FakeClient:
        """Async HTTP client raising network error."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def get(self, url, headers=None):
            """Simulate GET network failure during reconciliation."""
            raise httpx.ConnectError("GET unreachable")

        async def post(self, url, headers, json):
            """Raise network exception on POST."""
            raise httpx.ConnectError("Network unreachable")
    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-0",
    )

    assert result.ok is False
    assert result.error_class == "network_error"


@pytest.mark.asyncio
async def test_deliver_chunk_allowlist_and_pause_enforcement(tmp_path):
    """deliver_chunk rejects un-watchlisted targets and paused daemon state."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)
    ]
    store = Store(tmp_path / "state.db")
    poster = Poster(cfg, store, "token")

    # Un-watchlisted target
    unwatch_result = await poster.deliver_chunk(
        channel_id=999,
        target_channel_id=999,
        chunk="payload",
        nonce="nonce-0",
    )
    assert unwatch_result.ok is False
    assert unwatch_result.error_class == "target_not_found"

    # Paused daemon state
    store.set_paused(True)
    paused_result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-0",
    )
    assert paused_result.ok is False
    assert paused_result.error_class == "paused"


@pytest.mark.asyncio
async def test_deliver_chunk_hourly_cap_and_multi_chunk_accounting(
    tmp_path, monkeypatch
):
    """deliver_chunk reserves 1 hourly cap unit on chunk 1, without double-counting chunk 2."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=1)
    ]
    store = Store(tmp_path / "state.db")

    class FakeClient:
        """Async HTTP client returning success."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, json):
            """Return success."""
            return SimpleNamespace(
                status_code=200,
                headers={},
                text='{"id": "1"}',
                json=lambda: {"id": "1"},
            )

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    # First chunk of batch 1 -> reserves 1 cap slot
    res1 = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="chunk 1",
        nonce="batch1-0",
    )
    assert res1.ok is True
    assert store.posts_in_last_hour(111) == 1

    # Second chunk of batch 1 -> does not reserve an extra cap slot
    res2 = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="chunk 2",
        nonce="batch1-1",
    )
    assert res2.ok is True
    assert store.posts_in_last_hour(111) == 1

    # First chunk of batch 2 -> blocked because hourly cap (1) is exhausted
    res3 = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="chunk 1",
        nonce="batch2-0",
    )
    assert res3.ok is False
    assert res3.error_class == "hourly_cap_exceeded"
    assert store.posts_in_last_hour(111) == 1


def test_is_first_chunk_helper_logic():
    """_is_first_chunk recognizes 0-indexed nonces as first chunk, and higher indices as subsequent chunks."""
    assert _is_first_chunk("batch-0") is True
    assert _is_first_chunk("0123456789abcdef-0") is True
    assert _is_first_chunk("batch-1") is False
    assert _is_first_chunk("batch-2") is False
    assert _is_first_chunk("unindexed_nonce") is True
    assert _is_first_chunk("batch-abc") is True


@pytest.mark.asyncio
async def test_deliver_chunk_reconciles_unconfirmed_post_by_nonce(tmp_path, monkeypatch):
    """When POST fails on network error, deliver_chunk reconciles delivery if message is found by nonce."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)]
    store = Store(tmp_path / "state.db")

    class ReconcilingClient:
        def __init__(self, *args, **kwargs):
            self.post_called = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            if not self.post_called:
                # Initial pre-check: not found yet
                return SimpleNamespace(status_code=200, json=lambda: [])
            # After POST failed: server actually created it
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"id": "777888", "nonce": "nonce-reconcile-0", "content": "payload"}
                ],
            )

        async def post(self, url, headers, json):
            self.post_called = True
            raise httpx.ConnectError("Connection reset by peer after message commit")

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", ReconcilingClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="payload",
        nonce="nonce-reconcile-0",
    )

    assert result.ok is True
    assert result.discord_message_id == 777888
    assert result.status_code == 200


@pytest.mark.asyncio
async def test_deliver_chunk_reconciles_unconfirmed_post_by_content_and_reference(
    tmp_path, monkeypatch
):
    """When POST fails, deliver_chunk reconciles delivery by content and reply_to message reference."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)]
    store = Store(tmp_path / "state.db")

    class ContentReconcilingClient:
        def __init__(self, *args, **kwargs):
            self.post_called = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            if not self.post_called:
                return SimpleNamespace(status_code=200, json=lambda: [])
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {
                        "id": "999111",
                        "content": "chunk payload",
                        "message_reference": {"message_id": "444"},
                    }
                ],
            )

        async def post(self, url, headers, json):
            self.post_called = True
            raise httpx.TimeoutException("Read timeout waiting for response")

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", ContentReconcilingClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="chunk payload",
        nonce="nonce-ref-0",
        reply_to_message_id=444,
    )

    assert result.ok is True
    assert result.discord_message_id == 999111


@pytest.mark.asyncio
async def test_deliver_chunk_skips_post_if_already_in_channel_history(tmp_path, monkeypatch):
    """If chunk is already confirmed in channel history (e.g. late worker retry), deliver_chunk skips POST."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=3)]
    store = Store(tmp_path / "state.db")
    post_attempts = []

    class AlreadyDeliveredClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url, headers=None):
            return SimpleNamespace(
                status_code=200,
                json=lambda: [
                    {"id": "555666", "nonce": "existing-nonce-0", "content": "already delivered"}
                ],
            )

        async def post(self, url, headers, json):
            post_attempts.append(json)
            return SimpleNamespace(status_code=200, json=lambda: {"id": "999999"})

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", AlreadyDeliveredClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_chunk(
        channel_id=111,
        target_channel_id=111,
        chunk="already delivered",
        nonce="existing-nonce-0",
    )

    assert result.ok is True
    assert result.discord_message_id == 555666
    assert len(post_attempts) == 0
