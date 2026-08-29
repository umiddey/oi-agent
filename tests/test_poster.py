"""Tests for configurable Discord-safe reply chunking."""

from types import SimpleNamespace

import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.poster import Poster, split_message
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
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo",
                               post_hourly_cap=1)]
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
            return SimpleNamespace(status_code=200, text="")

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
async def test_poster_rejects_direct_single_message_cap(tmp_path, monkeypatch,
                                                        caplog):
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
