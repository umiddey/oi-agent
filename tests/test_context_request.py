"""Tests for chat-driven context window expansion (plan 20260920).

Covers the local intent gate, strict spec parsing, config clamping, the
runner's one-shot spec query, and the window-aware excerpt fetch.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pytest

from oi_agent.config import Config, WatchTarget, load_config, save_config
from oi_agent.opencode.runner import OpenCodeRunner
from oi_agent.watch.context_request import (
    DEFAULT_EXCERPT_LIMIT,
    ContextRequest,
    bounded_window,
    mentions_expansion,
    parse_explicit_request,
    parse_spec,
)
from oi_agent.watch.discord_client import _thread_excerpt


# ---------------------------------------------------------------- gate


@pytest.mark.parametrize(
    "question",
    [
        "read the last 100 messages and summarize",
        "what happened since 2 weeks ago?",
        "go back further, I need the earlier context",
        "scan the past 3 days of discussion",
        "check the last 5 replies about deploy",
        # date-form + typo asks (the estatehand feedback channel failures)
        "bro just messages frm 10th Sept 2026",
        "show me messages from 10 September",
        "anything since Sept 10?",
        "what did we say on 9/10?",
        "summarize 3 days ago",
    ],
)
def test_gate_matches_explicit_expansion_asks(question):
    assert mentions_expansion(question) is True


@pytest.mark.parametrize(
    "question",
    [
        "why did the deploy fail?",
        "summarize the thread",
        "audit the auth module",
        "review !12",
    ],
)
def test_gate_ignores_ordinary_questions(question):
    assert mentions_expansion(question) is False


def test_explicit_request_parses_exact_channel_phrase_without_model():
    request = parse_explicit_request(
        "try again to read the conversations from 10th Sept, it is about admin login",
        now=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    assert request == ContextRequest(since_date="2026-09-10")


def test_explicit_request_parses_count_and_relative_window():
    assert parse_explicit_request("read the last 100 messages") == ContextRequest(count=100)
    assert parse_explicit_request("read messages from 2 weeks ago") == ContextRequest(
        since_days=14
    )


# ---------------------------------------------------------------- parse_spec


def test_parse_spec_accepts_json_with_surrounding_prose():
    text = 'Sure! {"count": 100, "since_days": null} done.'
    assert parse_spec(text) == ContextRequest(count=100, since_days=None)


def test_parse_spec_rejects_missing_or_non_json():
    assert parse_spec(None) is None
    assert parse_spec("") is None
    assert parse_spec("count=100") is None
    assert parse_spec('["count", 100]') is None


def test_parse_spec_rejects_out_of_range_and_wrong_types():
    # Out-of-range or wrong-typed keys are dropped (fail-open at the caller,
    # which treats an all-null spec as "no widening").
    assert parse_spec('{"count": 0, "since_days": null}').count is None
    assert parse_spec('{"count": 999999, "since_days": null}').count is None
    assert parse_spec('{"count": "100", "since_days": null}').count is None
    assert parse_spec('{"count": true, "since_days": null}').count is None
    assert parse_spec('{"count": 100.7, "since_days": null}').count == 100


def test_parse_spec_defaults_missing_keys_to_none():
    assert parse_spec('{"count": 25}') == ContextRequest(count=25, since_days=None)


def test_parse_spec_accepts_iso_since_date():
    assert parse_spec('{"count": null, "since_days": null, "since_date": "2026-09-10"}') == (
        ContextRequest(since_date="2026-09-10")
    )
    assert parse_spec('{"since_date": "10-09-2026"}').since_date is None
    assert parse_spec('{"since_date": "2026-13-40"}').since_date is None
    assert parse_spec('{"since_date": 20260910}').since_date is None


def test_since_date_overrides_days_and_uses_day_start():
    request = ContextRequest(since_days=10, since_date="2026-09-10")
    limit, after = bounded_window(
        request, max_context_messages=200, max_context_age_days=30,
        now=datetime(2026, 9, 20, 22, 16, tzinfo=timezone.utc),
    )
    # Start of Sept 10, NOT now-minus-10-days (22:16 cutoff amputated the day).
    assert after == datetime(2026, 9, 10, 0, 0, tzinfo=timezone.utc)


def test_named_date_is_not_age_clamped():
    request = ContextRequest(since_date="2026-01-01")
    _, after = bounded_window(
        request, max_context_messages=200, max_context_age_days=30,
        now=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    assert after == datetime(2026, 1, 1, tzinfo=timezone.utc)




def test_date_window_uses_context_cap_without_explicit_count():
    limit, after = bounded_window(
        ContextRequest(since_date="2026-09-10"),
        max_context_messages=500,
        max_context_age_days=30,
    )
    assert limit == 500
    assert after == datetime(2026, 9, 10, tzinfo=timezone.utc)


def test_since_days_still_age_clamped():
    request = ContextRequest(since_days=1000)
    _, after = bounded_window(
        request, max_context_messages=200, max_context_age_days=30,
        now=datetime(2026, 9, 21, tzinfo=timezone.utc),
    )
    assert after == datetime(2026, 8, 22, tzinfo=timezone.utc)


# ---------------------------------------------------------------- clamp


def test_bounded_window_default_without_request():
    limit, after = bounded_window(None, max_context_messages=200, max_context_age_days=30)
    assert limit == DEFAULT_EXCERPT_LIMIT
    assert after is None


def test_bounded_window_clamps_to_config_caps():
    now = datetime(2026, 9, 20, 23, 0, tzinfo=timezone.utc)
    request = ContextRequest(count=10_000, since_days=1_000)
    limit, after = bounded_window(
        request,
        max_context_messages=200,
        max_context_age_days=30,
        now=now,
    )
    assert limit == 200
    assert after == now - timedelta(days=30)


def test_bounded_window_honors_undersized_count():
    request = ContextRequest(count=10, since_days=None)
    limit, _ = bounded_window(request, max_context_messages=200, max_context_age_days=30)
    assert limit == 10


# ---------------------------------------------------------------- runner


def _runner():
    return OpenCodeRunner(Config())


@pytest.mark.asyncio
async def test_infer_context_request_gate_miss_skips_model(monkeypatch):
    async def _explode(self, prompt, **kwargs):
        raise AssertionError("model must not be called without a gate hit")

    monkeypatch.setattr(OpenCodeRunner, "run_raw_prompt", _explode)
    assert await _runner().infer_context_request("why did the deploy fail?") is None


@pytest.mark.asyncio
async def test_infer_context_request_parses_model_json(monkeypatch):
    async def _reply(self, prompt, **kwargs):
        assert "since Monday" in prompt
        return '{"count":100,"since_days":14}'

    monkeypatch.setattr(OpenCodeRunner, "run_raw_prompt", _reply)
    request = await _runner().infer_context_request("please read since Monday")
    assert request == ContextRequest(count=100, since_days=14)


@pytest.mark.asyncio
async def test_infer_context_request_fails_open_on_garbage(monkeypatch):
    async def _garbage(self, prompt, **kwargs):
        return "I cannot help with that"

    monkeypatch.setattr(OpenCodeRunner, "run_raw_prompt", _garbage)
    assert await _runner().infer_context_request("please read since Monday") is None


@pytest.mark.asyncio
async def test_infer_context_request_fails_open_on_model_death(monkeypatch):
    async def _dead(self, prompt, **kwargs):
        return None

    monkeypatch.setattr(OpenCodeRunner, "run_raw_prompt", _dead)
    assert await _runner().infer_context_request("please read since Monday") is None


# ---------------------------------------------------------------- excerpt


@dataclass
class _Msg:
    message_id: int
    created_at: datetime
    author_name: str = "alice"
    attachments: list = field(default_factory=list)
    embeds: list = field(default_factory=list)
    content: str = "hello"

    @property
    def author(self):
        return type("A", (), {"display_name": self.author_name})()

    @property
    def id(self):
        return self.message_id


class _WindowChannel:
    """Fake channel whose history() honors datetime after/limit like discord.py."""

    def __init__(self, messages):
        self._messages = messages

    def history(self, limit=50, after=None, oldest_first=False):
        async def _gen():
            msgs = list(self._messages)
            if after is not None:
                msgs = [m for m in msgs if m.created_at > after]
            if limit is not None:
                msgs = msgs[:limit]
            for m in msgs:
                yield m

        return _gen()


@pytest.mark.asyncio
async def test_thread_excerpt_honors_limit_and_after():
    now = datetime.now(timezone.utc)
    messages = [_Msg(i, now - timedelta(hours=i)) for i in range(1, 9)]
    channel = _WindowChannel(messages)

    default_text = await _thread_excerpt(channel, bot_user_id=99)
    assert len(default_text.splitlines()) == 8

    widened = await _thread_excerpt(channel, 4, bot_user_id=99)
    assert len(widened.splitlines()) == 4

    cutoff = now - timedelta(hours=3)
    windowed = await _thread_excerpt(channel, 8, bot_user_id=99, after=cutoff)
    lines = windowed.splitlines()
    assert 0 < len(lines) <= 3
    assert all(str(m.message_id) not in windowed for m in messages[3:])


class _DirectionalChannel:
    """Fake channel honoring oldest_first like discord.py does server-side."""

    def __init__(self, messages):
        self._messages = sorted(messages, key=lambda m: m.message_id, reverse=True)

    def history(self, limit=50, after=None, oldest_first=False):
        async def _gen():
            msgs = sorted(self._messages, key=lambda m: m.message_id, reverse=not oldest_first)
            if after is not None:
                msgs = [m for m in msgs if m.created_at > after]
            if limit is not None:
                msgs = msgs[:limit]
            for m in msgs:
                yield m

        return _gen()


@pytest.mark.asyncio
async def test_since_window_returns_earliest_messages():
    """'from Sept 10 onwards' semantics: oldest messages in the window win.

    Regression for the estatehand failure: newest-first within a date window
    returns recent chatter and drops the requested start whenever the
    channel exceeds the cap.
    """
    now = datetime.now(timezone.utc)
    # 8 messages, one per day, oldest = id 1.
    messages = [_Msg(i, now - timedelta(days=8 - i)) for i in range(1, 9)]
    channel = _DirectionalChannel(messages)
    cutoff = now - timedelta(days=6)

    text = await _thread_excerpt(channel, 3, bot_user_id=99, after=cutoff)
    ids = [line.split("]")[0][1:] for line in text.splitlines()]
    assert ids == ["3", "4", "5"], "must return the EARLIEST window messages"

    # Count-only ask keeps newest semantics.
    recent = await _thread_excerpt(channel, 3, bot_user_id=99)
    ids = [line.split("]")[0][1:] for line in recent.splitlines()]
    assert ids == ["6", "7", "8"]


# ---------------------------------------------------------------- memory


def test_context_request_memory_remember_and_reuse():
    from oi_agent.watch.context_request import ContextRequestMemory

    memory = ContextRequestMemory(ttl_seconds=900)
    key = "conv:1:/repo"
    assert memory.recall(key, now=100.0) is None

    memory.remember(key, ContextRequest(count=None, since_days=10), now=100.0)
    assert memory.recall(key, now=500.0) == ContextRequest(count=None, since_days=10)

    # expired
    assert memory.recall(key, now=1000.0 + 901) is None


def test_context_request_memory_ignores_default_requests():
    from oi_agent.watch.context_request import ContextRequestMemory

    memory = ContextRequestMemory()
    memory.remember("k", ContextRequest())
    assert memory.recall("k") is None


# ---------------------------------------------------------------- config


def _valid_config() -> Config:
    cfg = Config(
        opencode_binary=shutil.which("true") or "/bin/true",
        opencode_model="provider/model",
    )
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/srv/repo")]
    return cfg


def test_config_round_trips_context_caps(tmp_path):
    cfg = _valid_config()
    cfg.max_context_messages = 500
    cfg.max_context_age_days = 14
    path = tmp_path / "config.toml"

    save_config(cfg, path)
    back = load_config(path)
    assert back.max_context_messages == 500
    assert back.max_context_age_days == 14


def test_config_rejects_context_caps_out_of_bounds(tmp_path):
    for key, value in (
        ("max_context_messages", 5_000),
        ("max_context_messages", 10),
        ("max_context_age_days", 0),
        ("max_context_age_days", 400),
    ):
        path = tmp_path / "config.toml"
        path.write_text(f"{key} = {value}\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_config(path)


def test_config_defaults_match_plan():
    cfg = Config()
    assert cfg.max_context_messages == 200
    assert cfg.max_context_age_days == 30
