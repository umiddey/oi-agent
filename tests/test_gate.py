"""Tests for the deterministic gate."""

from oi_agent.watch.gate import Action, evaluate


def test_bot_mention_responds():
    d = evaluate("hey check this", [42, 99], author_id=1,
                 bot_user_id=99, founder_ids=[], claim_keywords=["bug"])
    assert d.action == Action.RESPOND


def test_founder_message_responds():
    d = evaluate("hello there", [], author_id=7,
                 bot_user_id=99, founder_ids=[7], claim_keywords=[])
    assert d.action == Action.RESPOND


def test_claim_keyword_triggers_response():
    d = evaluate("the migration is deployed right?", [], author_id=5,
                 bot_user_id=99, founder_ids=[7],
                 claim_keywords=["migration"])
    assert d.action == Action.RESPOND


def test_chatter_ignored():
    d = evaluate("lol nice 😂", [], author_id=5,
                 bot_user_id=99, founder_ids=[7], claim_keywords=["bug"])
    assert d.action == Action.IGNORE
