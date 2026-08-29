"""Tests for the deterministic explicit-trigger gate."""

from oi_agent.watch.gate import Action, evaluate


def test_bot_mention_responds():
    """A direct bot mention admits the message."""
    d = evaluate("hey check this", [42, 99], bot_user_id=99)
    assert d.action == Action.RESPOND
    assert d.reason == "bot mentioned"


def test_oi_command_responds():
    """The command-style prefix admits a message without a mention."""
    d = evaluate("  OI: inspect the migration", [], bot_user_id=99)
    assert d.action == Action.RESPOND
    assert d.reason == "oi command"


def test_explicit_markers_respond():
    """Supported hashtags admit a message regardless of author."""
    for marker in ("#oi", "#audit", "#bugreport"):
        d = evaluate(f"please inspect this {marker}", [], bot_user_id=99)
        assert d.action == Action.RESPOND
        assert marker in d.reason


def test_founder_message_without_trigger_is_ignored():
    """Author identity no longer turns ordinary chatter into a trigger."""
    d = evaluate("hello there", [], bot_user_id=99)
    assert d.action == Action.IGNORE


def test_claim_keyword_without_trigger_is_ignored():
    """Domain words alone do not trigger an autonomous reply."""
    d = evaluate("the migration is deployed right?", [], bot_user_id=99)
    assert d.action == Action.IGNORE


def test_chatter_ignored():
    """Unmarked casual chat stays silent."""
    d = evaluate("lol nice", [], bot_user_id=99)
    assert d.action == Action.IGNORE
