"""Tests for configurable responder personality."""

from oi_agent.agent.responder import build_system_prompt


def test_personality_changes_tone_without_overriding_rules():
    """Configured voice is injected while fixed safety rules remain present."""
    prompt = build_system_prompt("Warm mentor; explain jargon with examples.")
    assert "Warm mentor; explain jargon with examples." in prompt
    assert "Answer only from the provided thread" in prompt
    assert "never override the evidence and safety rules" in prompt


def test_empty_personality_uses_witty_default_voice():
    """No personality falls back to the sharp-witty senior-dev voice."""
    prompt = build_system_prompt("")
    assert "sharp, witty senior engineer" in prompt
    # And a configured voice replaces that default entirely.
    custom = build_system_prompt("Warm mentor.")
    assert "sharp, witty senior engineer" not in custom


def test_sha_is_embedded_for_natural_closing():
    """The audited sha is woven into the rules so the model can close in-voice."""
    prompt = build_system_prompt("", sha="cca5066c6")
    assert "cca5066c6" in prompt
    assert "mechanical stamp" in prompt
