"""Tests for config load/save roundtrip and validation."""

import pytest

from oi_agent.config import (
    Config,
    LLMConfig,
    WatchTarget,
    effective_max_tool_iterations,
    load_config,
    save_config,
)


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_roundtrip(tmp_path):
    cfg = Config()
    cfg.personality = 'Warm "expert"\nUse bullets.'
    cfg.max_reply_chars = 10_000
    cfg.max_response_tokens = 8_192
    cfg.fallback = LLMConfig("http://a/v1", "m1", "K1")
    cfg.agent = LLMConfig("http://b/v1", "m2", "K2", disable_thinking=True)
    cfg.watches = [
        WatchTarget(channel_id=111, repo_path="/srv/repo",
                    founder_ids=[234567890123456789], bot_user_id=42,
                    post_hourly_cap=5),
        WatchTarget(channel_id=333, repo_path="/srv/other"),
    ]
    p = tmp_path / "config.toml"
    save_config(cfg, p)
    back = load_config(p)
    assert back.max_reply_chars == 10_000
    assert back.max_response_tokens == 8_192
    assert back.personality == 'Warm "expert"\nUse bullets.'
    w = back.watches[0]
    assert (w.channel_id, w.bot_user_id, w.post_hourly_cap) == (111, 42, 5)
    assert w.founder_ids == [234567890123456789]
    assert back.agent.disable_thinking is True
    assert back.fallback.disable_thinking is False
    assert back.target_for_channel(333).repo_path == "/srv/other"
    # Thread id lookup misses -> None handled by watcher via parent.
    assert back.target_for_channel(999999) is None


def test_watch_vocab_roundtrip(tmp_path):
    """search_aliases/preferred_tokens/search_stopwords survive save/load."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=1, repo_path="/srv/repo",
                    search_aliases={"kpi": ["metric", "dashboard"]},
                    preferred_tokens=["billing", "invoice"],
                    search_stopwords=["scratch", "archive"]),
    ]
    p = tmp_path / "config.toml"
    save_config(cfg, p)

    back = load_config(p)

    w = back.watches[0]
    assert w.search_aliases == {"kpi": ["metric", "dashboard"]}
    assert w.preferred_tokens == ["billing", "invoice"]
    assert w.search_stopwords == ["scratch", "archive"]


def test_watch_vocab_defaults_are_neutral(tmp_path):
    """A watch entry without vocabulary keys gets empty (neutral) defaults."""
    p = _write(tmp_path / "c.toml",
               '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n')

    cfg = load_config(p)

    w = cfg.watches[0]
    assert w.search_aliases == {}
    assert w.preferred_tokens == []
    assert w.search_stopwords == []


def test_missing_watch_field_rejected(tmp_path):
    p = _write(tmp_path / "c.toml", '[[watch]]\nchannel_id = 1\n')
    with pytest.raises(ValueError):
        load_config(p)


def test_no_watches_rejected(tmp_path):
    p = _write(tmp_path / "c.toml", "")
    with pytest.raises(ValueError):
        load_config(p)

def test_save_rejects_config_without_watches(tmp_path):
    """Every successfully saved config must be loadable by the same model."""
    path = tmp_path / "config.toml"

    with pytest.raises(ValueError, match="watch"):
        save_config(Config(), path)

    assert not path.exists()


@pytest.mark.parametrize("aliases", [
    '{ foo = "bar" }',
    '{ foo = [1, 2] }',
])
def test_malformed_search_alias_shapes_rejected(tmp_path, aliases):
    """Alias expansions must be TOML arrays of strings before normalization."""
    path = _write(
        tmp_path / "aliases.toml",
        "[[watch]]\n"
        "channel_id = 1\n"
        'repo_path = "/repo"\n'
        f"search_aliases = {aliases}\n",
    )

    with pytest.raises(ValueError, match="search_aliases"):
        load_config(path)


def test_max_response_tokens_defaults_when_absent(tmp_path):
    """A config without the key falls back to the 4000-token default."""
    p = _write(tmp_path / "c.toml",
               '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n')

    cfg = load_config(p)

    assert cfg.max_response_tokens == 4_000


@pytest.mark.parametrize("bad_toml", [
    "personality = [1]\n[[watch]]\nchannel_id = 1\nrepo_path = \"/r\"\n",
    "[[watch]]\nchannel_id = 1\nrepo_path = 42\n",
    "[[watch]]\nchannel_id = 1\nrepo_path = \"/r\"\nbot_user_id = true\n",
    "[[watch]]\nchannel_id = 1\nrepo_path = \"/r\"\nfounder_ids = [true]\n",
    "[agent]\ndisable_thinking = \"false\"\n[[watch]]\nchannel_id = 1\n"
    "repo_path = \"/r\"\n",
    # Collection types must not silently coerce: bare string -> not char list,
    # and non-string list elements are rejected.
    "[[watch]]\nchannel_id = 1\nrepo_path = \"/r\"\nclaim_keywords = \"bug\"\n",
    "[[watch]]\nchannel_id = 1\nrepo_path = \"/r\"\npreferred_tokens = [1]\n",
])
def test_malformed_toml_types_are_rejected(tmp_path, bad_toml):
    """Raw TOML types must be validated, not silently coerced."""
    p = _write(tmp_path / "c.toml", bad_toml)
    with pytest.raises(ValueError):
        load_config(p)


def test_save_survives_quotes_and_apostrophes(tmp_path):
    """Regression: hand-rolled TOML quoting once bricked configs.

    A quote in base_url or an apostrophe in a keyword must round-trip through
    save_config/load_config without producing unloadable TOML.
    """
    cfg = Config()
    cfg.agent = LLMConfig(base_url='https://x/"bad', model='m',
                          api_key_env="K")
    cfg.watches = [
        WatchTarget(channel_id=1, repo_path='/srv/o"brien',
                    claim_keywords=["it's", "bug's"]),
    ]
    p = tmp_path / "config.toml"

    save_config(cfg, p)

    back = load_config(p)  # must not raise TOMLDecodeError
    assert back.agent.base_url == 'https://x/"bad'
    assert back.watches[0].repo_path == '/srv/o"brien'
    assert back.watches[0].claim_keywords == ["it's", "bug's"]


def test_save_creates_missing_parent_dirs(tmp_path):
    """First-run regression: `oi init` on a fresh machine has no ~/.config/oi.

    The atomic-write change briefly dropped parent creation, so the tmp
    sibling write raised FileNotFoundError. The destination directory must
    be created before the temporary file is written.
    """
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=1, repo_path="/srv/repo")]
    nested = tmp_path / "missing" / "oi" / "config.toml"

    save_config(cfg, nested)  # must not raise

    assert load_config(nested).watches[0].channel_id == 1
    # No temp litter left behind.
    assert not (tmp_path / "missing" / "oi" / "config.toml.tmp").exists()


def test_invalid_on_disk_config_rejected_on_load(tmp_path):
    """Manual edits bypass save_config; the loader must still gate them.

    Reproduces the review's exact TOML: tiny limits, bogus wire style, and a
    negative post cap (which would silently block every reservation).
    """
    p = _write(
        tmp_path / "bad.toml",
        'max_reply_chars = 1\n'
        'max_response_tokens = 1\n\n'
        '[agent]\n'
        'base_url = "http://x/v1"\n'
        'model = "m"\n'
        'api_style = "bogus"\n\n'
        '[[watch]]\n'
        'channel_id = 1\n'
        'repo_path = "/repo"\n'
        'post_hourly_cap = -5\n',
    )

    with pytest.raises(ValueError):
        load_config(p)


def test_negative_bot_user_id_and_bad_vocab_rejected(tmp_path):
    p = _write(
        tmp_path / "bad2.toml",
        '[[watch]]\n'
        'channel_id = 1\n'
        'repo_path = "/repo"\n'
        'bot_user_id = -7\n'
        'search_stopwords = ["ok", ""]\n',
    )

    with pytest.raises(ValueError):
        load_config(p)


def test_max_tool_iterations_roundtrip(tmp_path):
    """Root knob always saved; watch override saved only when set."""
    cfg = Config()
    cfg.max_tool_iterations = 9
    cfg.watches = [
        WatchTarget(channel_id=1, repo_path="/srv/a", max_tool_iterations=12),
        WatchTarget(channel_id=2, repo_path="/srv/b"),
    ]
    p = tmp_path / "config.toml"
    save_config(cfg, p)

    text = p.read_text()
    back = load_config(p)

    assert back.max_tool_iterations == 9
    assert back.watches[0].max_tool_iterations == 12
    assert back.watches[1].max_tool_iterations is None
    # Root line + exactly one watch line (the None watch omits the key).
    assert text.count("max_tool_iterations") == 2


def test_max_tool_iterations_defaults_when_absent(tmp_path):
    """A config without the keys falls back to root 4 / inherit None."""
    p = _write(
        tmp_path / "c.toml",
        '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n',
    )

    cfg = load_config(p)

    assert cfg.max_tool_iterations == 4
    assert cfg.watches[0].max_tool_iterations is None
    assert effective_max_tool_iterations(cfg, cfg.watches[0]) == 4


@pytest.mark.parametrize("bad_toml", [
    'max_tool_iterations = 0\n',
    'max_tool_iterations = -1\n',
    'max_tool_iterations = 51\n',
    'max_tool_iterations = "5"\n',
    'max_tool_iterations = true\n',
])
def test_bad_root_max_tool_iterations_rejected_on_load(tmp_path, bad_toml):
    """The loader gates out-of-range and non-int root values."""
    p = _write(
        tmp_path / "c.toml",
        bad_toml + '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n',
    )

    with pytest.raises(ValueError):
        load_config(p)


@pytest.mark.parametrize("good_value", [1, 50])
def test_max_tool_iterations_boundary_accepted(tmp_path, good_value):
    """Both ends of the [1, 50] range load without error."""
    p = _write(
        tmp_path / "c.toml",
        f'max_tool_iterations = {good_value}\n'
        '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n',
    )

    cfg = load_config(p)

    assert cfg.max_tool_iterations == good_value


@pytest.mark.parametrize("bad_value", [0, -3, 51, "5", True, 2.5])
def test_bad_root_max_tool_iterations_rejected_on_save(tmp_path, bad_value):
    """save_config refuses invalid root values and writes nothing."""
    cfg = Config()
    cfg.max_tool_iterations = bad_value
    cfg.watches = [WatchTarget(channel_id=1, repo_path="/srv/a")]
    p = tmp_path / "config.toml"

    with pytest.raises(ValueError):
        save_config(cfg, p)
    assert not p.exists()


@pytest.mark.parametrize("bad_toml", [
    'max_tool_iterations = 0\n',
    'max_tool_iterations = -2\n',
    'max_tool_iterations = 51\n',
    'max_tool_iterations = "many"\n',
    'max_tool_iterations = false\n',
])
def test_bad_watch_max_tool_iterations_rejected_on_load(tmp_path, bad_toml):
    """A SET watch value must be an int within [1, 50]."""
    p = _write(
        tmp_path / "c.toml",
        '[[watch]]\nchannel_id = 1\nrepo_path = "/repo"\n' + bad_toml,
    )

    with pytest.raises(ValueError):
        load_config(p)


@pytest.mark.parametrize("bad_value", [0, 51, "5", False])
def test_bad_watch_max_tool_iterations_rejected_on_save(tmp_path, bad_value):
    """save_config refuses a SET watch value outside [1, 50] or non-int."""
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=1, repo_path="/srv/a",
                    max_tool_iterations=bad_value),
    ]
    p = tmp_path / "config.toml"

    with pytest.raises(ValueError):
        save_config(cfg, p)
    assert not p.exists()


def test_effective_max_tool_iterations_override_and_inherit():
    """Watch override wins when set; None inherits the global default."""
    cfg = Config(max_tool_iterations=6)
    overridden = WatchTarget(channel_id=1, repo_path="/a",
                             max_tool_iterations=15)
    inheriting = WatchTarget(channel_id=2, repo_path="/b")

    assert effective_max_tool_iterations(cfg, overridden) == 15
    assert effective_max_tool_iterations(cfg, inheriting) == 6
