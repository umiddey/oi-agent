"""TOML configuration model and init/run helpers.

Configuration supports any number of watch targets; each `[[watch]]` entry
binds one Discord channel to one repository plus its own rules.
"""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "oi" / "config.toml"
DEFAULT_SECRETS_PATH = Path.home() / ".config" / "oi" / ".env"

DEFAULT_PERSONALITY = (
    "A sharp, witty senior engineer. Dry humor, opinionated, direct. Call out "
    "bad ideas instead of nodding along — but wit never replaces evidence. Talk "
    "like a smart teammate in chat, not a report generator."
)

def load_secrets(path: Path = DEFAULT_SECRETS_PATH) -> None:
    """Load KEY=VALUE pairs into os.environ without overriding existing vars.

    Called by commands that talk to Discord/LLMs so users only paste secrets
    once during `oi init`.

    Args:
        path: Secrets file location.
    """
    import os

    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


@dataclass
class LLMConfig:
    """One LLM endpoint used for the agent or fallback response stage.

    Attributes:
        base_url: Provider root, e.g. https://llm.example/v1 or https://api.z.ai.
        model: Model identifier to request.
        api_key_env: Env var name holding the key (never the key itself).
        api_style: Wire protocol, ``openai`` or ``anthropic``.
        disable_thinking: When True, ask reasoning models to skip the hidden
            reasoning pass (vLLM/Qwen ``chat_template_kwargs.enable_thinking``)
            so the answer lands in ``content`` instead of the reasoning field.
    """

    base_url: str
    model: str
    api_key_env: str = "OI_LLM_API_KEY"
    api_style: str = "openai"
    disable_thinking: bool = False


@dataclass
class WatchTarget:
    """One watched Discord channel bound to one repository.

    Attributes:
        channel_id: Discord channel id to watch.
        repo_path: Local watched clone this channel's questions audit (OI
            runs git pull and writes .oi/ there; see README safety model).
        founder_ids: Author ids whose messages always pass the gate.
        bot_user_id: Bot application user id considered 'mentioned' in tags.
        post_hourly_cap: Max autonomous replies per trailing hour.
        claim_keywords: Lowercase substrings that force a response.
        search_stopwords: Extra domain words excluded from evidence search,
            merged over the built-in neutral filler-word list.
        search_aliases: Per-domain term expansion for repo search, e.g.
            {"kpi": ["metric"]} — keeps domain vocabulary in config
            instead of hardcoding one deployment's language into the pipeline.
        preferred_tokens: Path substrings that rank evidence files higher.
        max_tool_iterations: Per-watch agentic tool-loop depth override;
            ``None`` inherits the global :attr:`Config.max_tool_iterations`.
    """

    channel_id: int
    repo_path: str
    founder_ids: list[int] = field(default_factory=list)
    bot_user_id: int = 0
    post_hourly_cap: int = 3
    search_aliases: dict[str, list[str]] = field(default_factory=dict)
    preferred_tokens: list[str] = field(default_factory=list)
    search_stopwords: list[str] = field(default_factory=list)
    max_tool_iterations: int | None = None
    claim_keywords: list[str] = field(
        default_factory=lambda: [
            "plan", "commit", "migration", "deployed", "deploy",
            "broken", "bug", "merged", "release", "model", "schema",
        ]
    )


@dataclass
class Config:
    """Full daemon configuration.

    Attributes:
        discord_token_env: Env var holding the bot token.
        db_path: State database path.
        personality: User-selected response tone; safety/evidence rules remain
            authoritative.
        max_reply_chars: Total reply cap before Discord-safe chunking.
        max_response_tokens: LLM output-token budget for a repo-audit answer.
            Reasoning models spend part of this on hidden reasoning, so keep it
            comfortably above the visible answer length.
        max_tool_iterations: Global cap on agentic tool-loop rounds per audit
            (each assistant turn requesting tool calls is one round).
        fallback: Small-model endpoint used only after main-model failure.
        agent: Main-model endpoint config.
        watches: All watch targets.
    """

    discord_token_env: str = "OI_DISCORD_TOKEN"
    db_path: str = "~/.local/state/oi/state.db"
    personality: str = DEFAULT_PERSONALITY
    max_reply_chars: int = 2_000
    max_response_tokens: int = 4_000
    fallback: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            base_url="", model="", api_key_env="OI_FALLBACK_API_KEY"
        )
    )
    agent: LLMConfig = field(
        default_factory=lambda: LLMConfig(
            base_url="", model="", api_key_env="OI_AGENT_API_KEY"
        )
    )
    max_tool_iterations: int = 4
    watches: list[WatchTarget] = field(default_factory=list)

    def target_for_channel(self, channel_id: int) -> WatchTarget | None:
        """Resolve the watch target a message belongs to (incl. threads).

        Args:
            channel_id: The channel or thread id a message arrived in.

        Returns:
            Matching WatchTarget or None when the channel is not watched.
        """
        for t in self.watches:
            if t.channel_id == channel_id:
                return t
        return None


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate configuration from a TOML file.

    Args:
        path: Config file location.

    Returns:
        Parsed Config object.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On structurally invalid entries.
    """
    raw = tomllib.loads(path.read_text(encoding="utf-8"))

    def _str(container: dict, key: str, default: str) -> str:
        """Require a TOML string (or use the default); reject other types."""
        if key not in container:
            return default
        v = container[key]
        if not isinstance(v, str):
            raise ValueError(f"'{key}' must be a string, got {type(v).__name__}")
        return v

    def _int(container: dict, key: str, default: int) -> int:
        """Require a TOML integer; reject bool/str/float coercion."""
        if key not in container:
            return default
        v = container[key]
        # bool is a subclass of int — reject it explicitly.
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"'{key}' must be an integer, got {type(v).__name__}")
        return v

    def _bool(container: dict, key: str, default: bool) -> bool:
        """Require a TOML boolean; reject string/int coercion."""
        if key not in container:
            return default
        v = container[key]
        if not isinstance(v, bool):
            raise ValueError(f"'{key}' must be a boolean, got {type(v).__name__}")
        return v

    def _str_list(container: dict, key: str, default: list[str]) -> list[str]:
        """Require a TOML array of strings; reject a bare string or non-strings.

        Guards against silent coercion: a bare ``"bug"`` must not become
        ``["b", "u", "g"]``, and ``[1]`` must not become ``["1"]``.
        """
        if key not in container:
            return list(default)
        v = container[key]
        if not isinstance(v, list) or any(not isinstance(x, str) for x in v):
            raise ValueError(f"'{key}' must be a list of strings")
        return v

    cfg = Config(
        discord_token_env=_str(raw, "discord_token_env", "OI_DISCORD_TOKEN"),
        db_path=_str(raw, "db_path", cfg_default_db()),
        personality=_str(raw, "personality", DEFAULT_PERSONALITY),
        max_reply_chars=_int(raw, "max_reply_chars", 2_000),
        max_response_tokens=_int(raw, "max_response_tokens", 4_000),
        max_tool_iterations=_int(raw, "max_tool_iterations", 4),
    )
    fallback_raw = raw.get("fallback", {})
    for section, section_raw in (
        ("fallback", fallback_raw), ("agent", raw.get("agent", {}))
    ):
        if section_raw:
            s = section_raw
            setattr(cfg, section, LLMConfig(
                base_url=_str(s, "base_url", ""),
                model=_str(s, "model", ""),
                api_key_env=_str(
                    s, "api_key_env",
                    "OI_FALLBACK_API_KEY"
                    if section == "fallback" else "OI_AGENT_API_KEY",
                ),
                api_style=_str(s, "api_style", "openai"),
                disable_thinking=_bool(s, "disable_thinking", False),
            ))
    for w in raw.get("watch", []):
        for req in ("channel_id", "repo_path"):
            if req not in w:
                raise ValueError(f"watch entry missing '{req}': {w}")
        default_kw = [
            "plan", "commit", "migration", "deployed", "deploy",
            "broken", "bug", "merged", "release", "model", "schema",
        ]
        aliases_raw = w.get("search_aliases", {})
        if not isinstance(aliases_raw, dict):
            raise ValueError("search_aliases must be a table")
        for term, expansions in aliases_raw.items():
            if not isinstance(term, str) or not term.strip():
                raise ValueError(
                    "search_aliases keys must be non-empty strings")
            if not isinstance(expansions, list) or any(
                    not isinstance(value, str) or not value.strip()
                    for value in expansions):
                raise ValueError(f"search_aliases[{term!r}] must be a list "
                                 "of non-empty strings")
        founder_raw = w.get("founder_ids", [])
        if not isinstance(founder_raw, list) or any(
                isinstance(x, bool) or not isinstance(x, int)
                for x in founder_raw):
            raise ValueError("founder_ids must be a list of integers")
        cfg.watches.append(WatchTarget(
            channel_id=_int(w, "channel_id", 0),
            repo_path=_str(w, "repo_path", ""),
            founder_ids=list(founder_raw),
            bot_user_id=_int(w, "bot_user_id", 0),
            post_hourly_cap=_int(w, "post_hourly_cap", 3),
            claim_keywords=[k.lower()
                            for k in _str_list(w, "claim_keywords", default_kw)],
            search_aliases={term: list(expansions)
                            for term, expansions in aliases_raw.items()},
            preferred_tokens=_str_list(w, "preferred_tokens", []),
            search_stopwords=_str_list(w, "search_stopwords", []),
            # Absent key -> None -> inherits Config.max_tool_iterations.
            max_tool_iterations=_int(w, "max_tool_iterations", None),
        ))
    # Manual edits bypass save_config(), so the loader is the second gate:
    # invalid on-disk config must never reach the daemon.
    validate_config(cfg)
    return cfg




def validate_config(cfg: Config) -> None:
    """Raise ValueError when any configuration invariant is broken.

    Single source of truth for semantic validation: save_config() refuses to
    write anything that fails here, so every write path (init wizard,
    ``oi config set``, ``config watch-add``, programmatic saves) is covered
    and a rejected write leaves the previous file byte-for-byte intact.

    Args:
        cfg: Configuration to check.

    Raises:
        ValueError: With a precise message naming the first broken field.
    """
    for name in ("discord_token_env", "db_path"):
        if not str(getattr(cfg, name, "")).strip():
            raise ValueError(f"{name} must be a non-empty string")
    for name, minimum in (("max_reply_chars", 200),
                          ("max_response_tokens", 256)):
        value = getattr(cfg, name)
        # bool is an int subclass but never a legitimate limit.
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer (got {value!r})")
        if value < minimum:
            raise ValueError(f"{name} must be >= {minimum} (got {value})")
    value = cfg.max_tool_iterations
    # bool is an int subclass but never a legitimate loop depth.
    if isinstance(value, bool) or not isinstance(value, int) \
            or not 1 <= value <= 50:
        raise ValueError("max_tool_iterations must be an integer in "
                         f"[1, 50] (got {value!r})")
    for section in ("fallback", "agent"):
        llm = getattr(cfg, section)
        if llm.api_style not in ("openai", "anthropic"):
            raise ValueError(f"{section}.api_style must be "
                             f"'openai' or 'anthropic' (got {llm.api_style!r})")
        for attr in ("base_url", "model", "api_key_env"):
            if not isinstance(getattr(llm, attr), str):
                raise ValueError(f"{section}.{attr} must be a string")
    if not cfg.watches:
        raise ValueError("at least one watch target is required")
    seen_channels: set[int] = set()
    for t in cfg.watches:
        if isinstance(t.channel_id, bool) or not isinstance(t.channel_id, int) \
                or t.channel_id <= 0:
            raise ValueError("watch channel_id must be a positive integer "
                             f"(got {t.channel_id!r})")
        if t.channel_id in seen_channels:
            raise ValueError(f"duplicate watch channel_id {t.channel_id}")
        if isinstance(t.post_hourly_cap, bool) \
                or not isinstance(t.post_hourly_cap, int) \
                or t.post_hourly_cap < 1:
            raise ValueError("post_hourly_cap must be an integer >= 1 "
                             f"(got {t.post_hourly_cap!r})")
        if t.max_tool_iterations is not None:
            v = t.max_tool_iterations
            if isinstance(v, bool) or not isinstance(v, int) \
                    or not 1 <= v <= 50:
                raise ValueError(
                    "watch max_tool_iterations must be an integer in "
                    f"[1, 50] (got {v!r})")
        if isinstance(t.bot_user_id, bool) \
                or not isinstance(t.bot_user_id, int) or t.bot_user_id < 0:
            raise ValueError("bot_user_id must be a non-negative integer "
                             f"(got {t.bot_user_id!r})")
        for fid in t.founder_ids:
            if isinstance(fid, bool) or not isinstance(fid, int):
                raise ValueError(
                    f"founder_ids must be integers (got {fid!r})")
        for collection, label in ((t.claim_keywords, "claim_keywords"),
                                  (t.preferred_tokens, "preferred_tokens"),
                                  (t.search_stopwords, "search_stopwords")):
            for item in collection:
                if not isinstance(item, str) or not item.strip():
                    raise ValueError(f"{label} entries must be non-empty "
                                     f"strings (got {item!r})")
        for term, expansions in t.search_aliases.items():
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"search_aliases keys must be non-empty "
                                 f"strings (got {term!r})")
            if not isinstance(expansions, list) or any(
                    not isinstance(v, str) or not v.strip()
                    for v in expansions):
                raise ValueError(f"search_aliases[{term!r}] must be a list "
                                 f"of non-empty strings")


def cfg_default_db() -> str:
    """Return the default state db path as a string."""
    return "~/.local/state/oi/state.db"


def effective_max_tool_iterations(cfg: Config, target: WatchTarget) -> int:
    """Resolve the agentic tool-loop depth for one watch target.

    Args:
        cfg: Full configuration holding the global default.
        target: Watch target whose optional per-watch override wins.

    Returns:
        The watch's max_tool_iterations when set, else the config default.
    """
    if target.max_tool_iterations is not None:
        return target.max_tool_iterations
    return cfg.max_tool_iterations




def _toml_inline_table(mapping: dict[str, list[str]]) -> str:
    """Render a string->list mapping as a valid TOML inline table.

    json.dumps is not enough here: TOML separates key and value with ``=``,
    not JSON's ``:``. Strings and lists inside stay json.dumps-quoted.

    Args:
        mapping: e.g. ``{"kpi": ["metric"]}``.

    Returns:
        Inline-table text such as ``{ "kpi" = ["metric"] }``.
    """
    items = ", ".join(
        f"{json.dumps(k)} = {json.dumps(vs)}" for k, vs in mapping.items()
    )
    return "{ " + items + " }" if items else "{}"


def save_config(cfg: Config, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Validate and atomically write configuration to TOML.

    Validation runs BEFORE any bytes are touched, so an invalid config can
    never clobber a previously working one. The write lands in a temporary
    sibling file that is os.replace()d into place — readers see either the
    old or the new file, never a partial write.

    Args:
        cfg: Config to serialize.
        path: Destination file; parent dirs are created.

    Raises:
        ValueError: From validate_config() when an invariant is broken.
    """
    validate_config(cfg)
    lines = [
        f'discord_token_env = {json.dumps(cfg.discord_token_env)}',
        f'db_path = {json.dumps(cfg.db_path)}',
        f'personality = {json.dumps(cfg.personality)}',
        f"max_reply_chars = {cfg.max_reply_chars}",
        f"max_response_tokens = {cfg.max_response_tokens}",
        f"max_tool_iterations = {cfg.max_tool_iterations}",
        "",
        "[fallback]",
        f"base_url = {json.dumps(cfg.fallback.base_url)}",
        f"model = {json.dumps(cfg.fallback.model)}",
        f"api_key_env = {json.dumps(cfg.fallback.api_key_env)}",
        f"api_style = {json.dumps(cfg.fallback.api_style)}",
        f'disable_thinking = {str(cfg.fallback.disable_thinking).lower()}',
        "",
        "[agent]",
        f"base_url = {json.dumps(cfg.agent.base_url)}",
        f"model = {json.dumps(cfg.agent.model)}",
        f"api_key_env = {json.dumps(cfg.agent.api_key_env)}",
        f"api_style = {json.dumps(cfg.agent.api_style)}",
        f'disable_thinking = {str(cfg.agent.disable_thinking).lower()}',
        "",
    ]
    for t in cfg.watches:
        lines += [
            "[[watch]]",
            f"channel_id = {t.channel_id}",
            f"repo_path = {json.dumps(t.repo_path)}",
            f"founder_ids = {t.founder_ids}",
            f"bot_user_id = {t.bot_user_id}",
            f"post_hourly_cap = {t.post_hourly_cap}",
            f"claim_keywords = {json.dumps(t.claim_keywords)}",
            *([f"max_tool_iterations = {t.max_tool_iterations}"]
              if t.max_tool_iterations is not None else []),
            f"search_aliases = {_toml_inline_table(t.search_aliases)}",
            f"preferred_tokens = {json.dumps(t.preferred_tokens)}",
            f"search_stopwords = {json.dumps(t.search_stopwords)}",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_path, path)
