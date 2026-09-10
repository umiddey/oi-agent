"""Validated TOML configuration for the OpenCode-backed daemon."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "oi" / "config.toml"
DEFAULT_SECRETS_PATH = Path.home() / ".config" / "oi" / "secrets.env"
DEFAULT_PERSONALITY = "Talk like a smart teammate in chat, not a report generator."
REPLY_DELIVERY_MODES = ("chunked", "single_message")
ENVIRONMENT_MODES = ("workstation", "server")
WRITE_MODES = ("read-only", "docs-only")
DISCORD_MAX_MESSAGE_CHARS = 2_000
LEGACY_KEYS = {"agent", "llm", "api_key", "base_url", "search_aliases", "provider", "model"}


@dataclass
class WatchTarget:
    """One monitored Discord scope mapped to one or more local Git repositories."""

    channel_id: int = 0
    repo_path: str = ""
    bot_user_id: int = 0
    post_hourly_cap: int = 3
    guild_id: int = 0
    repos: list[str] = field(default_factory=list)
    ignored_channels: list[int] = field(default_factory=list)
    allowed_channels: list[int] = field(default_factory=list)

    @property
    def repo_paths(self) -> list[str]:
        """Return all watched repository paths in canonical order."""
        if self.repos:
            return list(self.repos)
        if self.repo_path:
            return [self.repo_path]
        return []

    @property
    def primary_repo(self) -> str:
        """Return the primary repository directory for OpenCode (--dir)."""
        paths = self.repo_paths
        return paths[0] if paths else self.repo_path

    @property
    def is_guild_watch(self) -> bool:
        """Return True if this target watches an entire Discord server (guild)."""
        return self.guild_id > 0 and self.channel_id == 0

    def matches_channel(self, channel_id: int, guild_id: int | None = None) -> bool:
        """Return True if this watch target applies to the given channel / guild."""
        if not self.is_guild_watch:
            return self.channel_id == channel_id

        if guild_id is not None and guild_id != self.guild_id:
            return False

        if self.ignored_channels and channel_id in self.ignored_channels:
            return False

        if self.allowed_channels and channel_id not in self.allowed_channels:
            return False

        return True


@dataclass
class Config:
    """Full OpenCode-only daemon configuration.

    OI enforces Discord safety rails, rate limits, burst coalescing, and
    process isolation, while OpenCode handles model execution.
    """

    discord_token_env: str = "OI_DISCORD_TOKEN"
    db_path: str = "~/.local/state/oi/state.db"
    personality: str = DEFAULT_PERSONALITY
    max_reply_chars: int = 2_000
    reply_delivery: str = "chunked"
    opencode_binary: str = "opencode"
    opencode_model: str = "openai/gpt-4o-mini"
    opencode_steps: int = 30
    opencode_timeout_seconds: int = 900
    environment_mode: str = "workstation"
    write_mode: str = "read-only"
    write_dirs: list[str] = field(default_factory=list)
    watches: list[WatchTarget] = field(default_factory=list)

    @property
    def resolved_environment_mode(self) -> str:
        """Resolve effective environment mode: strictly 'workstation' or 'server'."""
        if self.environment_mode in ("workstation", "server"):
            return self.environment_mode
        raise ValueError(
            f"Invalid environment_mode: {self.environment_mode!r}. "
            "Legacy 'auto' mode is deprecated. Run: oi config set environment_mode workstation|server"
        )
    def target_for_channel(self, channel_id: int, guild_id: int | None = None) -> WatchTarget | None:
        """Resolve a watch target by channel ID and optional guild ID.

        Direct channel-specific targets take precedence over guild-wide targets.
        """
        # 1. Exact channel match
        for target in self.watches:
            if not target.is_guild_watch and target.matches_channel(channel_id, guild_id):
                return target
        # 2. Fallback to guild-wide targets
        for target in self.watches:
            if target.is_guild_watch and target.matches_channel(channel_id, guild_id):
                return target
        return None


def _require_string(container: dict, key: str, default: str) -> str:
    """Read a TOML string, rejecting implicit type coercion."""
    if key not in container:
        return default
    value = container[key]
    if not isinstance(value, str):
        raise ValueError(f"'{key}' must be a string, got {type(value).__name__}")
    return value


def _require_int(container: dict, key: str, default: int) -> int:
    """Read a TOML integer, rejecting floats, booleans, and strings."""
    if key not in container:
        return default
    value = container[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"'{key}' must be an integer, got {type(value).__name__}")
    return value


def _require_string_list(container: dict, key: str) -> list[str]:
    """Read a TOML list of strings, rejecting non-list and non-string items."""
    if key not in container:
        return []
    value = container[key]
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be an array of strings, got {type(value).__name__}")
    res: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"items in '{key}' must be non-empty strings, got {item!r}")
        res.append(item)
    return res


def _require_int_list(container: dict, key: str) -> list[int]:
    """Read a TOML list of integers, rejecting non-list and non-integer items."""
    if key not in container:
        return []
    value = container[key]
    if not isinstance(value, list):
        raise ValueError(f"'{key}' must be an array of integers, got {type(value).__name__}")
    res: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"items in '{key}' must be integers, got {item!r}")
        if item <= 0:
            raise ValueError(f"channel IDs in '{key}' must be positive integers, got {item}")
        res.append(item)
    return res


def _require_write_dirs(container: dict) -> list[str]:
    """Read optional repo-relative doc-write scopes.

    Accepts a TOML array of strings or a comma-separated string (CLI
    convenience). Entries must be relative paths without ``..`` escapes;
    trailing slashes are stripped. Empty means .md/.txt may be written
    anywhere in the repo (secrets still denied at the policy layer).
    """
    if "write_dirs" not in container:
        return []
    value = container["write_dirs"]
    if isinstance(value, str):
        items = [p.strip() for p in value.split(",") if p.strip()]
    elif isinstance(value, list):
        items = value
    else:
        raise ValueError(
            "'write_dirs' must be an array of strings or comma-separated string, "
            f"got {type(value).__name__}"
        )
    res: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"items in 'write_dirs' must be non-empty strings, got {item!r}")
        if item.strip().startswith("/"):
            raise ValueError(f"write_dirs entries must be repo-relative without '..': {item!r}")
        norm = item.strip().strip("/")
        if not norm or norm == ".":
            raise ValueError(f"invalid write_dirs entry: {item!r}")
        parts = norm.split("/")
        if ".." in parts:
            raise ValueError(f"write_dirs entries must be repo-relative without '..': {item!r}")
        if any(not p for p in parts):
            raise ValueError(f"invalid write_dirs entry: {item!r}")
        res.append(norm)
    return res


def load_config(
    path: Path = DEFAULT_CONFIG_PATH,
    *,
    allow_legacy_environment_mode: bool = False,
) -> Config:
    """Load and validate an OpenCode-only TOML configuration.

    Args:
        path: Path to the configuration file.
        allow_legacy_environment_mode: Tolerate a legacy ``auto`` value so
            ``oi config set environment_mode`` can repair it. The caller must
            overwrite ``environment_mode`` with a valid value before saving;
            the placeholder ``workstation`` is substituted so validation
            passes and nothing legacy is ever persisted as-is.

    Returns:
        Validated Config dataclass instance.

    Raises:
        ValueError: On schema violations, legacy sections, or missing fields.
    """
    if not path.is_file():
        raise ValueError(f"config file does not exist: {path}")
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"invalid TOML in {path}: {exc}") from exc

    legacy_found: list[str] = [k for k in LEGACY_KEYS if k in raw]
    watches_raw = raw.get("watch") or raw.get("watches")
    if isinstance(watches_raw, list):
        for watch in watches_raw:
            if isinstance(watch, dict):
                legacy_found.extend(
                    [k for k in ("search_aliases", "tool_iterations", "tools", "manifest") if k in watch]
                )
    if legacy_found:
        raise ValueError(
            f"config contains legacy settings ({', '.join(sorted(set(legacy_found)))}): "
            "OI now uses native OpenCode; please rerun `oi init` to create a clean config."
        )

    if not watches_raw or not isinstance(watches_raw, list):
        raise ValueError(f"config at {path} must contain at least one [[watch]] table")

    legacy_environment_mode = isinstance(raw.get("environment_mode"), str) and (
        raw["environment_mode"].strip().casefold() == "auto"
    )
    if legacy_environment_mode and not allow_legacy_environment_mode:
        raise ValueError(
            "Legacy environment_mode 'auto' is deprecated. "
            "Run `oi config set environment_mode workstation` or `oi config set environment_mode server`."
        )
    cfg = Config(
        discord_token_env=_require_string(raw, "discord_token_env", "OI_DISCORD_TOKEN"),
        db_path=_require_string(raw, "db_path", cfg_default_db()),
        personality=_require_string(raw, "personality", DEFAULT_PERSONALITY),
        max_reply_chars=_require_int(raw, "max_reply_chars", 2_000),
        reply_delivery=_require_string(raw, "reply_delivery", "chunked"),
        opencode_binary=_require_string(raw, "opencode_binary", "opencode"),
        opencode_model=_require_string(raw, "opencode_model", "openai/gpt-4o-mini"),
        opencode_steps=_require_int(raw, "opencode_steps", 30),
        opencode_timeout_seconds=_require_int(
            raw, "opencode_timeout_seconds", 900
        ),
        environment_mode=(
            "workstation" if legacy_environment_mode
            else _require_string(raw, "environment_mode", "workstation")
        ),
        write_mode=_require_string(raw, "write_mode", "read-only"),
        write_dirs=_require_write_dirs(raw),
    )
    for watch in watches_raw:
        if not isinstance(watch, dict):
            raise ValueError("each watch entry must be a table")
        has_channel = "channel_id" in watch
        has_guild = "guild_id" in watch
        if (has_channel and has_guild) or (not has_channel and not has_guild):
            raise ValueError(
                f"watch entry must specify exactly one of 'channel_id' or 'guild_id': {watch}"
            )

        has_repo = "repo_path" in watch or "repos" in watch
        if not has_repo:
            raise ValueError(f"watch entry missing 'repo_path' or 'repos': {watch}")

        repos_list = _require_string_list(watch, "repos")
        ignored_list = _require_int_list(watch, "ignored_channels") or _require_int_list(watch, "ignore_channel_ids")
        allowed_list = _require_int_list(watch, "allowed_channels") or _require_int_list(watch, "allowed_channel_ids")

        cfg.watches.append(
            WatchTarget(
                channel_id=_require_int(watch, "channel_id", 0) if has_channel else 0,
                repo_path=(
                    _require_string(watch, "repo_path", "") if "repo_path" in watch
                    else (repos_list[0] if repos_list else "")
                ),
                bot_user_id=_require_int(watch, "bot_user_id", 0),
                post_hourly_cap=_require_int(watch, "post_hourly_cap", 3),
                guild_id=_require_int(watch, "guild_id", 0) if has_guild else 0,
                repos=repos_list,
                ignored_channels=ignored_list,
                allowed_channels=allowed_list,
            )
        )
    validate_config(cfg)
    return cfg


def _resolve_binary(binary: str) -> str | None:
    """Resolve an executable command or absolute/relative binary path."""
    if not binary.strip():
        return None
    expanded = Path(binary).expanduser()
    if expanded.is_file() and os.access(expanded, os.X_OK):
        return str(expanded.resolve())
    return shutil.which(binary)


def validate_config(cfg: Config) -> None:
    """Raise ``ValueError`` when an OpenCode config invariant is broken.

    Args:
        cfg: Configuration to validate before persistence or runtime use.
    """
    for name in ("discord_token_env", "db_path", "opencode_binary", "opencode_model"):
        value = getattr(cfg, name, "")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
    provider, separator, model = cfg.opencode_model.partition("/")
    if not separator or not provider.strip() or not model.strip():
        raise ValueError(
            "opencode_model must use the exact provider/model format "
            f"(got {cfg.opencode_model!r})"
        )
    if _resolve_binary(cfg.opencode_binary) is None:
        raise ValueError(f"opencode_binary is not executable: {cfg.opencode_binary!r}")
    if cfg.reply_delivery not in REPLY_DELIVERY_MODES:
        raise ValueError(
            f"reply_delivery must be one of {REPLY_DELIVERY_MODES!r} "
            f"(got {cfg.reply_delivery!r})"
        )
    if cfg.environment_mode not in ENVIRONMENT_MODES:
        raise ValueError(
            f"environment_mode must be one of {ENVIRONMENT_MODES!r} "
            f"(got {cfg.environment_mode!r})"
        )
    if cfg.write_mode not in WRITE_MODES:
        raise ValueError(
            f"write_mode must be one of {WRITE_MODES!r} (got {cfg.write_mode!r})"
        )
    if not isinstance(cfg.write_dirs, list):
        raise ValueError(f"write_dirs must be a list of strings (got {cfg.write_dirs!r})")
    for entry in cfg.write_dirs:
        if not isinstance(entry, str) or not entry.strip():
            raise ValueError(f"write_dirs item must be a non-empty string (got {entry!r})")
        if entry.strip().startswith("/"):
            raise ValueError(f"write_dirs entries must be repo-relative without '..': {entry!r}")
        norm = entry.strip().strip("/")
        if not norm or norm == "." or ".." in norm.split("/"):
            raise ValueError(f"write_dirs entries must be repo-relative without '..': {entry!r}")
    for name, minimum in (("max_reply_chars", 200), ("opencode_steps", 1),
                          ("opencode_timeout_seconds", 1)):
        value = getattr(cfg, name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer (got {value!r})")
        maximum = {"opencode_steps": 50, "opencode_timeout_seconds": 86_400}.get(name)
        if value < minimum or (maximum is not None and value > maximum):
            bound = f"[{minimum}, {maximum}]" if maximum else f">= {minimum}"
            raise ValueError(f"{name} must be in {bound} (got {value})")
    if cfg.reply_delivery == "single_message" and cfg.max_reply_chars > DISCORD_MAX_MESSAGE_CHARS:
        raise ValueError(
            "max_reply_chars must be <= 2000 when "
            f"reply_delivery='single_message' (got {cfg.max_reply_chars})"
        )
    if not cfg.watches:
        raise ValueError("at least one watch target is required")
    seen_channels: set[int] = set()
    seen_guilds: set[int] = set()
    for target in cfg.watches:
        if isinstance(target.channel_id, bool) or not isinstance(target.channel_id, int) or target.channel_id < 0:
            raise ValueError(f"channel_id must be a non-negative integer (got {target.channel_id!r})")
        if isinstance(target.guild_id, bool) or not isinstance(target.guild_id, int) or target.guild_id < 0:
            raise ValueError(f"guild_id must be a non-negative integer (got {target.guild_id!r})")

        if target.channel_id > 0 and target.guild_id > 0:
            raise ValueError(f"watch entry cannot specify both channel_id and guild_id: {target}")
        if target.channel_id <= 0 and target.guild_id <= 0:
            raise ValueError(f"watch entry must specify exactly one positive channel_id or guild_id: {target}")

        if target.channel_id > 0:
            if target.channel_id in seen_channels:
                raise ValueError(f"duplicate watch channel_id {target.channel_id}")
            seen_channels.add(target.channel_id)
        if target.is_guild_watch:
            if target.guild_id in seen_guilds:
                raise ValueError(f"duplicate watch guild_id {target.guild_id}")
            seen_guilds.add(target.guild_id)

        if not isinstance(target.repo_path, str):
            raise ValueError(f"repo_path must be a string (got {target.repo_path!r})")

        if not isinstance(target.repos, list):
            raise ValueError(f"repos must be a list of strings (got {target.repos!r})")
        for r in target.repos:
            if not isinstance(r, str) or not r.strip():
                raise ValueError(f"repos item must be a non-empty string (got {r!r})")

        paths = target.repo_paths
        if not paths:
            raise ValueError("watch entry must define at least one non-empty repo_path or repos item")
        for p in paths:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"watch repo path must be a non-empty string (got {p!r})")

        if not isinstance(target.ignored_channels, list):
            raise ValueError(f"ignored_channels must be a list of positive integers (got {target.ignored_channels!r})")
        for item in target.ignored_channels:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"ignored_channels item must be a positive integer (got {item!r})")

        if not isinstance(target.allowed_channels, list):
            raise ValueError(f"allowed_channels must be a list of positive integers (got {target.allowed_channels!r})")
        for item in target.allowed_channels:
            if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
                raise ValueError(f"allowed_channels item must be a positive integer (got {item!r})")

        if (
            isinstance(target.post_hourly_cap, bool) or not isinstance(target.post_hourly_cap, int)
            or target.post_hourly_cap < 1
        ):
            raise ValueError(f"post_hourly_cap must be an integer >= 1 (got {target.post_hourly_cap!r})")
        if isinstance(target.bot_user_id, bool) or not isinstance(target.bot_user_id, int) or target.bot_user_id < 0:
            raise ValueError(f"bot_user_id must be a non-negative integer (got {target.bot_user_id!r})")


def cfg_default_db() -> str:
    """Return the default SQLite state path."""
    return "~/.local/state/oi/state.db"


def save_config(cfg: Config, path: Path = DEFAULT_CONFIG_PATH) -> None:
    """Validate and atomically write an OpenCode-only TOML config.

    Args:
        cfg: Configuration to serialize.
        path: Destination path.
    """
    validate_config(cfg)
    lines = [
        f"discord_token_env = {json.dumps(cfg.discord_token_env)}",
        f"db_path = {json.dumps(cfg.db_path)}",
        f"personality = {json.dumps(cfg.personality)}",
        f"max_reply_chars = {cfg.max_reply_chars}",
        f"reply_delivery = {json.dumps(cfg.reply_delivery)}",
        f"opencode_binary = {json.dumps(cfg.opencode_binary)}",
        f"opencode_model = {json.dumps(cfg.opencode_model)}",
        f"opencode_steps = {cfg.opencode_steps}",
        f"opencode_timeout_seconds = {cfg.opencode_timeout_seconds}",
        f"environment_mode = {json.dumps(cfg.environment_mode)}",
        f"write_mode = {json.dumps(cfg.write_mode)}",
        f"write_dirs = {json.dumps(list(cfg.write_dirs))}",
    ]
    for target in cfg.watches:
        lines.append("")
        lines.append("[[watch]]")
        if target.is_guild_watch:
            lines.append(f"guild_id = {target.guild_id}")
            if target.ignored_channels:
                lines.append(f"ignored_channels = {target.ignored_channels}")
            if target.allowed_channels:
                lines.append(f"allowed_channels = {target.allowed_channels}")
        else:
            lines.append(f"channel_id = {target.channel_id}")

        if len(target.repo_paths) > 1 or target.repos:
            lines.append(f"repos = {json.dumps(target.repo_paths)}")
        else:
            lines.append(f"repo_path = {json.dumps(target.primary_repo)}")

        if target.bot_user_id:
            lines.append(f"bot_user_id = {target.bot_user_id}")
        if target.post_hourly_cap != 3:
            lines.append(f"post_hourly_cap = {target.post_hourly_cap}")

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp_path, path)
