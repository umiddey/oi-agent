"""OI Agent CLI: init wizard, index management, run/pause/resume.

Everything a deployment needs:
  oi init   -> guided config creation
  oi index  -> (re)build the manifest for one or all repos
  oi run    -> start watching
  oi pause / oi resume -> global kill switch
"""

from __future__ import annotations

import getpass
import logging
import os
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt

from .config import (
    DEFAULT_CONFIG_PATH, DEFAULT_PERSONALITY, DEFAULT_SECRETS_PATH, Config,
    LLMConfig, WatchTarget, load_config, load_secrets, save_config,
)
from .manifest.generator import generate_repo_manifest, refresh_repo_manifest
from .store import Store

app = typer.Typer(help="OI: standalone Discord feedback-channel respondent.")
console = Console()


# (label, default base_url, default model, needs_api_key)
PROVIDER_PRESETS = [
    ("Hetzner Qwen3.8-27B (free, experimental)",
     "https://inference.hetzner.com/api/v1", "Qwen3.8-27B", True),
    ("Hetzner Qwen3.6-35B MoE (free, experimental)",
     "https://inference.hetzner.com/api/v1", "Qwen/Qwen3.6-35B-A3B-FP8", True),
    ("OpenAI", "https://api.openai.com/v1", "gpt-4o-mini", True),
    ("OpenRouter", "https://openrouter.ai/api/v1", "qwen/qwen3-27b", True),
    ("Z.AI", "https://api.z.ai", "glm-4.6", True),
    ("Ollama (local)", "http://localhost:11434/v1", "qwen3:27b", False),
    ("Custom / other", "", "", True),
]


def _prompt_int(label: str, default: int) -> int:
    """Prompt for an integer, re-asking on bad input instead of crashing.

    Args:
        label: Prompt text.
        default: Value used when the user just presses enter.

    Returns:
        The entered integer.
    """
    while True:
        try:
            return int(str(typer.prompt(label, default=default)).strip())
        except (ValueError, TypeError):
            console.print("[red]must be a number[/red]")


def _prompt_founder_ids() -> list[int]:
    """Prompt for comma-separated Discord user ids until all parse.

    Returns:
        Parsed founder ids (empty list allowed).
    """
    while True:
        raw = typer.prompt("founder ids (comma separated)", default="")
        try:
            return [int(x) for x in raw.split(",") if x.strip()]
        except ValueError:
            console.print("[red]founder ids must be numbers[/red]")

@app.command()
def init() -> None:
    """Guided first-time setup: secrets + config + first indexes.

    Secrets are typed once (hidden input) and stored in ~/.config/oi/.env
    with 0600 permissions; config goes to ~/.config/oi/config.toml.
    """
    console.print("[bold]OI init[/bold] — ctrl-c aborts.")
    console.print("[dim]Secrets are stored locally, input is hidden.[/dim]")

    def ask_secret(label: str) -> str:
        """Prompt for a secret without echoing it.

        Args:
            label: Prompt label.

        Returns:
            Entered secret (may be empty to skip).
        """
        return getpass.getpass(f"{label} (input hidden): ")

    def ask_llm(label: str, default_key_env: str) -> LLMConfig:
        """Ask which provider to use, then prefill endpoint and wire protocol.

        Shows a numbered menu of known providers. Z.AI is configured with the
        Anthropic-compatible protocol; all other presets remain OpenAI-style.

        Args:
            label: Display label (fallback/agent).
            default_key_env: Fallback env var name for the API key.

        Returns:
            LLMConfig instance.
        """
        console.print(f"\n[bold]{label}[/bold] — pick a provider:")
        for i, (name, _, _, _) in enumerate(PROVIDER_PRESETS, 1):
            console.print(f"  {i}) {name}")
        choice = int(Prompt.ask(
            "Provider number", default="1",
            choices=[str(i) for i in range(1, len(PROVIDER_PRESETS) + 1)]))
        name, base_default, model_default, needs_key = \
            PROVIDER_PRESETS[choice - 1]

        base = Prompt.ask(f"[{label}] base_url", default=base_default)
        model = Prompt.ask(f"[{label}] model", default=model_default)
        if needs_key:
            key_env = Prompt.ask(f"[{label}] api key env var",
                                 default=default_key_env)
            key = ask_secret(f"[{label}] {key_env}")
            if key:
                _secrets.append(f"{key_env}={key}")
        else:
            key_env = "OI_NO_KEY"
        api_style = "anthropic" if name == "Z.AI" else "openai"
        return LLMConfig(base_url=base, model=model, api_key_env=key_env,
                         api_style=api_style)


    _secrets: list[str] = []
    discord_token = ask_secret("Discord bot token")
    if discord_token:
        _secrets.append(f"OI_DISCORD_TOKEN={discord_token}")

    cfg = Config()
    cfg.discord_token_env = Prompt.ask(
        "Env var name for the bot token", default="OI_DISCORD_TOKEN")
    cfg.personality = Prompt.ask(
        "Agent personality/tone", default=DEFAULT_PERSONALITY)
    cfg.max_response_tokens = _prompt_int(
        "Max response tokens (raise for reasoning models)",
        cfg.max_response_tokens)
    cfg.max_tool_iterations = _prompt_int(
        "Max tool iterations per audit", cfg.max_tool_iterations)

    cfg.fallback = ask_llm("fallback (small model)", "OI_FALLBACK_API_KEY")
    cfg.agent = ask_llm("agent (main model)", "OI_AGENT_API_KEY")


    while True:
        try:
            channel_id = int(typer.prompt(
                "Watch: Discord channel id (0 to finish)"))
        except (ValueError, TypeError):
            console.print("[red]channel id must be a number[/red]")
            continue
        if channel_id == 0:
            break
        raw_path = Prompt.ask(
            "Watch: repo path (enter = current directory)",
            default=str(Path.cwd()))
        repo_path = Path(raw_path).expanduser().resolve()
        if not (repo_path / ".git").exists():
            console.print(f"[red]{repo_path} is not a git clone, skipping[/red]")
            continue
        founders = _prompt_founder_ids()
        bot_user_id = _prompt_int("bot user id", 0)
        cap = _prompt_int("posts per hour cap", 3)
        cfg.watches.append(WatchTarget(
            channel_id=channel_id, repo_path=str(repo_path),
            founder_ids=founders, bot_user_id=bot_user_id,
            post_hourly_cap=cap,
        ))
    if not cfg.watches:
        console.print("[red]no channels configured; nothing written[/red]")
        raise typer.Exit(1)

    path = DEFAULT_CONFIG_PATH
    save_config(cfg, path)
    console.print(f"[green]wrote {path}[/green]")

    if _secrets:
        DEFAULT_SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_SECRETS_PATH.write_text("\n".join(_secrets) + "\n",
                                        encoding="utf-8")
        DEFAULT_SECRETS_PATH.chmod(0o600)
        console.print(f"[green]wrote {DEFAULT_SECRETS_PATH} (mode 600)[/green]")

    for t in cfg.watches:
        console.print("building index for [bold]{}[/bold]...".format(t.repo_path))
        result = generate_repo_manifest(Path(t.repo_path))
        console.print(f"  {result.file_count} files, sha {result.git_sha}")


def _load_or_die(path: Path) -> Config:
    """Load config (empty default when the file doesn't exist yet).

    Args:
        path: Config file location.

    Returns:
        Parsed Config, or a fresh empty one for a missing file.
    """
    try:
        return load_config(path) if path.exists() else Config()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]config error: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def index(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    force: bool = typer.Option(False, "--force", help="full rebuild"),
) -> None:
    """Build/refresh the manifest for every configured repo."""
    cfg = _load_or_die(config)
    for t in cfg.watches:
        repo = Path(t.repo_path)
        result = (generate_repo_manifest(repo) if force
                  else refresh_repo_manifest(repo))
        console.print(
            f"{repo}: {result.file_count} files @ {result.git_sha} "
            f"({result.duration_ms}ms)")


@app.command()
def run(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    stream: bool = typer.Option(
        False, "--stream",
        help="Request SSE streaming from the LLM provider (transport only). "
             "The full reply is still buffered before posting, so this does "
             "not lower user-visible latency. No content is logged.",
    ),
    log_answers: bool = typer.Option(
        False, "--log-answers",
        help="PRIVACY-SENSITIVE: write visible answer text (may quote private "
             "repository evidence) to the logs.",
    ),
) -> None:
    """Start the watcher daemon (blocking)."""
    from .watch.discord_client import run_daemon

    cfg = _load_or_die(config)
    load_secrets()
    token = os.environ.get(cfg.discord_token_env, "")
    if not token:
        console.print(f"[red]{cfg.discord_token_env} not set — "
                      "run `oi init` or export it[/red]")
        raise typer.Exit(1)
    store = Store(Path(cfg.db_path).expanduser())
    if log_answers:
        # Answer logging implies SSE transport; both switches documented in
        # llm.chat. Hidden reasoning content is never logged either way.
        os.environ["OI_LLM_LOG_ANSWERS"] = "1"
        console.print("[yellow]answer logging enabled — visible replies "
                      "will be written to logs[/yellow]")
    if stream or log_answers:
        os.environ["OI_LLM_STREAM_LOGS"] = "1"
    logging.basicConfig(level=logging.INFO)
    run_daemon(cfg, store, token)


@app.command()
def pause(config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")) -> None:
    """Kill switch: stop all posting until resumed."""
    store = Store(Path(_load_or_die(config).db_path).expanduser())
    store.set_paused(True)
    console.print("[yellow]paused[/yellow]")


@app.command()
def resume(config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")) -> None:
    """Resume posting after a pause."""
    store = Store(Path(_load_or_die(config).db_path).expanduser())
    store.set_paused(False)
    console.print("[green]resumed[/green]")


config_app = typer.Typer(help="View and edit configuration from the terminal.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Print the current configuration (secrets never shown)."""
    cfg = _load_or_die(config)
    console.print(f"[bold]config:[/bold] {config}")
    console.print(f"personality = {cfg.personality}")
    console.print(f"max_reply_chars = {cfg.max_reply_chars}")
    console.print(f"max_response_tokens = {cfg.max_response_tokens}")
    console.print(f"max_tool_iterations = {cfg.max_tool_iterations}")
    for name in ("fallback", "agent"):
        llm = getattr(cfg, name)
        key_set = bool(os.environ.get(llm.api_key_env) or
                       _secret_is_set(llm.api_key_env))
        console.print(f"[bold]{name}[/bold] {llm.base_url or '-'} | "
                      f"{llm.model or '-'} | key env {llm.api_key_env}"
                      f" {'(set)' if key_set else '(NOT set)'}")
    for t in cfg.watches:
        console.print(
            f"[watch] channel {t.channel_id} -> {t.repo_path} | "
            f"founders {t.founder_ids} | bot {t.bot_user_id} | "
            f"cap {t.post_hourly_cap}/h")


def _secret_is_set(key_env: str) -> bool:
    """Check whether a key exists in the secrets file without printing it.

    Args:
        key_env: Env var name to look up.

    Returns:
        True when present in ~/.config/oi/.env.
    """
    if not DEFAULT_SECRETS_PATH.exists():
        return False
    return any(line.startswith(f"{key_env}=") for line in
               DEFAULT_SECRETS_PATH.read_text(encoding="utf-8").splitlines())


def _coerce(value: str) -> str | int:
    """Convert numeric-looking CLI values to int.

    Args:
        value: Raw CLI argument.

    Returns:
        int when it parses as one, else the original string.
    """
    try:
        return int(value)
    except ValueError:
        return value


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="e.g. agent.model or db_path"),
    value: str = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Change one setting. Keys: personality, max_reply_chars,
    max_response_tokens, max_tool_iterations, discord_token_env, db_path,
    fallback.base_url|model|api_key_env|api_style|disable_thinking,
    agent.base_url|model|api_key_env|api_style|disable_thinking."""
    cfg = _load_or_die(config)
    section, _, leaf = key.partition(".")
    allowed_sections = {
        "personality", "max_reply_chars", "max_response_tokens",
        "max_tool_iterations",
        "discord_token_env", "db_path", "fallback", "agent",
    }
    llm_keys = {"base_url", "model", "api_key_env", "api_style",
                "disable_thinking"}
    if section not in allowed_sections:
        console.print(f"[red]unknown key '{key}'[/red]")
        raise typer.Exit(1)
    if section in ("fallback", "agent"):
        if leaf not in llm_keys:
            console.print(f"[red]unknown key '{key}' "
                          f"(use {'/'.join(sorted(llm_keys))})[/red]")
            raise typer.Exit(1)
        coerced = (value.strip().lower() in {"1", "true", "yes", "on"}
                   if leaf == "disable_thinking" else value)
        setattr(getattr(cfg, section), leaf, coerced)
    else:
        # Scalar keys take no dotted suffix; reject typos like
        # `personality.typo` instead of silently setting `personality`.
        if leaf:
            console.print(f"[red]'{section}' is a scalar setting; "
                          f"'{key}' has an unexpected '.{leaf}'[/red]")
            raise typer.Exit(1)
        setattr(cfg, section, _coerce(value))
    try:
        save_config(cfg, config)
    except ValueError as exc:
        # save_config validates before writing; the file on disk is untouched.
        console.print(f"[red]invalid value for {key}: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{key} = {value}[/green]")


@config_app.command("watch-add")
def config_watch_add(
    channel_id: int = typer.Option(..., help="Discord channel to watch"),
    repo_path: str = typer.Option(..., help="local repo clone path"),
    founder_ids: str = typer.Option("", help="comma-separated author ids"),
    bot_user_id: int = typer.Option(0, help="bot user id (mention trigger)"),
    post_hourly_cap: int = typer.Option(3, help="max replies per hour"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Add or replace a watch target from flags — no editor needed."""
    cfg = _load_or_die(config)
    resolved = Path(repo_path).expanduser().resolve()
    if not (resolved / ".git").exists():
        console.print(f"[red]{resolved} is not a git clone[/red]")
        raise typer.Exit(1)
    try:
        founders = [int(x) for x in founder_ids.split(",") if x.strip()]
    except ValueError:
        console.print("[red]founder ids must be numbers[/red]")
        raise typer.Exit(1)
    target = WatchTarget(
        channel_id=channel_id, repo_path=str(resolved),
        founder_ids=founders, bot_user_id=bot_user_id,
        post_hourly_cap=post_hourly_cap,
    )
    cfg.watches = [t for t in cfg.watches if t.channel_id != channel_id]
    cfg.watches.append(target)
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]invalid watch target: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]watching {channel_id} -> {target.repo_path}[/green]")
    console.print("run [bold]oi index[/bold] to build its manifest")


@config_app.command("watch-remove")
def config_watch_remove(
    channel_id: int = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Stop watching a channel."""
    cfg = _load_or_die(config)
    before = len(cfg.watches)
    cfg.watches = [t for t in cfg.watches if t.channel_id != channel_id]
    if len(cfg.watches) == before:
        console.print(f"[red]channel {channel_id} was not watched[/red]")
        raise typer.Exit(1)
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]cannot remove watch target: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]removed {channel_id}[/green]")


@config_app.command("secret")
def config_secret(
    name: str = typer.Argument(...,
                               help="env var name, e.g. OI_DISCORD_TOKEN"),
) -> None:
    """Set/replace one secret (hidden input) in ~/.config/oi/.env."""
    value = getpass.getpass(f"{name} (input hidden): ")
    if not value:
        console.print("[yellow]empty, nothing written[/yellow]")
        return
    DEFAULT_SECRETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if DEFAULT_SECRETS_PATH.exists():
        lines = [ln for ln in
                 DEFAULT_SECRETS_PATH.read_text(encoding="utf-8").splitlines()
                 if not ln.startswith(f"{name}=")]
    lines.append(f"{name}={value}")
    DEFAULT_SECRETS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    DEFAULT_SECRETS_PATH.chmod(0o600)
    console.print(f"[green]{name} updated[/green]")


if __name__ == "__main__":
    sys.exit(app())
