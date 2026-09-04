"""Command-line interface for the OpenCode-only OI daemon."""

from __future__ import annotations

import getpass
import logging
import os
import stat
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from .config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_PERSONALITY,
    DEFAULT_SECRETS_PATH,
    DISCORD_MAX_MESSAGE_CHARS,
    Config,
    WatchTarget,
    load_config,
    save_config,
)
from .daemon import (
    DEFAULT_DAEMON_USER,
    DEFAULT_SERVICE_NAME,
    DaemonError,
    stream_logs,
)
from .daemon import (
    doctor as run_doctor,
)
from .daemon import (
    install as install_daemon,
)
from .daemon import (
    lifecycle as daemon_lifecycle,
)
from .daemon import (
    uninstall as uninstall_daemon,
)
from .opencode.bootstrap import (
    INSTALL_DOCS_URL,
    OpenCodeBootstrapError,
    authenticate,
    authenticated,
    check_version,
    install_opencode,
    list_models,
)
from .opencode.paths import resolve_opencode_paths
from .store import Store

app = typer.Typer(
    help="OpenCode-only OI feedback-channel daemon and operator control CLI."
)
console = Console()


def _prompt_int(label: str, default: int) -> int:
    """Prompt for an integer value with a default."""
    while True:
        raw = Prompt.ask(label, default=str(default))
        try:
            val = int(raw)
            if val < 0:
                console.print("[red]must be non-negative[/red]")
                continue
            return val
        except ValueError:
            console.print("[red]must be an integer[/red]")


def _parse_id_list(raw: str, label: str) -> list[int]:
    """Parse comma-separated positive integer IDs."""
    if not raw.strip():
        return []
    res: list[int] = []
    try:
        for part in raw.split(","):
            part_str = part.strip()
            if not part_str:
                continue
            val = int(part_str)
            if val <= 0:
                console.print(f"[red]{label} IDs must be positive integers (> 0), got {val}[/red]")
                raise typer.Exit(1)
            res.append(val)
    except ValueError:
        console.print(f"[red]{label} must be comma-separated integers[/red]")
        raise typer.Exit(1)
    return res


def _ensure_opencode(requested_model: str | None, consent: bool) -> str:
    """Validate or bootstrap OpenCode dependency, auth, and model."""
    paths = resolve_opencode_paths()
    try:
        version = check_version("opencode", paths)
    except OpenCodeBootstrapError:
        if not consent:
            if not sys.stdin.isatty():
                console.print(
                    "[red]OpenCode is not installed on this system.[/red]\n"
                    f"Install it from {INSTALL_DOCS_URL} or run `oi init --install-opencode`."
                )
                raise typer.Exit(1)
            consent = typer.confirm(
                "OpenCode is not installed. Run official installer now?",
                default=False,
            )
            if not consent:
                raise typer.Exit(1)
        install_opencode(paths)
        version = check_version("opencode", paths)

    if not authenticated(version.binary, paths):
        console.print("[yellow]Authenticating OpenCode provider...[/yellow]")
        authenticate(version.binary, paths)

    models = list_models(version.binary, paths)
    if requested_model:
        if requested_model not in models:
            console.print(f"[yellow]warning: model {requested_model} not in detected models[/yellow]")
        return requested_model

    if models:
        return models[0]
    return "openai/gpt-4o-mini"


@app.command()
def init(
    model: str | None = typer.Option(None, "--model", help="Exact provider/model ID."),
    install_opencode: bool = typer.Option(
        False, "--install-opencode", help="Consent to the fixed official installer."
    ),
    yes: bool = typer.Option(
        False, "--yes", help="Non-interactive consent for OpenCode installation."
    ),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
    secrets: Path = typer.Option(DEFAULT_SECRETS_PATH, "--secrets"),
    install_service: bool = typer.Option(
        False, "--install-daemon", help="Install and start systemd daemon service."
    ),
    environment_mode: str | None = typer.Option(
        None, "--environment-mode", help="Execution environment perspective: workstation | server."
    ),
) -> None:
    """Create an OpenCode-only config after dependency/auth/model bootstrap."""
    console.print("[bold]OI init[/bold] — ctrl-c aborts.")
    consent = install_opencode or yes
    try:
        configured_model = _ensure_opencode(model, consent)
    except typer.Exit:
        raise

    def ask_secret(label: str) -> str:
        """Prompt for a secret without echoing it."""
        return getpass.getpass(f"{label} (input hidden): ")

    discord_token = ask_secret("Discord bot token")
    cfg = Config(opencode_model=configured_model)
    cfg.discord_token_env = Prompt.ask(
        "Env var name for the bot token", default="OI_DISCORD_TOKEN"
    )
    # Environment mode: default to server if install_service, else workstation, or user choice
    if environment_mode:
        if environment_mode not in ("workstation", "server"):
            console.print(f"[red]environment_mode must be 'workstation' or 'server', got {environment_mode!r}[/red]")
            raise typer.Exit(1)
        cfg.environment_mode = environment_mode
    else:
        default_env = "server" if install_service else "workstation"
        cfg.environment_mode = default_env
    cfg.personality = Prompt.ask(
        "Agent personality/tone", default=DEFAULT_PERSONALITY
    )
    while True:
        watch_type = Prompt.ask(
            "Watch target type [c=channel, g=guild/server, q=done]",
            choices=["c", "g", "q", "channel", "guild", "quit", "0"],
            default="c" if not cfg.watches else "q",
        ).lower()
        if watch_type in ("q", "quit", "0"):
            break

        channel_id = 0
        guild_id = 0
        ignored_channels: list[int] = []
        allowed_channels: list[int] = []

        if watch_type in ("c", "channel"):
            channel_id = _prompt_int("Discord channel id", 0)
            if channel_id <= 0:
                console.print("[yellow]skipping empty channel ID[/yellow]")
                continue
        elif watch_type in ("g", "guild"):
            guild_id = _prompt_int("Discord guild (server) id", 0)
            if guild_id <= 0:
                console.print("[yellow]skipping empty guild ID[/yellow]")
                continue
            ignored_raw = Prompt.ask(
                "Ignored channel IDs for guild watch (comma-separated, enter for none)",
                default="",
            )
            ignored_channels = _parse_id_list(ignored_raw, "ignored_channels")
            allowed_raw = Prompt.ask(
                "Allowed channel IDs for guild watch (comma-separated, enter for all)",
                default="",
            )
            allowed_channels = _parse_id_list(allowed_raw, "allowed_channels")

        raw_repos = Prompt.ask(
            "Watch: repo path(s) (comma-separated, enter = current directory)",
            default=str(Path.cwd()),
        )
        repo_paths: list[str] = []
        valid = True
        for part in raw_repos.split(","):
            p = part.strip()
            if not p:
                continue
            resolved = Path(p).expanduser().resolve()
            if not (resolved / ".git").exists():
                console.print(f"[red]{resolved} is not a git clone, skipping target[/red]")
                valid = False
                break
            repo_paths.append(str(resolved))
        if not valid or not repo_paths:
            continue

        cfg.watches.append(
            WatchTarget(
                channel_id=channel_id,
                guild_id=guild_id,
                repos=repo_paths,
                ignored_channels=ignored_channels,
                allowed_channels=allowed_channels,
                bot_user_id=_prompt_int("bot user id (0 for none)", 0),
                post_hourly_cap=_prompt_int("posts per hour cap", 3),
            )
        )
    if not cfg.watches:
        console.print("[red]no channels or guilds configured; nothing written[/red]")
        raise typer.Exit(1)
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]config error: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]wrote {config}[/green]")
    if discord_token:
        secrets.parent.mkdir(parents=True, exist_ok=True)
        secrets.write_text(
            f"{cfg.discord_token_env}={discord_token}\n", encoding="utf-8"
        )
        secrets.chmod(0o600)
        console.print(f"[green]wrote {secrets} (mode 600)[/green]")
    store = Store(Path(cfg.db_path).expanduser())
    store.set_installed_at(int(time.time()))
    store.close()
    should_install = install_service
    if not should_install and sys.stdin.isatty():
        should_install = (
            Prompt.ask("Install and start systemd daemon service now?", choices=["y", "n"], default="y") == "y"
        )
    if should_install:
        try:
            is_user = os.geteuid() != 0
            install_daemon(cfg, user_mode=is_user, config_path=config, secrets_path=secrets)
            mode_label = "user" if is_user else "system"
            console.print(f"[green]installed and started {mode_label} service oi.service[/green]")
        except Exception as exc:
            console.print(f"[yellow]could not auto-install daemon: {exc}[/yellow]")
            console.print("You can install it later with `oi daemon install`.")
def _load_or_die(path: Path) -> Config:
    """Load a config or exit with one actionable error."""
    try:
        return load_config(path) if path.exists() else Config()
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]config error: {exc}[/red]")
        raise typer.Exit(1)


@app.command()
def run(config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")) -> None:
    """Start the Discord watcher daemon."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s",
    )
    from .watch.discord_client import OIWatcher

    cfg = _load_or_die(config)
    load_secrets()
    token = os.environ.get(cfg.discord_token_env, "")
    if not token:
        console.print(
            f"[red]{cfg.discord_token_env} not set — run `oi init` or export it[/red]"
        )
        raise typer.Exit(1)
    store = Store(Path(cfg.db_path).expanduser())
    watcher = OIWatcher(cfg, store, token=token)
    watcher.run(token)


@app.command()
def pause(config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")) -> None:
    """Kill switch: stop all posting until resumed."""
    store = Store(Path(_load_or_die(config).db_path).expanduser())
    store.set_paused(True)
    store.close()
    console.print("[yellow]paused[/yellow]")


@app.command()
def resume(config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c")) -> None:
    """Clear the kill switch and resume posting."""
    store = Store(Path(_load_or_die(config).db_path).expanduser())
    store.set_paused(False)
    store.close()
    console.print("[green]resumed[/green]")


BOOTSTRAP_MAX_CHANNELS = 5
# Per-message collection bound for bootstrap; reuses the existing Discord char limit.
BOOTSTRAP_MAX_MESSAGE_CHARS = DISCORD_MAX_MESSAGE_CHARS


def _configured_bootstrap_channels(target: WatchTarget) -> list[int] | None:
    """Best-effort pre-connect resolution of scannable channel IDs for consent display.

    Returns the configured channel list, or ``None`` when the exact set can only
    be resolved after connecting (guild watch without an ``allowed_channels``
    list). Operational metadata only; never message content.
    """
    if not target.is_guild_watch:
        return [target.channel_id]
    if target.allowed_channels:
        return [c for c in target.allowed_channels if c not in target.ignored_channels]
    return None


def _stdin_isatty() -> bool:
    """Whether stdin is interactive; patched by tests to simulate a tty."""
    return sys.stdin.isatty()


def _print_dynamic_set_alternatives(target: WatchTarget, max_channels: int, *, third: str) -> None:
    """Print actionable alternatives for a dynamic (unresolved) channel set."""
    console.print("Choose one:")
    console.print(
        f"  1. configure allowed-channels on the watch target "
        f"(`oi config watch-add --guild-id {target.guild_id} --allowed-channels <ids>`)"
    )
    console.print(
        f"  2. pass --channels <id,id,...> with the exact IDs for this run (max {max_channels})"
    )
    console.print(f"  3. {third}")


def _parse_explicit_channels(raw: str) -> list[int]:
    """Parse a --channels value into a deduplicated, capped list of numeric IDs.

    Fail-closed on empty or non-numeric input. Duplicates are removed in order
    and more than ``BOOTSTRAP_MAX_CHANNELS`` IDs are capped to the first
    ``BOOTSTRAP_MAX_CHANNELS`` so a run never exceeds the advertised bound.
    """
    ids: list[int] = []
    for part in raw.split(","):
        part_str = part.strip()
        if not part_str:
            continue
        try:
            val = int(part_str)
        except ValueError:
            console.print(
                f"[red]--channels must be comma-separated numeric channel IDs, got {part_str!r}[/red]"
            )
            raise typer.Exit(1)
        if val <= 0:
            console.print(f"[red]--channels IDs must be positive integers (> 0), got {val}[/red]")
            raise typer.Exit(1)
        ids.append(val)
    if not ids:
        console.print("[red]--channels requires at least one numeric channel ID[/red]")
        raise typer.Exit(1)
    deduped = list(dict.fromkeys(ids))
    if len(deduped) > BOOTSTRAP_MAX_CHANNELS:
        capped = deduped[:BOOTSTRAP_MAX_CHANNELS]
        console.print(
            f"[yellow]--channels capped to the first {BOOTSTRAP_MAX_CHANNELS} IDs: "
            f"{', '.join(str(c) for c in capped)}[/yellow]"
        )
        deduped = capped
    return deduped


@app.command("bootstrap")
def bootstrap_cmd(
    scope: str = typer.Option(..., "--scope", "-s", help="Configured watch scope ID (channel ID or guild ID)."),
    limit: int = typer.Option(25, "--limit", "-l", help="Max messages per channel (max 50)."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview target scope/channels without reading or persisting."
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help=(
            "Non-interactive consent to analyze history. Valid only for an exact channel "
            "set (channel watch, configured allowed-channels, or --channels); a dynamic "
            "channel set fails closed with exit 1 before any connection."
        ),
    ),
    channels_opt: str | None = typer.Option(
        None, "--channels",
        help=(
            "Comma-separated exact channel IDs for this run; only valid for guild watches"
            " without a configured allowed-channels list. IDs must be numeric, duplicates"
            " are removed, and the list is capped at the first 5 IDs. With --channels,"
            " consent covers exactly these IDs and --yes may run non-interactively."
        ),
    ),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Bootstrap member profiles and team pulse from recent channel history with explicit consent.

    Consent is exact-known-channel. The consent matrix:

    - Exact channel set (channel watch, guild watch with a configured
      allowed-channels list, or explicit --channels): --yes runs
      non-interactively with zero prompts; interactive runs without --yes
      confirm once before connecting; non-interactive runs without --yes
      fail closed.
    - Dynamic channel set (guild watch without configured allowed-channels
      and without --channels): --yes fails closed immediately with exit 1,
      before any connection or setup, because --yes can never authorize an
      unresolved channel set; interactive runs without --yes connect
      read-only, display the resolved channel IDs with bounds, and require a
      second confirmation (default No) before any history is read, and
      declining aborts cleanly with exit code 0; non-interactive runs fail
      closed before any connection.
    - --dry-run never connects and only previews the intended scope.

    No prompt is ever reachable when --yes is passed, regardless of tty state.
    """
    import asyncio

    cfg = _load_or_die(config)
    target = None
    for t in cfg.watches:
        t_scope = str(t.guild_id if t.is_guild_watch else t.channel_id)
        if t_scope == str(scope):
            target = t
            break
    if target is None:
        console.print(f"[red]scope {scope!r} is not in configured watches[/red]")
        raise typer.Exit(1)

    bounded_limit = max(1, min(limit, 50))
    max_channels = BOOTSTRAP_MAX_CHANNELS
    max_total_msgs = bounded_limit * max_channels

    explicit_channels: list[int] | None = None
    if channels_opt is not None:
        if not target.is_guild_watch or target.allowed_channels:
            console.print(
                "[red]--channels applies only to guild watches without a configured "
                "allowed-channels list (this watch's exact channel set is already known)[/red]"
            )
            raise typer.Exit(1)
        explicit_channels = _parse_explicit_channels(channels_opt)

    configured_channels = (
        explicit_channels
        if explicit_channels is not None
        else _configured_bootstrap_channels(target)
    )
    dynamic_set = configured_channels is None

    console.print(f"[bold]Bootstrap Scope:[/bold] {scope} (is_guild_watch={target.is_guild_watch})")
    if dynamic_set:
        console.print(
            f"[bold]Channels:[/bold] up to {max_channels} guild text channels "
            f"(dynamic set: exact IDs resolve at connect and are re-confirmed before "
            f"any history read; ignored={target.ignored_channels or []})"
        )
    else:
        listed = ", ".join(str(c) for c in configured_channels) or "none"
        console.print(f"[bold]Channels ({len(configured_channels)}):[/bold] {listed}")
    console.print(
        f"[bold]Bounds:[/bold] max {bounded_limit} msgs/channel, "
        f"{max_total_msgs} messages total (enforced before analysis)"
    )
    console.print(
        "[bold]Retention:[/bold] extracted observations are stored in SQLite; "
        "raw message text is discarded immediately."
    )

    if dry_run:
        if dynamic_set:
            console.print(
                "[yellow]channel set is dynamic: exact channel IDs resolve at connect "
                "time (ignored channels excluded), so this preview cannot list them.[/yellow]"
            )
        console.print("[green]dry-run enabled: no history fetched, no model called, no memory written.[/green]")
        return

    interactive = _stdin_isatty()
    if dynamic_set and not interactive:
        console.print(
            "[red]cannot run non-interactively: this guild watch has no configured "
            "allowed-channels list, so the exact channel IDs to read cannot be known "
            "before connecting and cannot be consented to.[/red]"
        )
        _print_dynamic_set_alternatives(
            target, max_channels,
            third="run interactively (tty) to review the resolved channels before confirming",
        )
        raise typer.Exit(1)

    if dynamic_set and yes:
        # --yes means non-interactive pre-consent. It can never authorize a
        # channel set that is only resolvable after connecting, so fail closed
        # immediately: no prompt may be reachable with --yes regardless of tty.
        console.print(
            "[red]--yes cannot consent to a dynamic channel set: this guild watch has no "
            "configured allowed-channels list, so the exact channel IDs to read are only "
            "known after connecting.[/red]"
        )
        _print_dynamic_set_alternatives(
            target, max_channels,
            third=(
                "rerun interactively (tty) WITHOUT --yes to review the resolved "
                "channels and confirm before any history read"
            ),
        )
        raise typer.Exit(1)

    if not yes:
        if not interactive:
            console.print("[red]non-interactive execution requires --yes consent flag[/red]")
            raise typer.Exit(1)
        confirmed = Prompt.ask(
            "Proceed with reading channel history and synthesizing team memory?",
            choices=["y", "n"],
            default="n",
        ) == "y"
        if not confirmed:
            console.print("[yellow]bootstrap aborted by operator[/yellow]")
            return

    load_secrets()
    token = os.environ.get(cfg.discord_token_env, "")
    if not token:
        console.print(f"[red]{cfg.discord_token_env} not set[/red]")
        raise typer.Exit(1)

    import discord

    from .opencode.runner import OpenCodeRunner

    # Pinned contract (PIN-A): HistoryMessage + analyze_history live in reflection.engine.
    from .reflection.engine import HistoryMessage, ReflectionEngine

    store = Store(Path(cfg.db_path).expanduser())
    runner = OpenCodeRunner(cfg, store)
    engine = ReflectionEngine(runner, store)

    async def _run_bootstrap() -> dict:
        """Collect bounded chronological history, then run scoped analysis."""
        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        intents.guilds = True
        client = discord.Client(intents=intents)
        outcome: dict = {
            "guild_missing": False,
            "permission_failures": 0,
            "bot_excluded": 0,
            "system_excluded": 0,
            "empty_excluded": 0,
            "collected": [],
            "analysis": None,
            "analysis_error": "",
            "declined": False,
            "skipped_channels": [],
        }

        @client.event
        async def on_ready() -> None:
            channels_to_scan: list = []
            skipped_channels: list[int] = []
            if target.is_guild_watch:
                guild = discord.utils.get(client.guilds, id=target.guild_id)
                if guild is None:
                    outcome["guild_missing"] = True
                elif configured_channels is not None:
                    # Exact IDs were consented to pre-connect (configured list or
                    # --channels): resolve each one and skip/report the rest.
                    for cid in configured_channels:
                        ch_obj = discord.utils.get(getattr(guild, "text_channels", []), id=cid)
                        if ch_obj is not None and target.matches_channel(ch_obj.id, target.guild_id):
                            channels_to_scan.append(ch_obj)
                        else:
                            skipped_channels.append(cid)
                else:
                    for c in getattr(guild, "text_channels", []):
                        if target.matches_channel(c.id, target.guild_id):
                            channels_to_scan.append(c)
            else:
                ch = client.get_channel(target.channel_id)
                if ch is None:
                    try:
                        ch = await client.fetch_channel(target.channel_id)
                    except Exception:
                        outcome["permission_failures"] += 1
                if ch is not None:
                    channels_to_scan = [ch]

            channels_to_scan = channels_to_scan[:max_channels]
            outcome["skipped_channels"] = skipped_channels
            resolved_ids = ", ".join(str(c.id) for c in channels_to_scan) or "none"
            console.print(f"[bold]Resolved channels to scan ({len(channels_to_scan)}):[/bold] {resolved_ids}")

            if dynamic_set and channels_to_scan:
                # Two-stage consent: the exact set was unknowable before
                # connecting, so show it and re-confirm BEFORE any history read.
                console.print(
                    f"[bold]Exact scope for consent:[/bold] {len(channels_to_scan)} channel(s) "
                    f"[{resolved_ids}], max {bounded_limit} msgs/channel, "
                    f"{max_total_msgs} messages total"
                )
                confirmed = Prompt.ask(
                    "Read history from exactly these channels now?",
                    choices=["y", "n"],
                    default="n",
                ) == "y"
                if not confirmed:
                    outcome["declined"] = True
                    await client.close()
                    return

            collected: list[HistoryMessage] = []
            for ch in channels_to_scan:
                if len(collected) >= max_total_msgs:
                    break
                remaining = max_total_msgs - len(collected)
                batch = []  # newest-first, as discord.py yields history
                try:
                    async for m in ch.history(limit=min(bounded_limit, remaining)):
                        if getattr(m.author, "bot", False):
                            outcome["bot_excluded"] += 1
                            continue
                        if getattr(m, "type", None) != discord.MessageType.default:
                            outcome["system_excluded"] += 1
                            continue
                        if not (m.content or "").strip():
                            outcome["empty_excluded"] += 1
                            continue
                        batch.append(m)
                        if len(batch) >= remaining:
                            break
                except Exception:
                    outcome["permission_failures"] += 1
                    continue
                batch.reverse()  # analysis input must be chronological within each channel
                for m in batch:
                    created = getattr(m, "created_at", None)
                    collected.append(
                        HistoryMessage(
                            channel_id=str(ch.id),
                            author_id=str(m.author.id),
                            author_name=getattr(m.author, "display_name", getattr(m.author, "name", "member")),
                            timestamp_iso=created.isoformat() if created is not None else "",
                            content=(m.content or "")[:BOOTSTRAP_MAX_MESSAGE_CHARS],
                        )
                    )

            outcome["collected"] = collected
            if collected:
                console.print(f"Synthesizing dynamics for scope {scope} from {len(collected)} message(s)...")
                try:
                    outcome["analysis"] = await engine.analyze_history(
                        "discord",
                        scope,
                        collected,
                        max_messages=max_total_msgs,
                        max_total_chars=max_total_msgs * BOOTSTRAP_MAX_MESSAGE_CHARS,
                    )
                except ValueError as exc:
                    outcome["analysis_error"] = str(exc) or "history exceeded analysis bounds"
            await client.close()

        await client.start(token)
        return outcome

    try:
        result = asyncio.run(_run_bootstrap())
    except Exception as exc:
        console.print(f"[red]bootstrap failed: {exc}[/red]")
        raise typer.Exit(1)
    finally:
        store.close()

    if result["declined"]:
        console.print("[yellow]bootstrap aborted — no history read, nothing written, no model call.[/yellow]")
        return

    if result["skipped_channels"]:
        skipped_ids = ", ".join(str(c) for c in result["skipped_channels"])
        console.print(
            f"[yellow]skipped {len(result['skipped_channels'])} requested channel ID(s) "
            f"(not found or filtered by the watch): {skipped_ids}[/yellow]"
        )

    failures: list[str] = []
    if result["guild_missing"]:
        failures.append("guild not accessible to bot")
    if result["permission_failures"]:
        failures.append(f"{result['permission_failures']} channel(s) unreadable (permission or access failure)")
    if result["analysis_error"]:
        failures.append(f"history analysis rejected input bounds: {result['analysis_error']}")
    console.print(
        "[bold]Collection summary:[/bold] "
        f"collected={len(result['collected'])}/{max_total_msgs}, "
        f"bot-excluded={result['bot_excluded']}, "
        f"system-excluded={result['system_excluded']}, "
        f"empty-excluded={result['empty_excluded']}, "
        f"permission-failed={result['permission_failures']}"
    )
    if failures:
        for failure in failures:
            console.print(f"[red]bootstrap failure: {failure}[/red]")
        raise typer.Exit(1)

    if result["collected"]:
        analysis = result["analysis"]
        count = len(analysis.member_ids) if analysis is not None else 0
        console.print(f"[bold green]Bootstrap complete! Initialized {count} member profile(s).[/bold green]")
    else:
        console.print("[yellow]No accessible public messages found to analyze.[/yellow]")


@app.command()
def doctor(
    user: str | None = typer.Option(None, "--user", "-u", help="Service user name to check"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Run preflight health checks for OpenCode, model, repos, token, and state."""
    cfg = _load_or_die(config)
    load_secrets()
    failures = run_doctor(cfg, user=user)
    if failures:
        console.print("[bold red]Preflight check failed:[/bold red]")
        for f in failures:
            console.print(f"  [red]FAIL[/red] {f}")
        raise typer.Exit(1)
    console.print("[bold green]all checks passed[/bold green]")


def load_secrets(path: Path = DEFAULT_SECRETS_PATH) -> None:
    """Load key=value pairs into os.environ with restrictive permissions."""
    if not path.is_file():
        alt = path.parent / ".env"
        if alt.is_file():
            path = alt
        else:
            return
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        logging.warning("[secrets] %s has loose permissions (%o); should be 600", path, mode)
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


# ==============================================================================
# Config Subcommands
# ==============================================================================

config_app = typer.Typer(help="View and edit the OpenCode-only configuration.")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Print the non-secret configuration."""
    cfg = _load_or_die(config)
    console.print(f"[bold]config:[/bold] {config}")
    for name in (
        "discord_token_env", "db_path", "personality", "max_reply_chars",
        "reply_delivery", "opencode_binary", "opencode_model", "opencode_steps",
        "opencode_timeout_seconds", "environment_mode",
    ):
        console.print(f"{name} = {getattr(cfg, name)}")
    for target in cfg.watches:
        scope_str = f"guild {target.guild_id}" if target.is_guild_watch else f"channel {target.channel_id}"
        repos_str = ", ".join(target.repo_paths)
        filters = []
        if target.ignored_channels:
            filters.append(f"ignored={target.ignored_channels}")
        if target.allowed_channels:
            filters.append(f"allowed={target.allowed_channels}")
        filter_str = f" ({', '.join(filters)})" if filters else ""
        console.print(
            f"[watch] {scope_str}{filter_str} -> {repos_str} | cap {target.post_hourly_cap}/h"
        )


def _coerce(value: str) -> str | int:
    """Convert numeric CLI values to integers without broad coercion."""
    try:
        return int(value)
    except ValueError:
        return value


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="e.g. opencode_model, environment_mode, or db_path"),
    value: str = typer.Argument(...),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Change one OpenCode-only scalar setting atomically."""
    allowed = {
        "personality", "max_reply_chars", "reply_delivery", "discord_token_env",
        "db_path", "opencode_binary", "opencode_model", "opencode_steps",
        "opencode_timeout_seconds", "environment_mode",
    }
    if key not in allowed:
        console.print(f"[red]unknown key '{key}'[/red]")
        raise typer.Exit(1)
    try:
        # environment_mode is the documented repair path for legacy 'auto'
        # configs, so tolerate that one legacy value at load time; save_config
        # still validates the new value strictly before anything is written.
        cfg = (
            load_config(config, allow_legacy_environment_mode=(key == "environment_mode"))
            if config.exists()
            else Config()
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        console.print(f"[red]config error: {exc}[/red]")
        raise typer.Exit(1)
    setattr(cfg, key, _coerce(value))
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]invalid value for {key}: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]{key} = {value}[/green]")


@config_app.command("watch-add")
def config_watch_add(
    channel_id: int = typer.Option(0, help="Discord channel ID to watch (or 0 if using --guild-id)"),
    guild_id: int = typer.Option(0, help="Discord guild/server ID to watch (or 0 if using --channel-id)"),
    repo_path: str = typer.Option("", help="local repo clone path (single repo)"),
    repos: str = typer.Option("", help="comma-separated list of local repo clone paths (multi-repo)"),
    ignored_channels: str = typer.Option("", help="comma-separated channel IDs to ignore for guild watch"),
    allowed_channels: str = typer.Option("", help="comma-separated channel IDs to specifically allow for guild watch"),
    bot_user_id: int = typer.Option(0, help="bot user id (mention trigger)"),
    post_hourly_cap: int = typer.Option(3, help="max replies per hour"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Add or replace one watch target (channel or server-wide guild)."""
    cfg = _load_or_die(config)
    if channel_id < 0 or guild_id < 0:
        console.print("[red]channel ID and guild ID must be positive integers (> 0)[/red]")
        raise typer.Exit(1)
    if (channel_id > 0 and guild_id > 0) or (channel_id <= 0 and guild_id <= 0):
        console.print("[red]must specify exactly one positive --channel-id or --guild-id[/red]")
        raise typer.Exit(1)

    raw_paths: list[str] = []
    if repos.strip():
        raw_paths = [p.strip() for p in repos.split(",") if p.strip()]
    elif repo_path.strip():
        raw_paths = [repo_path.strip()]
    else:
        console.print("[red]must specify --repo-path or --repos[/red]")
        raise typer.Exit(1)

    resolved_paths: list[str] = []
    for p in raw_paths:
        resolved = Path(p).expanduser().resolve()
        if not (resolved / ".git").exists():
            console.print(f"[red]{resolved} is not a git clone[/red]")
            raise typer.Exit(1)
        resolved_paths.append(str(resolved))

    ignored_list = _parse_id_list(ignored_channels, "ignored_channels")
    allowed_list = _parse_id_list(allowed_channels, "allowed_channels")

    if channel_id > 0:
        cfg.watches = [t for t in cfg.watches if t.channel_id != channel_id]
    else:
        cfg.watches = [t for t in cfg.watches if t.guild_id != guild_id]

    cfg.watches.append(WatchTarget(
        channel_id=channel_id,
        repo_path=resolved_paths[0],
        bot_user_id=bot_user_id,
        post_hourly_cap=post_hourly_cap,
        guild_id=guild_id,
        repos=resolved_paths,
        ignored_channels=ignored_list,
        allowed_channels=allowed_list,
    ))
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]invalid watch target: {exc}[/red]")
        raise typer.Exit(1)
    scope_str = f"channel {channel_id}" if channel_id > 0 else f"guild {guild_id}"
    console.print(f"[green]watching {scope_str} -> {', '.join(resolved_paths)}[/green]")


@config_app.command("watch-remove")
def config_watch_remove(
    target_id: int = typer.Argument(..., help="channel ID or guild ID to stop watching"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Stop watching a channel or guild."""
    if target_id <= 0:
        console.print("[red]target ID must be a positive integer (> 0)[/red]")
        raise typer.Exit(1)
    cfg = _load_or_die(config)
    before = len(cfg.watches)
    cfg.watches = [t for t in cfg.watches if t.channel_id != target_id and t.guild_id != target_id]
    if len(cfg.watches) == before:
        console.print(f"[red]channel/guild {target_id} was not watched[/red]")
        raise typer.Exit(1)
    try:
        save_config(cfg, config)
    except ValueError as exc:
        console.print(f"[red]cannot remove watch target: {exc}[/red]")
        raise typer.Exit(1)
    console.print(f"[green]removed watch {target_id}[/green]")


@config_app.command("secret")
def config_secret(
    name: str = typer.Argument(..., help="configured Discord token env var"),
    value: str = typer.Argument(..., help="token secret value"),
    secrets: Path = typer.Option(DEFAULT_SECRETS_PATH, "--secrets"),
) -> None:
    """Store or update the bot token in mode-0600 secrets file."""
    secrets.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, str] = {}
    if secrets.is_file():
        for raw in secrets.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            existing[k.strip()] = v.strip().strip("'\"")
    existing[name] = value
    lines = [f"{k}={v}" for k, v in existing.items()]
    secrets.write_text("\n".join(lines) + "\n", encoding="utf-8")
    secrets.chmod(0o600)
    console.print(f"[green]{name} updated[/green]")


# ==============================================================================
# Queue Subcommands
# ==============================================================================

queue_app = typer.Typer(help="Inspect and manage durable delivery queue.")
app.add_typer(queue_app, name="queue")


@queue_app.command("status")
def queue_status(
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Display queue and outbox state metrics."""
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        stats = store.get_queue_stats()
        table = Table(title="Queue Status")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="magenta")

        table.add_row("Pending Jobs", str(stats.get("pending_jobs", 0)))
        table.add_row("Leased Jobs", str(stats.get("leased_jobs", 0)))
        table.add_row("Done Jobs", str(stats.get("done_jobs", 0)))
        table.add_row("Dead Jobs", str(stats.get("dead_jobs", 0)))
        table.add_row("Pending Batches", str(stats.get("pending_batches", 0)))
        table.add_row("Leased Batches", str(stats.get("leased_batches", 0)))
        table.add_row("Done Batches", str(stats.get("done_batches", 0)))
        table.add_row("Dead Batches", str(stats.get("dead_batches", 0)))
        table.add_row("Stale Leases", str(stats.get("stale_leases", 0)))
        console.print(table)
    finally:
        store.close()


@queue_app.command("retry")
def queue_retry(
    all_jobs: bool = typer.Option(False, "--all", help="Retry all dead jobs"),
    job_id: int | None = typer.Option(None, "--id", help="Specific message ID to retry"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Reset dead-lettered jobs back to pending."""
    if not all_jobs and job_id is None:
        console.print("[red]must specify --all or --id[/red]")
        raise typer.Exit(1)
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        ids: list[int] | None = None if all_jobs else ([job_id] if job_id is not None else None)
        count = store.retry_dead_jobs(ids)
        console.print(f"[green]retried {count} job(s)[/green]")
    finally:
        store.close()


@queue_app.command("purge")
def queue_purge(
    older_than_days: int = typer.Option(7, "--older-than-days", help="Purge records older than N days"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Purge completed jobs and batches older than threshold."""
    if older_than_days < 0:
        console.print("[red]--older-than-days must be non-negative[/red]")
        raise typer.Exit(1)
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        count = store.purge_completed(older_than_seconds=older_than_days * 86400)
        console.print(f"[green]purged {count} completed record(s)[/green]")
    finally:
        store.close()



# ==============================================================================
# Memory Subcommands
# ==============================================================================

memory_app = typer.Typer(help="Inspect, calibrate, prune, and reset dynamic team memory.")
app.add_typer(memory_app, name="memory")


@memory_app.command("status")
def memory_status(
    scope: str = typer.Option(..., "--scope", "-s", help="Target scope ID to inspect"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Display memory metrics for a specific scope."""
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        stats = store.get_memory_status("discord", scope)
        table = Table(title=f"Memory Status for Scope: {scope}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Schema Version", str(stats.get("schema_version", 2)))
        table.add_row("Member Profiles", str(stats.get("member_profiles_count", 0)))
        table.add_row("Active Transient State", str(stats.get("active_transient_count", 0)))
        table.add_row("Expired Transient State", str(stats.get("expired_transient_count", 0)))
        table.add_row("Has Active Team Pulse", str(stats.get("has_active_team_pulse", False)))
        table.add_row("Has Active Agent Calibration", str(stats.get("has_active_agent_calibration", False)))
        oldest = stats.get("oldest_profile_at")
        newest = stats.get("newest_profile_at")
        table.add_row("Oldest Profile Timestamp", str(oldest) if oldest else "none")
        table.add_row("Newest Profile Timestamp", str(newest) if newest else "none")
        console.print(table)
    finally:
        store.close()


@memory_app.command("show")
def memory_show(
    scope: str = typer.Option(..., "--scope", "-s", help="Target scope ID"),
    member: str = typer.Option(..., "--member", "-m", help="Member ID to inspect"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Show structured profile and confidence for a specific member in a scope."""
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        prof = store.get_member_profile("discord", scope, member)
        if not prof:
            console.print(f"[yellow]no profile found for member {member} in scope {scope}[/yellow]")
            return
        trans = store.get_transient_member_state("discord", scope, member)

        table = Table(title=f"Member Profile: {member} (Scope: {scope})")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Confidence", style="magenta")
        table.add_column("Evidence Count", style="yellow")

        comm = prof.get("communication", {})
        conf = prof.get("confidence", {})
        ev = prof.get("evidence_count", {})

        for k, v in comm.items():
            table.add_row(k, str(v), str(conf.get(k, 0.5)), str(ev.get(k, 1)))

        topics = ", ".join(prof.get("recurring_topics", [])) or "none"
        table.add_row("recurring_topics", topics, "-", "-")

        if trans:
            table.add_row("transient.energy", trans.get("energy", "unknown"), "-", "-")
            focus = ", ".join(trans.get("current_focus", [])) or "none"
            table.add_row("transient.current_focus", focus, "-", "-")

        console.print(table)
    finally:
        store.close()


@memory_app.command("reset-member")
def memory_reset_member(
    scope: str = typer.Option(..., "--scope", "-s", help="Target scope ID"),
    member: str = typer.Option(..., "--member", "-m", help="Member ID to reset"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive confirmation"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Delete all behavioral memory for a member in a scope."""
    if not yes:
        if not sys.stdin.isatty():
            console.print("[red]non-interactive execution requires --yes confirmation flag[/red]")
            raise typer.Exit(1)
        confirmed = Prompt.ask(
            f"Permanently delete memory for member {member} in scope {scope}?",
            choices=["y", "n"],
            default="n",
        ) == "y"
        if not confirmed:
            console.print("[yellow]aborted[/yellow]")
            return

    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        count = store.reset_member_memory("discord", scope, member)
        console.print(f"[green]deleted {count} record(s) for member {member} in scope {scope}[/green]")
    finally:
        store.close()


@memory_app.command("reset-scope")
def memory_reset_scope(
    scope: str = typer.Option(..., "--scope", "-s", help="Target scope ID to wipe"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive confirmation"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Delete all behavioral memory for an entire scope."""
    if not yes:
        if not sys.stdin.isatty():
            console.print("[red]non-interactive execution requires --yes confirmation flag[/red]")
            raise typer.Exit(1)
        confirmed = Prompt.ask(
            f"Permanently wipe ALL memory for scope {scope}?",
            choices=["y", "n"],
            default="n",
        ) == "y"
        if not confirmed:
            console.print("[yellow]aborted[/yellow]")
            return

    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        counts = store.reset_scope_memory("discord", scope)
        total = sum(counts.values())
        console.print(f"[green]wiped {total} record(s) for scope {scope}[/green]")
    finally:
        store.close()


@memory_app.command("prune")
def memory_prune(
    scope: str | None = typer.Option(None, "--scope", "-s", help="Optional scope to prune"),
    retention_days: int = typer.Option(90, "--retention-days", help="Max days for stable profiles (minimum 1)"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Prune expired transient records, team pulse, calibration, and stale stable profiles."""
    if retention_days < 1:
        console.print(
            "[red]--retention-days must be at least 1 "
            "(values below 1 would delete every stable profile)[/red]"
        )
        raise typer.Exit(1)
    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        counts = store.prune_memory(scope_id=scope, stable_retention_seconds=retention_days * 86400)
        total = sum(counts.values())
        console.print(f"[green]pruned {total} expired/stale memory record(s)[/green]")
    finally:
        store.close()


@memory_app.command("migrate-legacy")
def memory_migrate_legacy(
    scope: str = typer.Option(..., "--scope", "-s", help="Target scope to assign legacy profiles to"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive confirmation"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Migrate un-scoped legacy V1 profiles into V2 under a confirmed scope."""
    # Pinned contract (PIN-B): fail-closed migration error lives in store.
    from .store import LegacyMigrationError

    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        preview = store.preview_legacy_migration()
        if not preview:
            console.print("[yellow]no legacy V1 records found to migrate[/yellow]")
            return
        console.print(f"Found {len(preview)} legacy profile(s) to migrate to scope {scope}:")
        for p in preview:
            console.print(f"  - @{p['handle']} (ID: {p['member_id']})")
        if not yes:
            if not sys.stdin.isatty():
                console.print("[red]non-interactive execution requires --yes confirmation flag[/red]")
                raise typer.Exit(1)
            confirmed = Prompt.ask(
                f"Migrate these {len(preview)} records to scope {scope}?",
                choices=["y", "n"],
                default="n",
            ) == "y"
            if not confirmed:
                console.print("[yellow]aborted[/yellow]")
                return

        try:
            count = store.migrate_legacy_profiles(scope)
        except LegacyMigrationError as exc:
            skipped = getattr(exc, "skipped", 0)
            console.print(
                f"[red]legacy migration failed: skipped {skipped} corrupt record(s); nothing migrated.[/red]"
            )
            console.print(
                "Run `oi memory purge-legacy --yes` if you want to delete the legacy records instead."
            )
            raise typer.Exit(1)
        console.print(
            f"[bold green]successfully migrated {count} legacy profile(s) to scope {scope}.[/bold green]"
        )
        console.print("[green]legacy tables were dropped.[/green]")
    finally:
        store.close()


@memory_app.command("purge-legacy")
def memory_purge_legacy(
    yes: bool = typer.Option(False, "--yes", "-y", help="Non-interactive confirmation"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Permanently delete and drop all legacy V1 tables."""
    if not yes:
        if not sys.stdin.isatty():
            console.print("[red]non-interactive execution requires --yes confirmation flag[/red]")
            raise typer.Exit(1)
        confirmed = Prompt.ask(
            "Permanently delete and drop all legacy V1 tables?",
            choices=["y", "n"],
            default="n",
        ) == "y"
        if not confirmed:
            console.print("[yellow]aborted[/yellow]")
            return

    cfg = _load_or_die(config)
    store = Store(Path(cfg.db_path).expanduser())
    try:
        count = store.purge_legacy_profiles()
        console.print(f"[green]purged and dropped legacy tables ({count} profile records dropped)[/green]")
    finally:
        store.close()
# ==============================================================================
# Daemon Subcommands
# ==============================================================================

daemon_app = typer.Typer(help="Manage unprivileged systemd service unit.")
app.add_typer(daemon_app, name="daemon")


@daemon_app.command("install")
def daemon_install(
    user: str = typer.Option(DEFAULT_DAEMON_USER, "--user", "-u"),
    user_mode: bool = typer.Option(False, "--user-mode", help="Install as a systemd user service"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Render unit file without applying"),
    service_name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name"),
    config: Path = typer.Option(DEFAULT_CONFIG_PATH, "--config", "-c"),
) -> None:
    """Install and enable systemd user or system service."""
    cfg = _load_or_die(config)
    is_user = user_mode or (os.geteuid() != 0 and not dry_run)
    try:
        rendered = install_daemon(
            cfg, user=user, dry_run=dry_run, service_name=service_name, user_mode=is_user
        )
        if dry_run:
            console.print(rendered)
        else:
            mode_label = "user" if is_user else "system"
            console.print(f"[green]installed and started {mode_label} service {service_name}.service[/green]")
    except DaemonError as exc:
        console.print(f"[red]daemon install failed: {exc}[/red]")
        raise typer.Exit(1)


def _is_user_service(service_name: str) -> bool:
    return os.geteuid() != 0 or (Path.home() / f".config/systemd/user/{service_name}.service").exists()


@daemon_app.command("status")
def daemon_status(
    service_name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name"),
) -> None:
    """Show systemctl status for the daemon service."""
    try:
        out = daemon_lifecycle("status", service_name=service_name, user_mode=_is_user_service(service_name))
        console.print(out)
    except DaemonError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@daemon_app.command("restart")
def daemon_restart(
    service_name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name"),
) -> None:
    """Restart the daemon service."""
    try:
        daemon_lifecycle("restart", service_name=service_name, user_mode=_is_user_service(service_name))
        console.print(f"[green]restarted {service_name}.service[/green]")
    except DaemonError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)


@daemon_app.command("logs")
def daemon_logs(
    follow: bool = typer.Option(False, "--follow", "-f", help="Follow journal output interactively"),
    service_name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name"),
) -> None:
    """Show or follow journalctl logs for the daemon service."""
    is_user = _is_user_service(service_name)
    if follow:
        stream_logs(service_name=service_name, user_mode=is_user)
    else:
        try:
            out = daemon_lifecycle("logs", service_name=service_name, user_mode=is_user)
            console.print(out)
        except DaemonError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(1)
def daemon_uninstall(
    purge: bool = typer.Option(False, "--purge", help="Also remove service user account"),
    user: str = typer.Option(DEFAULT_DAEMON_USER, "--user", "-u"),
    service_name: str = typer.Option(DEFAULT_SERVICE_NAME, "--name"),
) -> None:
    """Stop, disable, and remove daemon service."""
    try:
        uninstall_daemon(service_name=service_name, purge=purge, user=user)
        console.print(f"[green]uninstalled {service_name}.service[/green]")
    except DaemonError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1)
