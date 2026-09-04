"""Unprivileged systemd installation and OpenCode-aware daemon diagnostics."""

from __future__ import annotations

import os
import pwd
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, DEFAULT_SECRETS_PATH, Config
from .opencode.bootstrap import (
    OpenCodeBootstrapError,
    authenticated,
    check_version,
    list_models,
)
from .opencode.paths import resolve_opencode_paths
from .store import Store

SERVICE_DIR = Path("/etc/systemd/system")
DEFAULT_SERVICE_NAME = "oi"
DEFAULT_DAEMON_USER = "oi"


class DaemonError(RuntimeError):
    """Raised for unsafe or incomplete daemon provisioning operations."""


@dataclass(frozen=True)
class DaemonIdentity:
    """Dedicated service account identity used by generated units."""

    user: str
    home: Path
    uid: int
    gid: int


def _identity(user: str, home: Path | None = None) -> DaemonIdentity:
    """Resolve a system user without copying personal operator state."""
    try:
        record = pwd.getpwnam(user)
    except KeyError as exc:
        if home is None:
            raise DaemonError(
                f"service user {user!r} does not exist; run daemon install as root"
            ) from exc
        return DaemonIdentity(user, home, -1, -1)
    return DaemonIdentity(user, Path(record.pw_dir), record.pw_uid, record.pw_gid)


def _quote(value: str | Path) -> str:
    """Quote one systemd Environment value."""
    text = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{text}"'


def render_unit(cfg: Config, identity: DaemonIdentity,
                *, service_name: str = DEFAULT_SERVICE_NAME,
                config_path: Path | None = None,
                secrets_path: Path | None = None,
                oi_binary: str = "/usr/local/bin/oi",
                user_mode: bool = False) -> str:
    """Render the systemd unit without performing filesystem actions.

    Args:
        cfg: Validated OpenCode-only configuration.
        identity: Dedicated unprivileged service user.
        service_name: Unit name without the ``.service`` suffix.
        config_path: OI config path owned by the daemon user.
        secrets_path: OI Discord-token env file.
        oi_binary: Deterministic OI executable path used by ExecStart and PATH.
        user_mode: Render as user-level service without User=/Group= directives.

    Returns:
        Complete systemd unit text.
    """
    config_path = config_path or identity.home / ".config/oi/config.toml"
    secrets_path = secrets_path or identity.home / ".config/oi/.env"
    paths = resolve_opencode_paths(identity.home)
    db_value = cfg.db_path
    db_path = (identity.home / db_value[2:].lstrip("/")
               if db_value.startswith("~/")
               else Path(db_value).expanduser())
    if not db_path.is_absolute():
        db_path = identity.home / db_path
    read_write = {
        config_path.parent,
        secrets_path.parent,
        db_path.parent,
        paths.config_home,
        paths.data_home,
        paths.state_home,
        paths.cache_home,
        *(Path(p).expanduser().resolve() for target in cfg.watches for p in target.repo_paths if p),
    }
    rw_paths = " ".join(sorted(str(path) for path in read_write))
    opencode_dir = ""
    opencode_bin = shutil.which(cfg.opencode_binary) or shutil.which("opencode")
    if opencode_bin:
        opencode_dir = f":{Path(opencode_bin).parent}"
    path_value = f"{Path(oi_binary).parent}:{identity.home}/.local/bin{opencode_dir}:/usr/local/bin:/usr/bin"
    if user_mode:
        return "\n".join([
            "[Unit]",
            "Description=OI Discord OpenCode auditor",
            "After=network-online.target",
            "Wants=network-online.target",
            "",
            "[Service]",
            "Type=simple",
            f"WorkingDirectory={identity.home}",
            f"Environment=HOME={_quote(identity.home)}",
            f"Environment=PATH={_quote(path_value)}",
            f"Environment=OI_CONFIG_PATH={_quote(config_path)}",
            f"Environment=OI_SECRETS_PATH={_quote(secrets_path)}",
            "Environment=OPENCODE_DISABLE_PROJECT_CONFIG=1",
            f"EnvironmentFile=-{secrets_path}",
            f"ExecStart={oi_binary} run --config {config_path}",
            "Restart=always",
            "RestartSec=10",
            "",
            "[Install]",
            "WantedBy=default.target",
            "",
        ])
    return "\n".join([
        "[Unit]",
        "Description=OI Discord OpenCode auditor",
        "After=network-online.target",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"User={identity.user}",
        f"Group={identity.user}",
        f"WorkingDirectory={identity.home}",
        f"Environment=HOME={_quote(identity.home)}",
        f"Environment=PATH={_quote(path_value)}",
        f"Environment=OI_CONFIG_PATH={_quote(config_path)}",
        f"Environment=OI_SECRETS_PATH={_quote(secrets_path)}",
        f"Environment=XDG_CONFIG_HOME={_quote(paths.config_home)}",
        f"Environment=XDG_DATA_HOME={_quote(paths.data_home)}",
        f"Environment=XDG_STATE_HOME={_quote(paths.state_home)}",
        f"Environment=XDG_CACHE_HOME={_quote(paths.cache_home)}",
        "Environment=OPENCODE_DISABLE_PROJECT_CONFIG=1",
        f"EnvironmentFile=-{secrets_path}",
        f"ExecStart={oi_binary} run --config {config_path}",
        "Restart=always",
        "RestartSec=10",
        "NoNewPrivileges=true",
        "PrivateTmp=true",
        "ProtectSystem=strict",
        "ProtectHome=read-only",
        "ProtectKernelTunables=true",
        "ProtectKernelModules=true",
        "ProtectControlGroups=true",
        "RestrictSUIDSGID=true",
        "LockPersonality=true",
        "RestrictRealtime=true",
        "RestrictNamespaces=true",
        f"ReadWritePaths={rw_paths}",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ])


def _run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one fixed system administration argv list."""
    return subprocess.run(command, capture_output=True, text=True,
                          check=check, timeout=120)


def ensure_user(identity: DaemonIdentity, *, dry_run: bool = False) -> None:
    """Create the dedicated system user and private directory hierarchy."""
    if identity.uid == -1:
        if dry_run:
            return
        command = ["useradd", "--system", "--create-home", "--shell",
                   "/usr/sbin/nologin", "--home-dir", str(identity.home),
                   identity.user]
        result = _run(command)
        if result.returncode != 0:
            raise DaemonError(f"could not create service user {identity.user!r}")
        identity = _identity(identity.user, identity.home)
    if dry_run:
        return
    for path in (
        identity.home / ".config",
        identity.home / ".config/oi",
        identity.home / ".local",
        identity.home / ".local/state",
        identity.home / ".local/state/oi",
        identity.home / ".local/share",
        identity.home / ".local/share/oi",
        identity.home / ".local/bin",
    ):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        path.chmod(0o700)
        if identity.uid != -1 and identity.gid != -1:
            try:
                shutil.chown(path, identity.user, identity.user)
            except (OSError, PermissionError):
                pass


def ensure_opencode_for_user(cfg: Config, identity: DaemonIdentity,
                             *, dry_run: bool = False) -> None:
    """Ensure the daemon user can execute OpenCode and owns auth state."""
    if dry_run:
        return
    paths = resolve_opencode_paths(identity.home)
    paths.ensure(uid=identity.uid, gid=identity.gid)
    for parent in (
        identity.home / ".local",
        identity.home / ".local/share",
        identity.home / ".local/share/oi",
        identity.home / ".local/share/oi/opencode",
    ):
        parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        parent.chmod(0o700)
        if identity.uid != -1 and identity.gid != -1:
            try:
                shutil.chown(parent, identity.user, identity.user)
            except (OSError, PermissionError):
                pass
    candidates = (
        identity.home / ".local/bin/opencode",
        Path("/usr/local/bin/opencode"),
        Path("/usr/bin/opencode"),
    )
    configured = Path(cfg.opencode_binary).expanduser()
    if configured.is_absolute():
        executable = configured if configured.is_file() and os.access(configured, os.X_OK) else None
    else:
        executable = next(
            (candidate for candidate in candidates
             if candidate.is_file() and os.access(candidate, os.X_OK)),
            None,
        )
    if executable is None or not executable.is_file() \
            or not os.access(executable, os.X_OK):
        raise DaemonError(
            f"OpenCode is not executable for service user {identity.user!r}; "
            f"install it into {identity.home}/.local/bin or /usr/local/bin"
        )
    auth_file = paths.data_home / "opencode" / "auth.json"
    if not auth_file.is_file():
        raise DaemonError(
            f"OpenCode auth is missing for {identity.user!r}; run "
            f"`sudo -u {identity.user} -H oi init --model "
            f"{cfg.opencode_model}` before starting the service"
        )

def _copy_owned(source: Path, destination: Path, identity: DaemonIdentity) -> None:
    """Copy one explicitly allowed OI file with restrictive permissions."""
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    shutil.copy2(source, destination)
    destination.chmod(0o600)
    shutil.chown(destination, identity.user, identity.user)


def install(cfg: Config, *, user: str = DEFAULT_DAEMON_USER,
            service_name: str = DEFAULT_SERVICE_NAME,
            config_path: Path = DEFAULT_CONFIG_PATH,
            secrets_path: Path = DEFAULT_SECRETS_PATH,
            dry_run: bool = False,
            service_dir: Path | None = None,
            user_mode: bool = False) -> str:
    """Provision the daemon user and write/enable a systemd unit.

    Returns:
        Rendered unit text; dry-run callers can print it directly.

    Raises:
        DaemonError: If root privileges (for system mode), config, or commands are missing.
    """
    if not user_mode and os.geteuid() != 0 and not dry_run:
        raise DaemonError("daemon install requires root; use sudo or --dry-run")

    oi_bin = shutil.which("oi") or "/usr/local/bin/oi"

    if user_mode:
        actual_user = os.getenv("USER") or pwd.getpwuid(os.getuid()).pw_name
        identity = DaemonIdentity(actual_user, Path.home(), os.getuid(), os.getgid())
        daemon_config = config_path if config_path.exists() else identity.home / ".config/oi/config.toml"
        daemon_secrets = secrets_path if secrets_path.exists() else identity.home / ".config/oi/.env"
        unit = render_unit(cfg, identity, service_name=service_name,
                           config_path=daemon_config, secrets_path=daemon_secrets,
                           oi_binary=oi_bin, user_mode=True)
        if dry_run:
            return unit
        dest_dir = service_dir or (identity.home / ".config/systemd/user")
        dest_dir.mkdir(parents=True, exist_ok=True)
        unit_path = dest_dir / f"{service_name}.service"
        unit_path.write_text(unit, encoding="utf-8")
        for command in (["systemctl", "--user", "daemon-reload"],
                        ["systemctl", "--user", "enable", "--now", f"{service_name}.service"]):
            result = _run(command)
            if result.returncode != 0:
                raise DaemonError(f"systemctl --user command failed: {result.stderr or result.stdout}")
        return unit

    identity = _identity(user, Path(f"/home/{user}"))
    ensure_user(identity, dry_run=dry_run)
    ensure_opencode_for_user(cfg, identity, dry_run=dry_run)
    daemon_config = identity.home / ".config/oi/config.toml"
    daemon_secrets = identity.home / ".config/oi/.env"
    unit = render_unit(cfg, identity, service_name=service_name,
                       config_path=daemon_config, secrets_path=daemon_secrets,
                       oi_binary=oi_bin, user_mode=False)
    if dry_run:
        return unit
    if not config_path.exists():
        raise DaemonError(f"missing OI config {config_path}; run `oi init` first")
    _copy_owned(config_path, daemon_config, identity)
    _copy_owned(secrets_path, daemon_secrets, identity)
    dest_dir = service_dir or SERVICE_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    unit_path = dest_dir / f"{service_name}.service"
    unit_path.write_text(unit, encoding="utf-8")
    unit_path.chmod(0o644)
    for command in (["systemctl", "daemon-reload"],
                    ["systemctl", "enable", "--now", f"{service_name}.service"]):
        result = _run(command)
        if result.returncode != 0:
            raise DaemonError(f"systemctl command failed: {command[1]}")
    return unit


def stream_logs(service_name: str = DEFAULT_SERVICE_NAME, *, user_mode: bool = False) -> None:
    """Stream journalctl output interactively without timeout."""
    command = (
        ["journalctl", "--user", "-u", f"{service_name}.service", "-f"]
        if user_mode
        else ["journalctl", "-u", f"{service_name}.service", "-f"]
    )
    try:
        subprocess.run(command, check=False)
    except KeyboardInterrupt:
        pass


def lifecycle(action: str, service_name: str = DEFAULT_SERVICE_NAME,
              *, user_mode: bool = False) -> str:
    """Run one non-destructive systemd/journalctl lifecycle command."""
    if action not in {"status", "restart", "logs"}:
        raise DaemonError(f"unsupported daemon action: {action}")
    command = (
        ["journalctl", "--user", "-u", f"{service_name}.service", "--no-pager"]
        if action == "logs" and user_mode
        else ["journalctl", "-u", f"{service_name}.service", "--no-pager"]
        if action == "logs"
        else (
            ["systemctl", "--user", action, f"{service_name}.service"]
            if user_mode
            else ["systemctl", action, f"{service_name}.service"]
        )
    )
    result = _run(command)
    output = (result.stdout or "").strip()
    if result.returncode != 0:
        raise DaemonError(output or f"{command[0]} exited {result.returncode}")
    return output


def uninstall(service_name: str = DEFAULT_SERVICE_NAME,
              *, purge: bool = False, user: str = DEFAULT_DAEMON_USER,
              service_dir: Path | None = None,
              user_mode: bool = False) -> None:
    """Stop/remove a service and optionally remove its dedicated user."""
    if user_mode:
        _run(["systemctl", "--user", "disable", "--now", f"{service_name}.service"])
        unit_dir = service_dir or (Path.home() / ".config/systemd/user")
        unit_path = unit_dir / f"{service_name}.service"
        if unit_path.exists():
            unit_path.unlink()
        _run(["systemctl", "--user", "daemon-reload"])
        return

    if os.geteuid() != 0:
        raise DaemonError("daemon uninstall requires root")
    dest_dir = service_dir or SERVICE_DIR
    _run(["systemctl", "disable", "--now", f"{service_name}.service"])
    unit_path = dest_dir / f"{service_name}.service"
    if unit_path.exists():
        unit_path.unlink()
    _run(["systemctl", "daemon-reload"])
    if purge:
        result = _run(["userdel", "--remove", user])
        if result.returncode != 0:
            raise DaemonError(f"could not remove service user {user!r}")


def doctor(cfg: Config, *, user: str | None = None) -> list[str]:
    """Run OpenCode, auth, model, repository, token, and state preflight checks.

    Args:
        cfg (Config): OpenCode and daemon configuration.
        user (str | None): Optional service user name to validate permissions for.

    Returns:
        list[str]: Human-readable failed checks; an empty list means all checks passed.
    """
    failures: list[str] = []
    identity = None
    if user:
        try:
            identity = _identity(user)
        except DaemonError as exc:
            failures.append(str(exc))
    paths = resolve_opencode_paths(identity.home if identity else None)
    try:
        version = check_version(cfg.opencode_binary, paths)
        if not authenticated(version.binary, paths):
            failures.append("no OpenCode provider is authenticated")
        if cfg.opencode_model not in list_models(version.binary, paths):
            failures.append(f"selected model unavailable: {cfg.opencode_model}")
    except OpenCodeBootstrapError as exc:
        failures.append(str(exc))
    for target in cfg.watches:
        repo = Path(target.repo_path).expanduser()
        if not repo.is_dir():
            failures.append(f"repository unreadable: {repo}")
        elif os.geteuid() == 0 and identity and identity.uid > 0:
            res = _run(["su", "-s", "/bin/sh", identity.user, "-c", f"test -r '{repo}'"])
            if res.returncode != 0:
                failures.append(f"repository is not readable by {identity.user}: {repo}")
        elif not os.access(repo, os.R_OK):
            failures.append(f"repository unreadable: {repo}")
    state_value = cfg.db_path
    state = (identity.home / state_value[2:].lstrip("/")
             if identity and state_value.startswith("~/")
             else Path(state_value).expanduser())
    if not state.is_absolute():
        state = (identity.home if identity else Path.cwd()) / state
    if not state.parent.exists():
        failures.append(f"state directory does not exist: {state.parent}")
    elif os.geteuid() == 0 and identity and identity.uid > 0:
        res = _run(["su", "-s", "/bin/sh", identity.user, "-c", f"test -w '{state.parent}'"])
        if res.returncode != 0:
            failures.append(f"state directory is not writable by {identity.user}: {state.parent}")
    elif not os.access(state.parent, os.W_OK):
        failures.append(f"state directory is not writable: {state.parent}")
    if state.exists() and os.access(state, os.R_OK):
        try:
            store = Store(state)
            try:
                stats = store.get_queue_stats()
                if stats.get("stale_leases", 0) > 0:
                    failures.append(f"store has {stats['stale_leases']} stale lease(s)")
                if stats.get("dead_jobs", 0) > 0:
                    failures.append(f"store has {stats['dead_jobs']} dead job(s)")
                if stats.get("dead_batches", 0) > 0:
                    failures.append(f"store has {stats['dead_batches']} dead batch(es)")
            finally:
                store.close()
        except Exception as exc:
            failures.append(f"unable to inspect store queue health: {exc}")
    secret_file = (
        identity.home / ".config/oi/.env"
        if identity else Path.home() / ".config/oi/.env"
    )
    token_set = bool(os.environ.get(cfg.discord_token_env))
    if secret_file.exists():
        token_set = token_set or any(
            line.startswith(f"{cfg.discord_token_env}=")
            for line in secret_file.read_text(encoding="utf-8").splitlines()
        )
    if not token_set:
        failures.append(f"Discord token env var is not set: {cfg.discord_token_env}")
    if identity:
        if identity.uid == -1:
            failures.append(f"service user does not exist: {identity.user}")
        elif identity.uid == 0:
            failures.append("service user must not be root")
        elif state.parent.exists() and state.parent.stat().st_uid != identity.uid:
            failures.append(f"state directory is not owned by {identity.user}: {state.parent}")
    return failures
