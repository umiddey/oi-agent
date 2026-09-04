"""Private filesystem locations used by every OI OpenCode invocation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

_ALLOWED_ENV_VARS = {
    # System & POSIX
    "PATH", "TMPDIR", "TEMP", "TMP", "TZ", "LANG", "LC_ALL", "LC_CTYPE",
    "LC_MESSAGES", "TERM", "COLORTERM", "SHELL",
    # Tooling & Runtime managers
    "BUN_INSTALL", "NVM_DIR", "NVM_BIN", "MISE_DATA_DIR", "MISE_CONFIG_DIR",
    "ASDF_DIR", "ASDF_DATA_DIR", "CARGO_HOME", "RUSTUP_HOME", "DENO_INSTALL",
    "PNPM_HOME",
    # Network, Proxy, & TLS
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    # Systemd execution metadata
    "SYSTEMD_EXEC_PID", "JOURNAL_STREAM",
}

_BLOCKED_TERMS = (
    "KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "PRIVATE",
)


def _is_secret_key(key: str) -> bool:
    """Return True if an environment key looks like a credential or secret."""
    upper = key.upper()
    return upper.startswith("OI_") or any(term in upper for term in _BLOCKED_TERMS)


@dataclass(frozen=True)
class OpenCodePaths:
    """Isolated HOME and XDG roots for one OI service user."""

    home: Path
    config_home: Path
    data_home: Path
    state_home: Path
    cache_home: Path

    def environment(self, base: dict[str, str] | None = None) -> dict[str, str]:
        """Return an isolated environment excluding parent secrets and personal config."""
        source = os.environ if base is None else base
        environment: dict[str, str] = {}
        for key, value in source.items():
            if base is None:
                if key in _ALLOWED_ENV_VARS and not _is_secret_key(key):
                    environment[key] = value
            else:
                if not _is_secret_key(key) and not key.startswith("OPENCODE_CONFIG"):
                    environment[key] = value
        environment.update({
            "HOME": str(self.home),
            "XDG_CONFIG_HOME": str(self.config_home),
            "XDG_DATA_HOME": str(self.data_home),
            "XDG_STATE_HOME": str(self.state_home),
            "XDG_CACHE_HOME": str(self.cache_home),
            "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        })
        return environment

    def ensure(self, uid: int = -1, gid: int = -1) -> None:
        """Create private roots with restrictive ownership-friendly modes."""
        for path in (self.home, self.config_home, self.data_home,
                     self.state_home, self.cache_home):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
            path.chmod(0o700)
            if uid != -1 or gid != -1:
                try:
                    os.chown(path, uid, gid)
                except (OSError, PermissionError):
                    pass


def resolve_opencode_paths(home: Path | None = None) -> OpenCodePaths:
    """Resolve the exact private roots shared by init, runtime, and daemon.

    Args:
        home: Service HOME; defaults to the current process HOME.

    Returns:
        Isolated OpenCode filesystem roots.
    """
    root = (home or Path.home()).expanduser().resolve()
    private = root / ".local" / "share" / "oi" / "opencode"
    return OpenCodePaths(
        home=private / "home",
        config_home=private / "config",
        data_home=private / "data",
        state_home=private / "state",
        cache_home=private / "cache",
    )
