"""Safe OpenCode dependency, authentication, and model bootstrap helpers."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .paths import OpenCodePaths

MIN_OPENCODE_VERSION = (1, 18, 0)
INSTALL_DOCS_URL = "https://opencode.ai/docs/#install"
_VERSION_RE = re.compile(r"(?<!\d)(\d+)\.(\d+)\.(\d+)(?!\d)")
_MODEL_RE = re.compile(r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_.-]+/[A-Za-z0-9_./:@+-]+)")


@dataclass(frozen=True)
class OpenCodeVersion:
    """Parsed compatible OpenCode version and the executable path."""

    binary: str
    version: tuple[int, int, int]
    raw: str


class OpenCodeBootstrapError(RuntimeError):
    """Raised when OpenCode cannot satisfy the OI runtime contract."""


def resolve_binary(binary: str) -> str:
    """Resolve an executable without shell interpolation.

    Args:
        binary: Command name or filesystem path.

    Returns:
        Absolute executable path.

    Raises:
        OpenCodeBootstrapError: If the executable is missing or not runnable.
    """
    if not isinstance(binary, str) or not binary.strip():
        raise OpenCodeBootstrapError("OpenCode binary is empty")
    candidate = Path(binary).expanduser()
    if candidate.parent != Path("."):
        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise OpenCodeBootstrapError(
                f"OpenCode binary cannot be resolved: {binary!r}"
            ) from exc
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return str(resolved)
    else:
        found = shutil.which(binary)
        if found:
            return str(Path(found).resolve())
    raise OpenCodeBootstrapError(
        f"OpenCode executable not found or not executable: {binary!r}; "
        f"install it from {INSTALL_DOCS_URL}"
    )


def check_version(binary: str, paths: OpenCodePaths) -> OpenCodeVersion:
    """Verify the executable responds with a supported semantic version.

    Args:
        binary: Configured executable command or path.
        paths: Isolated OpenCode environment.

    Returns:
        Parsed version information.

    Raises:
        OpenCodeBootstrapError: On execution failure or incompatible version.
    """
    resolved = resolve_binary(binary)
    try:
        completed = subprocess.run(
            [resolved, "--version"],
            env=paths.environment(),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenCodeBootstrapError(
            "OpenCode exists but `opencode --version` failed"
        ) from exc
    raw = (completed.stdout or "").strip()
    match = _VERSION_RE.search(raw)
    if completed.returncode != 0 or match is None:
        raise OpenCodeBootstrapError(
            "OpenCode returned no parseable compatible version"
        )
    version = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
    if version < MIN_OPENCODE_VERSION:
        minimum = ".".join(map(str, MIN_OPENCODE_VERSION))
        raise OpenCodeBootstrapError(
            f"OpenCode {'.'.join(map(str, version))} is too old; minimum is {minimum}"
        )
    return OpenCodeVersion(resolved, version, raw)


def install_command() -> list[str]:
    """Select one fixed official installer command available on this host."""
    if shutil.which("npm"):
        return ["npm", "install", "--global", "opencode-ai"]
    if platform.system() == "Darwin" and shutil.which("brew"):
        return ["brew", "install", "anomalyco/tap/opencode"]
    if shutil.which("mise"):
        return ["mise", "use", "--global", "npm:opencode-ai"]
    raise OpenCodeBootstrapError(
        f"No supported installer found; install OpenCode from {INSTALL_DOCS_URL}"
    )


def install_opencode(paths: OpenCodePaths) -> str:
    """Install OpenCode through a fixed official package-manager command.

    Args:
        paths: Isolated environment used for the installer subprocess.

    Returns:
        Resolved installed executable path.

    Raises:
        OpenCodeBootstrapError: If installation or post-install verification fails.
    """
    command = install_command()
    try:
        completed = subprocess.run(
            command,
            env=paths.environment(),
            check=False,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise OpenCodeBootstrapError("OpenCode installation could not start") from exc
    if completed.returncode != 0:
        raise OpenCodeBootstrapError(
            f"OpenCode installer exited with status {completed.returncode}"
        )
    return resolve_binary("opencode")


def _run(binary: str, args: list[str], paths: OpenCodePaths,
         *, inherit_stdio: bool = False) -> subprocess.CompletedProcess[str]:
    """Run one allowlisted OpenCode bootstrap command."""
    resolved = resolve_binary(binary)
    kwargs = {
        "env": paths.environment(),
        "check": False,
        "timeout": 600,
    }
    if not inherit_stdio:
        kwargs.update(capture_output=True, text=True)
    return subprocess.run([resolved, *args], **kwargs)


def authenticated(binary: str, paths: OpenCodePaths) -> bool:
    """Return whether OpenCode reports at least one configured provider."""
    try:
        completed = _run(binary, ["auth", "list"], paths)
    except (OpenCodeBootstrapError, OSError, subprocess.SubprocessError):
        return False
    if completed.returncode != 0:
        return False
    output = "\n".join((completed.stdout or "").splitlines()).strip()
    return bool(output and "no credentials" not in output.lower()
                and "not authenticated" not in output.lower())


def authenticate(binary: str, paths: OpenCodePaths) -> None:
    """Hand an interactive terminal to OpenCode authentication."""
    completed = _run(binary, ["auth", "login"], paths, inherit_stdio=True)
    if completed.returncode != 0 or not authenticated(binary, paths):
        raise OpenCodeBootstrapError(
            "OpenCode authentication was not completed; rerun `opencode auth login`"
        )


def list_models(binary: str, paths: OpenCodePaths) -> list[str]:
    """Return exact provider/model IDs from OpenCode's model listing."""
    try:
        completed = _run(binary, ["models"], paths)
    except (OpenCodeBootstrapError, OSError, subprocess.SubprocessError) as exc:
        raise OpenCodeBootstrapError("OpenCode model listing failed") from exc
    if completed.returncode != 0:
        raise OpenCodeBootstrapError("OpenCode model listing failed")
    models = sorted(set(_MODEL_RE.findall(completed.stdout or "")))
    return models


def bootstrap(binary: str, paths: OpenCodePaths, *, install: bool = False,
              login: bool = True) -> OpenCodeVersion:
    """Verify OpenCode and optionally install/authenticate it in isolation.

    Args:
        binary: Configured OpenCode executable.
        paths: Shared isolated OpenCode paths.
        install: Permit the selected package-manager installation path.
        login: Run interactive auth login when no provider is configured.

    Returns:
        Verified executable version.
    """
    paths.ensure()
    try:
        version = check_version(binary, paths)
    except OpenCodeBootstrapError:
        if not install:
            raise
        installed = install_opencode(paths)
        version = check_version(installed, paths)
    if login and not authenticated(version.binary, paths):
        authenticate(version.binary, paths)
    return version
