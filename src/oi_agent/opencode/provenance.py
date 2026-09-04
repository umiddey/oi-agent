"""Read-scoped repository refresh and snapshot provenance."""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a bounded read-only Git command without logging its output."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_pull(repo_path: str) -> bool:
    """Refresh a watched clone with a read-scoped fast-forward pull.

    Args:
        repo_path: Local clone path.

    Returns:
        ``True`` on successful pull, otherwise ``False``.
    """
    try:
        result = subprocess.run(
            ["git", "-C", repo_path, "-c", "pull.rebase=false", "pull",
             "--ff-only", "--autostash"],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError):
        logger.warning("[provenance] git pull unavailable")
        return False
    if result.returncode != 0:
        logger.warning("[provenance] git pull failed")
        return False
    return True


def audit_sha(repo_root: Path) -> str:
    """Return short HEAD SHA with an honest dirty suffix."""
    sha = _git(repo_root, "rev-parse", "--short", "HEAD") or "unknown"
    return f"{sha}-dirty" if _git(repo_root, "status", "--porcelain") else sha


def worktree_fingerprint(repo_root: Path) -> str:
    """Hash HEAD, status, and diff to detect any mid-run repository mutation."""
    head = _git(repo_root, "rev-parse", "HEAD") or "unknown"
    status = _git(repo_root, "status", "--porcelain") or ""
    diff = _git(repo_root, "diff", "HEAD") or ""
    if head == "unknown":
        return "unknown"
    payload = f"{head}\n{status}\n{diff}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:12]


MIDAUDIT_MUTATION_WARNING = "repository changed during the audit"


def mutation_warning(repo_root: Path, fingerprint_before: str) -> str:
    """Return the stable provenance warning when the worktree changed."""
    return (MIDAUDIT_MUTATION_WARNING
            if worktree_fingerprint(repo_root) != fingerprint_before else "")
