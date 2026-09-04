"""Immutable controller-owned base/head snapshots for one MR/PR audit.

For one explicitly resolved review reference, the trusted controller builds a
temporary mode-0700 workspace under OI's isolated state root, initializes a
bare object repository, fetches ONLY the provider-owned refs for the validated
number (fixed ref templates; never text derived from user input), verifies the
base/head relationship via the provider merge-result ref parents, archives
both verified trees with ``git archive``, and extracts them in pure Python
into read-only directories. The watched worktree is never mutated; only its
``remote.origin.url`` is read, and that URL is never logged.

All Git access is argv-only with fixed command templates, bounded captured
stdout/stderr, explicit timeouts, and sanitized failure classes. Snapshots are
never cached and live for exactly one audit. These public functions are
synchronous; the integration layer wraps them in a worker thread.

Failure classes (:class:`ReviewSnapshotError`; ``str(exc)`` is the class):

- ``review_provider_invalid`` / ``review_number_invalid``: defensive input checks.
- ``review_origin_unreadable``: ``remote.origin.url`` could not be read.
- ``review_init_failed``: temporary bare repository initialization failed.
- ``review_ref_unavailable``: the provider head ref does not exist.
- ``review_fetch_failed``: probe/fetch failed (auth, unreachable, transport).
- ``review_fetch_timeout``: probe/fetch exceeded the explicit timeout.
- ``review_head_unverified``: the head ref did not resolve to a commit SHA.
- ``review_base_unavailable``: provider supplies no trustworthy merge/base ref.
- ``review_base_unverified``: malformed merge commit, or its second parent is
  not exactly the verified head SHA.
- ``review_archive_failed``: ``git archive`` of a verified tree failed.
- ``review_extract_rejected``: unsafe or malformed archive member (path
  traversal, absolute path, device node, hardlink, sparse entry, escaping
  symlink, setuid/setgid bits, ``.git`` entry, oversized archive).
- ``review_diff_failed``: bounded changed-path manifest generation failed.
- ``review_snapshot_error``: unexpected boundary failure; never leaks stderr.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .paths import resolve_opencode_paths
from .review_target import (
    MAX_REVIEW_NUMBER,
    PROVIDER_GITHUB,
    PROVIDER_GITLAB,
    provider_ref_template,
    read_origin_url,
)

logger = logging.getLogger(__name__)

MAX_CHANGED_PATHS = 200

_LOCAL_REF_PREFIX = "refs/review"
_GIT_TIMEOUT_SECONDS = 60
_FETCH_TIMEOUT_SECONDS = 120
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_TAR_MEMBERS = 100_000
_MAX_TAR_TOTAL_BYTES = 512 * 1024 * 1024
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_ALLOWED_TAR_TYPES = (tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE, tarfile.SYMTYPE)
_READ_ONLY_DIR_MODE = 0o555
_READ_ONLY_FILE_MODE = 0o444


class ReviewSnapshotError(ValueError):
    """Bounded fail-closed snapshot failure; ``str(exc)`` is the class."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


@dataclass(frozen=True)
class ChangedPath:
    """One bounded changed-path manifest entry between verified SHAs.

    Attributes:
        status: Git name-status letter (``A``, ``M``, ``D``, ``R100``, ...).
        path: Changed path (rename/copy source).
        new_path: Rename/copy destination, when applicable.
    """

    status: str
    path: str
    new_path: str | None = None


@dataclass(frozen=True)
class ReviewSnapshot:
    """Immutable, read-only base/head materialization for one review audit.

    Attributes:
        provider: ``gitlab`` or ``github``.
        number: Validated MR/PR number.
        base_sha: Verified full base (merge-result first parent) SHA.
        head_sha: Verified full head SHA.
        base_dir: Read-only extracted base tree (no ``.git``).
        head_dir: Read-only extracted head tree (no ``.git``).
        changed_paths: Bounded changed-path manifest between base and head.
        manifest_truncated: Whether the manifest was honestly truncated.
    """

    provider: str
    number: int
    base_sha: str
    head_sha: str
    base_dir: Path
    head_dir: Path
    changed_paths: tuple[ChangedPath, ...]
    manifest_truncated: bool


@dataclass(frozen=True)
class _GitOutcome:
    returncode: int | None
    stdout: str
    timed_out: bool = False


def _git_env() -> dict[str, str]:
    """Return the ambient environment with interactive Git prompts disabled."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0"}


def _run_git(argv: list[str], timeout: int) -> _GitOutcome:
    """Run one bounded argv-only Git command without logging its output."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=_git_env(),
        )
    except subprocess.TimeoutExpired:
        return _GitOutcome(returncode=None, stdout="", timed_out=True)
    except (OSError, subprocess.SubprocessError):
        return _GitOutcome(returncode=None, stdout="")
    return _GitOutcome(returncode=result.returncode, stdout=result.stdout)


def _fail(failure_class: str, provider: str, number: int) -> ReviewSnapshotError:
    """Log the bounded failure class and return the exception to raise."""
    logger.warning(
        "[review] snapshot failed class=%s provider=%s number=%d",
        failure_class, provider, number,
    )
    return ReviewSnapshotError(failure_class)


def _snapshot_root() -> Path | None:
    """Return OI's isolated reviews root, or None to fall back to tempfile."""
    try:
        base = resolve_opencode_paths().state_home / "reviews"
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        base.chmod(0o700)
        return base
    except OSError:
        return None


def _validate_inputs(provider: str, number: int) -> None:
    """Defensively re-validate provider and number before any Git activity."""
    if provider not in (PROVIDER_GITLAB, PROVIDER_GITHUB):
        raise ReviewSnapshotError("review_provider_invalid")
    if isinstance(number, bool) or not isinstance(number, int):
        raise ReviewSnapshotError("review_number_invalid")
    if number < 1 or number > MAX_REVIEW_NUMBER:
        raise ReviewSnapshotError("review_number_invalid")


def _fetch_ref(
    bare: Path,
    origin: str,
    remote_ref: str,
    local_ref: str,
    *,
    required: bool,
    provider: str,
    number: int,
) -> bool:
    """Fetch one fixed provider-owned ref into the temporary bare repository.

    Args:
        bare: Temporary bare object repository.
        origin: The watched repository's origin URL (never logged).
        remote_ref: Fixed provider ref template with the validated integer.
        local_ref: Fixed local ref name inside the temporary repository.
        required: Whether an absent remote ref is itself a failure.
        provider: Provider name for bounded failure logging.
        number: Validated MR/PR number for bounded failure logging.

    Returns:
        ``True`` when the ref exists and was fetched.

    Raises:
        ReviewSnapshotError: Bounded fetch/unavailable/timeout classes.
    """
    probe = _run_git(
        ["git", "-C", str(bare), "ls-remote", origin, remote_ref],
        _FETCH_TIMEOUT_SECONDS,
    )
    if probe.timed_out:
        raise _fail("review_fetch_timeout", provider, number)
    if probe.returncode != 0:
        raise _fail("review_fetch_failed", provider, number)
    present = False
    for line in probe.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip() == remote_ref:
            present = True
            break
    if not present:
        if required:
            raise _fail("review_ref_unavailable", provider, number)
        return False
    fetch = _run_git(
        ["git", "-C", str(bare), "fetch", "--no-tags", origin, f"+{remote_ref}:{local_ref}"],
        _FETCH_TIMEOUT_SECONDS,
    )
    if fetch.timed_out:
        raise _fail("review_fetch_timeout", provider, number)
    if fetch.returncode != 0:
        raise _fail("review_fetch_failed", provider, number)
    return True


def _resolve_commit(bare: Path, ref_name: str) -> str | None:
    """Return the full 40-hex commit SHA a local ref points to, or None."""
    outcome = _run_git(
        ["git", "-C", str(bare), "rev-parse", f"{ref_name}^{{commit}}"],
        _GIT_TIMEOUT_SECONDS,
    )
    if outcome.timed_out or outcome.returncode != 0:
        return None
    sha = outcome.stdout.strip().lower()
    return sha if _SHA_RE.match(sha) else None


def _verified_base(bare: Path, head_sha: str) -> str | None:
    """Derive the base SHA from the merge-result ref, verifying head lineage.

    The provider merge-result commit must have exactly two parents; the first
    parent is the target/base SHA and the second parent must equal the
    verified head SHA.

    Args:
        bare: Temporary bare object repository.
        head_sha: Already-verified full head SHA.

    Returns:
        The verified base SHA, or ``None`` when the merge commit is malformed
        or its second parent does not match the head SHA.
    """
    merge_sha = _resolve_commit(bare, f"{_LOCAL_REF_PREFIX}/merge")
    if merge_sha is None:
        return None
    outcome = _run_git(
        ["git", "-C", str(bare), "rev-list", "--parents", "-n", "1", merge_sha],
        _GIT_TIMEOUT_SECONDS,
    )
    if outcome.timed_out or outcome.returncode != 0:
        return None
    tokens = outcome.stdout.split()
    if len(tokens) != 3 or tokens[0] != merge_sha:
        return None
    if not all(_SHA_RE.match(token) for token in tokens):
        return None
    if tokens[2] != head_sha:
        return None
    return tokens[1]


def _materialize(bare: Path, sha: str, dest: Path, role: str) -> None:
    """Archive one verified tree to a fixed temp path and extract it safely."""
    tar_path = dest.parent / f"{role}.tar"
    outcome = _run_git(
        ["git", "-C", str(bare), "archive", "--format=tar", f"--output={tar_path}", sha],
        _GIT_TIMEOUT_SECONDS,
    )
    if outcome.timed_out or outcome.returncode != 0:
        raise ReviewSnapshotError("review_archive_failed")
    try:
        _extract_snapshot_tar(tar_path, dest)
    finally:
        try:
            tar_path.unlink()
        except OSError:
            pass
    _make_read_only(dest)


def _validate_member_name(name: str) -> None:
    """Reject traversal, absolute paths, Windows drives, backslashes, .git."""
    posix = PurePosixPath(name)
    if posix.is_absolute() or ".." in posix.parts or "\\" in name or _DRIVE_RE.match(name):
        raise ReviewSnapshotError("review_extract_rejected")
    if any(part == ".git" for part in posix.parts):
        raise ReviewSnapshotError("review_extract_rejected")


def _validate_symlink(dest: Path, name: str, link_target: str, root_real: str) -> None:
    """Reject symlinks that are absolute or escape the snapshot root."""
    if not link_target or link_target.startswith("/"):
        raise ReviewSnapshotError("review_extract_rejected")
    parent_real = os.path.realpath(os.path.dirname(str(dest / name)))
    resolved = os.path.normpath(os.path.join(parent_real, link_target))
    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise ReviewSnapshotError("review_extract_rejected")


def _makedirs(target: Path) -> None:
    """Create a directory inside the snapshot, rejecting filesystem failures."""
    try:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        raise ReviewSnapshotError("review_extract_rejected") from None


def _extract_snapshot_tar(tar_path: Path, dest: Path) -> None:
    """Extract a ``git archive`` tar safely, rejecting any unsafe member.

    Rejects path traversal, absolute paths, Windows drives, device nodes,
    hardlinks, sparse entries, escaping or absolute symlinks, setuid/setgid
    bits, ``.git`` entries, and oversized archives. Extracted content is
    private (files 0o600, directories 0o700); the caller applies the
    read-only pass afterwards.

    Args:
        tar_path: Fixed temp path of the archived tree.
        dest: Extraction root (created with mode 0o700).

    Raises:
        ReviewSnapshotError: ``review_extract_rejected`` on any unsafe member.
    """
    dest.mkdir(mode=0o700)
    root_real = str(dest.resolve())
    total_bytes = 0
    members = 0
    with tarfile.open(tar_path, "r:") as archive:
        for member in archive:
            members += 1
            if members > _MAX_TAR_MEMBERS or member.type not in _ALLOWED_TAR_TYPES:
                raise ReviewSnapshotError("review_extract_rejected")
            name = (member.name or "").rstrip("/")
            if not name:
                continue
            _validate_member_name(name)
            if (member.mode & 0o6000) != 0:
                raise ReviewSnapshotError("review_extract_rejected")
            total_bytes += max(member.size, 0)
            if total_bytes > _MAX_TAR_TOTAL_BYTES:
                raise ReviewSnapshotError("review_extract_rejected")
            target = dest / name
            parent_real = os.path.normpath(os.path.dirname(str(target)))
            if parent_real != root_real and not parent_real.startswith(root_real + os.sep):
                raise ReviewSnapshotError("review_extract_rejected")
            if member.type == tarfile.DIRTYPE:
                _makedirs(target)
                continue
            if member.type == tarfile.SYMTYPE:
                _validate_symlink(dest, name, member.linkname, root_real)
                try:
                    os.symlink(member.linkname, target)
                except OSError:
                    raise ReviewSnapshotError("review_extract_rejected") from None
                continue
            source = archive.extractfile(member)
            if source is None:
                raise ReviewSnapshotError("review_extract_rejected")
            _makedirs(target.parent)
            with target.open("wb") as handle:
                shutil.copyfileobj(source, handle)


def _make_read_only(root: Path) -> None:
    """Freeze a snapshot tree for the audit: files 0o444, directories 0o555."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            try:
                os.chmod(os.path.join(dirpath, name), _READ_ONLY_FILE_MODE)
            except OSError:
                pass
        for name in dirnames:
            try:
                os.chmod(os.path.join(dirpath, name), _READ_ONLY_DIR_MODE)
            except OSError:
                pass
    try:
        root.chmod(_READ_ONLY_DIR_MODE)
    except OSError:
        pass


def _cleanup_root(root_str: str) -> None:
    """Remove one snapshot workspace, restoring write bits first."""
    root = Path(root_str)
    if not root.exists():
        return
    try:
        for dirpath, _dirnames, filenames in os.walk(root, topdown=True):
            try:
                os.chmod(dirpath, 0o700)
            except OSError:
                pass
            for name in filenames:
                try:
                    os.chmod(os.path.join(dirpath, name), 0o600)
                except OSError:
                    pass
    except OSError:
        pass
    try:
        shutil.rmtree(root)
    except OSError:
        logger.warning("[review] snapshot cleanup failed class=review_cleanup_failed")


def _changed_manifest(
    bare: Path, base_sha: str, head_sha: str
) -> tuple[tuple[ChangedPath, ...], bool]:
    """Build the bounded changed-path manifest between verified SHAs.

    Reads at most ``_MAX_MANIFEST_BYTES`` of NUL-separated ``git diff
    --name-status -z`` output and at most ``MAX_CHANGED_PATHS`` entries,
    reporting honest truncation. No patch contents are produced anywhere.

    Args:
        bare: Temporary bare object repository.
        base_sha: Verified base SHA.
        head_sha: Verified head SHA.

    Returns:
        ``(entries, truncated)``.

    Raises:
        ReviewSnapshotError: ``review_diff_failed`` when the diff fails.
    """
    argv = [
        "git", "-C", str(bare), "-c", "core.quotepath=false",
        "diff", "--name-status", "-z", base_sha, head_sha,
    ]
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_env(),
        )
    except OSError:
        raise ReviewSnapshotError("review_diff_failed") from None
    truncated = False
    raw = b""
    try:
        assert proc.stdout is not None
        raw = proc.stdout.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            truncated = True
            proc.kill()
        try:
            returncode = proc.wait(timeout=_GIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise ReviewSnapshotError("review_diff_failed") from None
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    if not truncated and returncode != 0:
        raise ReviewSnapshotError("review_diff_failed")
    fields = raw[:_MAX_MANIFEST_BYTES].decode("utf-8", errors="replace").split("\0")
    entries: list[ChangedPath] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            break
        if index >= len(fields):
            truncated = True
            break
        path = fields[index]
        index += 1
        new_path: str | None = None
        if status[0] in ("R", "C"):
            if index >= len(fields):
                truncated = True
                break
            new_path = fields[index] or None
            index += 1
        if len(entries) >= MAX_CHANGED_PATHS:
            truncated = True
            break
        entries.append(ChangedPath(status=status, path=path, new_path=new_path))
    return tuple(entries), truncated


class MaterializedReview:
    """Context manager owning one immutable snapshot workspace.

    Cleanup runs on success, exception, timeout, and cancellation through
    ``__exit__``/``close()``; :func:`weakref.finalize` removes the workspace
    if the object is garbage-collected without ``close()``.
    """

    def __init__(self, repo_path: str | Path, provider: str, number: int) -> None:
        """Materialize snapshots, removing all temp state on any failure.

        Args:
            repo_path: Configured watched repository with a readable origin.
            provider: ``gitlab`` or ``github``.
            number: Validated positive MR/PR number.

        Raises:
            ReviewSnapshotError: Bounded failure class (see module docstring).
        """
        self._closed = False
        self._tmp_root: Path | None = None
        self._finalizer: weakref.finalize | None = None
        try:
            self._snapshot = self._build(Path(repo_path).expanduser(), provider, number)
        except BaseException:
            self.close()
            raise
        if self._tmp_root is not None:
            self._finalizer = weakref.finalize(self, _cleanup_root, str(self._tmp_root))

    def _build(self, repo_path: Path, provider: str, number: int) -> ReviewSnapshot:
        _validate_inputs(provider, number)
        origin = read_origin_url(str(repo_path))
        if origin is None:
            raise _fail("review_origin_unreadable", provider, number)
        self._tmp_root = Path(tempfile.mkdtemp(prefix="oi-review-", dir=_snapshot_root()))
        try:
            bare = self._tmp_root / "objects.git"
            init = _run_git(["git", "init", "--bare", "-q", str(bare)], _GIT_TIMEOUT_SECONDS)
            if init.timed_out or init.returncode != 0:
                raise _fail("review_init_failed", provider, number)
            head_ref = provider_ref_template(provider, number, "head")
            merge_ref = provider_ref_template(provider, number, "merge")
            _fetch_ref(
                bare, origin, head_ref, f"{_LOCAL_REF_PREFIX}/head",
                required=True, provider=provider, number=number,
            )
            merge_present = _fetch_ref(
                bare, origin, merge_ref, f"{_LOCAL_REF_PREFIX}/merge",
                required=False, provider=provider, number=number,
            )
            head_sha = _resolve_commit(bare, f"{_LOCAL_REF_PREFIX}/head")
            if head_sha is None:
                raise _fail("review_head_unverified", provider, number)
            if not merge_present:
                raise _fail("review_base_unavailable", provider, number)
            base_sha = _verified_base(bare, head_sha)
            if base_sha is None:
                raise _fail("review_base_unverified", provider, number)
            base_dir = self._tmp_root / "base"
            head_dir = self._tmp_root / "head"
            _materialize(bare, base_sha, base_dir, "base")
            _materialize(bare, head_sha, head_dir, "head")
            manifest, truncated = _changed_manifest(bare, base_sha, head_sha)
        except ReviewSnapshotError:
            raise
        except (OSError, subprocess.SubprocessError, ValueError):
            logger.warning("[review] snapshot boundary failed class=review_snapshot_error")
            raise ReviewSnapshotError("review_snapshot_error") from None
        logger.info(
            "[review] snapshots provider=%s number=%d base=%s head=%s paths=%d truncated=%s",
            provider, number, base_sha[:12], head_sha[:12], len(manifest), truncated,
        )
        return ReviewSnapshot(
            provider=provider,
            number=number,
            base_sha=base_sha,
            head_sha=head_sha,
            base_dir=base_dir,
            head_dir=head_dir,
            changed_paths=manifest,
            manifest_truncated=truncated,
        )

    @property
    def snapshot(self) -> ReviewSnapshot:
        """The materialized snapshot; valid until :meth:`close`."""
        return self._snapshot

    def close(self) -> None:
        """Remove the snapshot workspace; idempotent and GC-safe."""
        if self._closed:
            return
        self._closed = True
        finalizer = self._finalizer
        self._finalizer = None
        if finalizer is not None:
            finalizer()
        elif self._tmp_root is not None:
            _cleanup_root(str(self._tmp_root))

    def __enter__(self) -> ReviewSnapshot:
        return self._snapshot

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def materialized_review(repo_path: str | Path, provider: str, number: int) -> MaterializedReview:
    """Materialize immutable base/head snapshots for one review audit.

    Args:
        repo_path: Configured watched repository with a readable origin.
        provider: ``gitlab`` or ``github``.
        number: Validated positive MR/PR number.

    Returns:
        Context manager yielding :class:`ReviewSnapshot`; cleanup is
        guaranteed on success, exception, timeout, and cancellation, and the
        watched worktree is never mutated.
    """
    return MaterializedReview(repo_path, provider, number)
