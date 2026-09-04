"""Immutable controller-owned base/head snapshots for one MR/PR audit.

For one explicitly resolved review reference, the trusted controller builds a
temporary mode-0700 workspace under OI's isolated state root, initializes a
bare object repository, records the origin URL ONLY in that repository's
config file (the URL can contain credentials and must never appear in process
arguments), fetches ONLY the provider-owned refs for the validated number
(fixed ref templates; never text derived from user input), verifies the
base/head relationship via the provider merge-result ref parents, and
materializes both verified trees as EXACT Git objects: entries come from
NUL-delimited ``git ls-tree -rz --full-tree`` and blob bytes stream through a
single ``git cat-file --batch`` process, so ``.gitattributes`` effects such as
``export-ignore`` or ``export-subst`` can never omit or alter reviewed
content. Size limits are enforced while streaming: the entry count before
materializing, then per-blob and cumulative bytes as payloads arrive. The
watched worktree is never mutated; only its ``remote.origin.url`` is read,
and that URL is never logged.

All Git access is argv-only with fixed command templates, bounded captured
stdout/stderr, explicit timeouts, and sanitized failure classes. Snapshots are
never cached and live for exactly one audit. These public functions are
synchronous; the integration layer wraps them in a worker thread.

Failure classes (:class:`ReviewSnapshotError`; ``str(exc)`` is the class):

- ``review_provider_invalid`` / ``review_number_invalid``: defensive input checks.
- ``review_origin_unreadable``: ``remote.origin.url`` could not be read.
- ``review_init_failed``: temporary bare repository initialization failed,
  including recording the origin URL in its config file.
- ``review_ref_unavailable``: the provider head ref does not exist.
- ``review_fetch_failed``: probe/fetch failed (auth, unreachable, transport).
- ``review_fetch_timeout``: probe/fetch exceeded the explicit timeout.
- ``review_head_unverified``: the head ref did not resolve to a commit SHA.
- ``review_base_unavailable``: provider supplies no trustworthy merge/base ref.
- ``review_base_unverified``: malformed merge commit, or its second parent is
  not exactly the verified head SHA.
- ``review_tree_failed``: reading a verified tree (ls-tree or the cat-file
  batch process) failed.
- ``review_extract_rejected``: unsafe or malformed tree entry (path traversal,
  absolute path, Windows drive, backslash path, ``.git`` entry, submodule
  entry, unusual mode, escaping/absolute/``.git``-entering symlink, malformed
  batch output, incomplete materialization).
- ``review_snapshot_too_large``: entry count, cumulative bytes, or one blob
  exceeds the snapshot size limits.
- ``review_diff_failed``: bounded changed-path manifest generation failed.
- ``review_snapshot_error``: unexpected boundary failure; never leaks stderr.
"""

from __future__ import annotations

import logging
import os
import re
import select
import shutil
import stat
import subprocess
import tempfile
import time
import weakref
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO

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
MAX_SNAPSHOT_TOTAL_BYTES = 512 * 1024 * 1024
MAX_SNAPSHOT_FILES = 50_000
MAX_SINGLE_BLOB_BYTES = 64 * 1024 * 1024

_LOCAL_REF_PREFIX = "refs/review"
_GIT_TIMEOUT_SECONDS = 60
_FETCH_TIMEOUT_SECONDS = 120
_MAX_MANIFEST_BYTES = 64 * 1024
_MAX_LS_TREE_BYTES = 32 * 1024 * 1024
_MAX_BATCH_HEADER_BYTES = 128
_CHUNK_BYTES = 64 * 1024
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_READ_ONLY_DIR_MODE = 0o555
_READ_ONLY_FILE_MODE = 0o444
_READ_ONLY_EXEC_FILE_MODE = 0o500


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


@dataclass(frozen=True)
class _TreeEntry:
    """One validated ``ls-tree -r`` record of a verified tree.

    Attributes:
        path: Slash-separated path relative to the tree root.
        sha: Full 40-hex blob SHA backing the entry.
        executable: Whether the blob mode is ``100755``.
        is_symlink: Whether the blob mode is ``120000`` (target is content).
    """

    path: str
    sha: str
    executable: bool
    is_symlink: bool

class _DeadlineReader:
    """Read a subprocess pipe through one absolute monotonic deadline."""

    def __init__(
        self, stream: BinaryIO, timeout: float, failure_class: str,
        malformed_class: str | None = None,
    ) -> None:
        self._fd = stream.fileno()
        self._deadline = time.monotonic() + timeout
        self._failure_class = failure_class
        self._malformed_class = malformed_class or failure_class
        self._buffer = bytearray()
        self._eof = False

    def remaining(self) -> float:
        """Return seconds remaining before the shared deadline."""
        remaining = self._deadline - time.monotonic()
        if remaining <= 0:
            raise ReviewSnapshotError(self._failure_class)
        return remaining

    def _fill(self) -> bool:
        """Read one available chunk, returning False only at EOF."""
        if self._eof:
            return False
        try:
            ready, _, _ = select.select([self._fd], [], [], self.remaining())
        except (OSError, ValueError):
            raise ReviewSnapshotError(self._failure_class) from None
        if not ready:
            raise ReviewSnapshotError(self._failure_class)
        try:
            chunk = os.read(self._fd, _CHUNK_BYTES)
        except OSError:
            raise ReviewSnapshotError(self._failure_class) from None
        if not chunk:
            self._eof = True
            return False
        self._buffer.extend(chunk)
        return True

    def read_to_eof(self, limit: int) -> bytes:
        """Read through EOF, retaining at most ``limit`` bytes."""
        while len(self._buffer) < limit and self._fill():
            pass
        return bytes(self._buffer[:limit])

    def read_exact(self, count: int) -> bytes:
        """Read exactly ``count`` bytes or fail on timeout/truncation."""
        while len(self._buffer) < count:
            if not self._fill():
                raise ReviewSnapshotError(self._malformed_class)
        result = bytes(self._buffer[:count])
        del self._buffer[:count]
        return result

    def readline(self, limit: int) -> bytes:
        """Read one LF-terminated line bounded by ``limit`` bytes."""
        while True:
            newline = self._buffer.find(b"\n")
            if newline != -1:
                if newline > limit:
                    raise ReviewSnapshotError(self._malformed_class)
                result = bytes(self._buffer[:newline])
                del self._buffer[:newline + 1]
                return result
            if len(self._buffer) > limit:
                raise ReviewSnapshotError(self._malformed_class)
            if not self._fill():
                raise ReviewSnapshotError(self._malformed_class)


def _stop_process(proc: subprocess.Popen[bytes]) -> None:
    """Kill and reap a subprocess without leaking cleanup failures."""
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=_GIT_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        pass


def _wait_before_deadline(
    proc: subprocess.Popen[bytes], reader: _DeadlineReader, failure_class: str,
) -> int:
    """Wait for a subprocess using the same deadline as its pipe reads."""
    try:
        return proc.wait(timeout=reader.remaining())
    except (OSError, subprocess.SubprocessError):
        _stop_process(proc)
        raise ReviewSnapshotError(failure_class) from None


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


def _validate_inputs(provider: str, number: int) -> None:
    """Defensively re-validate provider and number before any Git activity."""
    if provider not in (PROVIDER_GITLAB, PROVIDER_GITHUB):
        raise ReviewSnapshotError("review_provider_invalid")
    if isinstance(number, bool) or not isinstance(number, int):
        raise ReviewSnapshotError("review_number_invalid")
    if number < 1 or number > MAX_REVIEW_NUMBER:
        raise ReviewSnapshotError("review_number_invalid")


def _snapshot_root() -> Path | None:
    """Return OI's isolated reviews root, or None to fall back to tempfile."""
    try:
        base = resolve_opencode_paths().state_home / "reviews"
        base.mkdir(parents=True, exist_ok=True, mode=0o700)
        base.chmod(0o700)
        return base
    except OSError:
        return None


def _quote_git_config(value: str) -> str:
    """Quote one value for a git-config file: double quotes with escapes."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return '"' + escaped + '"'


def _write_remote_config(bare: Path, origin: str) -> bool:
    """Record ``remote.origin.url`` in the bare repository's config file.

    The URL may contain credentials and must never appear in process
    arguments, so it is written directly into the config file instead of via
    ``git config``/``git remote add`` (both would put it in argv). Values
    containing line breaks cannot be stored as one config line and fail
    closed.

    Args:
        bare: Newly initialized temporary bare repository.
        origin: The watched repository's origin URL (never logged).

    Returns:
        ``True`` when the remote section was written.
    """
    if not origin or "\n" in origin or "\r" in origin:
        return False
    section = '[remote "origin"]\n\turl = ' + _quote_git_config(origin) + "\n"
    try:
        with (bare / "config").open("a", encoding="utf-8") as handle:
            handle.write(section)
    except OSError:
        return False
    return True


def _fetch_ref(
    bare: Path,
    remote_ref: str,
    local_ref: str,
    *,
    required: bool,
    provider: str,
    number: int,
) -> bool:
    """Fetch one fixed provider-owned ref into the temporary bare repository.

    Only the remote NAME is passed on the command line; Git reads the origin
    URL from the config file written by :func:`_write_remote_config`, so a
    credentialed URL never appears in process arguments.

    Args:
        bare: Temporary bare object repository with ``remote.origin.url``.
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
        ["git", "-C", str(bare), "ls-remote", "origin", remote_ref],
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
        ["git", "-C", str(bare), "fetch", "--no-tags", "origin", f"+{remote_ref}:{local_ref}"],
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


def _validate_tree_path(name: str) -> None:
    """Reject traversal, absolute paths, Windows drives, backslashes, .git.

    Only a literal ``.git`` path component is rejected; ordinary files such
    as ``.gitmodules`` and ``.gitignore`` are allowed.
    """
    posix = PurePosixPath(name)
    if (
        not name
        or posix.is_absolute()
        or ".." in posix.parts
        or "\\" in name
        or _DRIVE_RE.match(name)
        or "//" in name
        or name == "."
    ):
        raise ReviewSnapshotError("review_extract_rejected")
    if any(part == ".git" for part in posix.parts):
        raise ReviewSnapshotError("review_extract_rejected")


def _parse_ls_tree(raw: str) -> tuple[_TreeEntry, ...]:
    """Parse NUL-delimited ``ls-tree -r`` records into validated entries.

    Only blobs with modes ``100644``, ``100755``, and ``120000`` are
    accepted; submodules (``160000``), trees, and any unusual mode fail the
    whole materialization.

    Raises:
        ReviewSnapshotError: ``review_extract_rejected`` on any malformed or
            disallowed record.
    """
    entries: list[_TreeEntry] = []
    for record in raw.split("\0"):
        if not record:
            continue
        meta, separator, path = record.partition("\t")
        if not separator:
            raise ReviewSnapshotError("review_extract_rejected")
        _validate_tree_path(path)
        fields = meta.split(" ")
        if len(fields) != 3:
            raise ReviewSnapshotError("review_extract_rejected")
        mode, obj_type, obj_sha = fields
        if not _SHA_RE.match(obj_sha):
            raise ReviewSnapshotError("review_extract_rejected")
        if obj_type == "blob" and mode in ("100644", "100755"):
            entries.append(
                _TreeEntry(
                    path=path, sha=obj_sha,
                    executable=mode == "100755", is_symlink=False,
                )
            )
        elif obj_type == "blob" and mode == "120000":
            entries.append(
                _TreeEntry(path=path, sha=obj_sha, executable=False, is_symlink=True)
            )
        else:
            raise ReviewSnapshotError("review_extract_rejected")
    return tuple(entries)


def _ls_tree_entries(bare: Path, sha: str) -> tuple[_TreeEntry, ...]:
    """List every tree entry with bounded, deadline-aware output reads."""
    argv = ["git", "-C", str(bare), "ls-tree", "-rz", "--full-tree", sha]
    try:
        proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_env(),
        )
    except OSError:
        raise ReviewSnapshotError("review_tree_failed") from None
    try:
        assert proc.stdout is not None
        reader = _DeadlineReader(proc.stdout, _GIT_TIMEOUT_SECONDS, "review_tree_failed")
        raw = reader.read_to_eof(_MAX_LS_TREE_BYTES + 1)
        if len(raw) > _MAX_LS_TREE_BYTES:
            raise ReviewSnapshotError("review_snapshot_too_large")
        if _wait_before_deadline(proc, reader, "review_tree_failed") != 0:
            raise ReviewSnapshotError("review_tree_failed")
    except ReviewSnapshotError:
        _stop_process(proc)
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
    return _parse_ls_tree(raw.decode("utf-8", errors="surrogateescape"))


def _reject_symlink_ancestors(path: str, symlink_paths: set[str]) -> None:
    """Reject entries located below a path that itself materialized as a
    symlink: nothing may be written through a link."""
    ancestor = ""
    for part in PurePosixPath(path).parts[:-1]:
        ancestor = f"{ancestor}/{part}" if ancestor else part
        if ancestor in symlink_paths:
            raise ReviewSnapshotError("review_extract_rejected")


def _spawn_cat_file(bare: Path) -> subprocess.Popen[bytes]:
    """Start one ``git cat-file --batch`` process for exact object payloads."""
    try:
        return subprocess.Popen(
            ["git", "-C", str(bare), "cat-file", "--batch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_git_env(),
        )
    except OSError:
        raise ReviewSnapshotError("review_tree_failed") from None


def _close_cat_file(proc: subprocess.Popen[bytes]) -> None:
    """Close and reap one cat-file process without leaking exceptions."""
    for stream in (proc.stdin, proc.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    _stop_process(proc)


def _read_batch_header(reader: _DeadlineReader) -> tuple[str, str, int]:
    """Read one bounded ``<oid> <type> <size>`` cat-file batch header."""
    fields = reader.readline(_MAX_BATCH_HEADER_BYTES).decode(
        "ascii", errors="replace",
    ).split(" ")
    if len(fields) != 3 or not _SHA_RE.match(fields[0]):
        raise ReviewSnapshotError("review_extract_rejected")
    oid, obj_type, size_text = fields
    if obj_type not in ("blob", "tree", "commit", "tag"):
        raise ReviewSnapshotError("review_extract_rejected")
    try:
        size = int(size_text)
    except ValueError:
        raise ReviewSnapshotError("review_extract_rejected") from None
    if size < 0:
        raise ReviewSnapshotError("review_extract_rejected")
    return oid, obj_type, size


def _entry_parent(dest: Path, name: str) -> Path:
    """Return the materialized parent directory for one validated entry."""
    parts = PurePosixPath(name).parts[:-1]
    return dest.joinpath(*parts) if parts else dest


def _makedirs(target: Path) -> None:
    """Create a directory inside the snapshot, rejecting filesystem failures."""
    try:
        target.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        raise ReviewSnapshotError("review_extract_rejected") from None


def _validate_symlink(dest: Path, name: str, link_target: str, root_real: str) -> None:
    """Reject absolute, escaping, or ``.git``-entering symlink targets."""
    if not link_target or link_target.startswith("/"):
        raise ReviewSnapshotError("review_extract_rejected")
    if any(part == ".git" for part in PurePosixPath(link_target).parts):
        raise ReviewSnapshotError("review_extract_rejected")
    parent_real = os.path.realpath(os.path.dirname(str(dest / name)))
    resolved = os.path.normpath(os.path.join(parent_real, link_target))
    if resolved != root_real and not resolved.startswith(root_real + os.sep):
        raise ReviewSnapshotError("review_extract_rejected")


def _write_blob_file(
    target: Path, reader: _DeadlineReader, size: int, executable: bool,
) -> None:
    """Stream one exact blob to a private file within the shared deadline."""
    _makedirs(target.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(target), flags, 0o700 if executable else 0o600)
    except OSError:
        raise ReviewSnapshotError("review_extract_rejected") from None
    try:
        remaining = size
        while remaining > 0:
            chunk = reader.read_exact(min(remaining, _CHUNK_BYTES))
            offset = 0
            while offset < len(chunk):
                written = os.write(fd, chunk[offset:])
                if written <= 0:
                    raise ReviewSnapshotError("review_extract_rejected")
                offset += written
            remaining -= len(chunk)
        if reader.read_exact(1) != b"\n":
            raise ReviewSnapshotError("review_extract_rejected")
    finally:
        os.close(fd)


def _write_entries(
    bare: Path, entries: tuple[_TreeEntry, ...], dest: Path, root_real: str,
) -> None:
    """Copy every verified tree entry exactly through one cat-file batch.

    Enforces the per-blob and cumulative size limits before and while each
    payload streams, validates symlink targets exactly like the entry paths,
    and verifies that the materialized path set equals the ls-tree entry set.

    Raises:
        ReviewSnapshotError: ``review_snapshot_too_large`` on a size breach;
            ``review_extract_rejected`` on any unsafe or incomplete entry.
    """
    symlink_paths = {entry.path for entry in entries if entry.is_symlink}
    materialized: set[str] = set()
    total_bytes = 0
    proc = _spawn_cat_file(bare)
    try:
        stdin = proc.stdin
        stdout = proc.stdout
        assert stdin is not None and stdout is not None
        reader = _DeadlineReader(
            stdout, _GIT_TIMEOUT_SECONDS, "review_tree_failed",
            "review_extract_rejected",
        )
        for entry in entries:
            stdin.write(entry.sha.encode("ascii") + b"\n")
            stdin.flush()
            oid, obj_type, size = _read_batch_header(reader)
            if oid != entry.sha or obj_type != "blob":
                raise ReviewSnapshotError("review_extract_rejected")
            if size > MAX_SINGLE_BLOB_BYTES:
                raise ReviewSnapshotError("review_snapshot_too_large")
            if total_bytes + size > MAX_SNAPSHOT_TOTAL_BYTES:
                raise ReviewSnapshotError("review_snapshot_too_large")
            total_bytes += size
            _reject_symlink_ancestors(entry.path, symlink_paths)
            target = dest.joinpath(*PurePosixPath(entry.path).parts)
            if entry.is_symlink:
                payload = reader.read_exact(size)
                if reader.read_exact(1) != b"\n":
                    raise ReviewSnapshotError("review_extract_rejected")
                _makedirs(_entry_parent(dest, entry.path))
                link_target = payload.decode("utf-8", errors="surrogateescape")
                _validate_symlink(dest, entry.path, link_target, root_real)
                try:
                    os.symlink(link_target, target)
                except OSError:
                    raise ReviewSnapshotError("review_extract_rejected") from None
            else:
                _write_blob_file(target, reader, size, entry.executable)
            materialized.add(entry.path)
        if materialized != {entry.path for entry in entries}:
            raise ReviewSnapshotError("review_extract_rejected")
    except ReviewSnapshotError:
        raise
    except OSError:
        raise ReviewSnapshotError("review_tree_failed") from None
    finally:
        _close_cat_file(proc)


def _materialize(bare: Path, sha: str, dest: Path) -> None:
    """Materialize one verified commit's exact tree read-only.

    Enumerates entries with ``git ls-tree -rz --full-tree`` and copies exact
    blob bytes through a single ``git cat-file --batch`` process; no archive
    is produced, so ``.gitattributes`` (``export-ignore``, ``export-subst``)
    can never omit or alter reviewed content.

    Args:
        bare: Temporary bare object repository holding the verified objects.
        sha: Verified full commit SHA.
        dest: Fresh directory to fill (created with mode 0o700).

    Raises:
        ReviewSnapshotError: Bounded tree/too-large/rejected classes.
    """
    entries = _ls_tree_entries(bare, sha)
    if len(entries) > MAX_SNAPSHOT_FILES:
        raise ReviewSnapshotError("review_snapshot_too_large")
    dest.mkdir(mode=0o700)
    root_real = str(dest.resolve())
    _write_entries(bare, entries, dest, root_real)
    _make_read_only(dest)


def _make_read_only(root: Path) -> None:
    """Freeze a snapshot tree: files 0o444 (0o500 when executable), dirs 0o555."""
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        for name in filenames:
            path = os.path.join(dirpath, name)
            try:
                executable = stat.S_IMODE(os.stat(path).st_mode) & 0o100
                os.chmod(
                    path,
                    _READ_ONLY_EXEC_FILE_MODE if executable else _READ_ONLY_FILE_MODE,
                )
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
    """Build the bounded changed-path manifest with deadline-aware I/O."""
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
    try:
        assert proc.stdout is not None
        reader = _DeadlineReader(proc.stdout, _GIT_TIMEOUT_SECONDS, "review_diff_failed")
        raw = reader.read_to_eof(_MAX_MANIFEST_BYTES + 1)
        truncated = len(raw) > _MAX_MANIFEST_BYTES
        if truncated:
            _stop_process(proc)
        elif _wait_before_deadline(proc, reader, "review_diff_failed") != 0:
            raise ReviewSnapshotError("review_diff_failed")
    except ReviewSnapshotError:
        _stop_process(proc)
        raise
    finally:
        if proc.stdout is not None:
            proc.stdout.close()
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
            if not _write_remote_config(bare, origin):
                raise _fail("review_init_failed", provider, number)
            head_ref = provider_ref_template(provider, number, "head")
            merge_ref = provider_ref_template(provider, number, "merge")
            _fetch_ref(
                bare, head_ref, f"{_LOCAL_REF_PREFIX}/head",
                required=True, provider=provider, number=number,
            )
            merge_present = _fetch_ref(
                bare, merge_ref, f"{_LOCAL_REF_PREFIX}/merge",
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
            _materialize(bare, base_sha, base_dir)
            _materialize(bare, head_sha, head_dir)
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
