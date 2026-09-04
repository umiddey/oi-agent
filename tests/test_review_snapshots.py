"""Tests for immutable controller-owned MR/PR base/head snapshots.

Uses temporary local Git repositories and hand-created provider refs; no
network and no real credentials. Asserts verification, safe extraction,
bounded manifests, argv discipline, worktree immutability, and cleanup on
every exit path.
"""

from __future__ import annotations

import gc
import io
import os
import stat
import subprocess
import tarfile
from pathlib import Path

import pytest

from oi_agent.opencode.paths import resolve_opencode_paths
from oi_agent.opencode.provenance import worktree_fingerprint
from oi_agent.opencode.review_snapshots import (
    MAX_CHANGED_PATHS,
    ReviewSnapshotError,
    _extract_snapshot_tar,
    materialized_review,
)
from oi_agent.opencode.review_target import PROVIDER_GITHUB, PROVIDER_GITLAB, provider_ref_template

_ALLOWED_VERBS = {
    "archive", "config", "diff", "fetch", "init", "ls-remote", "rev-list", "rev-parse",
}
_FORBIDDEN_VERBS = {
    "add", "am", "apply", "branch", "cherry-pick", "clean", "clone", "commit", "merge",
    "mv", "pull", "push", "rebase", "remote", "reset", "restore", "revert", "rm",
    "stash", "switch", "tag", "update-ref", "worktree",
}


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _bare_with_mr(
    tmp_path: Path,
    *,
    provider: str = PROVIDER_GITLAB,
    number: int = 188,
    with_symlinks: bool = False,
    feature_files: int = 0,
) -> tuple[Path, Path, str, str]:
    """Create a bare remote with base/head/merge commits and provider refs."""
    remote = tmp_path / "remote.git"
    _git("init", "-q", "--bare", str(remote))
    clone = tmp_path / "clone"
    clone.mkdir(parents=True)
    _git("init", "-q", "-b", "main", str(clone))
    _git("-C", str(clone), "config", "user.email", "t@example.com")
    _git("-C", str(clone), "config", "user.name", "Test")
    _git("-C", str(clone), "remote", "add", "origin", str(remote))
    (clone / "base.txt").write_text("base\n", encoding="utf-8")
    _git("-C", str(clone), "add", ".")
    _git("-C", str(clone), "commit", "-qm", "base")
    base_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "-b", "feature")
    (clone / "base.txt").write_text("head\n", encoding="utf-8")
    (clone / "head.txt").write_text("head\n", encoding="utf-8")
    for index in range(feature_files):
        (clone / f"extra_{index:04d}.txt").write_text("x\n", encoding="utf-8")
    if with_symlinks:
        os.symlink("/etc/passwd", clone / "escape_abs")
        os.symlink("../../outside", clone / "escape_rel")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "head")
    head_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge result", "feature")
    _git("-C", str(clone), "push", "-q", "origin", "main", "feature")
    _git(
        "-C", str(remote), "update-ref",
        provider_ref_template(provider, number, "head"), head_sha,
    )
    merge_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git(
        "-C", str(remote), "update-ref",
        provider_ref_template(provider, number, "merge"), merge_sha,
    )
    return remote, clone, base_sha, head_sha


def _set_provider_refs(remote: Path, provider: str, number: int, head: str, merge: str) -> None:
    _git("-C", str(remote), "update-ref", provider_ref_template(provider, number, "head"), head)
    _git("-C", str(remote), "update-ref", provider_ref_template(provider, number, "merge"), merge)


def _reviews_glob() -> list[Path]:
    reviews = resolve_opencode_paths().state_home / "reviews"
    if not reviews.exists():
        return []
    return list(reviews.glob("oi-review-*"))


@pytest.fixture()
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point OI's isolated state roots at the test tmp path."""
    home = tmp_path / "operator-home"
    monkeypatch.setenv("HOME", str(home))
    return home


# ==============================================================================
# 1. Successful materialization and verification
# ==============================================================================

def test_gitlab_snapshot_success_verification_and_cleanup(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        assert snapshot.provider == "gitlab"
        assert snapshot.number == 188
        assert snapshot.base_sha == base_sha
        assert snapshot.head_sha == head_sha
        assert snapshot.base_dir != snapshot.head_dir
        # Base and head content differ; both trees are readable.
        assert (snapshot.base_dir / "base.txt").read_text(encoding="utf-8") == "base\n"
        assert (snapshot.head_dir / "base.txt").read_text(encoding="utf-8") == "head\n"
        assert (snapshot.head_dir / "head.txt").read_text(encoding="utf-8") == "head\n"
        assert not (snapshot.base_dir / ".git").exists()
        assert not (snapshot.head_dir / ".git").exists()
        # Extracted trees are read-only and live under OI's isolated state root.
        assert stat.S_IMODE(snapshot.base_dir.stat().st_mode) == 0o555
        assert stat.S_IMODE(snapshot.head_dir.stat().st_mode) == 0o555
        head_file = snapshot.head_dir / "head.txt"
        assert stat.S_IMODE(head_file.stat().st_mode) == 0o444
        assert str(isolated_home) in str(snapshot.base_dir)
        # Changed-path manifest is correct and not truncated.
        statuses = {entry.path: entry.status for entry in snapshot.changed_paths}
        assert statuses == {"base.txt": "M", "head.txt": "A"}
        assert snapshot.manifest_truncated is False
        workspace = snapshot.base_dir.parent
        assert sorted(p.name for p in workspace.iterdir()) == ["base", "head", "objects.git"]
    # Worktree untouched; every temp file removed.
    assert worktree_fingerprint(clone) == fingerprint
    assert not workspace.exists()
    assert _reviews_glob() == []


def test_github_snapshot_success(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, base_sha, head_sha = _bare_with_mr(tmp_path, provider=PROVIDER_GITHUB)
    with materialized_review(clone, PROVIDER_GITHUB, 188) as snapshot:
        assert snapshot.base_sha == base_sha
        assert snapshot.head_sha == head_sha
        statuses = {entry.path: entry.status for entry in snapshot.changed_paths}
        assert statuses == {"base.txt": "M", "head.txt": "A"}


def test_manifest_truncates_honestly(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path, feature_files=210)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        assert len(snapshot.changed_paths) == MAX_CHANGED_PATHS
        assert snapshot.manifest_truncated is True


def test_internal_symlink_inside_root_is_allowed(tmp_path: Path, isolated_home: Path) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    os.symlink("base.txt", clone / "inside_link")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "internal link")
    new_head = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "feature")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge", "feature")
    new_merge = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "feature", "main")
    _set_provider_refs(remote, PROVIDER_GITLAB, 188, new_head, new_merge)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        link = snapshot.head_dir / "inside_link"
        assert os.readlink(link) == "base.txt"


# ==============================================================================
# 2. Verification failures (missing refs, malformed merge, SHA mismatch)
# ==============================================================================

def test_missing_head_ref_is_unavailable(tmp_path: Path, isolated_home: Path) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(remote), "update-ref", "-d", "refs/merge-requests/188/head")
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_ref_unavailable"
    assert _reviews_glob() == []


def test_missing_merge_ref_fails_without_default_branch_fallback(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(remote), "update-ref", "-d", "refs/merge-requests/188/merge")
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_base_unavailable"
    assert _reviews_glob() == []


def test_merge_ref_second_parent_mismatch_is_unverified(
    tmp_path: Path, isolated_home: Path,
) -> None:
    """A merge ref whose second parent is not the head SHA must fail closed."""
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "-b", "unrelated", "main~1")
    (clone / "other.txt").write_text("other\n", encoding="utf-8")
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "other")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "wrong merge", "unrelated")
    wrong_merge = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "unrelated", "main")
    head = _git("-C", str(remote), "rev-parse", "refs/merge-requests/188/head")
    _set_provider_refs(remote, PROVIDER_GITLAB, 188, head, wrong_merge)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_base_unverified"
    assert _reviews_glob() == []


def test_non_merge_commit_as_merge_ref_is_unverified(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    head = _git("-C", str(remote), "rev-parse", "refs/merge-requests/188/head")
    _git("-C", str(remote), "update-ref", "refs/merge-requests/188/merge", head)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_base_unverified"
    assert _reviews_glob() == []


def test_unknown_provider_fails_defensively(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, "gitea", 188)
    assert exc.value.failure_class == "review_provider_invalid"
    assert _reviews_glob() == []


def test_out_of_range_number_fails_defensively(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 0)
    assert exc.value.failure_class == "review_number_invalid"
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 99_999_999)
    assert exc.value.failure_class == "review_number_invalid"


# ==============================================================================
# 3. Origin and fetch failures
# ==============================================================================

def test_missing_origin_is_unreadable(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "remote", "remove", "origin")
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_origin_unreadable"
    assert _reviews_glob() == []


def test_unreachable_origin_is_fetch_failure(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "remote", "set-url", "origin", str(tmp_path / "void"))
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_fetch_failed"
    assert _reviews_glob() == []


def test_fetch_timeout_returns_bounded_class_and_cleans_up(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    real_run = subprocess.run

    def fake_run(argv: list[str], *args: object, **kwargs: object):
        if "fetch" in argv:
            raise subprocess.TimeoutExpired(cmd=argv, timeout=1)
        return real_run(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_fetch_timeout"
    assert _reviews_glob() == []


# ==============================================================================
# 4. Safe extraction: real commits and crafted tars
# ==============================================================================

def test_escaping_symlinks_in_head_are_rejected(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path, with_symlinks=True)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_extract_rejected"
    assert _reviews_glob() == []


def _write_tar(path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    with tarfile.open(path, "w") as archive:
        for info, payload in members:
            if payload is None:
                archive.addfile(info)
            else:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def _file_info(name: str, payload: bytes = b"data", mode: int = 0o644) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.size = len(payload)
    info.mode = mode
    return info


def _expect_rejected(tmp_path: Path, members: list[tuple[tarfile.TarInfo, bytes | None]]) -> None:
    tar_path = tmp_path / "crafted.tar"
    dest = tmp_path / "out"
    _write_tar(tar_path, members)
    with pytest.raises(ReviewSnapshotError) as exc:
        _extract_snapshot_tar(tar_path, dest)
    assert exc.value.failure_class == "review_extract_rejected"
    assert not (tmp_path / "evil").exists()
    assert not (dest / "evil").exists()


def test_extract_rejects_parent_traversal(tmp_path: Path) -> None:
    _expect_rejected(tmp_path, [(_file_info("../evil"), b"pwn")])


def test_extract_rejects_absolute_path(tmp_path: Path) -> None:
    _expect_rejected(tmp_path, [(_file_info("/abs/evil"), b"pwn")])


def test_extract_rejects_windows_drive_path(tmp_path: Path) -> None:
    _expect_rejected(tmp_path, [(_file_info("C:evil"), b"pwn")])


def test_extract_rejects_git_directory_entry(tmp_path: Path) -> None:
    _expect_rejected(tmp_path, [(_file_info(".git/config"), b"pwn")])


def test_extract_rejects_hardlink(tmp_path: Path) -> None:
    info = tarfile.TarInfo("evil")
    info.type = tarfile.LNKTYPE
    info.linkname = "base.txt"
    _expect_rejected(tmp_path, [(info, None)])


def test_extract_rejects_device_node(tmp_path: Path) -> None:
    info = tarfile.TarInfo("evil")
    info.type = tarfile.CHRTYPE
    info.devmajor = 1
    info.devminor = 3
    _expect_rejected(tmp_path, [(info, None)])


def test_extract_rejects_setuid_file(tmp_path: Path) -> None:
    _expect_rejected(tmp_path, [(_file_info("evil", mode=0o4755), b"pwn")])


def test_extract_rejects_absolute_symlink(tmp_path: Path) -> None:
    info = tarfile.TarInfo("evil")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"
    _expect_rejected(tmp_path, [(info, None)])


def test_extract_rejects_escaping_relative_symlink(tmp_path: Path) -> None:
    info = tarfile.TarInfo("sub/evil")
    info.type = tarfile.SYMTYPE
    info.linkname = "../../etc/passwd"
    _expect_rejected(tmp_path, [(info, None)])


# ==============================================================================
# 5. Cleanup on exception, close, and garbage collection
# ==============================================================================

def test_cleanup_after_exception_in_body(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    with pytest.raises(RuntimeError, match="boom"):
        with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
            workspace = snapshot.base_dir.parent
            assert workspace.exists()
            raise RuntimeError("boom")
    assert not workspace.exists()
    assert _reviews_glob() == []


def test_close_is_idempotent(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    review = materialized_review(clone, PROVIDER_GITLAB, 188)
    workspace = review.snapshot.base_dir.parent
    assert workspace.exists()
    review.close()
    review.close()
    assert not workspace.exists()


def test_garbage_collection_finalizer_cleans_up(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    review = materialized_review(clone, PROVIDER_GITLAB, 188)
    workspace = review._tmp_root  # noqa: SLF001 - test reaches in for GC coverage
    assert workspace is not None and workspace.exists()
    del review
    gc.collect()
    assert not workspace.exists()
    assert _reviews_glob() == []


# ==============================================================================
# 6. Git argv discipline: read-only verbs only
# ==============================================================================

def test_only_readonly_git_commands_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    real_run = subprocess.run
    real_popen = subprocess.Popen
    calls: list[list[str]] = []

    def recording_run(argv: list[str], *args: object, **kwargs: object):
        calls.append(list(argv))
        return real_run(argv, *args, **kwargs)

    def recording_popen(argv: list[str], *args: object, **kwargs: object):
        calls.append(list(argv))
        return real_popen(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    monkeypatch.setattr(subprocess, "Popen", recording_popen)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        assert snapshot.head_sha
    verbs: list[str] = []
    for argv in calls:
        assert argv[0] == "git"
        rest = argv[1:]
        while rest and rest[0] in ("-C", "-c"):
            rest = rest[2:]
        verbs.append(rest[0])
    assert verbs, "expected recorded git activity"
    assert not (set(verbs) - _ALLOWED_VERBS), sorted(set(verbs) - _ALLOWED_VERBS)
    assert set(verbs).isdisjoint(_FORBIDDEN_VERBS)
    # The only config access is the read-only origin lookup.
    for argv in calls:
        if "config" in argv:
            assert argv[argv.index("config") + 1:] == ["--get", "remote.origin.url"]
