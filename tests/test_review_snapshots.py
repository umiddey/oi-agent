"""Tests for immutable controller-owned MR/PR base/head snapshots.

Uses temporary local Git repositories and hand-created provider refs; no
network and no real credentials. Asserts verification, exact-object
materialization (``export-ignore``/``export-subst`` cannot hide or alter
content), crafted-tree rejection, streaming size limits, argv discipline
(origin URLs never appear in process arguments), bounded manifests, worktree
immutability, and cleanup on every exit path.
"""

from __future__ import annotations

import gc
import os
import stat
import subprocess
import time
from pathlib import Path

import pytest

from oi_agent.opencode import review_snapshots
from oi_agent.opencode.paths import resolve_opencode_paths
from oi_agent.opencode.provenance import worktree_fingerprint
from oi_agent.opencode.review_snapshots import (
    MAX_CHANGED_PATHS,
    ReviewSnapshotError,
    _reject_symlink_ancestors,
    _validate_tree_path,
    _write_remote_config,
    materialized_review,
)
from oi_agent.opencode.review_target import PROVIDER_GITHUB, PROVIDER_GITLAB, provider_ref_template

_ALLOWED_VERBS = {
    "cat-file", "config", "diff", "fetch", "init", "ls-remote", "ls-tree", "rev-list", "rev-parse",
}
_FORBIDDEN_VERBS = {
    "add", "am", "apply", "archive", "branch", "cherry-pick", "clean", "clone", "commit", "merge",
    "mv", "pull", "push", "rebase", "remote", "reset", "restore", "revert", "rm",
    "stash", "switch", "tag", "update-ref", "worktree",
}
_SHA = "b" * 40


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _git_input(*args: str, data: bytes) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, input=data)
    return result.stdout.decode().strip()


def _blob(remote: Path, content: bytes) -> str:
    """Write one blob into the remote object database and return its SHA."""
    return _git_input("-C", str(remote), "hash-object", "-w", "--stdin", data=content)


def _mktree(remote: Path, lines: list[str]) -> str:
    """Build one tree object in the remote from ``mode type sha\\tpath`` lines."""
    data = "".join(line + "\n" for line in lines).encode()
    return _git_input("-C", str(remote), "mktree", data=data)


def _commit_tree(
    remote: Path, tree: str, message: str, parents: tuple[str, ...] = (),
) -> str:
    argv = [
        "-C", str(remote), "-c", "user.email=t@example.com", "-c", "user.name=Test",
        "commit-tree", tree, "-m", message,
    ]
    for parent in parents:
        argv += ["-p", parent]
    return _git(*argv)


def _bare_with_mr(
    tmp_path: Path,
    *,
    provider: str = PROVIDER_GITLAB,
    number: int = 188,
    remote_name: str = "remote.git",
    with_symlinks: bool = False,
    feature_files: int = 0,
) -> tuple[Path, Path, str, str]:
    """Create a bare remote with base/head/merge commits and provider refs."""
    remote = tmp_path / remote_name
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


def _republish_feature(
    remote: Path, clone: Path, provider: str = PROVIDER_GITLAB, number: int = 188,
) -> None:
    """Commit pending clone changes on feature, re-merge, reset provider refs."""
    _git("-C", str(clone), "add", "-A")
    _git("-C", str(clone), "commit", "-qm", "feature update")
    new_head = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "feature")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge", "feature")
    new_merge = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "feature", "main")
    _set_provider_refs(remote, provider, number, new_head, new_merge)


def _install_crafted_head(
    remote: Path, clone: Path, base_sha: str, tree: str,
    provider: str = PROVIDER_GITLAB, number: int = 188,
) -> None:
    """Point the provider refs at a head commit with a hand-crafted tree."""
    head = _commit_tree(remote, tree, "crafted head")
    merge = _commit_tree(remote, tree, "merge", parents=(base_sha, head))
    _set_provider_refs(remote, provider, number, head, merge)


def _reviews_glob() -> list[Path]:
    reviews = resolve_opencode_paths().state_home / "reviews"
    if not reviews.exists():
        return []
    return list(reviews.glob("oi-review-*"))


def _expect_materialization_rejected(clone: Path, failure: str = "review_extract_rejected") -> None:
    fingerprint = worktree_fingerprint(clone)
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == failure
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint


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
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        link = snapshot.head_dir / "inside_link"
        assert os.readlink(link) == "base.txt"


def test_export_ignore_cannot_hide_files_from_snapshot(tmp_path: Path, isolated_home: Path) -> None:
    """P1-1 regression: export-ignore must not omit a changed file."""
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    (clone / ".gitattributes").write_text("hidden.py export-ignore\n", encoding="utf-8")
    (clone / "hidden.py").write_text("hidden-canary-content\n", encoding="utf-8")
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        hidden = snapshot.head_dir / "hidden.py"
        assert hidden.read_text(encoding="utf-8") == "hidden-canary-content\n"
        assert (snapshot.head_dir / ".gitattributes").exists()
        statuses = {entry.path: entry.status for entry in snapshot.changed_paths}
        assert statuses["hidden.py"] == "A"


def test_export_subst_cannot_alter_snapshot_bytes(tmp_path: Path, isolated_home: Path) -> None:
    """P1-1 regression: export-subst must not expand placeholders."""
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    (clone / ".gitattributes").write_text("subst.py export-subst\n", encoding="utf-8")
    (clone / "subst.py").write_text("$Format:%d$\n", encoding="utf-8")
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        subst = snapshot.head_dir / "subst.py"
        assert subst.read_text(encoding="utf-8") == "$Format:%d$\n"


def test_executable_file_keeps_owner_exec_bit(tmp_path: Path, isolated_home: Path) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    (clone / "run.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    os.chmod(clone / "run.sh", 0o755)
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        script = snapshot.head_dir / "run.sh"
        assert stat.S_IMODE(script.stat().st_mode) == 0o500
        assert script.read_text(encoding="utf-8") == "#!/bin/sh\n"


def test_gitmodules_file_is_materialized(tmp_path: Path, isolated_home: Path) -> None:
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    (clone / ".gitmodules").write_text('[submodule "x"]\n\tpath = x\n', encoding="utf-8")
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        modules = snapshot.head_dir / ".gitmodules"
        assert modules.read_text(encoding="utf-8") == '[submodule "x"]\n\tpath = x\n'


def test_head_snapshot_is_complete_against_ls_tree_and_manifest(
    tmp_path: Path, isolated_home: Path,
) -> None:
    """P1-1 regression: every changed path exists in the head snapshot, and
    the materialized entry count equals the verified tree's ls-tree count."""
    remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "checkout", "-q", "feature")
    (clone / "sub").mkdir()
    (clone / "sub" / "nested.txt").write_text("n\n", encoding="utf-8")
    os.symlink("base.txt", clone / "inside_link")
    _republish_feature(remote, clone)
    with materialized_review(clone, PROVIDER_GITLAB, 188) as snapshot:
        listing = subprocess.run(
            ["git", "-C", str(clone), "ls-tree", "-rz", "--full-tree", snapshot.head_sha],
            check=True, capture_output=True,
        ).stdout
        expected = len([record for record in listing.decode().split("\0") if record])
        seen = 0
        for dirpath, _dirnames, filenames in os.walk(snapshot.head_dir):
            seen += len(filenames)
        assert seen == expected
        statuses = {entry.path: entry.status for entry in snapshot.changed_paths}
        for path, status in statuses.items():
            if status in ("A", "M"):
                assert (snapshot.head_dir / path).exists(), path


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
# 4. Origin URL never appears in process arguments (P2-3)
# ==============================================================================

def test_origin_url_never_appears_in_process_arguments(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The origin URL (canary remote name) appears in zero argv elements."""
    remote, clone, _base, _head = _bare_with_mr(
        tmp_path, remote_name="remote-canary-S3cr3t-token.git",
    )
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
    assert calls
    for argv in calls:
        for element in argv:
            assert "S3cr3t-token" not in element
            assert str(remote) not in element
    # Networked commands address the remote by NAME, resolved from config.
    probes = [argv for argv in calls if "ls-remote" in argv]
    assert probes
    assert all("origin" in argv for argv in probes)


def test_embedded_credential_never_reaches_arguments(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token-bearing https origin stays out of argv even on fetch failure."""
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    token_url = "https://user:c4nary-t0ken@127.0.0.1:1/repo.git"
    _git("-C", str(clone), "remote", "set-url", "origin", token_url)
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
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_fetch_failed"
    assert calls
    for argv in calls:
        for element in argv:
            assert "c4nary-t0ken" not in element
            assert token_url not in element
    assert _reviews_glob() == []


def test_remote_config_quoting_round_trips(tmp_path: Path) -> None:
    bare = tmp_path / "quote.git"
    _git("init", "-q", "--bare", str(bare))
    url = 'https://user:to"ken@host/repo sp\\ace.git;x#y'
    assert _write_remote_config(bare, url) is True
    assert _git("-C", str(bare), "config", "--get", "remote.origin.url") == url


def test_remote_config_rejects_line_breaks(tmp_path: Path) -> None:
    bare = tmp_path / "newline.git"
    _git("init", "-q", "--bare", str(bare))
    injection = 'https://host/repo.git\n[remote "evil"]\n\turl = x'
    assert _write_remote_config(bare, injection) is False


def test_origin_with_line_break_fails_closed(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _git("-C", str(clone), "remote", "remove", "origin")
    _git("-C", str(clone), "config", "remote.origin.url", "https://host/repo.git\nevil")
    with pytest.raises(ReviewSnapshotError) as exc:
        materialized_review(clone, PROVIDER_GITLAB, 188)
    assert exc.value.failure_class == "review_init_failed"
    assert _reviews_glob() == []


# ==============================================================================
# 5. Safe materialization: crafted trees and path policy (P1-1)
# ==============================================================================

def test_escaping_symlinks_in_head_are_rejected(tmp_path: Path, isolated_home: Path) -> None:
    _remote, clone, _base, _head = _bare_with_mr(tmp_path, with_symlinks=True)
    _expect_materialization_rejected(clone)


@pytest.mark.parametrize(
    "name",
    ["../evil", "/abs/evil", ".git/config", ".git", "C:evil", "a\\b", "//evil", "."],
)
def test_tree_path_validator_rejects_hostile_names(name: str) -> None:
    with pytest.raises(ReviewSnapshotError):
        _validate_tree_path(name)


@pytest.mark.parametrize("name", [".gitmodules", ".gitignore", "ok.txt", "sub/file.txt", "..."])
def test_tree_path_validator_allows_ordinary_names(name: str) -> None:
    _validate_tree_path(name)


def test_symlink_directory_ancestor_check() -> None:
    with pytest.raises(ReviewSnapshotError):
        _reject_symlink_ancestors("d/inner.txt", {"d"})
    _reject_symlink_ancestors("d/inner.txt", {"other"})
    _reject_symlink_ancestors("top.txt", {"d"})


def test_parse_ls_tree_accepts_blob_symlink_and_nested_paths() -> None:
    entries = review_snapshots._parse_ls_tree(
        f"100644 blob {_SHA}\ttop.txt\0"
        f"100755 blob {_SHA}\tsub/run.sh\0"
        f"120000 blob {_SHA}\tlink\0"
    )
    assert [(entry.path, entry.executable, entry.is_symlink) for entry in entries] == [
        ("top.txt", False, False),
        ("sub/run.sh", True, False),
        ("link", False, True),
    ]


def test_parse_ls_tree_rejects_submodule_entry() -> None:
    with pytest.raises(ReviewSnapshotError):
        review_snapshots._parse_ls_tree(f"160000 commit {_SHA}\tsub\0")


def test_parse_ls_tree_rejects_unusual_mode() -> None:
    with pytest.raises(ReviewSnapshotError):
        review_snapshots._parse_ls_tree(f"100664 blob {_SHA}\tf.txt\0")


def test_parse_ls_tree_rejects_record_without_path() -> None:
    with pytest.raises(ReviewSnapshotError):
        review_snapshots._parse_ls_tree(f"100644 blob {_SHA}\0")


def test_parse_ls_tree_rejects_malformed_object_sha() -> None:
    with pytest.raises(ReviewSnapshotError):
        review_snapshots._parse_ls_tree("100644 blob zzz\tf.txt\0")


def test_crafted_tree_with_dotdot_path_is_rejected(tmp_path: Path, isolated_home: Path) -> None:
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    blob = _blob(remote, b"pwn")
    tree = _mktree(remote, [f"100644 blob {blob}\tok.txt", f"100644 blob {blob}\t.."])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_windows_drive_path_is_rejected(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    blob = _blob(remote, b"pwn")
    tree = _mktree(remote, [f"100644 blob {blob}\tC:evil"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_backslash_path_is_rejected(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    blob = _blob(remote, b"pwn")
    tree = _mktree(remote, ["100644 blob " + blob + "\ta" + chr(92) + "b"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_git_entry_is_rejected(tmp_path: Path, isolated_home: Path) -> None:
    """A symlink whose target enters .git is rejected (blob content is free-form)."""
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    blob = _blob(remote, b"hooks")
    target = _blob(remote, b".git/hooks")
    tree = _mktree(remote, [f"100644 blob {blob}\tok.txt", f"120000 blob {target}\tevil"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_absolute_symlink_is_rejected(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    target = _blob(remote, b"/etc/passwd")
    tree = _mktree(remote, [f"120000 blob {target}\tevil"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_escaping_symlink_is_rejected(
    tmp_path: Path, isolated_home: Path,
) -> None:
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    target = _blob(remote, b"../../outside")
    sub = _mktree(remote, [f"120000 blob {target}\tevil"])
    tree = _mktree(remote, [f"040000 tree {sub}\tsub"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


def test_crafted_tree_with_submodule_entry_is_rejected(
    tmp_path: Path, isolated_home: Path,
) -> None:
    """A 160000 commit entry fails the whole materialization."""
    remote, clone, base_sha, _head = _bare_with_mr(tmp_path)
    gitlink = _commit_tree(remote, _mktree(remote, [f"100644 blob {_blob(remote, b'x')}\tf"]), "sub")
    blob = _blob(remote, b"pwn")
    tree = _mktree(remote, [f"100644 blob {blob}\tok.txt", f"160000 commit {gitlink}\tsub"])
    _install_crafted_head(remote, clone, base_sha, tree)
    _expect_materialization_rejected(clone)


# ==============================================================================
# 6. Streaming size limits (P2-4)
# ==============================================================================

def test_single_blob_over_cap_is_rejected(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_snapshots, "MAX_SINGLE_BLOB_BYTES", 2)
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _expect_materialization_rejected(clone, "review_snapshot_too_large")


def test_total_bytes_over_cap_is_rejected(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_snapshots, "MAX_SNAPSHOT_TOTAL_BYTES", 6)
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _expect_materialization_rejected(clone, "review_snapshot_too_large")


def test_file_count_over_cap_is_rejected(
    tmp_path: Path, isolated_home: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(review_snapshots, "MAX_SNAPSHOT_FILES", 1)
    _remote, clone, _base, _head = _bare_with_mr(tmp_path)
    _expect_materialization_rejected(clone, "review_snapshot_too_large")

@pytest.mark.parametrize("method", ["read_exact", "read_to_eof"])
def test_subprocess_pipe_reads_obey_deadline(method: str) -> None:
    """A silent subprocess pipe cannot block snapshot work indefinitely."""
    read_fd, write_fd = os.pipe()
    started = time.monotonic()
    try:
        with os.fdopen(read_fd, "rb", buffering=0) as stream:
            reader = review_snapshots._DeadlineReader(
                stream, 0.01, "review_tree_failed",
            )
            with pytest.raises(ReviewSnapshotError) as exc:
                getattr(reader, method)(1)
        assert exc.value.failure_class == "review_tree_failed"
        assert time.monotonic() - started < 0.5
    finally:
        os.close(write_fd)


# ==============================================================================
# 7. Cleanup on exception, close, and garbage collection
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
# 8. Git argv discipline: read-only verbs only
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
    assert "cat-file" in verbs and "ls-tree" in verbs
    # The only config access is the read-only origin lookup.
    for argv in calls:
        if "config" in argv:
            assert argv[argv.index("config") + 1:] == ["--get", "remote.origin.url"]
