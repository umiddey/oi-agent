"""Tests for manifest generation and refresh."""

import subprocess

from oi_agent.manifest.generator import (
    generate_repo_manifest, load_manifest_summary, refresh_repo_manifest,
)


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args],
                   capture_output=True, check=True)


def _init_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    (repo / "app" / "service.py").write_text(
        '"""Domain service for widgets."""\n\n\nclass WidgetService:\n'
        '    """Handles widget lifecycle."""\n\n    def create(self):\n'
        '        """Create a widget."""\n        return 1\n')
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "init")
    return repo


def test_generate_indexes_python(tmp_path):
    repo = _init_repo(tmp_path)
    result = generate_repo_manifest(repo)
    text = result.manifest_path.read_text()
    assert result.file_count == 1
    assert "app/service.py" in text
    assert "Domain service for widgets" in text
    assert "WidgetService" in text
    assert f"git: {result.git_sha}" in text


def test_refresh_noop_when_unchanged(tmp_path):
    repo = _init_repo(tmp_path)
    first = generate_repo_manifest(repo)
    second = refresh_repo_manifest(repo)
    assert second.git_sha == first.git_sha
    assert second.duration_ms >= 0
    # Content unchanged.
    assert second.manifest_path.read_text() == \
        first.manifest_path.read_text()


def test_refresh_accepts_str_path(tmp_path):
    """Regression: responder passes repo_path as str — must not crash."""
    repo = _init_repo(tmp_path)
    generate_repo_manifest(repo)
    result = refresh_repo_manifest(str(repo))  # type: ignore[arg-type]
    assert result.git_sha


def test_generate_excludes_oi_via_local_exclude(tmp_path):
    """.oi/ is excluded via .git/info/exclude, not the tracked .gitignore."""
    repo = _init_repo(tmp_path)
    generate_repo_manifest(repo)
    exclude = (repo / ".git" / "info" / "exclude").read_text()
    assert ".oi/" in exclude
    # Idempotent: no duplicate entries on regenerate.
    generate_repo_manifest(repo)
    assert (repo / ".git" / "info" / "exclude").read_text().count(".oi/") == 1


def test_generate_does_not_dirty_the_worktree(tmp_path):
    """Read-only invariant: generating the manifest leaves git status clean."""
    repo = _init_repo(tmp_path)
    generate_repo_manifest(repo)
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        capture_output=True, text=True, check=True).stdout.strip()
    assert status == "", f"worktree dirtied by manifest generation: {status!r}"


def test_dirty_worktree_tags_sha(tmp_path):
    """Uncommitted edits make the audit SHA honest with a -dirty suffix."""
    repo = _init_repo(tmp_path)
    clean = generate_repo_manifest(repo)
    assert not clean.git_sha.endswith("-dirty")
    (repo / "app" / "service.py").write_text("x = 2\n")  # uncommitted change
    dirty = refresh_repo_manifest(repo)
    assert dirty.git_sha.endswith("-dirty")


def test_symlink_escaping_repo_is_not_indexed(tmp_path):
    """Security: a tracked symlink pointing outside the repo must not leak."""
    import os
    outside = tmp_path / "secret_outside.py"
    outside.write_text('"""EXTERNAL SECRET DOCSTRING."""\nTOKEN = 1\n')
    repo = _init_repo(tmp_path)
    os.symlink(outside, repo / "leak.py")
    _git(repo, "add", "leak.py")
    _git(repo, "-c", "core.symlinks=true", "commit", "-qm", "add symlink")

    result = generate_repo_manifest(repo)
    text = result.manifest_path.read_text()
    assert "EXTERNAL SECRET DOCSTRING" not in text
    assert "app/service.py" in text


def test_gitignored_files_are_not_indexed(tmp_path):
    """Security: files git ignores (globs) must never enter the manifest."""
    repo = _init_repo(tmp_path)
    (repo / "config").mkdir()
    (repo / "config" / "data.py").write_text("x = 1\n")
    # Glob pattern the old literal matcher could not honor.
    (repo / ".gitignore").write_text("config/*.py\n")
    _git(repo, "add", ".gitignore")
    _git(repo, "commit", "-qm", "ignore config")
    result = generate_repo_manifest(repo)
    text = result.manifest_path.read_text()
    assert "config/data.py" not in text
    assert "app/service.py" in text


def test_summary_is_compact_and_dir_grouped(tmp_path):
    repo = _init_repo(tmp_path)
    generate_repo_manifest(repo)
    summary = load_manifest_summary(repo)
    assert "# Code Map" in summary
    assert "app/" in summary
