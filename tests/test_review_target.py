"""Tests for safe MR/PR review-reference parsing and authorization.

Uses temporary local Git repositories and hand-created provider refs; no
network and no real credentials. Remote URLs never appear in logs.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from oi_agent.opencode.review_target import (
    MAX_REVIEW_NUMBER,
    PROVIDER_GITHUB,
    PROVIDER_GITLAB,
    ReviewRef,
    ReviewRefError,
    normalize_remote_url,
    parse_review_ref,
    provider_ref_template,
    read_origin_url,
    resolve_review_ref,
)

_GIT = ["git"]


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git("init", "-q", "-b", "main", str(path))
    _git("-C", str(path), "config", "user.email", "t@example.com")
    _git("-C", str(path), "config", "user.name", "Test")
    return path


def _repo_with_origin(path: Path, origin: str) -> Path:
    """Create a committed repo whose origin URL is set explicitly."""
    repo = _init_repo(path)
    (repo / "file.txt").write_text("content\n", encoding="utf-8")
    _git("-C", str(repo), "add", ".")
    _git("-C", str(repo), "commit", "-qm", "init")
    _git("-C", str(repo), "remote", "add", "origin", origin)
    return repo


def _bare_with_mr(
    tmp_path: Path,
    *,
    provider: str = PROVIDER_GITLAB,
    number: int = 188,
) -> tuple[Path, Path, str, str]:
    """Create a bare remote with base/head/merge commits and provider refs."""
    remote = tmp_path / "remote.git"
    _git("init", "-q", "--bare", str(remote))
    clone = _init_repo(tmp_path / "clone")
    _git("-C", str(clone), "remote", "add", "origin", str(remote))
    (clone / "base.txt").write_text("base\n", encoding="utf-8")
    _git("-C", str(clone), "add", ".")
    _git("-C", str(clone), "commit", "-qm", "base")
    base_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "-b", "feature")
    (clone / "head.txt").write_text("head\n", encoding="utf-8")
    _git("-C", str(clone), "add", ".")
    _git("-C", str(clone), "commit", "-qm", "head")
    head_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "checkout", "-q", "main")
    _git("-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge result", "feature")
    merge_sha = _git("-C", str(clone), "rev-parse", "HEAD")
    _git("-C", str(clone), "push", "-q", "origin", "main", "feature")
    _git("-C", str(remote), "update-ref", provider_ref_template(provider, number, "head"), head_sha)
    _git("-C", str(remote), "update-ref", provider_ref_template(provider, number, "merge"), merge_sha)
    return remote, clone, base_sha, head_sha


# ==============================================================================
# 1. Supported reference forms
# ==============================================================================

@pytest.mark.parametrize("text,provider,number", [
    ("please review MR 188", "gitlab", 188),
    ("MR 188 looks wrong", "gitlab", 188),
    ("mr 188", "gitlab", 188),
    ("Mr 188.", "gitlab", 188),
    ("MR: 188", "gitlab", 188),
    ("MR 007", "gitlab", 7),
    ("!188", "gitlab", 188),
    ("check !188 please", "gitlab", 188),
    ("(!188)", "gitlab", 188),
    ("!188.", "gitlab", 188),
    ("PR 188", "github", 188),
    ("pr 188", "github", 188),
    ("pull request 188", "github", 188),
    ("Pull Request 188", "github", 188),
    ("PR: 188", "github", 188),
    ("MR 10000000", "gitlab", 10_000_000),
])
def test_supported_keyword_forms(text: str, provider: str, number: int) -> None:
    ref = parse_review_ref(text)
    assert ref is not None
    assert ref.provider == provider
    assert ref.number == number
    assert ref.url is None


@pytest.mark.parametrize("text,provider,number", [
    ("https://gitlab.com/group/proj/-/merge_requests/188", "gitlab", 188),
    ("https://gitlab.com/group/proj/merge_requests/188", "gitlab", 188),
    ("see https://gitlab.example.com/sub/group/proj/-/merge_requests/7.", "gitlab", 7),
    ("https://github.com/owner/repo/pull/188", "github", 188),
    ("look at (https://github.com/owner/repo/pull/42)", "github", 42),
])
def test_supported_url_forms(text: str, provider: str, number: int) -> None:
    ref = parse_review_ref(text)
    assert ref is not None
    assert ref.provider == provider
    assert ref.number == number
    assert ref.url is not None and ref.url.startswith("http")


# ==============================================================================
# 2. Non-matches must never trigger resolution
# ==============================================================================

@pytest.mark.parametrize("text", [
    "188",
    "#188",
    "issue 188",
    "merge 188",
    "the MR I mentioned",
    "PRs and MRs everywhere",
    "https://gitlab.com/g/p/-/issues/188",
    "https://github.com/o/r/commit/abc1234567890abcdef1234567890abcdef123456",
    "https://example.com/some/page",
    "fix the !voting system",
    "",
])
def test_non_references_do_not_match(text: str) -> None:
    assert parse_review_ref(text) is None


# ==============================================================================
# 3. Invalid and malicious references fail closed
# ==============================================================================

@pytest.mark.parametrize("text", [
    "MR 0",
    "!0",
    "MR 00",
    "MR 99999999",
    "MR 10000001",
    "!99999999999999",
    f"MR {MAX_REVIEW_NUMBER + 1}",
    "MR #188",
    "MR !188",
    "MR 188;ls",
    "MR $(id)",
    "MR `id`",
    "!188/etc/passwd",
    "MR '188'",
])
def test_invalid_references_fail_closed(text: str) -> None:
    with pytest.raises(ReviewRefError) as exc:
        parse_review_ref(text)
    assert exc.value.failure_class == "review_ref_invalid"


def test_unsupported_forge_urls_fail_closed() -> None:
    for text in (
        "https://bitbucket.org/g/p/pull-requests/188",
        "https://gitea.example.com/o/r/pulls/188",
    ):
        with pytest.raises(ReviewRefError) as exc:
            parse_review_ref(text)
        assert exc.value.failure_class == "review_ref_unsupported"


def test_provider_host_mismatch_fails_unsupported() -> None:
    """A forge host contradicting the path-implied provider fails closed."""
    for text in (
        "https://github.com/a/b/merge_requests/1",
        "https://www.github.com/a/b/merge_requests/1",
        "https://gitlab.com/g/p/pull/188",
        "https://notebook.gitlab.com/g/p/pull/7",
    ):
        with pytest.raises(ReviewRefError) as exc:
            parse_review_ref(text)
        assert exc.value.failure_class == "review_ref_unsupported"


def test_self_hosted_hosts_keep_path_based_provider() -> None:
    """Self-hosted hosts keep inferring the provider from the path shape."""
    gitlab_ref = parse_review_ref(
        "https://gitlab.company.com/group/proj/merge_requests/188"
    )
    assert gitlab_ref is not None
    assert gitlab_ref.provider == "gitlab"
    assert gitlab_ref.number == 188
    github_ref = parse_review_ref("https://github.company.com/o/r/pull/42")
    assert github_ref is not None
    assert github_ref.provider == "github"
    assert github_ref.number == 42


def test_url_with_fragment_or_query_fails_closed() -> None:
    with pytest.raises(ReviewRefError) as fragment_exc:
        parse_review_ref("https://gitlab.com/g/p/-/merge_requests/188#note_123")
    assert fragment_exc.value.failure_class == "review_ref_invalid"
    with pytest.raises(ReviewRefError) as query_exc:
        parse_review_ref("https://github.com/o/r/pull/188?view=inline")
    assert query_exc.value.failure_class == "review_ref_invalid"


def test_malformed_mr_url_shape_fails_closed() -> None:
    with pytest.raises(ReviewRefError) as exc:
        parse_review_ref("https://gitlab.com/g/p/-/merge_requests/188/diffs")
    assert exc.value.failure_class == "review_ref_invalid"


# ==============================================================================
# 4. Multiple distinct references are ambiguous; duplicates collapse
# ==============================================================================

def test_multiple_distinct_references_fail_closed() -> None:
    for text in (
        "MR 188 and PR 200",
        "MR 188 or MR 189",
        "https://github.com/o/r/pull/188 and https://gitlab.com/g/p/-/merge_requests/188",
    ):
        with pytest.raises(ReviewRefError) as exc:
            parse_review_ref(text)
        assert exc.value.failure_class == "review_ref_multiple"


def test_duplicate_same_reference_is_single() -> None:
    ref = parse_review_ref("MR 188 or !188")
    assert ref is not None
    assert ref.provider == "gitlab"
    assert ref.number == 188


def test_keyword_matching_url_is_not_ambiguous() -> None:
    ref = parse_review_ref("PR 188 https://github.com/o/r/pull/188")
    assert ref is not None
    assert (ref.provider, ref.number) == ("github", 188)
    assert ref.url is not None


def test_conflicting_keyword_and_url_fail_closed() -> None:
    with pytest.raises(ReviewRefError) as exc:
        parse_review_ref("PR 5 https://github.com/o/r/pull/188")
    assert exc.value.failure_class == "review_ref_multiple"


# ==============================================================================
# 5. Explicit URL authorization against configured origins
# ==============================================================================

def test_explicit_url_resolves_to_matching_origin(tmp_path: Path) -> None:
    repo = _repo_with_origin(
        tmp_path / "repo", "https://gitlab.example.com/group/project.git"
    )
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/188")
    assert ref is not None
    target = resolve_review_ref(ref, [str(repo)])
    assert target.provider == "gitlab"
    assert target.number == 188
    assert target.repo_path == str(repo)


def test_explicit_url_scp_style_origin_matches(tmp_path: Path) -> None:
    repo = _repo_with_origin(tmp_path / "repo", "git@gitlab.example.com:group/project.git")
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/12")
    assert ref is not None
    assert resolve_review_ref(ref, [str(repo)]).repo_path == str(repo)


def test_explicit_url_foreign_project_mismatches(tmp_path: Path) -> None:
    repo = _repo_with_origin(
        tmp_path / "repo", "https://gitlab.example.com/other/project.git"
    )
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [str(repo)])
    assert exc.value.failure_class == "review_ref_url_mismatch"


def test_explicit_url_foreign_host_mismatches(tmp_path: Path) -> None:
    repo = _repo_with_origin(tmp_path / "repo", "https://gitlab.com/group/project.git")
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [str(repo)])
    assert exc.value.failure_class == "review_ref_url_mismatch"


def test_explicit_url_matching_multiple_origins_mismatches(tmp_path: Path) -> None:
    origin = "https://gitlab.example.com/group/project.git"
    repo_a = _repo_with_origin(tmp_path / "repoA", origin)
    repo_b = _repo_with_origin(tmp_path / "repoB", origin)
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [str(repo_a), str(repo_b)])
    assert exc.value.failure_class == "review_ref_url_mismatch"


def test_origin_credentials_never_logged(tmp_path: Path, caplog) -> None:
    repo = _repo_with_origin(
        tmp_path / "repo",
        "https://tokenizer:SECRET_TOKEN_VALUE@gitlab.example.com/group/project.git",
    )
    ref = parse_review_ref("https://gitlab.example.com/group/project/-/merge_requests/188")
    assert ref is not None
    with caplog.at_level(logging.DEBUG, logger="oi_agent"):
        target = resolve_review_ref(ref, [str(repo)])
    assert target.repo_path == str(repo)
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(
            parse_review_ref(
                "https://gitlab.example.com/other/project/-/merge_requests/188"
            ),
            [str(repo)],
        )
    assert "SECRET_TOKEN_VALUE" not in caplog.text
    assert "tokenizer@" not in caplog.text
    assert "gitlab.example.com" not in caplog.text
    assert str(exc.value) == "review_ref_url_mismatch"


# ==============================================================================
# 6. Bare-number resolution against configured repositories
# ==============================================================================

def test_bare_number_resolves_single_repo(tmp_path: Path) -> None:
    _remote, clone, _base_sha, _head_sha = _bare_with_mr(tmp_path)
    ref = parse_review_ref("MR 188")
    assert ref is not None
    target = resolve_review_ref(ref, [str(clone)])
    assert target.repo_path == str(clone)
    assert (target.provider, target.number) == ("gitlab", 188)


def test_bare_number_unavailable_when_no_repo_has_ref(tmp_path: Path) -> None:
    remote, clone, _base_sha, _head_sha = _bare_with_mr(tmp_path)
    _git("-C", str(remote), "update-ref", "-d", "refs/merge-requests/188/head")
    ref = parse_review_ref("MR 188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [str(clone)])
    assert exc.value.failure_class == "review_ref_unavailable"


def test_bare_number_unavailable_without_repositories() -> None:
    ref = parse_review_ref("MR 188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [])
    assert exc.value.failure_class == "review_ref_unavailable"


def test_bare_number_ambiguous_across_repositories(tmp_path: Path) -> None:
    _remote_a, clone_a, _b1, _h1 = _bare_with_mr(tmp_path / "a")
    _remote_b, clone_b, _b2, _h2 = _bare_with_mr(tmp_path / "b")
    ref = parse_review_ref("MR 188")
    assert ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [str(clone_a), str(clone_b)])
    assert exc.value.failure_class == "review_ref_ambiguous"


def test_provider_word_selects_provider_ref_template(tmp_path: Path) -> None:
    """`MR 188` probes only the GitLab template; `PR 188` only GitHub's."""
    remote, clone, _base_sha, _head_sha = _bare_with_mr(tmp_path, provider=PROVIDER_GITHUB)
    gitlab_ref = parse_review_ref("MR 188")
    assert gitlab_ref is not None
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(gitlab_ref, [str(clone)])
    assert exc.value.failure_class == "review_ref_unavailable"
    github_ref = parse_review_ref("PR 188")
    assert github_ref is not None
    assert resolve_review_ref(github_ref, [str(clone)]).repo_path == str(clone)
    assert "merge-requests" not in _git("-C", str(remote), "for-each-ref", "--format=%(refname)")


def test_duplicate_repo_paths_are_probed_once(tmp_path: Path) -> None:
    _remote, clone, _base_sha, _head_sha = _bare_with_mr(tmp_path)
    ref = parse_review_ref("MR 188")
    assert ref is not None
    target = resolve_review_ref(ref, [str(clone), str(clone)])
    assert target.repo_path == str(clone)


def test_invalid_provider_reference_fails_defensively() -> None:
    ref = ReviewRef(provider="gitea", number=5)
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [])
    assert exc.value.failure_class == "review_ref_invalid"


def test_out_of_range_reference_fails_defensively() -> None:
    ref = ReviewRef(provider=PROVIDER_GITLAB, number=MAX_REVIEW_NUMBER + 1)
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(ref, [])
    assert exc.value.failure_class == "review_ref_invalid"


# ==============================================================================
# 7. Remote helpers
# ==============================================================================

def test_provider_ref_templates_are_fixed() -> None:
    assert provider_ref_template("gitlab", 188, "head") == "refs/merge-requests/188/head"
    assert provider_ref_template("gitlab", 188, "merge") == "refs/merge-requests/188/merge"
    assert provider_ref_template("github", 188, "head") == "refs/pull/188/head"
    assert provider_ref_template("github", 188, "merge") == "refs/pull/188/merge"


def test_normalize_remote_url_forms() -> None:
    assert normalize_remote_url(
        "https://user:secret@gitlab.example.com/group/project.git/"
    ) == ("gitlab.example.com", "group/project")
    assert normalize_remote_url("git@github.com:owner/repo.git") == ("github.com", "owner/repo")
    assert normalize_remote_url("ssh://git@host:2222/srv/repo.git") == ("host", "srv/repo")
    assert normalize_remote_url("/srv/local/repo.git") is None


def test_read_origin_url_returns_value_without_logging(tmp_path: Path) -> None:
    repo = _repo_with_origin(tmp_path / "repo", "https://example.com/a/b.git")
    assert read_origin_url(str(repo)) == "https://example.com/a/b.git"
    assert read_origin_url(str(tmp_path / "missing")) is None


# ==============================================================================
# 8. Git argv discipline for the resolver layer
# ==============================================================================

_RESOLVER_ALLOWED_VERBS = {"config", "ls-remote", "rev-parse"}


def test_resolver_git_argv_allowlist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every resolver git argv is read-only, template-fixed, and user-text-free.

    Mirrors the recorder pattern from test_review_snapshots.py: subprocess
    run/Popen are monkeypatched with recording wrappers, a rejected
    metacharacter question must produce no git argv at all, and a bare-number
    resolution may probe only the fixed provider ref template interpolated
    with the validated integer.
    """
    _remote, clone, _base_sha, _head_sha = _bare_with_mr(tmp_path)
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

    # A rejected metacharacter question must fail closed before any git call.
    with pytest.raises(ReviewRefError) as exc:
        resolve_review_ref(
            parse_review_ref("MR 188;shutdown now `id` $(rm -rf) /etc/passwd"),
            [str(clone)],
        )
    assert exc.value.failure_class == "review_ref_invalid"
    assert calls == []

    ref = parse_review_ref("MR 188")
    assert ref is not None
    target = resolve_review_ref(ref, [str(clone)])
    assert (target.provider, target.number) == (PROVIDER_GITLAB, 188)
    assert target.repo_path == str(clone)

    assert calls, "expected recorded git activity"
    expected_head = provider_ref_template(PROVIDER_GITLAB, 188, "head")
    verbs: list[str] = []
    for argv in calls:
        assert argv[0] == "git"
        rest = argv[1:]
        while rest and rest[0] in ("-C", "-c"):
            rest = rest[2:]
        verbs.append(rest[0])
        assert rest[0] in _RESOLVER_ALLOWED_VERBS
        if rest[0] == "config":
            assert rest[1:] == ["--get", "remote.origin.url"]
        if rest[0] == "ls-remote":
            # Strictly the fixed template plus the validated integer; the
            # probed ref name is the only free-looking argv element and it is
            # template-owned.
            assert rest[1] == "origin"
            assert rest[2] == expected_head
    assert "ls-remote" in verbs
    joined = "\n".join(" ".join(argv) for argv in calls)
    for fragment in ("shutdown", "rm -rf", "/etc/passwd", "`", "$(", ";", "|", ">"):
        assert fragment not in joined
