"""Safe parsing and authorization of explicit MR/PR review references.

Discord text is untrusted and can never supply a Git command, remote URL, or
arbitrary ref. Only explicit review-reference forms in the latest question are
recognized:

- GitLab: ``MR 188``, ``!188``, canonical ``/merge_requests/188`` URLs.
- GitHub: ``PR 188``, ``pull request 188``, canonical ``/pull/188`` URLs.

The keyword words match case-insensitively. Thread context is never consulted;
callers pass exactly the latest question text. Bare numbers are resolved by
probing only configured watched repositories with bounded, argv-only
``git ls-remote origin <fixed template ref>`` commands. Explicit URLs are
authorized by matching their normalized host and project path against the
``origin`` remote of exactly one configured repository.

All Git access is argv-only with fixed command templates, explicit timeouts,
and bounded captured output. Remote URLs may contain credentials and are never
logged; only bounded metadata (provider, number, repository name, failure
class) is logged. These public functions are synchronous; the integration
layer wraps them in a worker thread when needed.

Failure classes (:class:`ReviewRefError`; ``str(exc)`` is the class):

- ``review_ref_invalid``: malformed numbers, zero/overflow, embedded ref
  syntax, shell metacharacters, or malformed canonical URLs.
- ``review_ref_unsupported``: URL shapes belonging to unsupported forges, or
  canonical URLs whose host contradicts the path-implied provider (e.g.
  ``github.com`` with a ``/merge_requests/`` path, or ``gitlab.com`` with a
  ``/pull/`` path). Self-hosted hosts keep path-based provider inference.
- ``review_ref_multiple``: several distinct review references in one question.
- ``review_ref_url_mismatch``: explicit URL matches no (or more than one)
  configured repository origin.
- ``review_ref_unavailable``: bare number found in no configured repository.
- ``review_ref_ambiguous``: bare number found in more than one configured
  repository; the user must include the repository or MR URL.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

PROVIDER_GITLAB = "gitlab"
PROVIDER_GITHUB = "github"
MAX_REVIEW_NUMBER = 10_000_000

_MAX_DIGITS = 8  # anything longer is overflow by construction
_GIT_TIMEOUT_SECONDS = 30
_MAX_TOKEN_LEN = 64
_TRAILING_PUNCT = set('.,;:!?)]}>"\'')
_METACHARS = set("#!$&;|<>`\\(){}[]\"'/@=")
_IGNORED_URL_SEGMENTS = {
    "issues", "issue", "commit", "commits", "blob", "tree", "blame", "raw",
    "snippets", "releases", "tags", "wikis",
}
_UNSUPPORTED_URL_SEGMENTS = {"pulls", "pull-requests"}

_KEYWORD_RE = re.compile(
    r"\b(?P<word>MR|PR|pull\s+request):?[ \t]+(?P<token>\S{1,%d})" % _MAX_TOKEN_LEN,
    re.IGNORECASE,
)
_BANG_RE = re.compile(r"(?<![\w!])!(?P<token>\S{1,%d})" % _MAX_TOKEN_LEN)
_URL_RE = re.compile(r"https?://[^\s<>\"`]+", re.IGNORECASE)
_SCP_REMOTE_RE = re.compile(r"^[^/@]+@([^:/]+):(.+)$")


class ReviewRefError(ValueError):
    """Bounded fail-closed review-reference failure; ``str(exc)`` is the class."""

    def __init__(self, failure_class: str) -> None:
        super().__init__(failure_class)
        self.failure_class = failure_class


@dataclass(frozen=True)
class ReviewRef:
    """One validated review reference parsed from the latest question.

    Attributes:
        provider: ``gitlab`` or ``github``.
        number: Positive bounded MR/PR number.
        url: The explicit canonical URL when the reference was a URL form.
    """

    provider: str
    number: int
    url: str | None = None


@dataclass(frozen=True)
class ReviewTarget:
    """One review reference resolved to a configured watched repository."""

    provider: str
    number: int
    repo_path: str


@dataclass(frozen=True)
class _GitOutcome:
    returncode: int | None
    stdout: str
    timed_out: bool = False


def _run_git(argv: list[str], timeout: int) -> _GitOutcome:
    """Run one bounded argv-only Git command without logging its output."""
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except subprocess.TimeoutExpired:
        return _GitOutcome(returncode=None, stdout="", timed_out=True)
    except (OSError, subprocess.SubprocessError):
        return _GitOutcome(returncode=None, stdout="")
    return _GitOutcome(returncode=result.returncode, stdout=result.stdout)


def read_origin_url(repo_path: str) -> str | None:
    """Return a repository's origin URL; never log it (may contain credentials).

    Args:
        repo_path: Local clone path.

    Returns:
        The ``remote.origin.url`` value, or ``None`` when unreadable.
    """
    outcome = _run_git(
        ["git", "-C", str(repo_path), "config", "--get", "remote.origin.url"],
        _GIT_TIMEOUT_SECONDS,
    )
    if outcome.timed_out or outcome.returncode != 0:
        return None
    value = outcome.stdout.strip()
    return value or None


def normalize_remote_url(url: str) -> tuple[str, str] | None:
    """Normalize a remote URL to a comparable (host, project path) pair.

    Accepts https/ssh/git URL forms and scp-style ``git@host:path`` remotes.
    The result never contains credentials.

    Args:
        url: Raw remote URL.

    Returns:
        Lowercased ``(host, project_path)`` without a ``.git`` suffix, or
        ``None`` for local or unparsable remotes that can never match an
        explicit review URL.
    """
    value = url.strip()
    if value.lower().startswith(("git://", "ssh://", "http://", "https://")):
        parsed = urlparse(value)
        host = (parsed.hostname or "").strip().lower()
        path = parsed.path
    else:
        scp = _SCP_REMOTE_RE.match(value)
        if scp is None:
            return None
        host = scp.group(1).strip().lower()
        path = scp.group(2)
    project = _normalize_project_path(path)
    if not host or not project:
        return None
    return (host, project)


def provider_ref_template(provider: str, number: int, kind: str) -> str:
    """Return the fixed provider-owned ref template for a validated number.

    Only the validated integer is interpolated; no user text can enter the
    ref name.

    Args:
        provider: ``gitlab`` or ``github``.
        number: Validated positive MR/PR number.
        kind: ``head`` or ``merge``.

    Returns:
        Fixed ref name such as ``refs/merge-requests/188/head``.

    Raises:
        ReviewRefError: ``review_ref_invalid`` on unknown provider or kind.
    """
    if provider == PROVIDER_GITLAB:
        base = f"refs/merge-requests/{number}"
    elif provider == PROVIDER_GITHUB:
        base = f"refs/pull/{number}"
    else:
        raise ReviewRefError("review_ref_invalid")
    if kind == "head":
        return f"{base}/head"
    if kind == "merge":
        return f"{base}/merge"
    raise ReviewRefError("review_ref_invalid")


def parse_review_ref(question: str) -> ReviewRef | None:
    """Parse the single explicit review reference in the latest question.

    Args:
        question: Latest untrusted Discord question text only.

    Returns:
        Parsed :class:`ReviewRef`, or ``None`` when the text contains no
        explicit review reference (ordinary numbers, ``#188``, issue links,
        and commit links never match).

    Raises:
        ReviewRefError: On malformed, unsupported, or ambiguous references.
    """
    if not question or not question.strip():
        return None
    url_refs: list[tuple[str, int, str, str, str]] = []

    def _scan_url(match: re.Match[str]) -> str:
        raw = _trim_url_token(match.group(0))
        parsed = _parse_review_url(raw)
        if parsed is not None:
            url_refs.append((*parsed, raw))
        return " "

    text = _URL_RE.sub(_scan_url, question)
    keyword_refs: list[tuple[str, int]] = []
    for match in _KEYWORD_RE.finditer(text):
        number = _token_number(match.group("token"))
        if number is not None:
            keyword_refs.append((_keyword_provider(match.group("word")), number))
    for match in _BANG_RE.finditer(text):
        number = _token_number(match.group("token"))
        if number is not None:
            keyword_refs.append((PROVIDER_GITLAB, number))

    unique_urls: dict[tuple[str, int, str, str], str] = {}
    for provider, number, host, project, raw in url_refs:
        unique_urls.setdefault((provider, number, host, project), raw)
    if len(unique_urls) > 1:
        raise ReviewRefError("review_ref_multiple")
    if unique_urls:
        (provider, number, _host, _project), raw = next(iter(unique_urls.items()))
        for keyword_provider, keyword_number in keyword_refs:
            if (keyword_provider, keyword_number) != (provider, number):
                raise ReviewRefError("review_ref_multiple")
        logger.info("[review] parsed provider=%s number=%d", provider, number)
        return ReviewRef(provider=provider, number=number, url=raw)

    unique_keywords = set(keyword_refs)
    if len(unique_keywords) > 1:
        raise ReviewRefError("review_ref_multiple")
    if unique_keywords:
        provider, number = next(iter(unique_keywords))
        logger.info("[review] parsed provider=%s number=%d", provider, number)
        return ReviewRef(provider=provider, number=number)
    return None


def resolve_review_target(question: str, repo_paths: Sequence[str]) -> ReviewTarget | None:
    """Parse and resolve the review reference in the latest question, if any.

    Args:
        question: Latest untrusted question text.
        repo_paths: Configured watched repository paths.

    Returns:
        Resolved :class:`ReviewTarget`, or ``None`` without any Git activity
        when the question contains no explicit review reference.

    Raises:
        ReviewRefError: Bounded resolution failure (see module docstring).
    """
    ref = parse_review_ref(question)
    if ref is None:
        return None
    return resolve_review_ref(ref, repo_paths)


def resolve_review_ref(ref: ReviewRef, repo_paths: Sequence[str]) -> ReviewTarget:
    """Resolve a parsed reference against configured watched repositories.

    Args:
        ref: Parsed review reference.
        repo_paths: Configured watched repository paths.

    Returns:
        The unique matching configured repository.

    Raises:
        ReviewRefError: ``review_ref_url_mismatch``, ``review_ref_unavailable``,
            or ``review_ref_ambiguous``.
    """
    _validate_ref(ref)
    ordered = _unique_repo_paths(repo_paths)
    if ref.url is not None:
        return _resolve_by_url(ref, ordered)
    return _resolve_by_number(ref, ordered)


def _keyword_provider(word: str) -> str:
    return PROVIDER_GITLAB if word.upper().startswith("MR") else PROVIDER_GITHUB


def _classify_token(token: str) -> tuple[str, int | None]:
    """Classify one keyword token as (kind, number).

    Kinds: ``valid`` (bounded positive integer), ``invalid`` (numeric but
    zero or out of range), ``suspicious`` (ref syntax or shell metacharacters,
    rejected outright), and ``plain`` (an ordinary word that is simply not a
    review reference and is skipped).
    """
    stripped = token[:-1] if len(token) > 1 and token[-1] in _TRAILING_PUNCT else token
    if not stripped or not stripped.isascii():
        return ("plain", None)
    if stripped.isdigit():
        if len(stripped) > _MAX_DIGITS:
            return ("invalid", None)
        number = int(stripped)
        if number < 1 or number > MAX_REVIEW_NUMBER:
            return ("invalid", None)
        return ("valid", number)
    if any(char in _METACHARS for char in stripped):
        return ("suspicious", None)
    return ("plain", None)


def _token_number(token: str) -> int | None:
    """Return the validated number for a keyword token, or None for plain words."""
    kind, number = _classify_token(token)
    if kind in ("invalid", "suspicious"):
        raise ReviewRefError("review_ref_invalid")
    return number


def _trim_url_token(url: str) -> str:
    """Strip sentence punctuation that cannot be part of a canonical URL."""
    trimmed = url.rstrip(".,;:!?")
    if trimmed.endswith(")") and "(" not in trimmed:
        trimmed = trimmed[:-1].rstrip(".,;:!?")
    return trimmed


def _normalize_project_path(project: str) -> str:
    value = project.strip().strip("/").lower()
    if value.endswith(".git"):
        value = value[:-4]
    return value.strip("/")


def _host_is(host: str, domain: str) -> bool:
    """Return True when host equals ``domain`` or is a subdomain of it."""
    return host == domain or host.endswith("." + domain)


def _provider_host_mismatch(host: str, provider: str) -> bool:
    """Detect a canonical URL whose host contradicts the path-implied provider.

    Only the two public forges are pinned to their hosts; self-hosted hosts
    keep path-based provider inference.
    """
    if _host_is(host, "github.com"):
        return provider != PROVIDER_GITHUB
    if _host_is(host, "gitlab.com"):
        return provider != PROVIDER_GITLAB
    return False


def _parse_review_url(url: str) -> tuple[str, int, str, str] | None:
    """Classify one URL token for review-reference shapes.

    Returns:
        ``(provider, number, host, project_path)`` for canonical GitLab/GitHub
        merge-request URLs, or ``None`` for URLs that are not review
        references at all (issue links, commit links, unrelated URLs).

    Raises:
        ReviewRefError: ``review_ref_invalid`` for malformed canonical shapes
            or URLs carrying query/fragment syntax; ``review_ref_unsupported``
            for other forges' pull-request URL shapes and for canonical shapes
            whose host contradicts the path-implied provider.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return None
    if parsed.query or parsed.fragment:
        raise ReviewRefError("review_ref_invalid")
    host = (parsed.hostname or "").strip().lower()
    raw_segments = [segment for segment in parsed.path.split("/") if segment]
    if not host or not raw_segments:
        return None
    segments = [segment.lower() for segment in raw_segments]
    if any(segment in _IGNORED_URL_SEGMENTS for segment in segments):
        return None
    for key, provider in (("merge_requests", PROVIDER_GITLAB), ("pull", PROVIDER_GITHUB)):
        if key not in segments:
            continue
        index = len(segments) - 1 - segments[::-1].index(key)
        number = _url_number(segments, index)
        project = raw_segments[:index]
        if project and project[-1] == "-":
            project = project[:-1]
        if not project:
            raise ReviewRefError("review_ref_invalid")
        if _provider_host_mismatch(host, provider):
            raise ReviewRefError("review_ref_unsupported")
        return (provider, number, host, _normalize_project_path("/".join(project)))
    if any(segment in _UNSUPPORTED_URL_SEGMENTS for segment in segments):
        raise ReviewRefError("review_ref_unsupported")
    return None


def _url_number(segments: list[str], key_index: int) -> int:
    """Validate the canonical numeric segment directly after the key segment."""
    tail = segments[key_index + 1:]
    if len(tail) != 1 or not tail[0].isascii() or not tail[0].isdigit():
        raise ReviewRefError("review_ref_invalid")
    if len(tail[0]) > _MAX_DIGITS:
        raise ReviewRefError("review_ref_invalid")
    number = int(tail[0])
    if number < 1 or number > MAX_REVIEW_NUMBER:
        raise ReviewRefError("review_ref_invalid")
    return number


def _validate_ref(ref: ReviewRef) -> None:
    """Defensively re-validate a reference before any Git activity."""
    if ref.provider not in (PROVIDER_GITLAB, PROVIDER_GITHUB):
        raise ReviewRefError("review_ref_invalid")
    if isinstance(ref.number, bool) or not isinstance(ref.number, int):
        raise ReviewRefError("review_ref_invalid")
    if ref.number < 1 or ref.number > MAX_REVIEW_NUMBER:
        raise ReviewRefError("review_ref_invalid")


def _unique_repo_paths(repo_paths: Sequence[str]) -> list[str]:
    """Return configured repository paths once each, in configured order."""
    ordered: list[str] = []
    seen: set[str] = set()
    for path in repo_paths:
        if not path or not str(path).strip():
            continue
        key = str(Path(str(path)).expanduser())
        if key not in seen:
            seen.add(key)
            ordered.append(str(path))
    return ordered


def _resolve_by_url(ref: ReviewRef, repo_paths: list[str]) -> ReviewTarget:
    """Authorize an explicit URL against configured repository origins."""
    parsed = _parse_review_url(ref.url or "")
    if parsed is None:
        raise ReviewRefError("review_ref_invalid")
    _, number, host, project = parsed
    matches: list[str] = []
    for repo_path in repo_paths:
        origin = read_origin_url(repo_path)
        if origin is None:
            continue
        if normalize_remote_url(origin) == (host, project):
            matches.append(repo_path)
    if len(matches) == 1:
        _log_resolved(ref.provider, number, matches[0])
        return ReviewTarget(provider=ref.provider, number=number, repo_path=matches[0])
    logger.warning(
        "[review] resolve failed class=review_ref_url_mismatch provider=%s number=%d repos=%d",
        ref.provider, number, len(repo_paths),
    )
    raise ReviewRefError("review_ref_url_mismatch")


def _resolve_by_number(ref: ReviewRef, repo_paths: list[str]) -> ReviewTarget:
    """Probe configured repositories for one fixed provider head ref."""
    head_ref = provider_ref_template(ref.provider, ref.number, "head")
    matches: list[str] = []
    for repo_path in repo_paths:
        if _ls_remote_has_ref(repo_path, head_ref):
            matches.append(repo_path)
    if len(matches) == 1:
        _log_resolved(ref.provider, ref.number, matches[0])
        return ReviewTarget(provider=ref.provider, number=ref.number, repo_path=matches[0])
    failure = "review_ref_unavailable" if not matches else "review_ref_ambiguous"
    logger.warning(
        "[review] resolve failed class=%s provider=%s number=%d repos=%d",
        failure, ref.provider, ref.number, len(repo_paths),
    )
    raise ReviewRefError(failure)


def _log_resolved(provider: str, number: int, repo_path: str) -> None:
    logger.info(
        "[review] resolved provider=%s number=%d repo=%s",
        provider, number, Path(repo_path).name,
    )


def _ls_remote_has_ref(repo_path: str, ref_name: str) -> bool:
    """Probe one configured repository's origin for one fixed provider ref."""
    outcome = _run_git(
        ["git", "-C", str(repo_path), "ls-remote", "origin", ref_name],
        _GIT_TIMEOUT_SECONDS,
    )
    if outcome.timed_out or outcome.returncode != 0:
        return False
    for line in outcome.stdout.splitlines():
        parts = line.split("\t", 1)
        if len(parts) == 2 and parts[1].strip() == ref_name:
            return True
    return False
