"""Jailed read-only repo tools exposed to the auditing pipeline.

Every function enforces path containment inside the repo root and runs no
shell. These are the ONLY ways the responder may touch the watched clone.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..manifest.generator import load_manifest_summary, manifest_git_sha


class ToolError(Exception):
    """Raised when a tool call violates containment or limits."""


# Filename suffixes/components that must never be read into a model prompt even
# when they live inside the repo root: git internals, env/secret files, keys.
_DENIED_SUFFIXES = (".env", ".pem", ".key", ".p12", ".pfx", ".keystore")


def _is_sensitive(rel: Path) -> bool:
    """Check whether a contained path points at secret/metadata content.

    Args:
        rel: Repo-relative path.

    Returns:
        True for hidden files/dirs (``.git``, dotfiles), ``.env*`` files, or
        known key/credential extensions.
    """
    parts = rel.parts
    if any(part.startswith(".") for part in parts):
        return True
    name = rel.name.lower()
    return name.startswith(".env") or name.endswith(_DENIED_SUFFIXES)


def _resolve(repo_root: Path, rel: str) -> Path:
    """Resolve a repo-relative path and verify it stays inside the root.

    Args:
        repo_root: Repo root path.
        rel: User/model supplied relative path.

    Returns:
        Absolute contained path.

    Raises:
        ToolError: On absolute paths, traversal, symlink escape, or a path that
            targets sensitive metadata/secret content.
    """
    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        raise ToolError(f"path escapes repo: {rel}")
    if _is_sensitive(p):
        raise ToolError(f"sensitive path denied: {rel}")
    full = (repo_root / p).resolve()
    root = repo_root.resolve()
    if not full.is_relative_to(root):
        raise ToolError(f"path escapes repo: {rel}")
    return full


def read_file(repo_root: str | Path, rel_path: str,
              max_bytes: int = 200_000, truncate: bool = False) -> str:
    """Read a contained file with an optional size-bounded truncation.

    Args:
        repo_root: Repo root.
        rel_path: Repo-relative path.
        max_bytes: Maximum returned bytes.
        truncate: Return a bounded prefix instead of rejecting large files.

    Returns:
        File content as text.

    Raises:
        ToolError: On escape or missing/oversized files when not truncating.
    """
    full = _resolve(Path(repo_root), rel_path)
    if not full.is_file():
        raise ToolError(f"not a file: {rel_path}")
    size = full.stat().st_size
    if size > max_bytes and not truncate:
        raise ToolError(f"file too large ({size}B): {rel_path}")
    if size > max_bytes:
        content = full.read_bytes()[:max_bytes].decode(
            "utf-8", errors="replace"
        )
        return content + "\n[... file evidence truncated ...]"
    return full.read_text(encoding="utf-8", errors="replace")


def grep(repo_root: str | Path, pattern: str,
         max_results: int = 30) -> list[str]:
    """Case-insensitive regex search for one pattern (see grep_multi).

    Args:
        repo_root: Repo root.
        pattern: Python-flavored regex.
        max_results: Result cap ('path:line: text' strings).

    Returns:
        Matching lines, capped.

    Raises:
        ToolError: On invalid regex.
    """
    return grep_multi(repo_root, [pattern], max_results)


def grep_multi(repo_root: str | Path, patterns: list[str],
               max_results: int = 30) -> list[str]:
    """Search several patterns in ONE process (git/ripgrep/python fallback).

    Collapsing per-term greps into a single multi-pattern scan avoids launching
    a dozen full-repo git processes per audit. All backends are
    case-insensitive so lowercased terms still match CamelCase code.

    Args:
        repo_root: Repo root.
        patterns: Python-flavored regexes; matched as an OR set.
        max_results: Result cap ('path:line: text' strings).

    Returns:
        Matching lines, capped.

    Raises:
        ToolError: On invalid regex.
    """
    root = Path(repo_root).resolve()
    patterns = [p for p in patterns if p]
    if not patterns:
        return []
    source_globs = [
        "*.py", "*.ts", "*.tsx", "*.js", "*.jsx",
        "*.vue", "*.svelte", "*.md",
    ]
    git_patterns = sum((["-e", p] for p in patterns), [])
    combined = "|".join(f"(?:{p})" for p in patterns)
    try:
        git_proc = subprocess.run(
            [
                "git", "-C", str(root), "grep", "-n", "-I", "-i", "-E",
                *git_patterns, "--", *source_globs,
                ":(exclude).agents/**", ":(exclude).claude/**",
                ":(exclude).oi/**", ":(exclude)node_modules/**",
                ":(exclude).venv/**",
            ],
            capture_output=True, text=True, check=False, timeout=20,
        )
        if git_proc.returncode in (0, 1):
            return [ln for ln in git_proc.stdout.splitlines()
                    if ln][:max_results]
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        proc = subprocess.run(
            [
                "rg", "--max-count", "5", "-n", "-i",
                *sum((["-g", glob] for glob in source_globs), []),
                "-g", "!.git/**", "-g", "!node_modules/**",
                "-g", "!.venv/**", "-g", "!__pycache__/**",
                "-g", "!.oi/**", combined,
            ],
            cwd=str(root), capture_output=True, text=True,
            check=False, timeout=20,
        )
    except FileNotFoundError:
        return _python_grep(root, combined, max_results)
    except subprocess.TimeoutExpired as exc:
        raise ToolError("grep timed out") from exc
    lines = [ln for ln in proc.stdout.splitlines() if ln][:max_results]
    return lines


def _python_grep(root: Path, pattern: str, max_results: int) -> list[str]:
    """Fallback grep without ripgrep (walks text files directly).

    Args:
        root: Repo root.
        pattern: Regex string.
        max_results: Cap on returned lines.

    Returns:
        Capped matching lines.

    Raises:
        ToolError: When the pattern is not valid regex.
    """
    import re
    try:
        rx = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"bad regex: {exc}") from exc
    out: list[str] = []
    skip_dirs = {".git", "node_modules", ".venv", ".oi", "__pycache__"}
    for f in sorted(root.rglob("*")):
        if not f.is_file() or f.suffix.lower() not in {
            ".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte", ".md",
        }:
            continue
        if any(part in skip_dirs for part in f.parts):
            continue
        try:
            for i, line in enumerate(
                f.read_text(encoding="utf-8", errors="replace").splitlines(), 1
            ):
                if rx.search(line):
                    out.append(f"{f.relative_to(root)}:{i}: {line.strip()}")
                    if len(out) >= max_results:
                        return out
        except OSError:
            continue
    return out


def manifest_summary(repo_root: str | Path) -> str:
    """Return the compact code map for prompt injection.

    Args:
        repo_root: Repo root.

    Returns:
        Summary markdown ('' when not yet generated).
    """
    return load_manifest_summary(Path(repo_root))


def audit_sha(repo_root: str | Path) -> str:
    """Return the git SHA the manifest was last built at.

    Args:
        repo_root: Repo root.

    Returns:
        Short SHA string stamped into every reply.
    """
    return manifest_git_sha(Path(repo_root))


_GREP_REPO_DESCRIPTION = (
    "Search the audited repository with case-insensitive regex patterns. "
    "Returns matching 'path:line: text' lines."
)
_READ_FILE_DESCRIPTION = (
    "Read one repo-relative file's text content. Large files are truncated "
    "to the byte budget instead of being refused."
)

# Model-facing schema surface. Kept as module constants so both wire styles
# describe exactly the same two tools and stay in sync by construction.
TOOL_SCHEMAS_OPENAI: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "grep_repo",
            "description": _GREP_REPO_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "patterns": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 12,
                        "description": "1 to 12 regex patterns",
                    },
                    "max_results": {
                        "type": "integer",
                        "default": 30,
                        "maximum": 100,
                        "description": "Max matched lines returned (1..100)",
                    },
                },
                "required": ["patterns"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": _READ_FILE_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Repo-relative file path",
                    },
                    "max_bytes": {
                        "type": "integer",
                        "default": 20000,
                        "maximum": 50000,
                        "description": "Byte budget before truncation (1..50000)",
                    },
                },
                "required": ["path"],
            },
        },
    },
]

TOOL_SCHEMAS_ANTHROPIC: list[dict] = [
    {
        "name": schema["function"]["name"],
        "description": schema["function"]["description"],
        "input_schema": schema["function"]["parameters"],
    }
    for schema in TOOL_SCHEMAS_OPENAI
]


_MAX_OUTPUT_CHARS = 40_000


def _clamp_int(value: object, default: int, low: int, high: int) -> int:
    """Coerce an untrusted integer-ish argument into a safe bounded int.

    Numeric bounds are CLAMPED, not rejected: a model asking for
    max_results=99999 gets 100 rather than a failed call. Non-int types
    fall back to the documented default.

    Args:
        value: Raw argument value from model JSON.
        default: Used when value is not an int (bools excluded).
        low: Inclusive lower bound.
        high: Inclusive upper bound.

    Returns:
        int: Bounded integer argument.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(low, min(high, value))


def execute_tool(repo_root, name: str, arguments: dict) -> str:
    """Run one model-requested tool call against a jailed repo view.

    Never raises: containment violations (ToolError), malformed arguments,
    unknown tools, and any unexpected exception are all converted into a
    returned "ERROR: ..." string so the calling loop can feed it back to the
    model for self-correction instead of killing the audit.

    Argument policy: one non-empty ``patterns`` string is normalized to a
    one-item list for providers that violate the advertised array schema.
    Other structural mistakes (missing/wrong-typed/empty fields, more than 12
    patterns) are rejected; purely numeric out-of-range values (max_results,
    max_bytes) are clamped into their documented ranges.

    Args:
        repo_root: Repo root the tools are jailed to.
        name: Tool name, "grep_repo" or "read_file".
        arguments: JSON-style dict of tool arguments from the model.

    Returns:
        str: Tool output (joined grep lines or file text), truncated to
        40k chars, or an "ERROR: ..." description of the failure.
    """
    out = _execute_tool_checked(repo_root, name, arguments)
    if len(out) > _MAX_OUTPUT_CHARS:
        cut = _MAX_OUTPUT_CHARS - len("\n[... tool output truncated ...]")
        out = out[:cut] + "\n[... tool output truncated ...]"
    return out


def _execute_tool_checked(repo_root, name: str, arguments: dict) -> str:
    """Dispatch one validated tool call; convert all failures to error text.

    Args:
        repo_root: Repo root the tools are jailed to.
        name: Tool name, "grep_repo" or "read_file".
        arguments: JSON-style dict of tool arguments from the model.

    Returns:
        str: Raw untruncated tool output, or an "ERROR: ..." description
        of the failure.
    """
    try:
        if not isinstance(arguments, dict):
            return f"ERROR: arguments must be an object, got {type(arguments).__name__}"
        if name == "grep_repo":
            patterns = arguments.get("patterns")
            if isinstance(patterns, str):
                # Some OpenAI-compatible models emit the one requested regex
                # directly despite the advertised string-array schema.
                patterns = [patterns]
            if not isinstance(patterns, list):
                return "ERROR: grep_repo requires 'patterns': a list of regex strings"
            if not patterns:
                return "ERROR: grep_repo needs at least 1 pattern"
            if any(not isinstance(p, str) or not p.strip() for p in patterns):
                return "ERROR: every pattern must be a non-empty string"
            if len(patterns) > 12:
                return "ERROR: at most 12 patterns per call"
            max_results = _clamp_int(
                arguments.get("max_results"), default=30, low=1, high=100
            )
            return "\n".join(grep_multi(repo_root, patterns, max_results))
        if name == "read_file":
            path = arguments.get("path")
            if not isinstance(path, str) or not path.strip():
                return "ERROR: read_file requires 'path': a repo-relative string"
            max_bytes = _clamp_int(
                arguments.get("max_bytes"), default=20000, low=1, high=50000
            )
            return read_file(repo_root, path, max_bytes=max_bytes, truncate=True)
        return f"ERROR: unknown tool: {name}"
    except ToolError as exc:
        return f"ERROR: {exc}"
    except Exception as exc:  # noqa: BLE001 - executor must never raise upward
        return f"ERROR: {type(exc).__name__}: {exc}"
