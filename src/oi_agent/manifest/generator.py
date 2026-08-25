"""Repo manifest: AST-derived codebase index for the auditing agent.

Adapted from an internal orchestrator project's repo-manifest generator:
output lives in `.oi/index.md` inside the watched clone, and only
stdlib is used. Generates a markdown map of every source file with its module
docstring and top-level symbols; refreshed incrementally via git diff so the
header SHA always proves freshness.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "index.md"
MANIFEST_DIRNAME = ".oi"

SOURCE_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".jsx", ".vue", ".svelte"}

SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", "dist", "build", ".next", ".nuxt",
    ".oi", "playwright-report", "test-results",
}

GROUPING_THRESHOLD = 10


@dataclass
class SymbolInfo:
    """One top-level symbol extracted from a source file.

    Attributes:
        name: Symbol name.
        kind: 'class', 'function', 'route' or 'export'.
        line: Definition line number.
        decorators: Decorator names applied.
        docstring: First docstring line, if any.
    """

    name: str
    kind: str
    line: int
    decorators: list[str] = field(default_factory=list)
    docstring: str = ""


@dataclass
class SymbolGroup:
    """A labeled cluster of related symbols inside one fat file.

    Attributes:
        label: Group label such as 'Routes'.
        symbols: Member symbol names.
    """

    label: str
    symbols: list[str]


@dataclass
class FileEntry:
    """Manifest entry for one source file.

    Attributes:
        rel_path: Path relative to repo root.
        loc: Lines of code.
        description: One-line purpose summary.
        symbols: Extracted top-level symbols.
        groups: Symbol groups for fat files.
    """

    rel_path: str
    loc: int
    description: str
    symbols: list[SymbolInfo] = field(default_factory=list)
    groups: list[SymbolGroup] = field(default_factory=list)


@dataclass
class ManifestResult:
    """Outcome of manifest generation or refresh.

    Attributes:
        manifest_path: Absolute path of the written manifest.
        file_count: Number of indexed source files.
        git_sha: Short HEAD SHA at generation time ('unknown' if unavailable).
        duration_ms: Wall time in milliseconds.
    """

    manifest_path: Path
    file_count: int
    git_sha: str
    duration_ms: int


def _git(repo_root: Path, *args: str) -> str | None:
    """Run a read-only git command and return stdout.

    Args:
        repo_root: Repository to run in.
        *args: Git arguments.

    Returns:
        Stripped stdout, or None when the command fails.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            capture_output=True, text=True, check=False, timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        # git missing or timed out: callers treat None as "unavailable".
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _head_sha(repo_root: Path) -> str:
    """Return the short HEAD SHA, tagged ``-dirty`` when the worktree differs.

    The SHA is OI's audit primitive: a reply stamped ``audited at X`` must have
    inspected code identifiable by X. Uncommitted changes are not described by
    HEAD, so a dirty worktree yields ``<sha>-dirty`` to keep the label honest.

    Args:
        repo_root: Repository root path.

    Returns:
        Short SHA, ``<sha>-dirty`` when uncommitted changes exist, or
        ``unknown`` when HEAD cannot be read.
    """
    sha = _git(repo_root, "rev-parse", "--short", "HEAD") or "unknown"
    if _git(repo_root, "status", "--porcelain"):
        return f"{sha}-dirty"
    return sha


def worktree_fingerprint(repo_root: Path) -> str:
    """Return a content fingerprint of the current worktree state.

    Unlike the manifest SHA (which only changes on a rebuild) or the dirty
    *flag* (which cannot distinguish two different uncommitted edits), this
    hashes HEAD plus the full unstaged/staged diff and status. Comparing it
    before and after evidence gathering detects ANY live mutation during the
    audit, even edits made while the tree was already dirty.

    Args:
        repo_root: Repository root path.

    Returns:
        Short hex digest of the live worktree state (``unknown`` if git fails).
    """
    head = _git(repo_root, "rev-parse", "HEAD") or "unknown"
    status = _git(repo_root, "status", "--porcelain") or ""
    diff = _git(repo_root, "diff", "HEAD") or ""
    if head == "unknown":
        return "unknown"
    payload = f"{head}\n{status}\n{diff}".encode("utf-8", errors="replace")
    return hashlib.sha1(payload).hexdigest()[:12]


def _git_ignored_set(repo_root: Path, rel_paths: list[str]) -> set[str] | None:
    """Ask git which of the given paths are ignored (authoritative).

    Uses ``git check-ignore --stdin``, which honors full gitignore semantics
    (globs, negation, nested/precedence rules) that a hand-rolled matcher
    cannot. This is the security-critical boundary: files git ignores (secrets,
    local config) must never be indexed into the model prompt.

    Args:
        repo_root: Repository root path.
        rel_paths: Candidate relative paths to test.

    Returns:
        Set of ignored relative paths, or None when git is unavailable or the
        directory is not a git repo (caller falls back to a conservative walk).
    """
    if not rel_paths:
        return set()
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "check-ignore", "--stdin"],
            input="\n".join(rel_paths), capture_output=True, text=True,
            check=False, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    # 0 = some ignored, 1 = none ignored; 128 = not a git repo / git error.
    if result.returncode not in (0, 1):
        return None
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def _is_ignored_fallback(rel_path: str) -> bool:
    """Conservative skip test used only when git check-ignore is unavailable.

    Cannot replicate gitignore glob semantics, so it errs toward skipping known
    non-source and hidden/metadata locations rather than pretending to honor
    arbitrary patterns.

    Args:
        rel_path: File path relative to repo root.

    Returns:
        True when the path is under a skip dir or a hidden/.git/.env location.
    """
    parts = Path(rel_path).parts
    if any(p in SKIP_DIRS for p in parts):
        return True
    return any(p.startswith(".") for p in parts)


def _decorator_name(node: ast.expr) -> str:
    """Render an AST decorator as a readable dotted name.

    Args:
        node: Decorator expression node.

    Returns:
        Dotted name string, '?' for unsupported forms.
    """
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parts = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if isinstance(current, ast.Name):
            parts.append(current.id)
        return ".".join(reversed(parts))
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    return "?"


ROUTE_DECORATORS = {
    "app.route", "router.get", "router.post",
    "router.put", "router.delete", "router.patch",
}


def _extract_python_symbols(source: str) -> tuple[str, list[SymbolInfo]]:
    """Extract module docstring and top-level symbols from Python source.

    Args:
        source: Full file content.

    Returns:
        Tuple of (first docstring line, symbol list).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", []

    docstring = ast.get_docstring(tree) or ""
    docstring = docstring.split("\n")[0].strip() if docstring else ""

    symbols: list[SymbolInfo] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            sym_doc = (ast.get_docstring(node) or "").split("\n")[0].strip()
            symbols.append(SymbolInfo(
                name=node.name, kind="class", line=node.lineno,
                decorators=[_decorator_name(d) for d in node.decorator_list],
                docstring=sym_doc,
            ))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            sym_doc = (ast.get_docstring(node) or "").split("\n")[0].strip()
            decorators = [_decorator_name(d) for d in node.decorator_list]
            kind = "route" if any(d in ROUTE_DECORATORS for d in decorators) \
                else "function"
            symbols.append(SymbolInfo(
                name=node.name, kind=kind, line=node.lineno,
                decorators=decorators, docstring=sym_doc,
            ))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    symbols.append(SymbolInfo(
                        name=target.id, kind="variable", line=node.lineno))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if not node.target.id.startswith("_"):
                symbols.append(SymbolInfo(
                    name=node.target.id, kind="variable", line=node.lineno))
    return docstring, symbols


_TS_PATTERNS = [
    (r"export\s+default\s+(?:function|class)\s+(\w+)", "export"),
    (r"export\s+(?:function|const|let|var)\s+(\w+)", "export"),
    (r"export\s+class\s+(\w+)", "class"),
    (r"export\s+interface\s+(\w+)", "export"),
    (r"export\s+type\s+(\w+)", "export"),
    (r"export\s+enum\s+(\w+)", "export"),
]


def _extract_ts_symbols(source: str) -> tuple[str, list[SymbolInfo]]:
    """Extract first comment line and exports from TS/JS/Vue/Svelte source.

    Args:
        source: Full file content.

    Returns:
        Tuple of (first comment line, export symbol list).
    """
    docstring = ""
    first_comment = re.match(
        r"^\s*(?://|/\*\*?)\s*(.+?)(?:\*/)?$", source, re.MULTILINE)
    if first_comment:
        docstring = first_comment.group(1).strip()

    symbols: list[SymbolInfo] = []
    seen: set[str] = set()
    for pattern, kind in _TS_PATTERNS:
        for match in re.finditer(pattern, source):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                symbols.append(SymbolInfo(
                    name=name, kind=kind,
                    line=source[: match.start()].count("\n") + 1))
    return docstring, symbols


def _group_symbols(symbols: list[SymbolInfo]) -> list[SymbolGroup]:
    """Group related symbols by prefix/decorator/kind heuristics.

    Only applied to fat files (> GROUPING_THRESHOLD symbols).

    Args:
        symbols: Flat symbol list.

    Returns:
        Labeled groups; empty for small files.
    """
    if len(symbols) <= GROUPING_THRESHOLD:
        return []

    groups: dict[str, list[str]] = {}
    grouped: set[str] = set()

    prefix_counts: dict[str, list[SymbolInfo]] = {}
    for sym in symbols:
        if "_" in sym.name:
            prefix = sym.name.split("_")[0]
            if len(prefix) >= 3:
                prefix_counts.setdefault(prefix, []).append(sym)
    for prefix, syms in prefix_counts.items():
        if len(syms) >= 2:
            groups[prefix.capitalize()] = [s.name for s in syms]
            grouped.update(s.name for s in syms)

    decorator_groups: dict[str, list[str]] = {}
    for sym in symbols:
        if sym.name in grouped:
            continue
        for dec in sym.decorators:
            if "route" in dec.lower():
                decorator_groups.setdefault("Routes", []).append(sym.name)
                break
    for label, names in decorator_groups.items():
        if len(names) >= 2:
            groups[label] = names
            grouped.update(names)

    buckets: dict[str, list[str]] = {}
    for sym in symbols:
        if sym.name not in grouped:
            buckets.setdefault(sym.kind, []).append(sym.name)
    for kind, names in buckets.items():
        label = ("Other" if len(names) < 2 else kind.capitalize() + "s")
        groups.setdefault(label, [])
        groups[label] += names

    return [SymbolGroup(label=k, symbols=v) for k, v in groups.items()]


def _infer_description(rel_path: str, symbols: list[SymbolInfo], loc: int) -> str:
    """Infer a one-line description when no module docstring exists.

    Uses path conventions (tests/, models/, middleware/) and symbol shapes.

    Args:
        rel_path: Relative file path.
        symbols: Extracted symbols.
        loc: Line count.

    Returns:
        Best-effort description string.
    """
    filename = Path(rel_path).stem
    parts = Path(rel_path).parts

    if filename.startswith("test_") or filename.endswith("_test"):
        count = sum(1 for s in symbols if s.name.startswith("test_"))
        subject = filename.replace("test_", "").replace("_test", "")
        return f"{count} tests for {subject}" if count else f"Tests for {subject}"

    if filename == "__init__":
        exports = [s.name for s in symbols[:5]]
        return f"Package init. Exports: {', '.join(exports)}" if exports \
            else "Package init"

    route_syms = [s for s in symbols if s.kind == "route"]
    if route_syms:
        names = ", ".join(s.name for s in route_syms[:5])
        suffix = f" (+{len(route_syms)-5} more)" if len(route_syms) > 5 else ""
        return f"Routes: {names}{suffix}"

    classes = [s for s in symbols if s.kind == "class"]
    if ("models" in parts or "model" in parts) and classes:
        names = ", ".join(c.name for c in classes[:5])
        suffix = f" (+{len(classes)-5} more)" if len(classes) > 5 else ""
        return f"Models: {names}{suffix}"

    exports = [s.name for s in symbols if not s.name.startswith("_")][:6]
    if exports:
        suffix = f" (+{len(symbols)-6} more)" if len(symbols) > 6 else ""
        return f"Exports: {', '.join(exports)}{suffix}"
    return f"{loc} LOC"


def _extract_file_entry(repo_root: Path, rel_path: str) -> FileEntry | None:
    """Build the manifest entry for one source file.

    Args:
        repo_root: Repo root.
        rel_path: Relative file path.

    Returns:
        FileEntry, or None when unreadable.
    """
    full_path = repo_root / rel_path
    # Containment: a tracked symlink can point outside the watched repo. Resolve
    # and require the target to stay under repo_root so the manifest never
    # indexes external file content (path-jail guarantee).
    try:
        resolved = full_path.resolve()
        if not resolved.is_relative_to(repo_root.resolve()):
            return None
    except OSError:
        return None
    try:
        source = full_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    loc = len(source.splitlines())
    suffix = full_path.suffix
    if suffix == ".py":
        docstring, symbols = _extract_python_symbols(source)
    elif suffix in SOURCE_EXTENSIONS:
        docstring, symbols = _extract_ts_symbols(source)
    else:
        return None

    description = docstring or _infer_description(rel_path, symbols, loc)
    return FileEntry(rel_path=rel_path, loc=loc, description=description,
                     symbols=symbols, groups=_group_symbols(symbols))


def collect_source_files(repo_root: Path) -> list[str]:
    """Walk the repo and list indexable source files.

    Respects SKIP_DIRS and the repo's own .gitignore.

    Args:
        repo_root: Repo root path.

    Returns:
        Sorted list of relative paths.
    """
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        # Prune excluded/hidden directories in place so os.walk never descends
        # into node_modules/.venv/.git etc. (rglob would stat their whole trees).
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not d.startswith(".")]
        for name in filenames:
            if Path(name).suffix not in SOURCE_EXTENSIONS or name.startswith("."):
                continue
            rel = str(Path(dirpath, name).relative_to(repo_root))
            candidates.append(rel)
    candidates.sort()
    ignored = _git_ignored_set(repo_root, candidates)
    if ignored is None:
        # git unavailable: fall back to the conservative filter.
        return [rel for rel in candidates if not _is_ignored_fallback(rel)]
    return [rel for rel in candidates if rel not in ignored]


def render_manifest(entries: list[FileEntry], git_sha: str,
                    repo_root: Path) -> str:
    """Render entries into grouped markdown with a provenance header.

    Args:
        entries: All file entries.
        git_sha: HEAD SHA stamped in the header.
        repo_root: Repo root for the title.

    Returns:
        Complete manifest markdown.
    """
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [
        f"# Repo Manifest — {repo_root.name}",
        f"<!-- generated: {now} | git: {git_sha} -->",
        f"<!-- files: {len(entries)} -->",
        "",
    ]
    dir_groups: dict[str, list[FileEntry]] = {}
    for e in entries:
        parts = Path(e.rel_path).parts
        top = parts[0] if len(parts) > 1 else "."
        dir_groups.setdefault(top, []).append(e)

    for d in sorted(dir_groups):
        lines += [f"## {d}/", ""]
        for e in dir_groups[d]:
            lines.append(f"- **{e.rel_path}** — {e.description}")
            if e.groups:
                for g in e.groups:
                    syms = ", ".join(g.symbols[:8])
                    suffix = f" (+{len(g.symbols)-8})" if len(g.symbols) > 8 else ""
                    lines.append(f"  - {g.label}: {syms}{suffix}")
            else:
                # Small file: show its key symbols inline so the map names them.
                key = [s.name for s in e.symbols
                       if s.kind in ("class", "route")][:5]
                if not key:
                    key = [s.name for s in e.symbols[:5]]
                if key:
                    lines.append(f"  - defines: {', '.join(key)}")
        lines.append("")
    return "\n".join(lines)


def _ensure_gitignored(repo_root: Path) -> None:
    """Make sure the .oi/ artifact dir is ignored, without dirtying the repo.

    Writes to ``.git/info/exclude`` (git's local, untracked exclude file)
    instead of the tracked ``.gitignore``. This keeps OI's own index out of git
    diffs while leaving the watched worktree byte-for-byte unchanged, so the
    audit SHA never reports ``-dirty`` because of OI itself.

    Falls back to a bare (non-git) directory silently: with no ``.git`` there is
    nothing to exclude.

    Args:
        repo_root: Repo root path.
    """
    info_dir = repo_root / ".git" / "info"
    if not (repo_root / ".git").is_dir():
        return
    info_dir.mkdir(parents=True, exist_ok=True)
    exclude = info_dir / "exclude"
    entry = ".oi/"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    if entry in {ln.strip() for ln in existing.splitlines()}:
        return
    block = f"\n# oi-agent index\n{entry}\n"
    with exclude.open("a", encoding="utf-8") as fh:
        fh.write(block if existing and not existing.endswith("\n")
                 else block.lstrip("\n"))


def generate_repo_manifest(repo_root: Path) -> ManifestResult:
    """Rebuild the whole manifest from scratch into <repo>/.oi/index.md.

    Args:
        repo_root: Absolute repo root path.

    Returns:
        ManifestResult with path, counts, SHA and duration.
    """
    t0 = time.monotonic()
    repo_root = Path(repo_root)
    sha = _head_sha(repo_root)
    entries = []
    for rel in collect_source_files(repo_root):
        entry = _extract_file_entry(repo_root, rel)
        if entry:
            entries.append(entry)
    manifest_dir = repo_root / MANIFEST_DIRNAME
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / MANIFEST_FILENAME
    manifest_path.write_text(render_manifest(entries, sha, repo_root),
                             encoding="utf-8")
    _ensure_gitignored(repo_root)
    duration = int((time.monotonic() - t0) * 1000)
    logger.info("[manifest] generated %d files in %dms (%s)",
                len(entries), duration, sha)
    return ManifestResult(manifest_path, len(entries), sha, duration)


def _parse_manifest_sha(manifest_path: Path) -> str | None:
    """Extract the recorded SHA from a manifest header.

    Args:
        manifest_path: Existing manifest file.

    Returns:
        SHA string or None.
    """
    if not manifest_path.exists():
        return None
    try:
        header = manifest_path.read_text(encoding="utf-8")[:500]
    except OSError:
        return None
    match = re.search(r"git:\s*(\S+)", header)
    return match.group(1) if match else None


def refresh_repo_manifest(repo_root: Path) -> ManifestResult:
    """Incrementally refresh the manifest after a pull.

    No-op when the SHA is unchanged; falls back to a full rebuild whenever
    the previous SHA cannot be diffed against.

    Args:
        repo_root: Absolute repo root path.

    Returns:
        ManifestResult describing the resulting manifest state.
    """
    t0 = time.monotonic()
    repo_root = Path(repo_root)
    manifest_path = repo_root / MANIFEST_DIRNAME / MANIFEST_FILENAME
    old_sha = _parse_manifest_sha(manifest_path)
    current = _head_sha(repo_root)
    # A dirty worktree means grep/read will inspect uncommitted code that HEAD
    # does not describe. The committed-diff fast paths below compare manifest
    # SHA against HEAD and would silently keep stale entries (an empty
    # `diff old..HEAD` looks like "only non-source changes"). Any dirty tree
    # therefore forces a full rebuild so entries match the inspected bytes.
    if current.endswith("-dirty"):
        return generate_repo_manifest(repo_root)
    if old_sha == current and manifest_path.exists():
        header = manifest_path.read_text(encoding="utf-8")[:500]
        m = re.search(r"files:\s*(\d+)", header)
        return ManifestResult(manifest_path, int(m.group(1)) if m else 0,
                              current, int((time.monotonic()-t0)*1000))
    if old_sha:
        changed = _git(repo_root, "diff", "--name-only", f"{old_sha}..HEAD")
        if changed is not None:
            src_changed = [f for f in changed.splitlines()
                           if Path(f).suffix in SOURCE_EXTENSIONS]
            if not src_changed:
                # Only non-source changes: bump SHA/timestamp, keep entries.
                content = manifest_path.read_text(encoding="utf-8")
                now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                content = re.sub(r"git:\s*\S+", f"git: {current}", content)
                content = re.sub(r"generated:\s*[\dT:Z-]+",
                                 f"generated: {now}", content)
                manifest_path.write_text(content, encoding="utf-8")
                m = re.search(r"files:\s*(\d+)", content[:500])
                return ManifestResult(manifest_path,
                                      int(m.group(1)) if m else 0, current,
                                      int((time.monotonic()-t0)*1000))
            logger.info("[manifest] %d source files changed since %s",
                        len(src_changed), old_sha)
    return generate_repo_manifest(repo_root)


def load_manifest_summary(repo_root: Path) -> str:
    """Build a compact directory-level summary for system-prompt injection.

    Parses the stored manifest (no re-walk) and stays small enough to live
    permanently in the agent prompt.

    Args:
        repo_root: Repo root containing .oi/index.md.

    Returns:
        Summary markdown, or '' when no manifest exists yet.
    """
    manifest_path = repo_root / MANIFEST_DIRNAME / MANIFEST_FILENAME
    if not manifest_path.exists():
        return ""
    try:
        content = manifest_path.read_text(encoding="utf-8")
    except OSError:
        return ""

    dir_files: dict[str, list[tuple[str, str]]] = {}
    for line in content.splitlines():
        if line.startswith("- **") and "** — " in line:
            rel = line.split("**")[1]
            desc = line.split("** — ", 1)[1]
            top = Path(rel).parts[0] if len(Path(rel).parts) > 1 else "."
            dir_files.setdefault(top, []).append((rel, desc))

    lines = ["# Code Map", "",
             "Directory overview. For file detail grep `.oi/index.md`.", ""]
    for d in sorted(dir_files):
        files = dir_files[d]
        subdirs: dict[str, int] = {}
        for rel, _ in files:
            parts = Path(rel).parts
            sub = "/".join(parts[:-1]) if len(parts) > 2 else "(root)"
            subdirs[sub] = subdirs.get(sub, 0) + 1
        top_subs = sorted(subdirs.items(), key=lambda x: -x[1])[:8]
        sub_text = ", ".join(f"{k}({v})" for k, v in top_subs)
        extra = f", +{len(subdirs)-8} more" if len(subdirs) > 8 else ""
        sample = "; ".join(d_ for _, d_ in files[:3])
        lines.append(f"- **{d}/** — {len(files)} files | {sub_text}{extra}")
        lines.append(f"  - e.g. {sample[:200]}")
    return "\n".join(lines)


def manifest_git_sha(repo_root: Path) -> str:
    """Return the SHA the current manifest was built at.

    Args:
        repo_root: Repo root path.

    Returns:
        SHA string or 'none'.
    """
    sha = _parse_manifest_sha(repo_root / MANIFEST_DIRNAME / MANIFEST_FILENAME)
    return sha or "none"
