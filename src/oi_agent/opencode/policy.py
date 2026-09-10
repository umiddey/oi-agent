"""OpenCode permission policy for the native OI agent.

Default is read-only; ``write_mode="docs-only"`` optionally grants
``edit``/``write`` for ``*.md``/``*.txt`` files only (secrets stay denied,
code/bash stay denied). ``write_dirs`` optionally narrows doc writes to
repo-relative directories; empty means any .md/.txt in the repo.
"""

from __future__ import annotations

from pathlib import Path

_READ_TOOLS = ("read", "glob", "list")
_WRITE_MODES = ("read-only", "docs-only")
_DOC_EXTENSIONS = (".md", ".txt")
_DOC_TOOLS = ("edit", "write")
_DENIED_TOOLS = (
    "edit", "write", "patch", "bash", "task", "skill", "webfetch",
    "websearch", "question", "grep", "mcp", "todolist",
)
_SECRET_PATTERNS = (
    ".git/**", ".env", ".env.*", "**/.env", "**/.env.*", "**/*.env",
    "**/*.pem", "**/*.key", "**/*.keytab", "**/*.p12", "**/*.pfx",
    "**/*.crt", "**/*.cer", "**/id_rsa", "**/id_ed25519",
    "**/*credentials*", "**/*secret*", "**/*token*",
)


def _workspace_rules(
    *,
    deny_secrets: bool = True,
    additional_roots: list[Path] | None = None,
) -> dict[str, str]:
    """Build rules for OpenCode's workspace-relative tool inputs.

    Args:
        deny_secrets: Whether to enforce secret pattern blocking.
        additional_roots: Optional auxiliary repository roots to protect.

    Returns:
        dict[str, str]: Permission rules mapping patterns to allow/deny.
    """
    rules: dict[str, str] = {"*": "allow"}
    if deny_secrets:
        for pattern in _SECRET_PATTERNS:
            rules[pattern] = "deny"
        if additional_roots:
            for extra in additional_roots:
                p_str = str(extra.expanduser().resolve())
                for pattern in _SECRET_PATTERNS:
                    rules[f"{p_str}/{pattern}"] = "deny"
                    rules[f"{p_str}/**/{pattern}"] = "deny"
    return rules


def normalize_write_dirs(write_dirs: list[str] | tuple[str, ...] | None) -> list[str]:
    """Normalize optional repo-relative doc-write scopes.

    Args:
        write_dirs: Raw directory entries from config.

    Returns:
        Cleaned relative paths (no trailing slash, no ``.``/``..``/absolute).
        Invalid entries are dropped so the policy fails closed to a narrower
        scope; config validation rejects them earlier with a clear error.
    """
    cleaned: list[str] = []
    for entry in write_dirs or ():
        if not isinstance(entry, str):
            continue
        if entry.strip().startswith("/"):
            continue
        norm = entry.strip().strip("/")
        if not norm or norm == ".":
            continue
        if ".." in norm.split("/"):
            continue
        if any(not part for part in norm.split("/")):
            continue
        if norm not in cleaned:
            cleaned.append(norm)
    return cleaned


def _doc_write_rules(write_dirs: list[str]) -> dict[str, str]:
    """Build fail-closed edit/write rules for .md/.txt docs only.

    Args:
        write_dirs: Normalized repo-relative scopes; empty means any
            ``*.md``/``*.txt`` in the workspace.

    Returns:
        Permission rules mapping patterns to allow/deny.
    """
    rules: dict[str, str] = {"*": "deny"}
    if write_dirs:
        for scope in write_dirs:
            for ext in _DOC_EXTENSIONS:
                rules[f"{scope}/**/*{ext}"] = "allow"
                rules[f"{scope}/*{ext}"] = "allow"
    else:
        for ext in _DOC_EXTENSIONS:
            rules[f"**/*{ext}"] = "allow"
            rules[f"*{ext}"] = "allow"
    for pattern in _SECRET_PATTERNS:
        rules[pattern] = "deny"
    return rules


def build_permission_policy(
    repo_root: Path,
    additional_repo_roots: list[Path] | None = None,
    *,
    write_mode: str = "read-only",
    write_dirs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Return an explicit fail-closed OpenCode permission map.

    OpenCode matches read/glob/list inputs relative to its ``--dir``
    workspace. When additional_repo_roots are provided, they are explicitly
    granted read-only access under external_directory, while all edit/bash/mcp
    tools and secret patterns remain denied across all roots.

    Args:
        repo_root: Canonical primary repository root used as OpenCode cwd.
        additional_repo_roots: Optional auxiliary repository roots.
        write_mode: ``"read-only"`` (default) or ``"docs-only"`` (allow
            ``edit``/``write`` for ``*.md``/``*.txt`` only).
        write_dirs: Optional repo-relative scopes for doc writes; empty
            means any .md/.txt. Unknown modes fail closed to read-only.

    Returns:
        JSON-compatible permission configuration.
    """
    _ = repo_root.expanduser().resolve()
    policy: dict[str, object] = {"*": "deny"}
    policy.update({
        tool: _workspace_rules(deny_secrets=True, additional_roots=additional_repo_roots)
        for tool in _READ_TOOLS
    })
    policy["lsp"] = "allow"
    denied = [tool for tool in _DENIED_TOOLS if tool not in _DOC_TOOLS]
    policy.update({tool: "deny" for tool in denied})
    if write_mode == "docs-only":
        policy.update({
            tool: _doc_write_rules(normalize_write_dirs(write_dirs))
            for tool in _DOC_TOOLS
        })
    else:
        policy.update({tool: "deny" for tool in _DOC_TOOLS})

    if additional_repo_roots:
        ext_rules: dict[str, str] = {"*": "deny"}
        for extra in additional_repo_roots:
            p_str = str(extra.expanduser().resolve())
            ext_rules[p_str] = "allow"
            ext_rules[f"{p_str}/**"] = "allow"
        policy["external_directory"] = ext_rules
    else:
        policy["external_directory"] = "deny"

    return policy


def build_agent_config(
    model: str,
    steps: int,
    repo_root: Path,
    prompt: str,
    additional_repo_roots: list[Path] | None = None,
    *,
    write_mode: str = "read-only",
    write_dirs: list[str] | tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Build the native primary ``agent.oi`` OpenCode configuration.

    Args:
        model: Model provider identifier.
        steps: Step limit.
        repo_root: Canonical primary repository root.
        prompt: Embedded system prompt.
        additional_repo_roots: Optional auxiliary repository roots.
        write_mode: ``"read-only"`` or ``"docs-only"`` (unknown fails closed).
        write_dirs: Optional repo-relative scopes for doc writes.

    Returns:
        JSON-compatible agent configuration.
    """
    docs = write_mode == "docs-only"
    return {
        "agent": {
            "oi": {
                "description": (
                    "Discord repository auditor with docs-only write"
                    if docs else "Read-only Discord repository auditor"
                ),
                "mode": "primary",
                "model": model,
                "prompt": prompt,
                "steps": steps,
                "permission": build_permission_policy(
                    repo_root,
                    additional_repo_roots=additional_repo_roots,
                    write_mode=write_mode,
                    write_dirs=write_dirs,
                ),
            }
        }
    }
