"""OpenCode permission policy for the native read-only OI agent."""

from __future__ import annotations

from pathlib import Path

_READ_TOOLS = ("read", "glob", "list")
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


def build_permission_policy(
    repo_root: Path,
    additional_repo_roots: list[Path] | None = None,
) -> dict[str, object]:
    """Return an explicit fail-closed OpenCode permission map.

    OpenCode matches read/glob/list inputs relative to its ``--dir``
    workspace. When additional_repo_roots are provided, they are explicitly
    granted read-only access under external_directory, while all edit/bash/mcp
    tools and secret patterns remain denied across all roots.

    Args:
        repo_root: Canonical primary repository root used as OpenCode cwd.
        additional_repo_roots: Optional auxiliary repository roots.

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
    policy.update({tool: "deny" for tool in _DENIED_TOOLS})

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
) -> dict[str, object]:
    """Build the native primary ``agent.oi`` OpenCode configuration.

    Args:
        model: Model provider identifier.
        steps: Step limit.
        repo_root: Canonical primary repository root.
        prompt: Embedded system prompt.
        additional_repo_roots: Optional auxiliary repository roots.

    Returns:
        JSON-compatible agent configuration.
    """
    return {
        "agent": {
            "oi": {
                "description": "Read-only Discord repository auditor",
                "mode": "primary",
                "model": model,
                "prompt": prompt,
                "steps": steps,
                "permission": build_permission_policy(
                    repo_root, additional_repo_roots=additional_repo_roots
                ),
            }
        }
    }
