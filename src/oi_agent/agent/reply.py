"""Reply value object shared by the runner and Discord poster."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reply:
    """One bounded response and its audit/session metadata.

    Args:
        text: Visible Discord reply body.
        sha: Repository SHA used for the audit, or ``none``.
        ok: Whether the response is a genuine successful audit.
        session_id: OpenCode session ID returned by a successful run.
        error_class: Bounded machine-readable failure category.
        attachments: Absolute paths of .md/.txt docs written during the run
            for Discord file upload (docs-only mode, bounded, best-effort).
    """

    text: str
    sha: str
    ok: bool = True
    session_id: str | None = None
    error_class: str | None = None
    attachments: tuple[str, ...] = ()
