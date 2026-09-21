"""Chat-driven THREAD CONTEXT window expansion.

Pure logic only: intent gate, strict spec parsing, and clamping against
config caps. OpenCode invocation lives in the runner; Discord fetching in
``discord_client._thread_excerpt``. Any parse/validation failure fails open
to the default 40-message excerpt window.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# Default excerpt window; mirrors discord_client.EXCERPT_LIMIT (kept here so
# clamping never depends on the Discord layer).
DEFAULT_EXCERPT_LIMIT = 40

# Hard sanity limits for the model-authored spec before config clamping.
MAX_SPEC_COUNT = 10_000
MAX_SPEC_SINCE_DAYS = 3_650

# Gate: only explicit-looking expansion asks may trigger the one-shot spec
# query. Conservative by design so ordinary mentions never pay for it.
_EXPANSION_HINTS = (
    re.compile(r"\b\d+\s*(?:messages?|msgs?|replies?)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:last|past|previous|prior)\s+(?:\d+\s+)?(?:day|week|month|hour)s?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bsince\s+\S+", re.IGNORECASE),
    re.compile(r"\bgo\s+(?:back|further)\b", re.IGNORECASE),
    re.compile(
        r"\bread\s+(?:the\s+)?(?:whole|entire|full|complete|earlier)\b",
        re.IGNORECASE,
    ),
    # "messages frm 10th Sept 2026" / "since Sept 10" — date-form asks,
    # including common typos (frm), with the date before or after the month.
    re.compile(
        r"\b(?:from|since|frm|after)\s+(?:the\s+)?(?:\d{1,2}(?:st|nd|rd|th)?[\s./-]+)?"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}(?:st|nd|rd|th)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b"),
    # "3 days ago" / "2 weeks ago"
    re.compile(
        r"\b\d+\s*(?:d|w|m|days?|weeks?|months?|hours?)\s+ago\b",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class ContextRequest:
    """Model-interpreted history window ask from one user message."""

    count: int | None = None
    since_days: int | None = None
    # ISO YYYY-MM-DD; a named date means "since the START of that day" and
    # takes priority over since_days — whole-day semantics, no hour-loss.
    since_date: str | None = None

    @property
    def is_default(self) -> bool:
        """Return True when no dimension widens the window."""
        return self.count is None and self.since_days is None and self.since_date is None


def mentions_expansion(question: str) -> bool:
    """Return True when the question looks like a history-expansion request."""
    return any(hint.search(question) for hint in _EXPANSION_HINTS)
_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}
_MONTH_PATTERN = "(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + ")"
_NAMED_DATE_RE = re.compile(
    rf"\b(?:from|since|after|frm)\s+(?:the\s+)?"
    rf"(?:(?P<day>\d{{1,2}})(?:st|nd|rd|th)?[\s./-]+"
    rf"(?P<month>{_MONTH_PATTERN})(?:[\s,/-]+(?P<year>\d{{4}}))?"
    rf"|(?P<month_first>{_MONTH_PATTERN})[\s./-]+"
    rf"(?P<day_first>\d{{1,2}})(?:st|nd|rd|th)?"
    rf"(?:[\s,/-]+(?P<year_first>\d{{4}}))?)\b",
    re.IGNORECASE,
)
_RELATIVE_WINDOW_RE = re.compile(
    r"\b(?P<number>\d+)\s*(?P<unit>hours?|days?|weeks?|months?)\s+ago\b",
    re.IGNORECASE,
)
_COUNT_WINDOW_RE = re.compile(
    r"\b(?:last|past|previous|prior|read)\s+(?P<count>\d+)\s+"
    r"(?:messages?|msgs?|replies?)\b",
    re.IGNORECASE,
)


def parse_explicit_request(
    question: str, *, now: datetime | None = None
) -> ContextRequest | None:
    """Parse common history requests locally without an LLM round-trip.

    Args:
        question: Current Discord message text.
        now: Injectable UTC clock for deterministic relative-date tests.

    Returns:
        ContextRequest for a recognized explicit request, otherwise None so
        the model-backed fallback can interpret less literal language.
    """
    count_match = _COUNT_WINDOW_RE.search(question)
    count = int(count_match.group("count")) if count_match else None
    date_match = _NAMED_DATE_RE.search(question)
    since_date: str | None = None
    if date_match:
        day = int(date_match.group("day") or date_match.group("day_first"))
        month_name = (date_match.group("month") or date_match.group("month_first")).lower()
        year = int(
            date_match.group("year")
            or date_match.group("year_first")
            or (now or datetime.now(timezone.utc)).year
        )
        try:
            since_date = datetime(year, _MONTHS[month_name], day).strftime("%Y-%m-%d")
        except ValueError:
            since_date = None
    relative_match = _RELATIVE_WINDOW_RE.search(question)
    since_days: int | None = None
    if relative_match and since_date is None:
        number = int(relative_match.group("number"))
        unit = relative_match.group("unit").lower()
        since_days = number * (
            7 if unit.startswith("week") else
            30 if unit.startswith("month") else
            1
        )
        if unit.startswith("hour"):
            since_days = max(1, (number + 23) // 24)
    if count is None and since_date is None and since_days is None:
        if re.search(
            r"\b(?:read earlier|go back further|whole thread|entire thread)\b",
            question,
            re.IGNORECASE,
        ):
            count = 200
    if count is None and since_date is None and since_days is None:
        return None
    return ContextRequest(count=count, since_days=since_days, since_date=since_date)


def parse_spec(text: str | None) -> ContextRequest | None:
    """Parse the model's JSON spec; return None on any deviation.

    Args:
        text: Raw model output, expected to contain exactly one JSON object
            ``{"count": int|null, "since_days": int|null}``.

    Returns:
        ContextRequest or None when the output is absent, non-JSON, or out
        of range (fail-open to the default window).
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        raw = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    count = _bounded_int(raw.get("count"), MAX_SPEC_COUNT)
    since_days = _bounded_int(raw.get("since_days"), MAX_SPEC_SINCE_DAYS)
    since_date = _parse_iso_date(raw.get("since_date"))
    return ContextRequest(count=count, since_days=since_days, since_date=since_date)


def _parse_iso_date(value: object) -> str | None:
    """Return the value as a validated YYYY-MM-DD string or None."""
    if not isinstance(value, str):
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        return None
    return value


def _bounded_int(value: object, ceiling: int) -> int | None:
    """Coerce a spec value to a bounded positive int or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = int(value)
    if number < 1 or number > ceiling:
        return None
    return number


def bounded_window(
    request: ContextRequest | None,
    *,
    max_context_messages: int,
    max_context_age_days: int,
    now: datetime | None = None,
) -> tuple[int, datetime | None]:
    """Clamp a spec against config caps.

    Args:
        request: Parsed spec; None means the default window.
        max_context_messages: Hard cap on fetched messages.
        max_context_age_days: Hard cap on look-back days.
        now: Injectable clock for deterministic tests.

    Returns:
        (limit, after): fetch limit and optional server-side cutoff for
        ``_thread_excerpt``.
    """
    limit = DEFAULT_EXCERPT_LIMIT
    after: datetime | None = None
    if request is not None:
        if request.count is not None:
            limit = min(request.count, max(1, max_context_messages))
        elif request.since_date is not None or request.since_days is not None:
            # A date/relative window without an explicit count means "the
            # requested window", so use the configured cap instead of the
            # ordinary 40-message excerpt.
            limit = max(1, max_context_messages)
        if request.since_date is not None:
            # A named date means the START of that day (whole-day semantics):
            # since_days arithmetic would cut into the day and drop the
            # conversation the user asked for. Bounded by the fetch limit
            # regardless, so no age clamp here.
            try:
                after = datetime.strptime(request.since_date, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                after = None
        if after is None and request.since_days is not None:
            current = now or datetime.now(timezone.utc)
            after = current - timedelta(days=min(request.since_days, max(1, max_context_age_days)))
    return limit, after


class ContextRequestMemory:
    """Conversation-scoped, in-process recall of the last expansion ask.

    Anaphora ("do it this time", "and the plan for that?") refers to a
    window request made in a PREVIOUS turn, so a freshly remembered request
    re-applies to follow-up questions that carry no expansion keywords of
    their own. In-memory only: numeric window params, never prompt text;
    expires by TTL and on daemon restart, so nothing persists in SQLite.
    """

    def __init__(self, ttl_seconds: int = 900) -> None:
        """Create the memory store.

        Args:
            ttl_seconds: How long a remembered request stays reusable.
        """
        self._ttl_seconds = ttl_seconds
        self._items: dict[str, tuple[ContextRequest, float]] = {}

    def remember(self, key: str, request: ContextRequest, now: float | None = None) -> None:
        """Store a non-default request for the conversation key.

        Args:
            key: Conversation-scoped key (conversation_id + repo path).
            request: Parsed request; default requests are ignored.
            now: Injectable clock for deterministic tests.
        """
        if request is None or request.is_default:
            return
        stamp = now if now is not None else time.time()
        self._items[key] = (request, stamp)

    def recall(self, key: str, now: float | None = None) -> ContextRequest | None:
        """Return the fresh remembered request for the key, or None.

        Args:
            key: Conversation-scoped key.
            now: Injectable clock for deterministic tests.

        Returns:
            ContextRequest or None when absent or expired.
        """
        entry = self._items.get(key)
        if entry is None:
            return None
        request, stamp = entry
        current = now if now is not None else time.time()
        if current - stamp > self._ttl_seconds:
            del self._items[key]
            return None
        return request
