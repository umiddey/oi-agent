"""SQLite-backed state store: pause flag, hourly post counters, rolling memory.

One database file serves all watch targets; rows are keyed by channel id so
each channel keeps independent rate limits and memory.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    ts INTEGER NOT NULL,
    channel_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS memory (
    channel_id INTEGER PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    exchanges TEXT NOT NULL DEFAULT '[]',
    updated_at INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_posts_channel_ts ON posts(channel_id, ts);
-- Migration: the raw message archive is gone. Runtime never read it back
-- (Discord itself is the canonical thread history), so keeping it only meant
-- unbounded growth. Existing databases lose the table on first open.
DROP TABLE IF EXISTS messages;
"""


class Store:
    """Small sqlite wrapper for daemon state (all methods synchronous)."""

    def __init__(self, db_path: Path) -> None:
        """Open (and initialize) the state database.

        Args:
            db_path: Path to the sqlite file; parent dirs are created.
        """
        db_path.parent.mkdir(parents=True, exist_ok=True)
        # timeout: a losing writer waits for the lock instead of raising
        # "database is locked". isolation_level=None lets us drive explicit
        # BEGIN IMMEDIATE transactions for the atomic cap reservation.
        self._conn = sqlite3.connect(db_path, timeout=10,
                                     isolation_level=None)
        self._conn.executescript(_SCHEMA)



    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # -- kill switch -------------------------------------------------

    def is_paused(self) -> bool:
        """Return True when posting is globally paused via `oi pause`."""
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key='paused'"
        ).fetchone()
        return bool(row and row[0] == "1")

    def set_paused(self, paused: bool) -> None:
        """Set or clear the global pause flag.

        Args:
            paused: True to stop all posting, False to resume.
        """
        val = "1" if paused else "0"
        self._conn.execute(
            "INSERT INTO kv(key,value) VALUES('paused',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (val,),
        )
        self._conn.commit()

    # -- rate limiting -------------------------------------------------

    def posts_in_last_hour(self, channel_id: int) -> int:
        """Count messages posted by the daemon for this channel in the last hour.

        Args:
            channel_id: Watch target channel id.

        Returns:
            Number of posts within the trailing 3600 seconds.
        """
        cutoff = int(time.time()) - 3600
        row = self._conn.execute(
            "SELECT COUNT(*) FROM posts WHERE channel_id=? AND ts>?",
            (channel_id, cutoff),
        ).fetchone()
        return int(row[0])

    def reserve_post(self, channel_id: int, cap: int) -> bool:
        """Atomically reserve one unit of a channel's hourly posting cap.

        Check-and-insert happens inside one synchronous transaction, so two
        concurrent reply attempts cannot both squeeze past the same cap slot.
        The same transaction deletes timestamps outside the trailing hour,
        making the posts table bounded by construction: rows only ever exist
        for reservations inside the last hour, at most `cap` per channel.

        Args:
            channel_id: Watch target channel id.
            cap: Max posts allowed in the trailing hour.

        Returns:
            True when the slot was reserved, False when the cap is exhausted.
        """
        now = int(time.time())
        cutoff = now - 3600
        # BEGIN IMMEDIATE takes the write lock up front, so a second connection
        # cannot read the same count and also reserve. Without it, deferred
        # transactions let both readers pass the cap check (cross-connection
        # TOCTOU). The loser blocks on the lock (connect timeout) then retries.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute("DELETE FROM posts WHERE ts <= ?", (cutoff,))
            count = self._conn.execute(
                "SELECT COUNT(*) FROM posts WHERE channel_id=? AND ts>?",
                (channel_id, cutoff),
            ).fetchone()[0]
            if int(count) >= cap:
                self._conn.execute("ROLLBACK")
                return False
            self._conn.execute(
                "INSERT INTO posts(ts, channel_id) VALUES(?,?)",
                (now, channel_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return True


    # -- rolling recall memory -------------------------------------------

    def get_memory(self, channel_id: int) -> tuple[str, list[dict]]:
        """Load the rolling memory for a watch target.

        Args:
            channel_id: Watch target channel id.

        Returns:
            (summary, exchanges): long-term digest string and a list of
            recent {q, a, sha, ts} exchange dicts.
        """
        row = self._conn.execute(
            "SELECT summary, exchanges FROM memory WHERE channel_id=?",
            (channel_id,),
        ).fetchone()
        if not row:
            return "", []
        import json
        try:
            exchanges = json.loads(row[1])
        except (json.JSONDecodeError, TypeError):
            exchanges = []
        return row[0], exchanges

    def set_memory(self, channel_id: int, summary: str,
                   exchanges: list[dict]) -> None:
        """Persist the rolling memory for a watch target.

        Args:
            channel_id: Watch target channel id.
            summary: Long-term digest text.
            exchanges: Recent exchange dicts (JSON-serialized).
        """
        import json
        self._conn.execute(
            "INSERT INTO memory(channel_id,summary,exchanges,updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(channel_id) DO UPDATE SET "
            "summary=excluded.summary, exchanges=excluded.exchanges, "
            "updated_at=excluded.updated_at",
            (channel_id, summary, json.dumps(exchanges), int(time.time())),
        )
        self._conn.commit()
