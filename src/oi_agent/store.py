"""SQLite state for pause, posting caps, OpenCode sessions, and durable delivery queues."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .dynamics import MemorySnapshot

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS posts (
    ts INTEGER NOT NULL,
    channel_id INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_posts_channel_ts ON posts(channel_id, ts);
CREATE TABLE IF NOT EXISTS opencode_sessions (
    conversation_id INTEGER NOT NULL,
    repo_path TEXT NOT NULL,
    session_id TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    PRIMARY KEY (conversation_id, repo_path)
);
CREATE INDEX IF NOT EXISTS idx_opencode_sessions_updated
    ON opencode_sessions(updated_at);
CREATE TABLE IF NOT EXISTS discord_scopes (
    scope_id TEXT PRIMARY KEY,
    owner_channel_id INTEGER NOT NULL,
    last_seen_message_id INTEGER NOT NULL,
    updated_at INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS mention_jobs (
    source_message_id INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    owner_channel_id INTEGER NOT NULL,
    repo_path TEXT NOT NULL,
    status TEXT NOT NULL,
    batch_id TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    available_at INTEGER NOT NULL,
    lease_until INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_error_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_mention_jobs_lookup
    ON mention_jobs(conversation_id, status, available_at);
CREATE INDEX IF NOT EXISTS idx_mention_jobs_batch_id
    ON mention_jobs(batch_id);
CREATE TABLE IF NOT EXISTS delivery_batches (
    batch_id TEXT PRIMARY KEY,
    conversation_id INTEGER NOT NULL,
    owner_channel_id INTEGER NOT NULL,
    repo_path TEXT NOT NULL,
    status TEXT NOT NULL,
    candidate_session_id TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    available_at INTEGER NOT NULL,
    lease_until INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_error_class TEXT
);
CREATE INDEX IF NOT EXISTS idx_delivery_batches_lookup
    ON delivery_batches(status, available_at);
CREATE TABLE IF NOT EXISTS outbound_chunks (
    batch_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    nonce TEXT UNIQUE NOT NULL,
    body TEXT,
    body_sha256 TEXT NOT NULL,
    discord_message_id INTEGER,
    sent_at INTEGER,
    PRIMARY KEY (batch_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_outbound_chunks_nonce
    ON outbound_chunks(nonce);
CREATE TABLE IF NOT EXISTS member_profiles_v2 (
    platform TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    handle TEXT NOT NULL,
    profile_json TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    last_interaction_at INTEGER NOT NULL,
    PRIMARY KEY (platform, scope_id, member_id),
    CHECK (length(profile_json) <= 4096)
);
CREATE INDEX IF NOT EXISTS idx_member_profiles_v2_scope
    ON member_profiles_v2(platform, scope_id, last_interaction_at);
CREATE TABLE IF NOT EXISTS transient_member_state (
    platform TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    member_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    expires_at INTEGER NOT NULL,
    PRIMARY KEY (platform, scope_id, member_id),
    CHECK (length(state_json) <= 2048)
);
CREATE INDEX IF NOT EXISTS idx_transient_member_state_expires
    ON transient_member_state(expires_at);
CREATE TABLE IF NOT EXISTS team_pulse_v2 (
    platform TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    pulse_json TEXT NOT NULL,
    observed_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    PRIMARY KEY (platform, scope_id),
    CHECK (length(pulse_json) <= 2048)
);
CREATE INDEX IF NOT EXISTS idx_team_pulse_v2_expires
    ON team_pulse_v2(expires_at);
CREATE TABLE IF NOT EXISTS agent_calibration (
    platform TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    calibration_json TEXT NOT NULL,
    updated_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    PRIMARY KEY (platform, scope_id),
    CHECK (length(calibration_json) <= 2048)
);
CREATE INDEX IF NOT EXISTS idx_agent_calibration_expires
    ON agent_calibration(expires_at);
DROP TABLE IF EXISTS memory;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS digests;
"""


class LegacyMigrationError(ValueError):
    """Raised when legacy V1 migration encounters an invalid record (audit F6/PIN-3).

    Migration fails closed: no rows are written and no legacy tables are dropped.
    Attributes:
        migrated: Number of records migrated (always 0 on failure).
        skipped: Number of legacy records that failed parse/validate.
    """

    def __init__(self, migrated: int, skipped: int) -> None:
        super().__init__(
            f"Legacy migration failed closed: {skipped} legacy record(s) could not be "
            f"validated; nothing was written and no legacy tables were dropped. "
            f"Preview records, fix or remove them, or run `oi memory purge-legacy --yes` "
            f"to discard the quarantined legacy data."
        )
        self.migrated = migrated
        self.skipped = skipped


def _convert_legacy_row(
    platform: Any,
    member_id: Any,
    handle: Any,
    raw_json: Any,
    updated_at: Any,
) -> tuple[str, str, str, str]:
    """Convert one legacy V1 row to a validated v2 profile tuple.

    Raises ValueError when the row cannot be parsed or validated; callers fail
    closed on any error (plan Phase 6: validate every imported record).
    """
    from .dynamics import (
        CommunicationProfile,
        MemberProfile,
        sanitize_label,
        to_compact_json,
        validate_member_profile,
    )

    if not isinstance(platform, str) or not platform.strip():
        raise ValueError("legacy platform must be a non-empty string")
    if not isinstance(member_id, str) or not member_id.strip():
        raise ValueError("legacy member_id must be a non-empty string")
    if not isinstance(raw_json, str):
        raise ValueError("legacy profile_json must be a string")
    updated = _validate_epoch(updated_at)
    data = json.loads(raw_json)
    if not isinstance(data, dict):
        raise ValueError("legacy profile_json must decode to an object")

    try:
        prof_obj = validate_member_profile(data)
    except Exception:
        # Construct a default V2 profile from V1 data.
        comm = CommunicationProfile(
            directness=0.5,
            detail_preference="balanced",
            challenge_preference="direct",
            humor_preference="medium",
            decision_style="recommendation",
        )
        conf = {
            k: 0.5
            for k in ("directness", "detail_preference", "challenge_preference", "humor_preference", "decision_style")
        }
        ev = {k: 1 for k in conf}
        obs = {k: updated for k in conf}
        topics: list[str] = []
        for k in ("key_competencies", "active_focus", "focus_areas"):
            raw_list = data.get(k)
            if isinstance(raw_list, list):
                for item in raw_list:
                    # Fail closed: an unsanitizable topic invalidates the row.
                    topics.append(sanitize_label(item, max_len=40))
        candidate = MemberProfile(
            schema_version=2,
            communication=comm,
            confidence=conf,
            evidence_count=ev,
            last_observed_at=obs,
            recurring_topics=topics[:5],
            created_at=updated,
            updated_at=updated,
            last_interaction_at=updated,
        )
        # Validate every imported record before it may be written (plan Phase 6).
        prof_obj = validate_member_profile(candidate.to_dict())

    clean_handle = sanitize_label(handle if isinstance(handle, str) else "member", max_len=64) or "member"
    if not clean_handle:
        clean_handle = "member"
    compact_json = to_compact_json(prof_obj)
    return platform, member_id, clean_handle, compact_json


def _validate_epoch(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("legacy timestamps must be non-negative integers")
    return value


def _migrate_legacy_rows(
    conn: sqlite3.Connection,
    target_scope_id: str,
) -> int:
    """Convert and write all legacy rows under target_scope_id in one transaction.

    Returns the migrated count. Raises LegacyMigrationError (leaving the
    transaction rolled back) if any row fails; operational tables are untouched.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='member_profiles'"
    )
    if not cur.fetchone():
        return 0

    rows = conn.execute(
        "SELECT platform, member_id, handle, profile_json, updated_at FROM member_profiles"
    ).fetchall()
    if not rows:
        conn.execute("DROP TABLE IF EXISTS member_profiles")
        conn.execute("DROP TABLE IF EXISTS team_pulse")
        conn.execute("DROP TABLE IF EXISTS agent_reflections")
        return 0

    converted: list[tuple[str, str, str, str, int]] = []
    skipped = 0
    for r in rows:
        platform, member_id, handle, raw_json, updated_at = r
        try:
            plat, mid, clean_handle, compact_json = _convert_legacy_row(
                platform, member_id, handle, raw_json, updated_at
            )
        except Exception:
            skipped += 1
            continue
        converted.append((plat, mid, clean_handle, compact_json, int(updated_at)))

    if skipped:
        raise LegacyMigrationError(migrated=0, skipped=skipped)

    for plat, mid, clean_handle, compact_json, updated in converted:
        conn.execute(
            "INSERT INTO member_profiles_v2(platform, scope_id, member_id, handle, profile_json, created_at, "
            "updated_at, last_interaction_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(platform, scope_id, member_id) DO UPDATE SET "
            "handle=excluded.handle, profile_json=excluded.profile_json, updated_at=excluded.updated_at, "
            "last_interaction_at=excluded.last_interaction_at",
            (plat, str(target_scope_id), mid, clean_handle, compact_json, updated, updated, updated),
        )
    conn.execute("DROP TABLE IF EXISTS member_profiles")
    conn.execute("DROP TABLE IF EXISTS team_pulse")
    conn.execute("DROP TABLE IF EXISTS agent_reflections")
    return len(converted)



def _ensure_schema_compatibility(conn: sqlite3.Connection) -> None:
    """Apply additive migrations required by the current SQLite schema."""
    columns = {
        row[1]
        for row in conn.execute("PRAGMA table_info(member_profiles_v2)").fetchall()
    }
    if "created_at" not in columns:
        conn.execute(
            "ALTER TABLE member_profiles_v2 "
            "ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0"
        )
        conn.execute("UPDATE member_profiles_v2 SET created_at=updated_at")


class Store:
    """Synchronous SQLite wrapper for daemon safety state and durable delivery queues."""

    def __init__(self, db_path: Path) -> None:
        """Open and initialize the state database.

        Args:
            db_path (Path): Filesystem path to the SQLite database file.

        Returns:
            None: No return value.
        """
        self._conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        _ensure_schema_compatibility(self._conn)

    def close(self) -> None:
        """Close the SQLite connection.

        Returns:
            None: No return value.
        """
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run a block inside one IMMEDIATE transaction (audit F8).

        Commits on success; rolls back and re-raises on any failure so a
        mid-apply error cannot leave partial state. Nested use joins the outer
        transaction.
        """
        if self._conn.in_transaction:
            yield self._conn
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def is_paused(self) -> bool:
        """Return whether global posting is paused.

        Returns:
            bool: True if global posting is paused, False otherwise.
        """
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key='paused'"
        ).fetchone()
        return bool(row and row[0] == "1")

    def set_paused(self, paused: bool) -> None:
        """Set or clear the global posting kill switch.

        Args:
            paused (bool): True to pause all outbound posting, False to resume.

        Returns:
            None: No return value.
        """
        self._conn.execute(
            "INSERT INTO kv(key,value) VALUES('paused',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1" if paused else "0",),
        )

    def get_installed_at(self) -> int:
        """Return unix timestamp when agent was installed/initialized."""
        row = self._conn.execute(
            "SELECT value FROM kv WHERE key='installed_at'"
        ).fetchone()
        if row:
            try:
                return int(row[0])
            except ValueError:
                pass
        now = int(time.time())
        self._conn.execute(
            "INSERT INTO kv(key,value) VALUES('installed_at',?) "
            "ON CONFLICT(key) DO NOTHING",
            (str(now),),
        )
        return now

    def set_installed_at(self, ts: int) -> None:
        """Set or update the installation timestamp cutoff."""
        self._conn.execute(
            "INSERT INTO kv(key,value) VALUES('installed_at',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(int(ts)),),
        )

    def get_member_profile(self, platform: str, scope_id: str, member_id: str) -> dict | None:
        """Retrieve a member's scoped dynamic profile traits."""
        row = self._conn.execute(
            "SELECT profile_json FROM member_profiles_v2 WHERE platform=? AND scope_id=? AND member_id=?",
            (platform, str(scope_id), str(member_id)),
        ).fetchone()
        if not row:
            return None
        try:
            from .dynamics import validate_member_profile
            data = json.loads(row[0])
            return validate_member_profile(data).to_dict()
        except Exception:
            return None

    def upsert_member_profile(
        self,
        platform: str,
        scope_id: str,
        member_id: str,
        handle: str,
        profile: Any,
    ) -> None:
        """Persist or update dynamic profile attributes for one team member in a scope.

        Every input (dict or dataclass) is re-validated through the dynamics
        validators (audit F7) so canonical compact JSON is the only thing ever
        persisted.
        """
        from .dynamics import (
            MemberProfile,
            sanitize_label,
            to_compact_json,
            validate_member_profile,
        )

        if isinstance(profile, MemberProfile):
            prof_obj = validate_member_profile(profile.to_dict())
        elif isinstance(profile, dict):
            prof_obj = validate_member_profile(profile)
        else:
            raise ValueError(f"Invalid profile type: {type(profile).__name__}")

        clean_handle = sanitize_label(handle or "member", max_len=64)
        compact_json = to_compact_json(prof_obj)
        now = prof_obj.updated_at
        last_interaction = prof_obj.last_interaction_at
        self._conn.execute(
            "INSERT INTO member_profiles_v2(platform, scope_id, member_id, handle, profile_json, created_at, "
            "updated_at, last_interaction_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(platform, scope_id, member_id) DO UPDATE SET "
            "handle=excluded.handle, profile_json=excluded.profile_json, updated_at=excluded.updated_at, "
            "last_interaction_at=excluded.last_interaction_at",
            (
                platform, str(scope_id), str(member_id), clean_handle,
                compact_json, prof_obj.created_at, now, last_interaction,
            ),
        )

    def list_member_profiles(
        self,
        platform: str,
        scope_id: str,
        member_ids: list[str] | None = None,
    ) -> list[dict]:
        """List scoped member profiles, optionally filtered by member IDs.

        Global platform-only listing is strictly prohibited.
        """
        from .dynamics import validate_member_profile

        if not scope_id:
            raise ValueError("scope_id is required to list member profiles")

        if member_ids:
            placeholders = ",".join("?" for _ in member_ids)
            rows = self._conn.execute(
                f"SELECT member_id, handle, profile_json FROM member_profiles_v2 "
                f"WHERE platform=? AND scope_id=? AND member_id IN ({placeholders}) ORDER BY updated_at DESC",
                [platform, str(scope_id), *(str(m) for m in member_ids)],
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT member_id, handle, profile_json FROM member_profiles_v2 "
                "WHERE platform=? AND scope_id=? ORDER BY updated_at DESC LIMIT 50",
                (platform, str(scope_id)),
            ).fetchall()
        results = []
        for r in rows:
            try:
                raw = json.loads(r[2])
                prof = validate_member_profile(raw).to_dict()
                prof["member_id"] = r[0]
                prof["handle"] = r[1]
                results.append(prof)
            except Exception:
                continue
        return results

    def get_transient_member_state(self, platform: str, scope_id: str, member_id: str) -> dict | None:
        """Retrieve transient member state if not expired."""
        now = int(time.time())
        row = self._conn.execute(
            "SELECT state_json FROM transient_member_state "
            "WHERE platform=? AND scope_id=? AND member_id=? AND expires_at > ?",
            (platform, str(scope_id), str(member_id), now),
        ).fetchone()
        if not row:
            return None
        try:
            from .dynamics import validate_transient_member_state
            return validate_transient_member_state(json.loads(row[0])).to_dict()
        except Exception:
            return None

    def upsert_transient_member_state(
        self,
        platform: str,
        scope_id: str,
        member_id: str,
        state: Any,
    ) -> None:
        """Persist short-lived transient member state.

        Dataclass inputs are re-validated before persistence (audit F7).
        """
        from .dynamics import (
            TransientMemberState,
            to_compact_json,
            validate_transient_member_state,
        )

        if isinstance(state, TransientMemberState):
            obj = validate_transient_member_state(state.to_dict())
        elif isinstance(state, dict):
            obj = validate_transient_member_state(state)
        else:
            raise ValueError(f"Invalid transient state type: {type(state).__name__}")

        compact_json = to_compact_json(obj)
        self._conn.execute(
            "INSERT INTO transient_member_state(platform, scope_id, member_id, state_json, expires_at) "
            "VALUES(?, ?, ?, ?, ?) ON CONFLICT(platform, scope_id, member_id) DO UPDATE SET "
            "state_json=excluded.state_json, expires_at=excluded.expires_at",
            (platform, str(scope_id), str(member_id), compact_json, obj.expires_at),
        )

    def get_team_pulse(self, platform: str, scope_id: str) -> dict | None:
        """Retrieve the rolling team pulse for a scope if not expired."""
        now = int(time.time())
        row = self._conn.execute(
            "SELECT pulse_json FROM team_pulse_v2 WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, str(scope_id), now),
        ).fetchone()
        if not row:
            return None
        try:
            from .dynamics import validate_team_pulse
            return validate_team_pulse(json.loads(row[0])).to_dict()
        except Exception:
            return None

    def upsert_team_pulse(self, platform: str, scope_id: str, pulse: Any) -> None:
        """Persist or update rolling team momentum and focus.

        Dataclass inputs are re-validated before persistence (audit F7).
        """
        from .dynamics import TeamPulse, to_compact_json, validate_team_pulse

        if isinstance(pulse, TeamPulse):
            obj = validate_team_pulse(pulse.to_dict())
        elif isinstance(pulse, dict):
            obj = validate_team_pulse(pulse)
        else:
            raise ValueError(f"Invalid team pulse type: {type(pulse).__name__}")

        compact_json = to_compact_json(obj)
        self._conn.execute(
            "INSERT INTO team_pulse_v2(platform, scope_id, pulse_json, observed_at, expires_at) "
            "VALUES(?, ?, ?, ?, ?) ON CONFLICT(platform, scope_id) DO UPDATE SET "
            "pulse_json=excluded.pulse_json, observed_at=excluded.observed_at, expires_at=excluded.expires_at",
            (platform, str(scope_id), compact_json, obj.observed_at, obj.expires_at),
        )

    def get_agent_calibration(self, platform: str, scope_id: str) -> dict | None:
        """Retrieve the agent's current calibration if not expired."""
        now = int(time.time())
        row = self._conn.execute(
            "SELECT calibration_json FROM agent_calibration WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, str(scope_id), now),
        ).fetchone()
        if not row:
            return None
        try:
            from .dynamics import validate_agent_calibration
            return validate_agent_calibration(json.loads(row[0])).to_dict()
        except Exception:
            return None

    def upsert_agent_calibration(self, platform: str, scope_id: str, calibration: Any) -> None:
        """Persist or update the agent's calibration directive.

        Dataclass inputs are re-validated before persistence (audit F7).
        """
        from .dynamics import (
            AgentCalibration,
            to_compact_json,
            validate_agent_calibration,
        )

        if isinstance(calibration, AgentCalibration):
            obj = validate_agent_calibration(calibration.to_dict())
        elif isinstance(calibration, dict):
            obj = validate_agent_calibration(calibration)
        else:
            raise ValueError(f"Invalid agent calibration type: {type(calibration).__name__}")

        compact_json = to_compact_json(obj)
        self._conn.execute(
            "INSERT INTO agent_calibration(platform, scope_id, calibration_json, updated_at, expires_at) "
            "VALUES(?, ?, ?, ?, ?) ON CONFLICT(platform, scope_id) DO UPDATE SET "
            "calibration_json=excluded.calibration_json, updated_at=excluded.updated_at, "
            "expires_at=excluded.expires_at",
            (platform, str(scope_id), compact_json, obj.updated_at, obj.expires_at),
        )

    def prune_memory(
        self,
        scope_id: str | None = None,
        stable_retention_seconds: int = 90 * 86400,
    ) -> dict[str, int]:
        """Prune expired transient records, expired pulse, expired calibration, and stale profiles.

        Returns counts of deleted records by table.
        """
        now = int(time.time())
        stable_cutoff = now - stable_retention_seconds
        counts = {
            "transient_member_state": 0,
            "team_pulse": 0,
            "agent_calibration": 0,
            "member_profiles": 0,
        }
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if scope_id:
                cur = self._conn.execute(
                    "DELETE FROM transient_member_state WHERE scope_id=? AND expires_at <= ?",
                    (str(scope_id), now),
                )
                counts["transient_member_state"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM team_pulse_v2 WHERE scope_id=? AND expires_at <= ?",
                    (str(scope_id), now),
                )
                counts["team_pulse"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM agent_calibration WHERE scope_id=? AND expires_at <= ?",
                    (str(scope_id), now),
                )
                counts["agent_calibration"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM member_profiles_v2 WHERE scope_id=? AND last_interaction_at <= ?",
                    (str(scope_id), stable_cutoff),
                )
                counts["member_profiles"] = cur.rowcount
            else:
                cur = self._conn.execute(
                    "DELETE FROM transient_member_state WHERE expires_at <= ?",
                    (now,),
                )
                counts["transient_member_state"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM team_pulse_v2 WHERE expires_at <= ?",
                    (now,),
                )
                counts["team_pulse"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM agent_calibration WHERE expires_at <= ?",
                    (now,),
                )
                counts["agent_calibration"] = cur.rowcount

                cur = self._conn.execute(
                    "DELETE FROM member_profiles_v2 WHERE last_interaction_at <= ?",
                    (stable_cutoff,),
                )
                counts["member_profiles"] = cur.rowcount
            self._conn.execute("COMMIT")
            return counts
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def reset_member_memory(self, platform: str, scope_id: str, member_id: str) -> int:
        """Delete all profile and transient state for a specific member in a scope."""
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            c1 = self._conn.execute(
                "DELETE FROM member_profiles_v2 WHERE platform=? AND scope_id=? AND member_id=?",
                (platform, str(scope_id), str(member_id)),
            ).rowcount
            c2 = self._conn.execute(
                "DELETE FROM transient_member_state WHERE platform=? AND scope_id=? AND member_id=?",
                (platform, str(scope_id), str(member_id)),
            ).rowcount
            self._conn.execute("COMMIT")
            return c1 + c2
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def reset_scope_memory(self, platform: str, scope_id: str) -> dict[str, int]:
        """Delete all memory records for an entire scope."""
        counts = {}
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            counts["member_profiles"] = self._conn.execute(
                "DELETE FROM member_profiles_v2 WHERE platform=? AND scope_id=?",
                (platform, str(scope_id)),
            ).rowcount
            counts["transient_member_state"] = self._conn.execute(
                "DELETE FROM transient_member_state WHERE platform=? AND scope_id=?",
                (platform, str(scope_id)),
            ).rowcount
            counts["team_pulse"] = self._conn.execute(
                "DELETE FROM team_pulse_v2 WHERE platform=? AND scope_id=?",
                (platform, str(scope_id)),
            ).rowcount
            counts["agent_calibration"] = self._conn.execute(
                "DELETE FROM agent_calibration WHERE platform=? AND scope_id=?",
                (platform, str(scope_id)),
            ).rowcount
            self._conn.execute("COMMIT")
            return counts
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_memory_status(self, platform: str, scope_id: str) -> dict[str, Any]:
        """Return summary metrics for memory in a scope."""
        now = int(time.time())
        prof_row = self._conn.execute(
            "SELECT COUNT(*), MIN(created_at), MAX(updated_at) FROM member_profiles_v2 WHERE platform=? AND scope_id=?",
            (platform, str(scope_id)),
        ).fetchone()
        trans_row = self._conn.execute(
            "SELECT COUNT(*) FROM transient_member_state WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, str(scope_id), now),
        ).fetchone()
        trans_expired = self._conn.execute(
            "SELECT COUNT(*) FROM transient_member_state WHERE platform=? AND scope_id=? AND expires_at <= ?",
            (platform, str(scope_id), now),
        ).fetchone()
        pulse_row = self._conn.execute(
            "SELECT COUNT(*) FROM team_pulse_v2 WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, str(scope_id), now),
        ).fetchone()
        calib_row = self._conn.execute(
            "SELECT COUNT(*) FROM agent_calibration WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, str(scope_id), now),
        ).fetchone()
        from .dynamics import MEMORY_SCHEMA_VERSION

        return {
            "schema_version": MEMORY_SCHEMA_VERSION,
            "member_profiles_count": prof_row[0] if prof_row else 0,
            "oldest_profile_at": prof_row[1] if prof_row and prof_row[1] else None,
            "newest_profile_at": prof_row[2] if prof_row and prof_row[2] else None,
            "active_transient_count": trans_row[0] if trans_row else 0,
            "expired_transient_count": trans_expired[0] if trans_expired else 0,
            "has_active_team_pulse": bool(pulse_row and pulse_row[0] > 0),
            "has_active_agent_calibration": bool(calib_row and calib_row[0] > 0),
        }

    def memory_snapshot(
        self,
        platform: str,
        scope_id: str,
        *,
        participant_member_ids: Sequence[str] | None = None,
        max_profiles: int = 10,
        now: float | None = None,
    ) -> MemorySnapshot:
        """Build a render-safe, scope-isolated memory snapshot (pinned contract PIN-2).

        Includes only unexpired transient/pulse/calibration records and only
        profiles whose overall confidence clears CONFIDENCE_FLOOR with at least
        one render-ready field. Raises ValueError on empty scope_id.
        """
        from .dynamics import (
            MemorySnapshot,
            build_calibration_view,
            build_profile_view,
            build_team_pulse_view,
            build_transient_view,
            validate_agent_calibration,
            validate_member_profile,
            validate_team_pulse,
            validate_transient_member_state,
        )

        if not scope_id:
            raise ValueError("scope_id is required for a memory snapshot")
        scope_key = str(scope_id)
        now_i = int(now) if now is not None else int(time.time())

        member_filter = ""
        params: list[Any] = [platform, scope_key]
        if participant_member_ids:
            ids = [str(m) for m in participant_member_ids if str(m)]
            if ids:
                member_filter = f" AND member_id IN ({','.join('?' for _ in ids)})"
                params.extend(ids)

        rows = self._conn.execute(
            f"SELECT member_id, handle, profile_json FROM member_profiles_v2 "
            f"WHERE platform=? AND scope_id=?{member_filter} "
            f"ORDER BY last_interaction_at DESC LIMIT ?",
            params + [int(max_profiles)],
        ).fetchall()
        profile_views = []
        handle_by_member: dict[str, str] = {}
        for member_id, handle, profile_json in rows:
            try:
                prof = validate_member_profile(json.loads(profile_json))
            except Exception:
                continue
            handle_by_member[str(member_id)] = str(handle)
            view = build_profile_view(member_id, handle, prof)
            if view is not None:
                profile_views.append(view)

        transient_rows = self._conn.execute(
            f"SELECT member_id, state_json FROM transient_member_state "
            f"WHERE platform=? AND scope_id=?{member_filter} AND expires_at > ? "
            f"ORDER BY expires_at DESC LIMIT 50",
            params + [now_i],
        ).fetchall()
        transient_views = []
        for member_id, state_json in transient_rows:
            try:
                state = validate_transient_member_state(json.loads(state_json))
            except Exception:
                continue
            view = build_transient_view(member_id, handle_by_member.get(str(member_id), "member"), state)
            if view is not None:
                transient_views.append(view)

        pulse_view = None
        pulse_row = self._conn.execute(
            "SELECT pulse_json FROM team_pulse_v2 WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, scope_key, now_i),
        ).fetchone()
        if pulse_row:
            try:
                pulse_view = build_team_pulse_view(validate_team_pulse(json.loads(pulse_row[0])))
            except Exception:
                pulse_view = None

        calibration_view = None
        calib_row = self._conn.execute(
            "SELECT calibration_json FROM agent_calibration WHERE platform=? AND scope_id=? AND expires_at > ?",
            (platform, scope_key, now_i),
        ).fetchone()
        if calib_row:
            try:
                calibration_view = build_calibration_view(
                    validate_agent_calibration(json.loads(calib_row[0]))
                )
            except Exception:
                calibration_view = None

        generated_at = datetime.now(UTC).isoformat()
        return MemorySnapshot(
            profiles=tuple(profile_views),
            transient=tuple(transient_views),
            team_pulse=pulse_view,
            calibration=calibration_view,
            generated_at=generated_at,
        )

    def preview_legacy_migration(self) -> list[dict[str, Any]]:
        """Preview inaccessible legacy V1 profiles without migrating."""
        # Check if legacy table exists
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='member_profiles'"
        )
        if not cur.fetchone():
            return []
        rows = self._conn.execute(
            "SELECT platform, member_id, handle, profile_json, updated_at FROM member_profiles ORDER BY updated_at DESC"
        ).fetchall()
        results = []
        for r in rows:
            results.append({
                "platform": r[0],
                "member_id": r[1],
                "handle": r[2],
                "updated_at": r[4],
            })
        return results

    def migrate_legacy_profiles(self, target_scope_id: str) -> int:
        """Migrate legacy V1 profiles to V2 under an explicit scope, failing closed.

        If ANY legacy row fails parse/validate, nothing is written and no legacy
        tables are dropped; LegacyMigrationError is raised (audit F6/PIN-3).
        """
        if not target_scope_id:
            raise ValueError("target_scope_id is required")
        with self.transaction() as conn:
            return _migrate_legacy_rows(conn, target_scope_id)

    def purge_legacy_profiles(self) -> int:
        """Purge and drop all legacy V1 tables."""
        total = 0
        with self.transaction() as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='member_profiles'"
            )
            if cur.fetchone():
                c = conn.execute("SELECT COUNT(*) FROM member_profiles").fetchone()[0]
                total += int(c)
                conn.execute("DROP TABLE member_profiles")
            conn.execute("DROP TABLE IF EXISTS team_pulse")
            conn.execute("DROP TABLE IF EXISTS agent_reflections")
        return total

    def posts_in_last_hour(self, channel_id: int) -> int:
        """Count reservations for one channel in the trailing hour.

        Args:
            channel_id (int): Snowflake ID of the target channel.

        Returns:
            int: Number of reserved posts in the trailing hour.
        """
        cutoff = int(time.time()) - 3600
        row = self._conn.execute(
            "SELECT COUNT(*) FROM posts WHERE channel_id=? AND ts>?",
            (channel_id, cutoff),
        ).fetchone()
        return int(row[0])

    def reserve_post(self, channel_id: int, cap: int) -> bool:
        """Atomically reserve one hourly post slot.

        Args:
            channel_id (int): Snowflake ID of the target channel.
            cap (int): Maximum allowed posts per hour.

        Returns:
            bool: True if reservation was successful, False if rate limited.
        """
        now = int(time.time())
        cutoff = now - 3600
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
            return True
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    @staticmethod
    def _repo_key(repo_path: str | Path) -> str:
        """Canonicalize a repository key without reading repository contents.

        Args:
            repo_path (str | Path): Path to the audited repository.

        Returns:
            str: Normalized absolute filesystem path string.
        """
        return str(Path(repo_path).expanduser().resolve())

    def get_opencode_session(self, conversation_id: int,
                             repo_path: str | Path) -> str | None:
        """Return the stored OpenCode session ID for a conversation/repo pair.

        Args:
            conversation_id (int): Snowflake ID of the channel or thread conversation.
            repo_path (str | Path): Path to the audited repository.

        Returns:
            str | None: Stored OpenCode session ID, or None if not set.
        """
        row = self._conn.execute(
            "SELECT session_id FROM opencode_sessions "
            "WHERE conversation_id=? AND repo_path=?",
            (conversation_id, self._repo_key(repo_path)),
        ).fetchone()
        return row[0] if row else None

    def set_opencode_session(self, conversation_id: int, repo_path: str | Path,
                             session_id: str) -> None:
        """Persist one successful OpenCode session mapping.

        Args:
            conversation_id (int): Snowflake ID of the channel or thread conversation.
            repo_path (str | Path): Path to the audited repository.
            session_id (str): OpenCode session identifier string.

        Returns:
            None: No return value.
        """
        if not session_id.strip():
            raise ValueError("session_id must be non-empty")
        self._conn.execute(
            "INSERT INTO opencode_sessions(conversation_id,repo_path,session_id,updated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(conversation_id,repo_path) DO UPDATE SET "
            "session_id=excluded.session_id, updated_at=excluded.updated_at",
            (conversation_id, self._repo_key(repo_path), session_id, int(time.time())),
        )

    def clear_opencode_session(self, conversation_id: int,
                               repo_path: str | Path) -> None:
        """Delete a stale OpenCode session mapping.

        Args:
            conversation_id (int): Snowflake ID of the channel or thread conversation.
            repo_path (str | Path): Path to the audited repository.

        Returns:
            None: No return value.
        """
        self._conn.execute(
            "DELETE FROM opencode_sessions WHERE conversation_id=? AND repo_path=?",
            (conversation_id, self._repo_key(repo_path)),
        )

    def admit_mention_job(self, source_message_id: int, conversation_id: int,
                          owner_channel_id: int, repo_path: str | Path,
                          available_at: int | None = None) -> bool:
        """Admit a new mention job to the queue if not already recorded.

        Args:
            source_message_id (int): Snowflake ID of the Discord trigger message.
            conversation_id (int): Snowflake ID of the channel or thread conversation.
            owner_channel_id (int): Snowflake ID of the parent/owner channel.
            repo_path (str | Path): Path to the audited repository.
            available_at (int | None): Epoch timestamp when the job becomes available. Defaults to now.

        Returns:
            bool: True if the job was admitted, False if it was already recorded.
        """
        now = int(time.time())
        avail = now if available_at is None else available_at
        repo_key = self._repo_key(repo_path)
        cur = self._conn.execute(
            "INSERT OR IGNORE INTO mention_jobs ("
            "source_message_id, conversation_id, owner_channel_id, repo_path, "
            "status, batch_id, attempt_count, available_at, lease_until, "
            "created_at, updated_at, last_error_class"
            ") VALUES (?, ?, ?, ?, 'pending', NULL, 0, ?, 0, ?, ?, NULL)",
            (source_message_id, conversation_id, owner_channel_id, repo_key, avail, now, now),
        )
        return cur.rowcount > 0

    def update_scope_cursor(self, scope_id: str, owner_channel_id: int,
                            last_seen_message_id: int) -> None:
        """Update the last seen message ID cursor for a discord scope.

        Args:
            scope_id (str): Unique scope identifier string (e.g. channel or thread ID).
            owner_channel_id (int): Snowflake ID of the owner channel.
            last_seen_message_id (int): Snowflake ID of the latest processed message.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        self._conn.execute(
            "INSERT INTO discord_scopes (scope_id, owner_channel_id, last_seen_message_id, updated_at) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(scope_id) DO UPDATE SET "
            "owner_channel_id = excluded.owner_channel_id, "
            "last_seen_message_id = excluded.last_seen_message_id, "
            "updated_at = excluded.updated_at "
            "WHERE excluded.last_seen_message_id > discord_scopes.last_seen_message_id",
            (str(scope_id), owner_channel_id, last_seen_message_id, now),
        )

    def get_scope_cursor(self, scope_id: str) -> int | None:
        """Retrieve the last seen message ID cursor for a discord scope.

        Args:
            scope_id (str): Unique scope identifier string.

        Returns:
            int | None: Last seen message ID, or None if scope has no recorded cursor.
        """
        row = self._conn.execute(
            "SELECT last_seen_message_id FROM discord_scopes WHERE scope_id = ?",
            (str(scope_id),),
        ).fetchone()
        return int(row[0]) if row is not None else None

    def claim_next_conversation_jobs(
        self,
        lease_duration_seconds: int = 60,
        exclude_conversations: list[int] | set[int] | None = None,
    ) -> tuple[int, list[dict]] | None:
        """Atomically claim the next eligible batch of mention jobs for a single conversation.

        Args:
            lease_duration_seconds (int): Number of seconds to hold the lease. Defaults to 60.
            exclude_conversations (list[int] | set[int] | None): Conversation IDs to exclude.

        Returns:
            tuple[int, list[dict]] | None: Tuple of (conversation_id, list of job dictionaries) or None.
        """
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            exclude_clause = ""
            params: list[Any] = [now, now]
            if exclude_conversations:
                ex_list = list(exclude_conversations)
                placeholders_ex = ",".join("?" * len(ex_list))
                exclude_clause = f"AND conversation_id NOT IN ({placeholders_ex}) "
                params.extend(ex_list)

            row = self._conn.execute(
                f"SELECT conversation_id FROM mention_jobs "
                f"WHERE ((status = 'pending' AND available_at <= ?) "
                f"   OR (status = 'leased' AND lease_until < ? AND batch_id IS NULL)) "
                f"{exclude_clause}"
                f"ORDER BY created_at ASC, source_message_id ASC LIMIT 1",
                params,
            ).fetchone()
            if not row:
                self._conn.execute("COMMIT")
                return None
            conv_id = int(row[0])

            cursor = self._conn.execute(
                "SELECT source_message_id, conversation_id, owner_channel_id, repo_path, "
                "status, batch_id, attempt_count, available_at, lease_until, created_at, "
                "updated_at, last_error_class FROM mention_jobs "
                "WHERE conversation_id = ? "
                "  AND ((status = 'pending' AND available_at <= ?) "
                "       OR (status = 'leased' AND lease_until < ? AND batch_id IS NULL)) "
                "ORDER BY created_at ASC, source_message_id ASC",
                (conv_id, now, now),
            )
            columns = [col[0] for col in cursor.description]
            jobs = [dict(zip(columns, r)) for r in cursor.fetchall()]
            if not jobs:
                self._conn.execute("COMMIT")
                return None

            msg_ids = [j["source_message_id"] for j in jobs]
            new_lease = now + lease_duration_seconds
            placeholders = ",".join("?" * len(msg_ids))
            self._conn.execute(
                f"UPDATE mention_jobs SET status = 'leased', lease_until = ?, updated_at = ? "
                f"WHERE source_message_id IN ({placeholders})",
                (new_lease, now, *msg_ids),
            )
            for j in jobs:
                j["status"] = "leased"
                j["lease_until"] = new_lease
                j["updated_at"] = now

            self._conn.execute("COMMIT")
            return (conv_id, jobs)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def claim_more_pending_jobs(
        self, conversation_id: int, lease_duration_seconds: int = 180
    ) -> list[dict[str, Any]]:
        """Lease any additional pending mention jobs that arrived for a conversation.

        Args:
            conversation_id (int): Conversation identifier.
            lease_duration_seconds (int): Lease duration in seconds. Defaults to 180.

        Returns:
            list[dict[str, Any]]: Newly leased job records.
        """
        now = int(time.time())
        new_lease = now + lease_duration_seconds
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "SELECT source_message_id, conversation_id, owner_channel_id, repo_path, "
                "status, batch_id, attempt_count, available_at, lease_until, created_at, "
                "updated_at, last_error_class FROM mention_jobs "
                "WHERE conversation_id = ? AND status = 'pending' "
                "ORDER BY created_at ASC, source_message_id ASC",
                (conversation_id,),
            )
            columns = [col[0] for col in cursor.description]
            more_jobs = [dict(zip(columns, r)) for r in cursor.fetchall()]
            if more_jobs:
                more_msg_ids = [j["source_message_id"] for j in more_jobs]
                placeholders = ",".join("?" * len(more_msg_ids))
                self._conn.execute(
                    f"UPDATE mention_jobs SET status = 'leased', lease_until = ?, updated_at = ? "
                    f"WHERE source_message_id IN ({placeholders})",
                    (new_lease, now, *more_msg_ids),
                )
                for j in more_jobs:
                    j["status"] = "leased"
                    j["lease_until"] = new_lease
                    j["updated_at"] = now
            self._conn.execute("COMMIT")
            return more_jobs
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def renew_job_leases(self, source_message_ids: list[int],
                         lease_duration_seconds: int = 60) -> None:
        """Renew the lease duration for a set of currently leased mention jobs.

        Args:
            source_message_ids (list[int]): List of source message IDs whose leases are renewed.
            lease_duration_seconds (int): Number of seconds to extend the lease. Defaults to 60.

        Returns:
            None: No return value.
        """
        if not source_message_ids:
            return
        now = int(time.time())
        new_lease = now + lease_duration_seconds
        placeholders = ",".join("?" * len(source_message_ids))
        self._conn.execute(
            f"UPDATE mention_jobs SET lease_until = ?, updated_at = ? "
            f"WHERE source_message_id IN ({placeholders}) AND status = 'leased'",
            (new_lease, now, *source_message_ids),
        )

    def commit_outbox(self, batch_id: str, conversation_id: int,
                      owner_channel_id: int, repo_path: str | Path,
                      source_message_ids: list[int], chunks: list[str],
                      candidate_session_id: str | None) -> None:
        """Commit a generated response batch and its chunks to the outbound queue.

        Args:
            batch_id (str): Unique identifier for the delivery batch.
            conversation_id (int): Snowflake ID of the conversation.
            owner_channel_id (int): Snowflake ID of the owner channel.
            repo_path (str | Path): Path to the audited repository.
            source_message_ids (list[int]): List of trigger message IDs included in this batch.
            chunks (list[str]): Text message chunks to deliver.
            candidate_session_id (str | None): OpenCode session ID to persist upon completion.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        repo_key = self._repo_key(repo_path)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "INSERT INTO delivery_batches ("
                "batch_id, conversation_id, owner_channel_id, repo_path, "
                "status, candidate_session_id, attempt_count, available_at, "
                "lease_until, created_at, updated_at, last_error_class"
                ") VALUES (?, ?, ?, ?, 'pending', ?, 0, ?, 0, ?, ?, NULL)",
                (batch_id, conversation_id, owner_channel_id, repo_key, candidate_session_id, now, now, now),
            )
            prefix = batch_id[:16]
            for idx, chunk in enumerate(chunks):
                nonce = f"{prefix}-{idx}"
                sha = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
                self._conn.execute(
                    "INSERT INTO outbound_chunks ("
                    "batch_id, chunk_index, nonce, body, body_sha256, "
                    "discord_message_id, sent_at"
                    ") VALUES (?, ?, ?, ?, ?, NULL, NULL)",
                    (batch_id, idx, nonce, chunk, sha),
                )
            if source_message_ids:
                placeholders = ",".join("?" * len(source_message_ids))
                self._conn.execute(
                    f"UPDATE mention_jobs SET batch_id = ?, status = 'leased', updated_at = ? "
                    f"WHERE source_message_id IN ({placeholders})",
                    (batch_id, now, *source_message_ids),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def claim_pending_outbox_batch(self, lease_duration_seconds: int = 60) -> dict | None:
        """Claim the next pending or expired-lease outbound delivery batch.

        Args:
            lease_duration_seconds (int): Number of seconds to lease the batch. Defaults to 60.

        Returns:
            dict | None: Batch dictionary including chunks and source message IDs, or None.
        """
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = self._conn.execute(
                "SELECT batch_id, conversation_id, owner_channel_id, repo_path, status, "
                "candidate_session_id, attempt_count, available_at, lease_until, "
                "created_at, updated_at, last_error_class FROM delivery_batches "
                "WHERE (status = 'pending' AND available_at <= ?) "
                "   OR (status = 'leased' AND lease_until < ?) "
                "ORDER BY created_at ASC, batch_id ASC LIMIT 1",
                (now, now),
            )
            row = cursor.fetchone()
            if not row:
                self._conn.execute("COMMIT")
                return None
            columns = [col[0] for col in cursor.description]
            batch = dict(zip(columns, row))
            batch_id = batch["batch_id"]
            new_lease = now + lease_duration_seconds

            self._conn.execute(
                "UPDATE delivery_batches SET status = 'leased', lease_until = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (new_lease, now, batch_id),
            )
            self._conn.execute(
                "UPDATE mention_jobs SET status = 'leased', lease_until = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (new_lease, now, batch_id),
            )
            batch["status"] = "leased"
            batch["lease_until"] = new_lease
            batch["updated_at"] = now

            chunk_cur = self._conn.execute(
                "SELECT batch_id, chunk_index, nonce, body, body_sha256, "
                "discord_message_id, sent_at FROM outbound_chunks "
                "WHERE batch_id = ? ORDER BY chunk_index ASC",
                (batch_id,),
            )
            chunk_cols = [col[0] for col in chunk_cur.description]
            batch["chunks"] = [dict(zip(chunk_cols, r)) for r in chunk_cur.fetchall()]

            job_rows = self._conn.execute(
                "SELECT source_message_id FROM mention_jobs WHERE batch_id = ? ORDER BY created_at ASC",
                (batch_id,),
            ).fetchall()
            batch["source_message_ids"] = [r[0] for r in job_rows]

            self._conn.execute("COMMIT")
            return batch
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def record_chunk_delivery(self, batch_id: str, chunk_index: int,
                              discord_message_id: int) -> None:
        """Record the successful Discord delivery of an individual message chunk.

        Args:
            batch_id (str): Unique delivery batch identifier.
            chunk_index (int): Zero-based index of the delivered chunk.
            discord_message_id (int): Snowflake ID of the delivered Discord message.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        self._conn.execute(
            "UPDATE outbound_chunks SET discord_message_id = ?, sent_at = ? "
            "WHERE batch_id = ? AND chunk_index = ?",
            (discord_message_id, now, batch_id, chunk_index),
        )

    def complete_delivery_batch(self, batch_id: str, repo_path: str | Path,
                                candidate_session_id: str | None) -> None:
        """Mark a delivery batch and its mention jobs as done, scrubbing chunk bodies.

        Args:
            batch_id (str): Unique delivery batch identifier.
            repo_path (str | Path): Path to the audited repository.
            candidate_session_id (str | None): OpenCode session ID to persist, if any.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        repo_key = self._repo_key(repo_path)
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "UPDATE delivery_batches SET status = 'done', updated_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )
            self._conn.execute(
                "UPDATE mention_jobs SET status = 'done', updated_at = ? WHERE batch_id = ?",
                (now, batch_id),
            )
            self._conn.execute(
                "UPDATE outbound_chunks SET body = NULL WHERE batch_id = ?",
                (batch_id,),
            )
            if candidate_session_id and candidate_session_id.strip():
                row = self._conn.execute(
                    "SELECT conversation_id FROM delivery_batches WHERE batch_id = ?",
                    (batch_id,),
                ).fetchone()
                if row:
                    conv_id = int(row[0])
                    self._conn.execute(
                        "INSERT INTO opencode_sessions(conversation_id, repo_path, session_id, updated_at) "
                        "VALUES(?, ?, ?, ?) ON CONFLICT(conversation_id, repo_path) DO UPDATE SET "
                        "session_id = excluded.session_id, updated_at = excluded.updated_at",
                        (conv_id, repo_key, candidate_session_id.strip(), now),
                    )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def fail_jobs(self, source_message_ids: list[int], error_class: str,
                  is_permanent: bool, retry_delay_seconds: int = 10,
                  max_attempts: int = 5) -> None:
        """Record a failure for a set of mention jobs, scheduling retry or dead-lettering.

        Args:
            source_message_ids (list[int]): List of source message IDs that failed.
            error_class (str): Classification label of the error.
            is_permanent (bool): Whether the failure is non-retryable.
            retry_delay_seconds (int): Delay in seconds before the next retry attempt. Defaults to 10.
            max_attempts (int): Maximum retry attempts before dead-lettering. Defaults to 5.

        Returns:
            None: No return value.
        """
        if not source_message_ids:
            return
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            placeholders = ",".join("?" * len(source_message_ids))
            rows = self._conn.execute(
                f"SELECT source_message_id, attempt_count FROM mention_jobs "
                f"WHERE source_message_id IN ({placeholders})",
                source_message_ids,
            ).fetchall()
            for msg_id, attempt_count in rows:
                new_attempt = attempt_count + 1
                if is_permanent or new_attempt >= max_attempts:
                    status = "dead"
                    avail = now
                else:
                    status = "pending"
                    avail = now + retry_delay_seconds
                self._conn.execute(
                    "UPDATE mention_jobs SET status = ?, attempt_count = ?, available_at = ?, "
                    "lease_until = 0, last_error_class = ?, updated_at = ? "
                    "WHERE source_message_id = ?",
                    (status, new_attempt, avail, error_class, now, msg_id),
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def fail_batch(self, batch_id: str, error_class: str,
                   is_permanent: bool, retry_delay_seconds: int = 10,
                   max_attempts: int = 5) -> None:
        """Record a failure for an outbound delivery batch and its linked mention jobs.

        Args:
            batch_id (str): Delivery batch identifier that failed.
            error_class (str): Classification label of the error.
            is_permanent (bool): Whether the failure is non-retryable.
            retry_delay_seconds (int): Delay in seconds before the next retry attempt. Defaults to 10.
            max_attempts (int): Maximum retry attempts before dead-lettering. Defaults to 5.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            row = self._conn.execute(
                "SELECT attempt_count FROM delivery_batches WHERE batch_id = ?",
                (batch_id,),
            ).fetchone()
            attempt_count = row[0] if row else 0
            new_attempt = attempt_count + 1
            if is_permanent or new_attempt >= max_attempts:
                status = "dead"
                avail = now
            else:
                status = "pending"
                avail = now + retry_delay_seconds

            self._conn.execute(
                "UPDATE delivery_batches SET status = ?, attempt_count = ?, available_at = ?, "
                "lease_until = 0, last_error_class = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (status, new_attempt, avail, error_class, now, batch_id),
            )
            self._conn.execute(
                "UPDATE mention_jobs SET status = ?, attempt_count = attempt_count + 1, "
                "available_at = ?, lease_until = 0, last_error_class = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (status, avail, error_class, now, batch_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def defer_jobs(self, source_message_ids: list[int], error_class: str,
                   delay_seconds: int = 15) -> None:
        """Defer mention jobs without incrementing their attempt count.

        Used for pause, rate-limiting, and external backpressure where the job
        was not attempted or failed, but merely delayed.

        Args:
            source_message_ids (list[int]): List of source message IDs to defer.
            error_class (str): Bounded label (e.g. 'paused', 'hourly_cap_exceeded').
            delay_seconds (int): Time in seconds before jobs become available again. Defaults to 15.

        Returns:
            None: No return value.
        """
        if not source_message_ids:
            return
        now = int(time.time())
        avail = now + delay_seconds
        placeholders = ",".join("?" * len(source_message_ids))
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                f"UPDATE mention_jobs SET status = 'pending', lease_until = 0, "
                f"available_at = ?, last_error_class = ?, updated_at = ? "
                f"WHERE source_message_id IN ({placeholders})",
                (avail, error_class, now, *source_message_ids),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def defer_batch(self, batch_id: str, error_class: str,
                    delay_seconds: int = 15) -> None:
        """Defer a delivery batch and its linked jobs without incrementing attempt count.

        Args:
            batch_id (str): Delivery batch identifier to defer.
            error_class (str): Bounded label (e.g. 'paused', 'hourly_cap_exceeded').
            delay_seconds (int): Time in seconds before batch becomes available again. Defaults to 15.

        Returns:
            None: No return value.
        """
        now = int(time.time())
        avail = now + delay_seconds
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._conn.execute(
                "UPDATE delivery_batches SET status = 'pending', lease_until = 0, "
                "available_at = ?, last_error_class = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (avail, error_class, now, batch_id),
            )
            self._conn.execute(
                "UPDATE mention_jobs SET status = 'pending', lease_until = 0, "
                "available_at = ?, last_error_class = ?, updated_at = ? "
                "WHERE batch_id = ?",
                (avail, error_class, now, batch_id),
            )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def get_queue_stats(self) -> dict[str, int]:
        """Calculate and return aggregate queue and outbox statistics.

        Returns:
            dict[str, int]: Dictionary of queue counters across jobs, batches, and stale leases.
        """
        now = int(time.time())
        stats = {
            "pending_jobs": 0,
            "leased_jobs": 0,
            "done_jobs": 0,
            "dead_jobs": 0,
            "pending_batches": 0,
            "leased_batches": 0,
            "done_batches": 0,
            "dead_batches": 0,
            "stale_leases": 0,
        }
        for row in self._conn.execute("SELECT status, COUNT(*) FROM mention_jobs GROUP BY status").fetchall():
            key = f"{row[0]}_jobs"
            if key in stats:
                stats[key] = int(row[1])

        for row in self._conn.execute("SELECT status, COUNT(*) FROM delivery_batches GROUP BY status").fetchall():
            key = f"{row[0]}_batches"
            if key in stats:
                stats[key] = int(row[1])

        stale_jobs = self._conn.execute(
            "SELECT COUNT(*) FROM mention_jobs WHERE status = 'leased' AND lease_until < ?",
            (now,),
        ).fetchone()[0]
        stale_batches = self._conn.execute(
            "SELECT COUNT(*) FROM delivery_batches WHERE status = 'leased' AND lease_until < ?",
            (now,),
        ).fetchone()[0]
        stats["stale_leases"] = int(stale_jobs) + int(stale_batches)
        return stats

    def retry_dead_jobs(self, source_message_ids: list[int] | None = None) -> int:
        """Reset dead-lettered jobs and associated batches back to pending status.

        Args:
            source_message_ids (list[int] | None): Specific message IDs to retry, or None for all dead jobs.

        Returns:
            int: Total number of dead jobs reset to pending.
        """
        now = int(time.time())
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if source_message_ids is None:
                batch_rows = self._conn.execute(
                    "SELECT DISTINCT batch_id FROM mention_jobs WHERE status = 'dead' AND batch_id IS NOT NULL"
                ).fetchall()
                cur = self._conn.execute(
                    "UPDATE mention_jobs SET status = 'pending', attempt_count = 0, "
                    "available_at = ?, lease_until = 0, last_error_class = NULL, updated_at = ? "
                    "WHERE status = 'dead'",
                    (now, now),
                )
                count = cur.rowcount
                if batch_rows:
                    b_ids = [r[0] for r in batch_rows]
                    placeholders = ",".join("?" * len(b_ids))
                    self._conn.execute(
                        f"UPDATE delivery_batches SET status = 'pending', attempt_count = 0, "
                        f"available_at = ?, lease_until = 0, last_error_class = NULL, updated_at = ? "
                        f"WHERE batch_id IN ({placeholders})",
                        (now, now, *b_ids),
                    )
                self._conn.execute(
                    "UPDATE delivery_batches SET status = 'pending', attempt_count = 0, "
                    "available_at = ?, lease_until = 0, last_error_class = NULL, updated_at = ? "
                    "WHERE status = 'dead'",
                    (now, now),
                )
            else:
                if not source_message_ids:
                    self._conn.execute("COMMIT")
                    return 0
                placeholders = ",".join("?" * len(source_message_ids))
                batch_rows = self._conn.execute(
                    f"SELECT DISTINCT batch_id FROM mention_jobs "
                    f"WHERE status = 'dead' AND source_message_id IN ({placeholders}) AND batch_id IS NOT NULL",
                    source_message_ids,
                ).fetchall()
                cur = self._conn.execute(
                    f"UPDATE mention_jobs SET status = 'pending', attempt_count = 0, "
                    f"available_at = ?, lease_until = 0, last_error_class = NULL, updated_at = ? "
                    f"WHERE status = 'dead' AND source_message_id IN ({placeholders})",
                    (now, now, *source_message_ids),
                )
                count = cur.rowcount
                if batch_rows:
                    b_ids = [r[0] for r in batch_rows]
                    b_placeholders = ",".join("?" * len(b_ids))
                    self._conn.execute(
                        f"UPDATE delivery_batches SET status = 'pending', attempt_count = 0, "
                        f"available_at = ?, lease_until = 0, last_error_class = NULL, updated_at = ? "
                        f"WHERE batch_id IN ({b_placeholders})",
                        (now, now, *b_ids),
                    )
            self._conn.execute("COMMIT")
            return int(count)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def purge_completed(self, older_than_seconds: int = 86400) -> int:
        """Purge completed mention jobs and delivery batches older than a given threshold.

        Args:
            older_than_seconds (int): Age threshold in seconds for completed records. Defaults to 86400.

        Returns:
            int: Number of purged mention jobs.
        """
        now = int(time.time())
        cutoff = now - older_than_seconds
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            done_batches = [
                r[0] for r in self._conn.execute(
                    "SELECT batch_id FROM delivery_batches WHERE status = 'done' AND updated_at < ?",
                    (cutoff,),
                ).fetchall()
            ]
            if done_batches:
                placeholders = ",".join("?" * len(done_batches))
                self._conn.execute(
                    f"DELETE FROM outbound_chunks WHERE batch_id IN ({placeholders})",
                    done_batches,
                )
                self._conn.execute(
                    f"DELETE FROM delivery_batches WHERE batch_id IN ({placeholders})",
                    done_batches,
                )
            cur = self._conn.execute(
                "DELETE FROM mention_jobs WHERE status = 'done' AND updated_at < ?",
                (cutoff,),
            )
            purged_count = cur.rowcount
            self._conn.execute("COMMIT")
            return int(purged_count)
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
