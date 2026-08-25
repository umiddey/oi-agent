"""Tests for the state store."""

from oi_agent.store import Store


def test_pause_roundtrip(tmp_path):
    s = Store(tmp_path / "state.db")
    assert not s.is_paused()
    s.set_paused(True)
    assert s.is_paused()
    s.set_paused(False)
    assert not s.is_paused()


def test_hourly_cap_reservation(tmp_path):
    s = Store(tmp_path / "state.db")
    assert s.posts_in_last_hour(111) == 0
    for _ in range(3):
        assert s.reserve_post(111, cap=3) is True
    assert s.reserve_post(222, cap=3) is True
    assert s.posts_in_last_hour(111) == 3
    assert s.posts_in_last_hour(222) == 1
    # Cap exhausted: further reservations rejected.
    assert s.reserve_post(111, cap=3) is False


def test_posts_table_bounded_by_construction(tmp_path):
    """Reservation prunes stale rows in the same transaction that counts.

    A months-old database with legacy rows must not grow without bound:
    every reservation first deletes everything outside the trailing hour.
    """
    import sqlite3
    import time

    db = tmp_path / "state.db"
    s = Store(db)
    old = int(time.time()) - 7200  # two hours ago
    conn = sqlite3.connect(db)
    conn.executemany("INSERT INTO posts(ts, channel_id) VALUES(?,?)",
                     [(old, 111)] * 500)
    conn.commit()
    conn.close()

    assert s.reserve_post(111, cap=3) is True
    assert s.posts_in_last_hour(111) == 1  # only the fresh reservation left

    conn = sqlite3.connect(db)
    total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    conn.close()
    assert total == 1  # 500 stale rows gone, not just excluded from counts


def test_legacy_messages_table_dropped_on_open(tmp_path):
    """The raw message archive is removed by migration on first open."""
    import sqlite3

    db = tmp_path / "state.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()

    Store(db)  # open triggers the migration

    conn = sqlite3.connect(db)
    names = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "messages" not in names


def test_memory_roundtrip_and_update(tmp_path):
    from oi_agent.agent.responder import (
        MEMORY_EXCHANGES_KEPT, MEMORY_SUMMARY_CAP, update_memory,
    )

    s = Store(tmp_path / "state.db")
    summary, exchanges = s.get_memory(111)
    assert summary == "" and exchanges == []

    # Flood past the kept-window: overflow must fold into the summary.
    for i in range(MEMORY_EXCHANGES_KEPT + 6):
        update_memory(
            s, 111, f"open question number {i}?",
            f"decision {i} must hold", f"sha{i}",
        )
    summary, exchanges = s.get_memory(111)
    assert len(exchanges) == MEMORY_EXCHANGES_KEPT
    assert "open question number 0" in summary
    assert len(summary) <= MEMORY_SUMMARY_CAP

    # Heavy payloads: summary stays capped, window stays bounded.
    for i in range(20):
        update_memory(s, 111, "q" * 400, "a" * 800, "sha-heavy")
    summary, exchanges = s.get_memory(111)
    assert len(exchanges) == MEMORY_EXCHANGES_KEPT
    assert len(summary) <= MEMORY_SUMMARY_CAP + 100

    # Persistence across store reopen.
    s2 = Store(tmp_path / "state.db")
    summary2, exchanges2 = s2.get_memory(111)
    assert summary2 == summary and len(exchanges2) == len(exchanges)


def test_memory_compaction_preserves_salient_facts(tmp_path):
    """Dropped exchanges compact to salient facts, not raw character slices.

    The old fold appended `q[:150] -> a[:250]` and later cut an arbitrary
    mid-fact chunk off the summary head. Compaction must keep decisions,
    open questions, paths, symbols and SHAs, drop social chatter, and trim
    whole oldest lines only.
    """
    from oi_agent.agent.responder import _compact_exchange

    line = _compact_exchange(
        "should we ship it?",
        "Chatter chatter.\nWe decided the migration must wait. "
        "See src/oi_agent/store.py and `reserve_post`. "
        "Audited at fecdf0e.",
    )
    assert "decided" in line
    assert "must wait" in line
    assert "src/oi_agent/store.py" in line
    assert "reserve_post" in line          # backtick symbol kept
    assert "fecdf0e" in line               # sha kept
    assert "Chatter" not in line           # noise dropped
    assert len(line) <= 500

def test_memory_compaction_drops_nonsalient_chatter(tmp_path):
    """Overflow chatter disappears instead of becoming durable memory."""
    from oi_agent.agent.responder import (
        MEMORY_EXCHANGES_KEPT, _compact_exchange, update_memory,
    )

    assert _compact_exchange("hello there", "nice to meet you") == ""

    store = Store(tmp_path / "state.db")
    for index in range(MEMORY_EXCHANGES_KEPT + 1):
        update_memory(
            store, 111, f"hello there {index}", "nice to meet you", "abc1234",
        )

    summary, exchanges = store.get_memory(111)
    assert summary == ""
    assert [exchange["q"] for exchange in exchanges] == [
        "hello there 1", "hello there 2",
    ]


def test_summary_trim_drops_whole_lines_not_mid_fact_slices(tmp_path):
    """Cap overflow removes the OLDEST summary lines intact."""
    from oi_agent.agent.responder import (
        MEMORY_EXCHANGES_KEPT, MEMORY_SUMMARY_CAP, update_memory,
    )

    s = Store(tmp_path / "state.db")
    # Each folded exchange becomes one long, unique summary line.
    for i in range(MEMORY_EXCHANGES_KEPT + 30):
        update_memory(s, 111,
                      f"constraint number {i} must hold",
                      f"decision {i}: " + f"d{i}" * 120 + " done",
                      f"sha{i:07d}")
    summary, exchanges = s.get_memory(111)
    assert len(exchanges) == MEMORY_EXCHANGES_KEPT
    body = summary.splitlines()
    assert body[0].startswith("(older memory trimmed)")
    facts = [ln for ln in body if ln.startswith("- ")]
    assert facts and all(ln.endswith("done") for ln in facts)
    # Newest folded fact survived; oldest were dropped whole.
    assert "constraint number 29" in facts[-1]
