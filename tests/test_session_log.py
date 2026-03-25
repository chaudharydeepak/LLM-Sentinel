"""
Tests for SessionLog: write operations, read operations, multi-session listing,
insights computation, and phase detection.
"""

import time
import tempfile
from pathlib import Path

import pytest

from llm_sentinel.session_log import SessionLog, ConnectionEvent, _detect_phases, _most_common


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path):
    """Fresh in-memory-ish SessionLog using a temp file."""
    return SessionLog(db_path=tmp_path / "test.db")


@pytest.fixture
def db2(tmp_path):
    """Second SessionLog pointing at the same DB (different session_id)."""
    return SessionLog(db_path=tmp_path / "test.db")


# ---------------------------------------------------------------------------
# Basic write / read
# ---------------------------------------------------------------------------

class TestRecordOpened:
    def test_returns_row_id(self, db):
        rid = db.record_opened(123, "ollama", "1.2.3.4", "example.com", 443, "tcp")
        assert isinstance(rid, int)
        assert rid > 0

    def test_event_is_opened(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        events = db.recent_events()
        assert any(e.id == rid and e.event == "opened" for e in events)

    def test_duration_is_null(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.duration_s is None

    def test_fields_stored_correctly(self, db):
        db.record_opened(42, "ollama", "8.8.8.8", "dns.google", 443, "tcp")
        ev = db.recent_events()[0]
        assert ev.pid == 42
        assert ev.process_name == "ollama"
        assert ev.remote_ip == "8.8.8.8"
        assert ev.hostname == "dns.google"
        assert ev.port == 443
        assert ev.protocol == "tcp"


class TestRecordSeen:
    def test_updates_last_seen(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        before = db.recent_events()[0].last_seen
        time.sleep(0.05)
        db.record_seen(rid)
        after = db.recent_events()[0].last_seen
        assert after > before

    def test_does_not_change_event_to_seen(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        db.record_seen(rid)
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.event == "opened"  # must NOT become "seen"

    def test_multiple_seen_calls_update_last_seen(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        for _ in range(3):
            time.sleep(0.02)
            db.record_seen(rid)
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.last_seen > ev.ts


class TestRecordClosed:
    def test_sets_event_to_closed(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        db.record_closed(rid)
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.event == "closed"

    def test_fills_duration_s(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.05)
        db.record_closed(rid)
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.duration_s is not None
        assert ev.duration_s >= 0.04

    def test_noop_on_unknown_id(self, db):
        db.record_closed(99999)  # should not raise


class TestUpdateHostname:
    def test_updates_when_hostname_equals_ip(self, db):
        rid = db.record_opened(1, "p", "1.2.3.4", "1.2.3.4", 443, "tcp")
        db.update_hostname(rid, "resolved.example.com")
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.hostname == "resolved.example.com"

    def test_does_not_overwrite_existing_hostname(self, db):
        rid = db.record_opened(1, "p", "1.2.3.4", "already.resolved.com", 443, "tcp")
        db.update_hostname(rid, "something.else.com")
        ev = next(e for e in db.recent_events() if e.id == rid)
        assert ev.hostname == "already.resolved.com"


# ---------------------------------------------------------------------------
# Multi-session listing
# ---------------------------------------------------------------------------

class TestListSessions:
    def test_single_session(self, db):
        db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        sessions = db.list_sessions()
        assert len(sessions) == 1
        assert sessions[0]["is_current"] is True

    def test_multiple_sessions_ordered_newest_first(self, tmp_path):
        db_path = tmp_path / "multi.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.05)
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "q", "2.2.2.2", "h2", 80, "tcp")

        sessions = s2.list_sessions()
        assert len(sessions) == 2
        assert sessions[0]["started_at"] >= sessions[1]["started_at"]

    def test_current_session_flagged(self, tmp_path):
        db_path = tmp_path / "multi.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "q", "2.2.2.2", "h2", 80, "tcp")

        sessions = s2.list_sessions()
        current = [s for s in sessions if s["is_current"]]
        not_current = [s for s in sessions if not s["is_current"]]
        assert len(current) == 1
        assert len(not_current) == 1
        assert current[0]["session_id"] == s2.session_id

    def test_conn_count_per_session(self, tmp_path):
        db_path = tmp_path / "multi.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        s1.record_opened(1, "p", "2.2.2.2", "h2", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(1, "p", "3.3.3.3", "h3", 443, "tcp")

        sessions = s2.list_sessions()
        by_id = {s["session_id"]: s for s in sessions}
        assert by_id[s1.session_id]["conn_count"] == 2
        assert by_id[s2.session_id]["conn_count"] == 1

    def test_unique_ips_per_session(self, tmp_path):
        db_path = tmp_path / "multi.db"
        s = SessionLog(db_path=db_path)
        s.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        s.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")  # same IP again
        s.record_opened(1, "p", "2.2.2.2", "h2", 443, "tcp")

        sessions = s.list_sessions()
        assert sessions[0]["unique_ips"] == 2

    def test_empty_db_returns_empty_list(self, db):
        assert db.list_sessions() == []


# ---------------------------------------------------------------------------
# get_session_events
# ---------------------------------------------------------------------------

class TestGetSessionEvents:
    def test_returns_events_for_given_session(self, tmp_path):
        db_path = tmp_path / "s.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "q", "2.2.2.2", "h2", 80, "tcp")

        evts = s2.get_session_events(s1.session_id)
        assert len(evts) == 1
        assert evts[0].remote_ip == "1.1.1.1"

    def test_ordered_oldest_first(self, db):
        db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.02)
        db.record_opened(1, "p", "2.2.2.2", "h2", 443, "tcp")
        evts = db.get_session_events(db.session_id)
        assert evts[0].ts <= evts[1].ts

    def test_unknown_session_returns_empty(self, db):
        assert db.get_session_events("nonexistent_session") == []


# ---------------------------------------------------------------------------
# recent_events
# ---------------------------------------------------------------------------

class TestRecentEvents:
    def test_respects_limit(self, db):
        for i in range(10):
            db.record_opened(i, "p", f"1.1.1.{i}", "h", 443, "tcp")
        assert len(db.recent_events(limit=5)) == 5

    def test_newest_first(self, db):
        db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.02)
        db.record_opened(2, "p", "2.2.2.2", "h", 443, "tcp")
        evts = db.recent_events()
        assert evts[0].ts >= evts[1].ts

    def test_only_current_session(self, tmp_path):
        db_path = tmp_path / "s.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "q", "9.9.9.9", "h2", 443, "tcp")

        evts = s2.recent_events()
        assert all(e.session_id == s2.session_id for e in evts)
        assert all(e.remote_ip == "9.9.9.9" for e in evts)


# ---------------------------------------------------------------------------
# insights
# ---------------------------------------------------------------------------

class TestInsights:
    def test_empty_session_returns_empty_dict(self, db):
        assert db.insights() == {}

    def test_basic_stats(self, db):
        db.record_opened(1, "ollama", "1.1.1.1", "Cloudflare", 443, "tcp")
        db.record_opened(1, "ollama", "2.2.2.2", "Google", 443, "tcp")
        ins = db.insights()
        assert ins["total_connections"] == 2
        # unique_hosts = union of unique IPs + unique hostnames
        assert ins["unique_hosts"] >= 2

    def test_most_contacted(self, db):
        db.record_opened(1, "p", "1.1.1.1", "Cloudflare", 443, "tcp")
        db.record_opened(1, "p", "1.1.1.1", "Cloudflare", 443, "tcp")
        db.record_opened(1, "p", "2.2.2.2", "Google", 443, "tcp")
        ins = db.insights()
        assert ins["most_contacted"] == "Cloudflare"

    def test_longest_connection(self, db):
        rid = db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.05)
        db.record_closed(rid)
        ins = db.insights()
        assert ins["longest"] is not None
        assert ins["longest"].duration_s >= 0.04

    def test_session_age_grows(self, db):
        db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.05)
        ins = db.insights()
        assert ins["session_age_s"] >= 0.04

    def test_phases_present(self, db):
        db.record_opened(1, "p", "1.1.1.1", "h", 443, "tcp")
        ins = db.insights()
        assert "phases" in ins
        assert isinstance(ins["phases"], list)


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

class TestDetectPhases:
    def _make_event(self, ts, ip="1.1.1.1"):
        return ConnectionEvent(
            id=1, session_id="s", event="opened",
            ts=ts, last_seen=ts,
            pid=1, process_name="p",
            remote_ip=ip, hostname="h",
            port=443, protocol="tcp", duration_s=None,
        )

    def test_empty_returns_empty(self):
        assert _detect_phases([]) == []

    def test_startup_phase_within_30s(self):
        now = time.time()
        events = [self._make_event(now + i) for i in range(5)]
        phases = _detect_phases(events)
        names = [p["name"] for p in phases]
        assert "startup" in names

    def test_no_startup_phase_for_late_events(self):
        # First event at t=0, second at t=35 — only the second appears after 30s
        # gap from session start. Startup only triggers for events in first 30s.
        now = time.time()
        e_early = self._make_event(now)
        e_late = self._make_event(now + 35)
        phases = _detect_phases([e_early, e_late])
        startup = [p for p in phases if p["name"] == "startup"]
        # startup phase should only count early events
        assert startup[0]["count"] == 1

    def test_download_phase_on_burst(self):
        now = time.time()
        # 5 events in the same second
        events = [self._make_event(now, ip=f"1.1.1.{i}") for i in range(5)]
        phases = _detect_phases(events)
        names = [p["name"] for p in phases]
        assert "download" in names

    def test_inference_phase_on_gap(self):
        now = time.time()
        e1 = self._make_event(now)
        e1.last_seen = now + 1
        e2 = self._make_event(now + 120)  # 120s gap
        phases = _detect_phases([e1, e2])
        names = [p["name"] for p in phases]
        assert "inference" in names

    def test_phases_sorted_by_time(self):
        now = time.time()
        e1 = self._make_event(now)
        e1.last_seen = now + 1
        e2 = self._make_event(now + 120)
        phases = _detect_phases([e1, e2])
        for i in range(len(phases) - 1):
            assert phases[i]["ts"] <= phases[i + 1]["ts"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TestMostCommon:
    def test_returns_most_frequent(self):
        assert _most_common(["a", "b", "a", "a", "b"]) == "a"

    def test_single_item(self):
        assert _most_common(["x"]) == "x"

    def test_empty_returns_none(self):
        assert _most_common([]) is None


# ---------------------------------------------------------------------------
# ConnectionEvent helpers
# ---------------------------------------------------------------------------

class TestConnectionEvent:
    def _ev(self, **kwargs):
        defaults = dict(
            id=1, session_id="s", event="opened",
            ts=time.time(), last_seen=time.time(),
            pid=1, process_name="p",
            remote_ip="1.1.1.1", hostname="h",
            port=443, protocol="tcp", duration_s=None,
        )
        defaults.update(kwargs)
        return ConnectionEvent(**defaults)

    def test_fmt_duration_seconds(self):
        ev = self._ev(duration_s=45.0)
        assert ev.fmt_duration() == "45s"

    def test_fmt_duration_minutes(self):
        ev = self._ev(duration_s=90.0)
        assert "m" in ev.fmt_duration()

    def test_fmt_duration_hours(self):
        ev = self._ev(duration_s=7200.0)
        assert "h" in ev.fmt_duration()

    def test_fmt_duration_uses_active_for_s_when_no_duration(self):
        now = time.time()
        ev = self._ev(ts=now - 10, last_seen=now, duration_s=None)
        result = ev.fmt_duration()
        assert "s" in result or "m" in result
