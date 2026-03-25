"""
Tests for the web dashboard API endpoints.
Uses FastAPI's TestClient — no real server needed.
"""

import time
import pytest
from fastapi.testclient import TestClient

from llm_sentinel.web import app, update_state, _state, _lock
from llm_sentinel.session_log import SessionLog


@pytest.fixture(autouse=True)
def clear_state():
    """Reset shared web state between tests."""
    with _lock:
        _state.clear()
    yield
    with _lock:
        _state.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sl(tmp_path):
    return SessionLog(db_path=tmp_path / "test.db")


def _push_state(sl, processes=None, scan=1):
    """Helper: push minimal state to the web module."""
    update_state(processes or [], sl, scan, 3.0)


# ---------------------------------------------------------------------------
# GET /api/state
# ---------------------------------------------------------------------------

class TestGetState:
    def test_empty_state_returns_200(self, client):
        r = client.get("/api/state")
        assert r.status_code == 200

    def test_empty_state_has_scan_zero(self, client):
        r = client.get("/api/state")
        assert r.json().get("scan") == 0

    def test_session_log_not_in_response(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert "_session_log" not in data

    def test_alert_false_when_no_external(self, client, sl):
        _push_state(sl)
        assert client.get("/api/state").json()["alert"] is False

    def test_scan_count_increments(self, client, sl):
        _push_state(sl, scan=5)
        assert client.get("/api/state").json()["scan"] == 5

    def test_updated_at_present(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert "updated_at" in data

    def test_processes_list(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert isinstance(data["processes"], list)

    def test_external_list(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert isinstance(data["external"], list)

    def test_history_list(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert isinstance(data["history"], list)

    def test_insights_dict(self, client, sl):
        _push_state(sl)
        data = client.get("/api/state").json()
        assert isinstance(data["insights"], dict)


# ---------------------------------------------------------------------------
# GET /api/sessions
# ---------------------------------------------------------------------------

class TestGetSessions:
    def test_no_state_returns_empty_list(self, client):
        r = client.get("/api/sessions")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_session_list(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        sessions = client.get("/api/sessions").json()
        assert len(sessions) == 1

    def test_session_has_required_fields(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        s = client.get("/api/sessions").json()[0]
        for field in ("id", "started", "duration", "conn_count", "unique_ips",
                      "processes", "is_current"):
            assert field in s, f"missing field: {field}"

    def test_current_session_flagged(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        sessions = client.get("/api/sessions").json()
        assert any(s["is_current"] for s in sessions)

    def test_multiple_sessions(self, client, tmp_path):
        db_path = tmp_path / "s.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "ollama", "2.2.2.2", "h2", 443, "tcp")
        _push_state(s2)
        sessions = client.get("/api/sessions").json()
        assert len(sessions) == 2

    def test_newest_session_first(self, client, tmp_path):
        db_path = tmp_path / "s.db"
        s1 = SessionLog(db_path=db_path)
        s1.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        s2 = SessionLog(db_path=db_path)
        s2.record_opened(2, "ollama", "2.2.2.2", "h2", 443, "tcp")
        _push_state(s2)
        sessions = client.get("/api/sessions").json()
        # current session (s2) should be first
        assert sessions[0]["is_current"] is True

    def test_conn_count_in_response(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        sl.record_opened(1, "ollama", "2.2.2.2", "h2", 443, "tcp")
        _push_state(sl)
        sessions = client.get("/api/sessions").json()
        assert sessions[0]["conn_count"] == 2

    def test_processes_field_contains_name(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        sessions = client.get("/api/sessions").json()
        assert "ollama" in sessions[0]["processes"]


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}
# ---------------------------------------------------------------------------

class TestGetSessionDetail:
    def test_unknown_session_returns_empty(self, client, sl):
        _push_state(sl)
        r = client.get("/api/sessions/nonexistent_session")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_events_for_session(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        assert len(events) == 1

    def test_event_has_required_fields(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "example.com", 443, "tcp")
        _push_state(sl)
        ev = client.get(f"/api/sessions/{sl.session_id}").json()[0]
        for field in ("time", "event", "process", "hostname", "ip", "port", "duration"):
            assert field in ev, f"missing field: {field}"

    def test_seen_events_excluded(self, client, sl):
        rid = sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        sl.record_seen(rid)
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        assert all(e["event"] != "seen" for e in events)

    def test_still_open_duration(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        assert events[0]["duration"] == "still open"

    def test_closed_event_has_duration(self, client, sl):
        rid = sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        time.sleep(0.05)
        sl.record_closed(rid)
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        ev = next(e for e in events if e["event"] == "closed")
        assert ev["duration"] != "still open"
        assert ev["duration"] != "—"

    def test_event_field_open_for_still_open(self, client, sl):
        sl.record_opened(1, "ollama", "1.1.1.1", "h", 443, "tcp")
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        assert events[0]["event"] == "open"

    def test_hostname_shown(self, client, sl):
        sl.record_opened(1, "ollama", "34.36.133.15", "bc.googleusercontent.com", 443, "tcp")
        _push_state(sl)
        events = client.get(f"/api/sessions/{sl.session_id}").json()
        assert events[0]["hostname"] == "bc.googleusercontent.com"

    def test_no_state_returns_empty(self, client):
        r = client.get("/api/sessions/anything")
        assert r.status_code == 200
        assert r.json() == []


# ---------------------------------------------------------------------------
# GET / and /sessions pages
# ---------------------------------------------------------------------------

class TestPages:
    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "LLM Sentinel" in r.text

    def test_index_has_sessions_link(self, client):
        r = client.get("/")
        assert "/sessions" in r.text

    def test_sessions_page_returns_html(self, client):
        r = client.get("/sessions")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]
        assert "Session History" in r.text

    def test_sessions_page_has_back_link(self, client):
        r = client.get("/sessions")
        assert 'href="/"' in r.text
