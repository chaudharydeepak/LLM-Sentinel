"""
Tests for AuthDB: user management, password hashing, session tokens.
"""

import sqlite3
import time

import pytest

from llm_sentinel.auth import AuthDB, _hash_password, _verify_password, SESSION_TTL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:", check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture
def auth(conn):
    return AuthDB(conn)


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

class TestPasswordHashing:
    def test_hash_is_not_plaintext(self):
        h = _hash_password("secret")
        assert "secret" not in h

    def test_verify_correct_password(self):
        h = _hash_password("hunter2")
        assert _verify_password("hunter2", h) is True

    def test_verify_wrong_password(self):
        h = _hash_password("hunter2")
        assert _verify_password("wrong", h) is False

    def test_different_hashes_for_same_password(self):
        h1 = _hash_password("abc")
        h2 = _hash_password("abc")
        assert h1 != h2  # unique salt each time

    def test_both_verify_to_true(self):
        h1 = _hash_password("abc")
        h2 = _hash_password("abc")
        assert _verify_password("abc", h1) is True
        assert _verify_password("abc", h2) is True

    def test_verify_bad_stored_hash_returns_false(self):
        assert _verify_password("password", "notahash") is False

    def test_verify_empty_password(self):
        h = _hash_password("")
        assert _verify_password("", h) is True
        assert _verify_password("x", h) is False


# ---------------------------------------------------------------------------
# User management
# ---------------------------------------------------------------------------

class TestUsers:
    def test_has_users_false_initially(self, auth):
        assert auth.has_users() is False

    def test_has_users_true_after_create(self, auth):
        auth.create_user("admin", "pass")
        assert auth.has_users() is True

    def test_verify_user_correct(self, auth):
        auth.create_user("alice", "secret")
        assert auth.verify_user("alice", "secret") is True

    def test_verify_user_wrong_password(self, auth):
        auth.create_user("alice", "secret")
        assert auth.verify_user("alice", "wrong") is False

    def test_verify_user_unknown_username(self, auth):
        assert auth.verify_user("nobody", "pass") is False

    def test_create_user_replaces_existing(self, auth):
        auth.create_user("admin", "old")
        auth.create_user("admin", "new")
        assert auth.verify_user("admin", "new") is True
        assert auth.verify_user("admin", "old") is False

    def test_multiple_users(self, auth):
        auth.create_user("alice", "a1")
        auth.create_user("bob", "b2")
        assert auth.verify_user("alice", "a1") is True
        assert auth.verify_user("bob", "b2") is True
        assert auth.verify_user("alice", "b2") is False


# ---------------------------------------------------------------------------
# Session tokens
# ---------------------------------------------------------------------------

class TestSessions:
    def test_create_session_returns_token(self, auth):
        auth.create_user("admin", "pass")
        token = auth.create_session("admin")
        assert isinstance(token, str)
        assert len(token) > 20

    def test_validate_session_returns_username(self, auth):
        auth.create_user("admin", "pass")
        token = auth.create_session("admin")
        assert auth.validate_session(token) == "admin"

    def test_validate_empty_token_returns_none(self, auth):
        assert auth.validate_session("") is None

    def test_validate_unknown_token_returns_none(self, auth):
        assert auth.validate_session("not_a_real_token") is None

    def test_revoke_session_invalidates_token(self, auth):
        auth.create_user("admin", "pass")
        token = auth.create_session("admin")
        auth.revoke_session(token)
        assert auth.validate_session(token) is None

    def test_revoke_nonexistent_token_noop(self, auth):
        auth.revoke_session("fake_token")  # should not raise

    def test_multiple_sessions_for_same_user(self, auth):
        auth.create_user("admin", "pass")
        t1 = auth.create_session("admin")
        t2 = auth.create_session("admin")
        assert t1 != t2
        assert auth.validate_session(t1) == "admin"
        assert auth.validate_session(t2) == "admin"

    def test_tokens_are_unique(self, auth):
        auth.create_user("admin", "pass")
        tokens = {auth.create_session("admin") for _ in range(20)}
        assert len(tokens) == 20

    def test_expired_session_returns_none(self, auth, conn):
        # Insert a token that expired in the past
        conn.execute(
            "INSERT INTO auth_sessions (token, username, created_at, expires_at) VALUES (?,?,?,?)",
            ("expired_token", "admin", time.time() - 100, time.time() - 1),
        )
        conn.commit()
        assert auth.validate_session("expired_token") is None

    def test_expired_session_is_purged_on_validate(self, auth, conn):
        conn.execute(
            "INSERT INTO auth_sessions (token, username, created_at, expires_at) VALUES (?,?,?,?)",
            ("old_token", "admin", time.time() - 100, time.time() - 1),
        )
        conn.commit()
        auth.validate_session("old_token")
        row = conn.execute("SELECT 1 FROM auth_sessions WHERE token='old_token'").fetchone()
        assert row is None

    def test_purge_expired_removes_old_tokens(self, auth, conn):
        conn.execute(
            "INSERT INTO auth_sessions (token, username, created_at, expires_at) VALUES (?,?,?,?)",
            ("tok1", "admin", time.time() - 200, time.time() - 100),
        )
        conn.commit()
        auth.purge_expired()
        row = conn.execute("SELECT 1 FROM auth_sessions WHERE token='tok1'").fetchone()
        assert row is None

    def test_purge_expired_keeps_valid_tokens(self, auth):
        auth.create_user("admin", "pass")
        token = auth.create_session("admin")
        auth.purge_expired()
        assert auth.validate_session(token) == "admin"

    def test_session_ttl_is_reasonable(self):
        assert SESSION_TTL >= 3600  # at least 1 hour


# ---------------------------------------------------------------------------
# Schema isolation (tables exist after init)
# ---------------------------------------------------------------------------

class TestSchema:
    def test_users_table_exists(self, conn):
        AuthDB(conn)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "users" in tables

    def test_auth_sessions_table_exists(self, conn):
        AuthDB(conn)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "auth_sessions" in tables

    def test_multiple_init_calls_are_idempotent(self, conn):
        AuthDB(conn)
        AuthDB(conn)  # should not raise or corrupt


# ---------------------------------------------------------------------------
# Web middleware integration via TestClient
# ---------------------------------------------------------------------------

class TestWebAuth:
    """Test that the middleware actually blocks unauthenticated requests."""

    @pytest.fixture(autouse=True)
    def setup_auth_in_web(self, auth, monkeypatch):
        import llm_sentinel.web as web_mod
        monkeypatch.setattr(web_mod, "_auth", auth)
        auth.create_user("admin", "testpass")
        yield
        monkeypatch.setattr(web_mod, "_auth", None)

    def test_unauthenticated_page_redirects_to_login(self):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        client = TestClient(app, follow_redirects=False)
        r = client.get("/")
        assert r.status_code == 302
        assert "/login" in r.headers["location"]

    def test_unauthenticated_api_returns_401(self):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        client = TestClient(app, follow_redirects=False)
        r = client.get("/api/state")
        assert r.status_code == 401

    def test_login_page_accessible_without_auth(self):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        client = TestClient(app, follow_redirects=False)
        r = client.get("/login")
        assert r.status_code == 200

    def test_successful_login_sets_cookie(self, auth):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        client = TestClient(app, follow_redirects=False)
        r = client.post("/api/login", data={"username": "admin", "password": "testpass"})
        assert r.status_code == 302
        assert "sentinel_session" in r.cookies

    def test_failed_login_redirects_with_error(self):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        client = TestClient(app, follow_redirects=False)
        r = client.post("/api/login", data={"username": "admin", "password": "wrongpass"})
        assert r.status_code == 302
        assert "error" in r.headers["location"]

    def test_authenticated_request_allowed(self, auth):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        token = auth.create_session("admin")
        client = TestClient(app, follow_redirects=False)
        r = client.get("/api/state", cookies={"sentinel_session": token})
        assert r.status_code == 200

    def test_logout_clears_cookie(self, auth):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        token = auth.create_session("admin")
        client = TestClient(app, follow_redirects=False)
        r = client.get("/logout", cookies={"sentinel_session": token})
        assert r.status_code == 302
        # Cookie should be cleared (empty or deleted)
        assert client.cookies.get("sentinel_session") in (None, "")

    def test_logout_invalidates_token(self, auth):
        from fastapi.testclient import TestClient
        from llm_sentinel.web import app
        token = auth.create_session("admin")
        client = TestClient(app, follow_redirects=False)
        client.get("/logout", cookies={"sentinel_session": token})
        assert auth.validate_session(token) is None
