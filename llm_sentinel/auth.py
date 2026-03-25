"""
Authentication for the web dashboard.

Stores users and session tokens in the same SQLite DB as connection logs.
Passwords are hashed with scrypt (built-in Python 3.6+, memory-hard).
Session tokens are 32-byte URL-safe random strings with a configurable TTL.
"""

import hashlib
import os
import secrets
import sqlite3
import time
from typing import Optional

SESSION_TTL = 8 * 3600  # 8 hours

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    username      TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL,
    created_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS auth_sessions (
    token      TEXT PRIMARY KEY,
    username   TEXT NOT NULL,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL
);
"""


class AuthDB:
    """
    Manages users and browser session tokens.
    Shares the sentinel's existing sqlite3.Connection so everything
    lives in one file (~/.llm_sentinel/sessions.db).
    """

    def __init__(self, conn: sqlite3.Connection):
        self._conn = conn
        conn.executescript(_SCHEMA)
        conn.commit()

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    def has_users(self) -> bool:
        return bool(self._conn.execute("SELECT 1 FROM users LIMIT 1").fetchone())

    def create_user(self, username: str, password: str) -> None:
        """Create or replace a user with a freshly hashed password."""
        now = time.time()
        self._conn.execute(
            "INSERT OR REPLACE INTO users (username, password_hash, created_at)"
            " VALUES (?, ?, ?)",
            (username, _hash_password(password), now),
        )
        self._conn.commit()

    def verify_user(self, username: str, password: str) -> bool:
        row = self._conn.execute(
            "SELECT password_hash FROM users WHERE username=?", (username,)
        ).fetchone()
        if not row:
            return False
        return _verify_password(password, row[0])

    # ------------------------------------------------------------------
    # Session tokens
    # ------------------------------------------------------------------

    def create_session(self, username: str) -> str:
        """Mint a new session token, persist it, and return it."""
        token = secrets.token_urlsafe(32)
        now = time.time()
        self._conn.execute(
            "INSERT INTO auth_sessions (token, username, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (token, username, now, now + SESSION_TTL),
        )
        self._conn.commit()
        return token

    def validate_session(self, token: str) -> Optional[str]:
        """Return the username if the token exists and has not expired, else None."""
        if not token:
            return None
        row = self._conn.execute(
            "SELECT username, expires_at FROM auth_sessions WHERE token=?",
            (token,),
        ).fetchone()
        if not row:
            return None
        if time.time() > row["expires_at"]:
            self.revoke_session(token)
            return None
        return row["username"]

    def revoke_session(self, token: str) -> None:
        self._conn.execute("DELETE FROM auth_sessions WHERE token=?", (token,))
        self._conn.commit()

    def purge_expired(self) -> None:
        """Delete all expired tokens — call occasionally to keep the table small."""
        self._conn.execute("DELETE FROM auth_sessions WHERE expires_at < ?", (time.time(),))
        self._conn.commit()


# ------------------------------------------------------------------
# Password hashing (scrypt, no third-party deps)
# ------------------------------------------------------------------

def _hash_password(password: str) -> str:
    salt = os.urandom(16)
    key = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
    return salt.hex() + ":" + key.hex()


def _verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, key_hex = stored.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(key_hex)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1)
        return secrets.compare_digest(actual, expected)
    except Exception:
        return False
