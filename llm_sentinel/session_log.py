"""
Persistent SQLite session log.

Every external connection lifecycle is recorded:
  - 'opened'  : first time a (pid, remote_ip, port) is seen
  - 'closed'  : connection no longer present; duration_s filled in
  - 'seen'    : still active on a subsequent scan (heartbeat, updates last_seen)

The log survives across sentinel restarts. Default location: ~/.llm_sentinel/sessions.db
"""

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_DB = Path.home() / ".llm_sentinel" / "sessions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS connections (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    event        TEXT    NOT NULL,   -- 'opened' | 'closed' | 'seen'
    ts           REAL    NOT NULL,   -- unix timestamp of this event
    last_seen    REAL    NOT NULL,   -- updated on each 'seen'
    pid          INTEGER NOT NULL,
    process_name TEXT    NOT NULL,
    remote_ip    TEXT    NOT NULL,
    hostname     TEXT    NOT NULL,
    port         INTEGER NOT NULL,
    protocol     TEXT    NOT NULL,
    duration_s   REAL                -- NULL until closed
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_session  ON connections (session_id);
CREATE INDEX IF NOT EXISTS idx_ts       ON connections (ts);
CREATE INDEX IF NOT EXISTS idx_ip       ON connections (remote_ip);
"""


@dataclass
class ConnectionEvent:
    id: int
    session_id: str
    event: str
    ts: float
    last_seen: float
    pid: int
    process_name: str
    remote_ip: str
    hostname: str
    port: int
    protocol: str
    duration_s: Optional[float]

    def age_s(self) -> float:
        return time.time() - self.ts

    def active_for_s(self) -> float:
        return (self.last_seen if self.duration_s is None else self.ts + self.duration_s) - self.ts

    def fmt_duration(self) -> str:
        s = self.duration_s if self.duration_s is not None else self.active_for_s()
        if s < 60:
            return f"{s:.0f}s"
        if s < 3600:
            return f"{s/60:.1f}m"
        return f"{s/3600:.1f}h"


class SessionLog:
    def __init__(self, db_path: Path = DEFAULT_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._session_id = self._init_session()

    def _init_session(self) -> str:
        # Include nanosecond component so multiple instances in the same
        # process/second (e.g., during tests) get distinct session IDs.
        sid = f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{time.monotonic_ns() % 1_000_000_000}"
        self._conn.execute(
            "INSERT OR REPLACE INTO meta VALUES ('last_session', ?)", (sid,)
        )
        self._conn.commit()
        return sid

    @property
    def session_id(self) -> str:
        return self._session_id

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def record_opened(self, pid: int, process_name: str, remote_ip: str,
                      hostname: str, port: int, protocol: str) -> int:
        now = time.time()
        cur = self._conn.execute(
            """INSERT INTO connections
               (session_id, event, ts, last_seen, pid, process_name,
                remote_ip, hostname, port, protocol)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (self._session_id, "opened", now, now,
             pid, process_name, remote_ip, hostname, port, protocol),
        )
        self._conn.commit()
        return cur.lastrowid

    def record_seen(self, row_id: int) -> None:
        # Only update last_seen — preserve the original event type ('opened')
        self._conn.execute(
            "UPDATE connections SET last_seen=? WHERE id=?",
            (time.time(), row_id),
        )
        self._conn.commit()

    def record_closed(self, row_id: int) -> None:
        now = time.time()
        row = self._conn.execute(
            "SELECT ts FROM connections WHERE id=?", (row_id,)
        ).fetchone()
        if row:
            duration = now - row["ts"]
            self._conn.execute(
                "UPDATE connections SET event='closed', last_seen=?, duration_s=? WHERE id=?",
                (now, duration, row_id),
            )
            self._conn.commit()

    def update_hostname(self, row_id: int, hostname: str) -> None:
        self._conn.execute(
            "UPDATE connections SET hostname=? WHERE id=? AND hostname=remote_ip",
            (hostname, row_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def recent_events(self, limit: int = 30) -> list[ConnectionEvent]:
        rows = self._conn.execute(
            """SELECT * FROM connections
               WHERE session_id=?
               ORDER BY ts DESC LIMIT ?""",
            (self._session_id, limit),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def all_session_events(self) -> list[ConnectionEvent]:
        rows = self._conn.execute(
            "SELECT * FROM connections WHERE session_id=? ORDER BY ts",
            (self._session_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def unique_destinations(self, session_only: bool = True) -> list[dict]:
        """Return unique (remote_ip, hostname, port) with first_seen, last_seen, hit_count."""
        where = "WHERE session_id=?" if session_only else ""
        params = (self._session_id,) if session_only else ()
        rows = self._conn.execute(
            f"""SELECT remote_ip, hostname, port, protocol,
                       MIN(ts) as first_seen, MAX(last_seen) as last_seen,
                       COUNT(*) as hit_count,
                       SUM(CASE WHEN event='closed' THEN duration_s ELSE 0 END) as total_s
                FROM connections {where}
                GROUP BY remote_ip, port
                ORDER BY first_seen DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def insights(self) -> dict:
        """Compute summary stats for the current session."""
        events = self.all_session_events()
        if not events:
            return {}

        unique_ips = {e.remote_ip for e in events}
        unique_hosts = {e.hostname for e in events if e.hostname != e.remote_ip}
        total_connections = len(events)
        closed = [e for e in events if e.event == "closed"]
        avg_duration = (
            sum(e.duration_s for e in closed) / len(closed) if closed else None
        )
        longest = max(closed, key=lambda e: e.duration_s, default=None)
        most_active_proc = _most_common(e.process_name for e in events)
        most_contacted = _most_common(e.hostname or e.remote_ip for e in events)

        # Detect phases based on connection patterns
        phases = _detect_phases(events)

        session_start = min(e.ts for e in events)
        session_age_s = time.time() - session_start

        return {
            "unique_ips": len(unique_ips),
            "unique_hosts": len(unique_hosts | unique_ips),
            "total_connections": total_connections,
            "avg_duration_s": avg_duration,
            "longest": longest,
            "most_active_proc": most_active_proc,
            "most_contacted": most_contacted,
            "phases": phases,
            "session_age_s": session_age_s,
        }

    def list_sessions(self) -> list[dict]:
        """Return summary of all sessions, newest first."""
        rows = self._conn.execute("""
            SELECT
                session_id,
                MIN(ts)        AS started_at,
                MAX(last_seen) AS last_active,
                COUNT(*)       AS conn_count,
                COUNT(DISTINCT remote_ip) AS unique_ips,
                GROUP_CONCAT(DISTINCT process_name) AS processes
            FROM connections
            GROUP BY session_id
            ORDER BY started_at DESC
        """).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["is_current"] = d["session_id"] == self._session_id
            result.append(d)
        return result

    def get_session_events(self, session_id: str) -> list[ConnectionEvent]:
        """Return all events for a given session, oldest first."""
        rows = self._conn.execute(
            "SELECT * FROM connections WHERE session_id=? ORDER BY ts",
            (session_id,),
        ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def close(self):
        self._conn.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> ConnectionEvent:
        return ConnectionEvent(
            id=row["id"],
            session_id=row["session_id"],
            event=row["event"],
            ts=row["ts"],
            last_seen=row["last_seen"],
            pid=row["pid"],
            process_name=row["process_name"],
            remote_ip=row["remote_ip"],
            hostname=row["hostname"],
            port=row["port"],
            protocol=row["protocol"],
            duration_s=row["duration_s"],
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _most_common(iterable) -> Optional[str]:
    counts: dict[str, int] = {}
    for item in iterable:
        counts[item] = counts.get(item, 0) + 1
    return max(counts, key=counts.get) if counts else None


def _detect_phases(events: list[ConnectionEvent]) -> list[dict]:
    """
    Infer behavioral phases from the connection timeline:
      - 'startup'    : connections in first 30s
      - 'download'   : burst of connections to CDN (>3 simultaneous)
      - 'idle'       : low connection rate, single persistent keep-alive
      - 'inference'  : no external connections (connections closed)
    """
    if not events:
        return []

    phases = []
    session_start = min(e.ts for e in events)

    # Startup phase
    startup = [e for e in events if e.ts - session_start < 30]
    if startup:
        phases.append({
            "name": "startup",
            "label": "Startup / update check",
            "ts": session_start,
            "count": len(startup),
        })

    # Download phase: multiple CDN connections opened in a short window
    by_second: dict[int, list] = {}
    for e in events:
        bucket = int(e.ts)
        by_second.setdefault(bucket, []).append(e)
    bursts = [(t, evs) for t, evs in by_second.items() if len(evs) >= 3]
    if bursts:
        burst_ts = min(t for t, _ in bursts)
        burst_count = sum(len(evs) for _, evs in bursts)
        phases.append({
            "name": "download",
            "label": "Model download (CDN burst)",
            "ts": float(burst_ts),
            "count": burst_count,
        })

    # Inference phase: gaps where no new connections were opened
    if len(events) >= 2:
        sorted_events = sorted(events, key=lambda e: e.ts)
        for i in range(1, len(sorted_events)):
            gap = sorted_events[i].ts - sorted_events[i - 1].last_seen
            if gap > 60:
                phases.append({
                    "name": "inference",
                    "label": "Local inference (no external traffic)",
                    "ts": sorted_events[i - 1].last_seen,
                    "count": 0,
                })
                break

    return sorted(phases, key=lambda p: p["ts"])
