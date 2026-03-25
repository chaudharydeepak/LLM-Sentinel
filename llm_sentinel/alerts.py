"""
Alert management — tracks, deduplicates, and logs alerts for external connections.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("llm_sentinel")


@dataclass
class Alert:
    pid: int
    process_name: str
    remote_addr: str
    remote_port: int
    protocol: str
    timestamp: float = field(default_factory=time.time)
    count: int = 1

    @property
    def key(self) -> str:
        return f"{self.pid}:{self.remote_addr}:{self.remote_port}"

    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))


class AlertManager:
    """
    Tracks alerts, deduplicates repeated ones, and logs them.
    Keeps the last `max_alerts` unique alerts in memory.
    """

    def __init__(self, max_alerts: int = 200, log_to_file: Optional[str] = None):
        self._seen: dict[str, Alert] = {}   # key -> Alert
        self._ordered: list[str] = []       # insertion-order keys
        self.max_alerts = max_alerts
        self._log_to_file = log_to_file

        if log_to_file:
            fh = logging.FileHandler(log_to_file)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(fh)
            logger.setLevel(logging.WARNING)

    def record(self, pid: int, process_name: str, remote_addr: str,
               remote_port: int, protocol: str) -> Alert:
        """Record an external connection alert. Returns the Alert object."""
        alert = Alert(
            pid=pid,
            process_name=process_name,
            remote_addr=remote_addr,
            remote_port=remote_port,
            protocol=protocol,
        )
        key = alert.key

        if key in self._seen:
            self._seen[key].count += 1
            self._seen[key].timestamp = alert.timestamp
            return self._seen[key]

        self._seen[key] = alert
        self._ordered.append(key)

        # Trim oldest if over limit
        if len(self._ordered) > self.max_alerts:
            oldest = self._ordered.pop(0)
            self._seen.pop(oldest, None)

        logger.warning(
            "EXTERNAL CONNECTION: pid=%d process=%s remote=%s:%d proto=%s",
            pid, process_name, remote_addr, remote_port, protocol,
        )
        return alert

    def get_alerts(self) -> list[Alert]:
        """Return alerts newest-first."""
        return [self._seen[k] for k in reversed(self._ordered) if k in self._seen]

    def clear(self):
        self._seen.clear()
        self._ordered.clear()

    @property
    def total_count(self) -> int:
        return len(self._seen)


