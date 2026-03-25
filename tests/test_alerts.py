"""
Tests for llm_sentinel.alerts — AlertManager deduplication, ordering, trimming, logging.
"""

import time
import pytest
from unittest.mock import patch

from llm_sentinel.alerts import Alert, AlertManager


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:
    def test_key_is_pid_addr_port(self):
        a = Alert(pid=123, process_name="ollama", remote_addr="1.2.3.4",
                  remote_port=443, protocol="tcp")
        assert a.key == "123:1.2.3.4:443"

    def test_key_uniqueness(self):
        a1 = Alert(pid=1, process_name="x", remote_addr="1.1.1.1", remote_port=80, protocol="tcp")
        a2 = Alert(pid=1, process_name="x", remote_addr="1.1.1.1", remote_port=443, protocol="tcp")
        a3 = Alert(pid=2, process_name="x", remote_addr="1.1.1.1", remote_port=80, protocol="tcp")
        assert a1.key != a2.key
        assert a1.key != a3.key

    def test_formatted_time_looks_like_hhmmss(self):
        a = Alert(pid=1, process_name="x", remote_addr="1.1.1.1",
                  remote_port=80, protocol="tcp", timestamp=time.time())
        fmt = a.formatted_time()
        parts = fmt.split(":")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---------------------------------------------------------------------------
# AlertManager — basic recording
# ---------------------------------------------------------------------------

class TestAlertManagerRecord:
    def setup_method(self):
        self.am = AlertManager()

    def test_records_new_alert(self):
        alert = self.am.record(pid=1, process_name="ollama",
                               remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        assert alert.pid == 1
        assert alert.remote_addr == "8.8.8.8"
        assert self.am.total_count == 1

    def test_deduplicates_repeated_alert(self):
        for _ in range(5):
            self.am.record(pid=1, process_name="ollama",
                           remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        assert self.am.total_count == 1
        alerts = self.am.get_alerts()
        assert alerts[0].count == 5

    def test_separate_ports_are_distinct_alerts(self):
        self.am.record(pid=1, process_name="ollama",
                       remote_addr="8.8.8.8", remote_port=80, protocol="tcp")
        self.am.record(pid=1, process_name="ollama",
                       remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        assert self.am.total_count == 2

    def test_separate_pids_same_addr_are_distinct(self):
        self.am.record(pid=1, process_name="ollama",
                       remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        self.am.record(pid=2, process_name="vllm",
                       remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        assert self.am.total_count == 2

    def test_clear_resets_everything(self):
        self.am.record(pid=1, process_name="ollama",
                       remote_addr="8.8.8.8", remote_port=443, protocol="tcp")
        self.am.clear()
        assert self.am.total_count == 0
        assert self.am.get_alerts() == []


# ---------------------------------------------------------------------------
# AlertManager — ordering
# ---------------------------------------------------------------------------

class TestAlertManagerOrdering:
    def test_get_alerts_newest_first(self):
        am = AlertManager()
        am.record(pid=1, process_name="a", remote_addr="1.1.1.1", remote_port=80, protocol="tcp")
        am.record(pid=2, process_name="b", remote_addr="2.2.2.2", remote_port=80, protocol="tcp")
        am.record(pid=3, process_name="c", remote_addr="3.3.3.3", remote_port=80, protocol="tcp")
        alerts = am.get_alerts()
        pids = [a.pid for a in alerts]
        assert pids == [3, 2, 1]

    def test_repeated_alert_timestamp_updated(self):
        am = AlertManager()
        am.record(pid=1, process_name="ollama",
                  remote_addr="1.1.1.1", remote_port=80, protocol="tcp")
        t_before = am.get_alerts()[0].timestamp
        time.sleep(0.05)
        am.record(pid=1, process_name="ollama",
                  remote_addr="1.1.1.1", remote_port=80, protocol="tcp")
        t_after = am.get_alerts()[0].timestamp
        assert t_after >= t_before


# ---------------------------------------------------------------------------
# AlertManager — max_alerts trimming
# ---------------------------------------------------------------------------

class TestAlertManagerTrimming:
    def test_trims_oldest_when_over_limit(self):
        am = AlertManager(max_alerts=3)
        for i in range(5):
            am.record(pid=i, process_name="x",
                      remote_addr=f"{i}.{i}.{i}.{i}", remote_port=80, protocol="tcp")
        # Only 3 most recent should remain
        assert am.total_count == 3
        pids = {a.pid for a in am.get_alerts()}
        assert pids == {2, 3, 4}

    def test_exactly_at_limit_keeps_all(self):
        am = AlertManager(max_alerts=3)
        for i in range(3):
            am.record(pid=i, process_name="x",
                      remote_addr=f"{i}.{i}.{i}.{i}", remote_port=80, protocol="tcp")
        assert am.total_count == 3


# ---------------------------------------------------------------------------
# AlertManager — file logging
# ---------------------------------------------------------------------------

class TestAlertManagerLogging:
    def test_logs_to_file(self, tmp_path):
        log_file = str(tmp_path / "sentinel.log")
        am = AlertManager(log_to_file=log_file)
        am.record(pid=42, process_name="ollama",
                  remote_addr="8.8.8.8", remote_port=443, protocol="tcp")

        with open(log_file) as f:
            content = f.read()

        assert "EXTERNAL CONNECTION" in content
        assert "8.8.8.8" in content
        assert "ollama" in content

    def test_repeated_alerts_not_duplicated_in_log(self, tmp_path):
        log_file = str(tmp_path / "sentinel.log")
        am = AlertManager(log_to_file=log_file)
        for _ in range(3):
            am.record(pid=1, process_name="ollama",
                      remote_addr="8.8.8.8", remote_port=443, protocol="tcp")

        with open(log_file) as f:
            lines = [l for l in f.readlines() if "EXTERNAL CONNECTION" in l]
        # Only logged once (first occurrence)
        assert len(lines) == 1
