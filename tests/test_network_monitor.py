"""
Tests for llm_sentinel.network_monitor — IP classification and connection fetching.
"""

import socket
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import psutil

from llm_sentinel.network_monitor import (
    _is_external,
    get_connections_for_pid,
    get_all_llm_connections,
    Connection,
)


# ---------------------------------------------------------------------------
# _is_external — IP classification
# ---------------------------------------------------------------------------

class TestIsExternal:
    # -- Should return False (safe / local) --

    def test_loopback_v4(self):
        assert _is_external("127.0.0.1") is False

    def test_loopback_v4_any(self):
        assert _is_external("127.255.255.255") is False

    def test_loopback_v6(self):
        assert _is_external("::1") is False

    def test_private_10(self):
        assert _is_external("10.0.0.1") is False

    def test_private_172_16(self):
        assert _is_external("172.16.0.1") is False

    def test_private_172_31(self):
        assert _is_external("172.31.255.254") is False

    def test_private_192_168(self):
        assert _is_external("192.168.1.100") is False

    def test_link_local_v4(self):
        assert _is_external("169.254.1.1") is False

    def test_link_local_v6(self):
        assert _is_external("fe80::1") is False

    def test_ipv6_ula(self):
        assert _is_external("fd00::1") is False

    def test_unspecified_v4(self):
        assert _is_external("0.0.0.0") is False

    def test_unspecified_v6(self):
        assert _is_external("::") is False

    def test_empty_string(self):
        assert _is_external("") is False

    def test_wildcard_star(self):
        assert _is_external("*") is False

    # -- Should return True (external) --

    def test_google_dns(self):
        assert _is_external("8.8.8.8") is True

    def test_cloudflare_dns(self):
        assert _is_external("1.1.1.1") is True

    def test_public_ipv6(self):
        assert _is_external("2606:4700:4700::1111") is True

    def test_arbitrary_public(self):
        assert _is_external("52.86.200.100") is True

    def test_just_outside_172_range(self):
        # 172.15.x.x is NOT in the 172.16/12 range → external
        assert _is_external("172.15.0.1") is True

    def test_172_32_is_external(self):
        # 172.32.x.x is outside 172.16/12 → external
        assert _is_external("172.32.0.1") is True

    def test_invalid_ip_treated_as_not_external(self):
        assert _is_external("not-an-ip") is False


# ---------------------------------------------------------------------------
# Connection dataclass
# ---------------------------------------------------------------------------

class TestConnection:
    def test_risk_label_external(self):
        c = Connection(pid=1, local_addr="0.0.0.0:11434", remote_addr="8.8.8.8:443",
                       remote_ip="8.8.8.8", remote_port=443, status="ESTABLISHED",
                       is_external=True)
        assert c.risk_label == "EXTERNAL"

    def test_risk_label_local(self):
        c = Connection(pid=1, local_addr="127.0.0.1:11434", remote_addr="127.0.0.1:54321",
                       remote_ip="127.0.0.1", remote_port=54321, status="ESTABLISHED",
                       is_external=False)
        assert c.risk_label == "local"


# ---------------------------------------------------------------------------
# get_connections_for_pid — mocked psutil
# ---------------------------------------------------------------------------

def _make_sconn(laddr_ip, laddr_port, raddr_ip, raddr_port, status="ESTABLISHED", kind=1):
    """Build a mock psutil sconn named-tuple."""
    conn = MagicMock()
    conn.laddr = MagicMock()
    conn.laddr.ip = laddr_ip
    conn.laddr.port = laddr_port
    if raddr_ip:
        conn.raddr = MagicMock()
        conn.raddr.ip = raddr_ip
        conn.raddr.port = raddr_port
    else:
        conn.raddr = None
    conn.status = status
    conn.type = kind  # 1 = TCP, 2 = UDP
    return conn


class TestGetConnectionsForPid:
    def test_returns_empty_on_access_denied(self):
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.side_effect = psutil.AccessDenied(1)
            result = get_connections_for_pid(1)
        assert result == []

    def test_returns_empty_on_no_such_process(self):
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.side_effect = psutil.NoSuchProcess(1)
            result = get_connections_for_pid(1)
        assert result == []

    def test_skips_connections_without_remote(self):
        conn = _make_sconn("0.0.0.0", 11434, None, None)
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = [conn]
            result = get_connections_for_pid(1)
        assert result == []

    def test_local_connection_not_external(self):
        conn = _make_sconn("127.0.0.1", 11434, "127.0.0.1", 54321)
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = [conn]
            result = get_connections_for_pid(1)
        assert len(result) == 1
        assert result[0].is_external is False

    def test_external_connection_flagged(self):
        conn = _make_sconn("0.0.0.0", 11434, "8.8.8.8", 443)
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = [conn]
            result = get_connections_for_pid(1)
        assert len(result) == 1
        assert result[0].is_external is True
        assert result[0].remote_ip == "8.8.8.8"
        assert result[0].remote_port == 443

    def test_protocol_tcp(self):
        conn = _make_sconn("127.0.0.1", 11434, "192.168.1.1", 80, kind=1)
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = [conn]
            result = get_connections_for_pid(1)
        assert result[0].protocol == "tcp"

    def test_protocol_udp(self):
        conn = _make_sconn("127.0.0.1", 11434, "192.168.1.1", 53, kind=2)
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = [conn]
            result = get_connections_for_pid(1)
        assert result[0].protocol == "udp"

    def test_multiple_connections(self):
        conns = [
            _make_sconn("127.0.0.1", 11434, "127.0.0.1", 54321),  # local
            _make_sconn("0.0.0.0", 11434, "8.8.8.8", 443),         # external
            _make_sconn("0.0.0.0", 11434, "1.1.1.1", 80),          # external
        ]
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = conns
            result = get_connections_for_pid(1)
        assert len(result) == 3
        external = [c for c in result if c.is_external]
        assert len(external) == 2


# ---------------------------------------------------------------------------
# get_all_llm_connections
# ---------------------------------------------------------------------------

class TestGetAllLlmConnections:
    def test_returns_dict_keyed_by_pid(self):
        with patch("llm_sentinel.network_monitor.psutil.Process") as MockProc:
            MockProc.return_value.net_connections.return_value = []
            result = get_all_llm_connections([1, 2, 3])
        assert set(result.keys()) == {1, 2, 3}

    def test_empty_pids_returns_empty_dict(self):
        result = get_all_llm_connections([])
        assert result == {}
