"""
Monitors network connections for tracked LLM processes.
Classifies connections as local (safe) or external (flagged).
"""

import ipaddress
import psutil
from dataclasses import dataclass
from typing import Optional

# Private/loopback ranges — connections to these are considered safe
_LOCAL_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),     # loopback
    ipaddress.ip_network("::1/128"),           # IPv6 loopback
    ipaddress.ip_network("10.0.0.0/8"),        # private
    ipaddress.ip_network("172.16.0.0/12"),     # private
    ipaddress.ip_network("192.168.0.0/16"),    # private
    ipaddress.ip_network("169.254.0.0/16"),    # link-local
    ipaddress.ip_network("fc00::/7"),          # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),         # IPv6 link-local
]


@dataclass
class Connection:
    pid: int
    local_addr: str
    remote_addr: str
    remote_ip: str
    remote_port: int
    status: str
    is_external: bool
    protocol: str = "tcp"

    @property
    def risk_label(self) -> str:
        return "EXTERNAL" if self.is_external else "local"


def _is_external(ip_str: str) -> bool:
    """Return True if the IP is a routable (non-private) address."""
    if not ip_str or ip_str in ("", "0.0.0.0", "::", "*"):
        return False
    try:
        addr = ipaddress.ip_address(ip_str)
        if addr.is_loopback or addr.is_private or addr.is_link_local or addr.is_unspecified:
            return False
        for network in _LOCAL_NETWORKS:
            if addr in network:
                return False
        return True
    except ValueError:
        return False


def get_connections_for_pid(pid: int) -> list[Connection]:
    """Return all active network connections for a given PID."""
    connections = []
    try:
        proc = psutil.Process(pid)
        for conn in proc.net_connections(kind="inet"):
            if not conn.raddr:
                continue  # skip listening sockets with no remote

            remote_ip = conn.raddr.ip if conn.raddr else ""
            remote_port = conn.raddr.port if conn.raddr else 0
            local_str = f"{conn.laddr.ip}:{conn.laddr.port}" if conn.laddr else ""
            remote_str = f"{remote_ip}:{remote_port}" if remote_ip else ""

            connections.append(
                Connection(
                    pid=pid,
                    local_addr=local_str,
                    remote_addr=remote_str,
                    remote_ip=remote_ip,
                    remote_port=remote_port,
                    status=conn.status or "NONE",
                    is_external=_is_external(remote_ip),
                    protocol="tcp" if conn.type == 1 else "udp",
                )
            )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    return connections


def get_all_llm_connections(pids: list[int]) -> dict[int, list[Connection]]:
    """Return connections grouped by PID for a list of LLM process PIDs."""
    return {pid: get_connections_for_pid(pid) for pid in pids}
