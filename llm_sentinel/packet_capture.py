"""
TLS SNI capture via packet sniffing.

Parses TLS ClientHello packets to extract the Server Name Indication (SNI) —
the hostname the process is connecting to, visible in plaintext before encryption.

Correlates SNI with PIDs by matching the packet's source port against psutil's
connection table (which maps source ports → PIDs).

Requires root/sudo for pcap access on macOS/Linux.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

_available = False
try:
    import scapy.all as scapy
    _available = True
except ImportError:
    pass


@dataclass
class SNIEvent:
    ts: float
    pid: Optional[int]
    process_name: str
    src_port: int
    dst_ip: str
    dst_port: int
    sni: str                    # actual hostname from TLS ClientHello
    tls_version: str = "TLS"

    def formatted_time(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.ts))

    @property
    def url_hint(self) -> str:
        """Best guess at the URL base."""
        proto = "https" if self.dst_port == 443 else f":{self.dst_port}"
        return f"{proto}://{self.sni}"


class SNICapture:
    """
    Background packet sniffer that extracts TLS SNI from ClientHello packets
    and correlates them with PIDs via psutil's connection table.
    """

    def __init__(self, max_events: int = 200):
        self._events: deque[SNIEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        # Cache: (src_ip, src_port) -> (pid, process_name)
        # Refreshed from psutil on each packet
        self._port_map: dict[tuple, tuple] = {}
        self._port_map_lock = threading.Lock()

    # ------------------------------------------------------------------
    # TLS ClientHello parser
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_sni(data: bytes) -> Optional[tuple[str, str]]:
        """
        Parse a TLS ClientHello and return (sni, tls_version) or None.

        TLS Record layout:
          [0]     content_type  (0x16 = handshake)
          [1-2]   legacy_version
          [3-4]   record_length
          [5]     handshake_type (0x01 = ClientHello)
          [6-8]   handshake_length
          [9-10]  client_version
          [11-42] random (32 bytes)
          [43]    session_id_len
          ...     session_id
          ...     cipher_suites_len (2 bytes)
          ...     cipher_suites
          ...     compression_methods_len (1 byte)
          ...     compression_methods
          ...     extensions_len (2 bytes)
          ...     extensions[]
        """
        try:
            if len(data) < 6:
                return None
            if data[0] != 0x16:           # not TLS handshake
                return None
            if data[5] != 0x01:           # not ClientHello
                return None

            tls_ver_map = {
                (0x03, 0x01): "TLS 1.0",
                (0x03, 0x02): "TLS 1.1",
                (0x03, 0x03): "TLS 1.2",
                (0x03, 0x04): "TLS 1.3",
            }
            tls_ver = tls_ver_map.get((data[1], data[2]), "TLS")

            pos = 43                       # after fixed header + random

            # Skip session ID
            if pos >= len(data):
                return None
            sid_len = data[pos]
            pos += 1 + sid_len

            # Skip cipher suites
            if pos + 2 > len(data):
                return None
            cs_len = int.from_bytes(data[pos:pos+2], "big")
            pos += 2 + cs_len

            # Skip compression methods
            if pos >= len(data):
                return None
            cm_len = data[pos]
            pos += 1 + cm_len

            # Extensions
            if pos + 2 > len(data):
                return None
            ext_total = int.from_bytes(data[pos:pos+2], "big")
            pos += 2
            ext_end = pos + ext_total

            sni_found = None
            while pos + 4 <= ext_end and pos + 4 <= len(data):
                ext_type = int.from_bytes(data[pos:pos+2], "big")
                ext_len  = int.from_bytes(data[pos+2:pos+4], "big")
                pos += 4

                if ext_type == 0x0000:     # SNI extension
                    # SNI list length (2) + name_type (1) + name_length (2) + name
                    if pos + 5 > len(data):
                        break
                    name_type = data[pos+2]
                    if name_type == 0x00:  # host_name
                        name_len = int.from_bytes(data[pos+3:pos+5], "big")
                        name = data[pos+5:pos+5+name_len].decode("ascii", errors="ignore")
                        sni_found = name

                elif ext_type == 0x002b:  # supported_versions (TLS 1.3 indicator)
                    # list_len (1) + versions (2 each)
                    if pos < len(data):
                        n = data[pos]
                        for i in range(n // 2):
                            off = pos + 1 + i * 2
                            if off + 2 <= len(data):
                                v = (data[off], data[off+1])
                                if v == (0x03, 0x04):
                                    tls_ver = "TLS 1.3"
                                    break

                pos += ext_len

            if sni_found:
                return sni_found, tls_ver

        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Port → PID mapping
    # ------------------------------------------------------------------

    def refresh_port_map(self, tracked_pids: list[int]):
        """Update the src_port → (pid, name) map from psutil connections."""
        import psutil
        new_map = {}
        for pid in tracked_pids:
            try:
                p = psutil.Process(pid)
                name = p.name()
                for conn in p.net_connections(kind="inet"):
                    if conn.laddr:
                        new_map[(conn.laddr.ip, conn.laddr.port)] = (pid, name)
                        new_map[("", conn.laddr.port)] = (pid, name)  # wildcard ip
            except Exception:
                pass
        with self._port_map_lock:
            self._port_map = new_map

    def _lookup_pid(self, src_port: int) -> tuple[Optional[int], str]:
        with self._port_map_lock:
            for key_ip in ("", "0.0.0.0", "::"):
                result = self._port_map.get((key_ip, src_port))
                if result:
                    return result
            # Try any IP
            for (ip, port), val in self._port_map.items():
                if port == src_port:
                    return val
        return None, "unknown"

    # ------------------------------------------------------------------
    # Packet handler
    # ------------------------------------------------------------------

    def _handle_packet(self, pkt):
        try:
            if not pkt.haslayer(scapy.TCP):
                return
            tcp = pkt[scapy.TCP]
            if not tcp.payload:
                return

            raw = bytes(tcp.payload)
            result = self._parse_sni(raw)
            if not result:
                return

            sni, tls_ver = result
            src_port = tcp.sport
            dst_port = tcp.dport

            # Destination IP
            if pkt.haslayer(scapy.IP):
                dst_ip = pkt[scapy.IP].dst
            elif pkt.haslayer(scapy.IPv6):
                dst_ip = pkt[scapy.IPv6].dst
            else:
                dst_ip = "?"

            pid, name = self._lookup_pid(src_port)

            event = SNIEvent(
                ts=time.time(),
                pid=pid,
                process_name=name,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                sni=sni,
                tls_version=tls_ver,
            )
            with self._lock:
                self._events.appendleft(event)

        except Exception:
            pass

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start(self, interface: str = None):
        if not _available:
            raise RuntimeError("scapy is not installed — run: pip install scapy")
        self._running = True
        self._thread = threading.Thread(
            target=self._sniff_loop,
            args=(interface,),
            daemon=True,
            name="sni-capture",
        )
        self._thread.start()

    def _sniff_loop(self, interface):
        # Filter for TCP packets to port 443 (HTTPS) and 80 (HTTP)
        bpf = "tcp and (dst port 443 or dst port 80 or dst port 8080 or dst port 11434)"
        try:
            scapy.sniff(
                iface=interface,
                filter=bpf,
                prn=self._handle_packet,
                store=False,
                stop_filter=lambda _: not self._running,
            )
        except Exception as e:
            self._running = False

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recent_events(self, limit: int = 50) -> list[SNIEvent]:
        with self._lock:
            return list(self._events)[:limit]

    def is_running(self) -> bool:
        return self._running and bool(self._thread and self._thread.is_alive())

    @property
    def available(self) -> bool:
        return _available


def is_root() -> bool:
    import os
    return os.geteuid() == 0
