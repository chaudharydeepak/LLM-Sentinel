"""
Non-blocking, cached resolver: reverse DNS + known-org fallback.
Lookups run in a background thread pool so they never stall the dashboard.
"""

import ipaddress
import socket
import threading
from concurrent.futures import ThreadPoolExecutor

_cache: dict[str, str] = {}
_pending: set[str] = set()
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dns")

# Well-known IP ranges → org label, checked when PTR lookup fails or returns raw IP.
# Ordered most-specific first.
_KNOWN_ORGS: list[tuple[str, str]] = [
    # Cloudflare
    ("2606:4700::/32",      "Cloudflare"),
    ("2606:4700:2ff9::1",   "Cloudflare"),   # specific anycast addr
    ("1.1.1.0/24",          "Cloudflare DNS"),
    ("1.0.0.0/24",          "Cloudflare DNS"),
    # Google
    ("2001:4860::/32",      "Google"),
    ("34.0.0.0/8",          "Google Cloud"),
    ("35.0.0.0/8",          "Google Cloud"),
    ("142.250.0.0/15",      "Google"),
    ("172.217.0.0/16",      "Google"),
    # AWS / CloudFront CDN
    ("2600:9000::/20",      "AWS CloudFront"),   # CloudFront IPv6
    ("2600:1f18::/36",      "AWS"),              # AWS IPv6
    ("13.32.0.0/15",        "AWS CloudFront"),
    ("13.35.0.0/16",        "AWS CloudFront"),
    ("52.0.0.0/8",          "AWS"),
    ("54.0.0.0/8",          "AWS"),
    ("3.0.0.0/8",           "AWS"),
    ("18.0.0.0/8",          "AWS"),
    ("13.0.0.0/8",          "AWS"),
    # Azure
    ("20.0.0.0/8",          "Azure"),
    ("40.0.0.0/8",          "Azure"),
    # GitHub
    ("2606:50c0::/32",      "GitHub"),
    ("140.82.112.0/20",     "GitHub"),
    ("185.199.108.0/22",    "GitHub"),
    ("192.30.252.0/22",     "GitHub"),
    # Hugging Face
    ("18.200.0.0/13",       "HuggingFace (AWS)"),
    # Meta
    ("157.240.0.0/16",      "Meta"),
]

_parsed_orgs: list[tuple[ipaddress._BaseNetwork, str]] = []

def _build_org_table():
    for cidr, label in _KNOWN_ORGS:
        try:
            _parsed_orgs.append((ipaddress.ip_network(cidr, strict=False), label))
        except ValueError:
            # Handle single IPs like "2606:4700:2ff9::1"
            try:
                addr = ipaddress.ip_address(cidr)
                bits = 128 if addr.version == 6 else 32
                _parsed_orgs.append((ipaddress.ip_network(f"{cidr}/{bits}", strict=False), label))
            except ValueError:
                pass

_build_org_table()


def _known_org(ip: str) -> str | None:
    try:
        addr = ipaddress.ip_address(ip)
        for net, label in _parsed_orgs:
            if addr in net:
                return label
    except ValueError:
        pass
    return None


def _resolve(ip: str) -> None:
    result = None
    try:
        host = socket.gethostbyaddr(ip)[0]
        # If the PTR record just echoes back the IP (common for CDN IPs), use org fallback
        if host != ip:
            result = host
    except Exception:
        pass

    if result is None:
        result = _known_org(ip) or ip

    with _lock:
        _cache[ip] = result
        _pending.discard(ip)


def hostname(ip: str) -> str:
    """
    Return the best human-readable label for an IP:
      1. Cached PTR hostname (if different from raw IP)
      2. Known org label (Cloudflare, Google Cloud, AWS, etc.)
      3. Raw IP (while lookup is still pending)
    """
    with _lock:
        if ip in _cache:
            return _cache[ip]
        if ip not in _pending:
            _pending.add(ip)
            _executor.submit(_resolve, ip)
    return _known_org(ip) or ip
