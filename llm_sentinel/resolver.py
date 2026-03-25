"""
Non-blocking, cached reverse DNS resolver.
Lookups run in a background thread pool so they never stall the dashboard.
"""

import socket
import threading
from concurrent.futures import ThreadPoolExecutor

_cache: dict[str, str] = {}
_pending: set[str] = set()
_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="dns")


def _resolve(ip: str) -> None:
    try:
        host = socket.gethostbyaddr(ip)[0]
    except Exception:
        host = ip  # fall back to raw IP if DNS fails
    with _lock:
        _cache[ip] = host
        _pending.discard(ip)


def hostname(ip: str) -> str:
    """
    Return the resolved hostname for an IP, or the raw IP while lookup is pending.
    First call for an unseen IP kicks off an async lookup; subsequent calls return
    the cached result once it completes.
    """
    with _lock:
        if ip in _cache:
            return _cache[ip]
        if ip not in _pending:
            _pending.add(ip)
            _executor.submit(_resolve, ip)
    return ip  # return raw IP until resolved
