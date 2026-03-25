"""
Integration tests — spin up real LLM processes (via Docker) and verify the sentinel
correctly detects them and classifies their connections.

Platform notes:
  - On Linux: Docker containers run natively; host psutil sees container PIDs directly.
  - On macOS (Docker Desktop): containers run inside a Linux VM. Container PIDs are
    invisible to the macOS host psutil. Process detection only works for natively-installed
    LLMs (e.g. `brew install ollama`). Connection inspection inside containers uses
    `docker exec ss` instead.

Tests are skipped automatically if Docker is unavailable or images are missing.
"""

import platform
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error

import pytest
import psutil

from llm_sentinel.process_monitor import get_llm_processes
from llm_sentinel.network_monitor import get_connections_for_pid, _is_external


IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _docker_available() -> bool:
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _image_exists(image: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True, timeout=10,
    )
    return result.returncode == 0


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_http(url: str, timeout: float = 60.0, interval: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as resp:
                if resp.status < 500:
                    return True
        except Exception:
            pass
        time.sleep(interval)
    return False


def _run_container(image: str, name: str, ports: dict,
                   env: dict = None, extra_args: list = None) -> str:
    cmd = ["docker", "run", "-d", "--name", name]
    for host_port, container_port in ports.items():
        cmd += ["-p", f"{host_port}:{container_port}"]
    for k, v in (env or {}).items():
        cmd += ["-e", f"{k}={v}"]
    cmd += (extra_args or [])
    cmd.append(image)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"docker run failed: {result.stderr}")
    return result.stdout.strip()


def _stop_and_remove(name: str):
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)


def _container_is_running(name: str) -> bool:
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.State.Running}}", name],
        capture_output=True, text=True, timeout=5,
    )
    return result.stdout.strip() == "true"


def _container_listening_ports(container_name: str) -> set[int]:
    """
    Get listening TCP ports from inside the container by reading /proc/net/tcp
    and /proc/net/tcp6. Works in any Linux container regardless of which tools
    are installed (ss/netstat not required).

    /proc/net/tcp format: sl local_address rem_address st ...
      - local_address is hex-encoded IP:PORT (big-endian for port)
      - state 0A (hex) = LISTEN
    """
    ports = set()
    for proc_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        result = subprocess.run(
            ["docker", "exec", container_name, "cat", proc_file],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines()[1:]:  # skip header
            parts = line.split()
            if len(parts) < 4:
                continue
            state = parts[3]
            if state != "0A":   # 0A = LISTEN
                continue
            local_addr = parts[1]  # e.g. "00000000:2CAA" or IPv6 equivalent
            try:
                port_hex = local_addr.rsplit(":", 1)[-1]
                ports.add(int(port_hex, 16))
            except (ValueError, IndexError):
                pass
    return ports


def _container_established_connections(container_name: str) -> list[dict]:
    """
    Get ESTABLISHED TCP connections from inside the container by reading
    /proc/net/tcp and /proc/net/tcp6. State 01 (hex) = ESTABLISHED.

    Remote address is hex-encoded. For IPv4: the 4-byte address is stored
    in little-endian order in /proc/net/tcp.
    """
    import struct
    import socket as _socket

    def _decode_ipv4(hex_addr: str) -> str:
        try:
            packed = bytes.fromhex(hex_addr)
            return _socket.inet_ntop(_socket.AF_INET, packed[::-1])  # little-endian
        except Exception:
            return hex_addr

    def _decode_ipv6(hex_addr: str) -> str:
        try:
            # 32 hex chars, stored as 4 little-endian 32-bit words
            words = [hex_addr[i:i+8] for i in range(0, 32, 8)]
            packed = b"".join(bytes.fromhex(w)[::-1] for w in words)
            return _socket.inet_ntop(_socket.AF_INET6, packed)
        except Exception:
            return hex_addr

    conns = []
    for proc_file, is_v6 in (("/proc/net/tcp", False), ("/proc/net/tcp6", True)):
        result = subprocess.run(
            ["docker", "exec", container_name, "cat", proc_file],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines()[1:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            state = parts[3]
            if state != "01":  # 01 = ESTABLISHED
                continue
            remote_field = parts[2]  # remote_address hex
            try:
                hex_ip, hex_port = remote_field.rsplit(":", 1)
                remote_ip = _decode_ipv6(hex_ip) if is_v6 else _decode_ipv4(hex_ip)
                remote_port = int(hex_port, 16)
                conns.append({
                    "local": parts[1],
                    "remote": f"{remote_ip}:{remote_port}",
                    "remote_ip": remote_ip,
                    "is_external": _is_external(remote_ip),
                })
            except (ValueError, IndexError):
                pass
    return conns


def _get_all_pids_in_container(container_name: str) -> list[int]:
    """
    Get PIDs of container processes.
    On Linux these are real host PIDs. On macOS they are VM-internal PIDs
    not visible to the host psutil — callers must handle this.
    """
    result = subprocess.run(
        ["docker", "top", container_name, "-o", "pid"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode != 0:
        return []
    pids = []
    for line in result.stdout.strip().splitlines()[1:]:
        try:
            pids.append(int(line.strip()))
        except ValueError:
            pass
    return pids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker daemon not available",
)

requires_linux_docker = pytest.mark.skipif(
    not (_docker_available() and IS_LINUX),
    reason="Requires Linux with native Docker (macOS Docker Desktop runs containers in a VM, "
           "making container PIDs invisible to host psutil)",
)


@pytest.fixture(scope="module")
def ollama_container():
    if not _docker_available():
        pytest.skip("Docker not available")
    if not _image_exists("ollama/ollama"):
        pytest.skip("ollama/ollama image not pulled — run: docker pull ollama/ollama")

    name = "sentinel-test-ollama"
    port = _free_port()
    _stop_and_remove(name)

    _run_container(image="ollama/ollama", name=name, ports={port: 11434})

    ready = _wait_for_http(f"http://127.0.0.1:{port}/api/tags", timeout=60)
    if not ready:
        _stop_and_remove(name)
        pytest.skip("Ollama container did not become ready in 60s")

    yield {"name": name, "port": port}
    _stop_and_remove(name)


@pytest.fixture(scope="module")
def localai_container():
    """
    Uses localai/localai:latest (no pre-loaded models) which starts in ~2s.
    The aio-cpu image loads many models and takes >5 min — unsuitable for tests.
    """
    if not _docker_available():
        pytest.skip("Docker not available")
    if not _image_exists("localai/localai:latest"):
        pytest.skip("localai image not pulled — run: docker pull localai/localai:latest")

    name = "sentinel-test-localai"
    port = _free_port()
    _stop_and_remove(name)

    _run_container(image="localai/localai:latest", name=name, ports={port: 8080})

    ready = _wait_for_http(f"http://127.0.0.1:{port}/readyz", timeout=30)
    if not ready:
        _stop_and_remove(name)
        pytest.skip("LocalAI container did not become ready in 30s")

    yield {"name": name, "port": port}
    _stop_and_remove(name)


# ---------------------------------------------------------------------------
# Mock LLM subprocess fixture (no Docker, all platforms)
# ---------------------------------------------------------------------------

@pytest.fixture()
def mock_ollama_process(tmp_path):
    """
    Launches a real Python subprocess with 'ollama' in its cmdline that
    listens on loopback. Verifies the sentinel detects it and classifies
    all its connections as local.
    """
    script = tmp_path / "mock_ollama.py"
    script.write_text(
        "import socket, time, sys\n"
        "s = socket.socket()\n"
        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
        "s.bind(('127.0.0.1', 0))\n"
        "s.listen(1)\n"
        "port = s.getsockname()[1]\n"
        "sys.stdout.write(str(port) + '\\n')\n"
        "sys.stdout.flush()\n"
        "time.sleep(30)\n"
    )
    # Pass script path as argument so 'ollama' appears in cmdline
    proc = subprocess.Popen(
        [sys.executable, str(script), "--ollama-mock"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    port_line = proc.stdout.readline().decode().strip()
    port = int(port_line)
    yield {"pid": proc.pid, "port": port, "proc": proc}
    proc.terminate()
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Tests: Ollama via Docker
# ---------------------------------------------------------------------------

@requires_docker
class TestOllamaViaDocker:
    def test_ollama_container_is_running(self, ollama_container):
        """Docker container reports as running."""
        assert _container_is_running(ollama_container["name"])

    def test_ollama_http_api_is_reachable(self, ollama_container):
        """Ollama API /api/tags responds 200 on the mapped host port."""
        port = ollama_container["port"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/tags", timeout=5) as resp:
            assert resp.status == 200

    def test_ollama_listens_on_11434_inside_container(self, ollama_container):
        """Ollama must be listening on port 11434 inside the container (verified via docker exec ss)."""
        ports = _container_listening_ports(ollama_container["name"])
        assert 11434 in ports, \
            f"Ollama not listening on 11434 inside container. Found ports: {ports}"

    def test_ollama_has_no_external_connections(self, ollama_container):
        """
        A freshly-started Ollama with no loaded model should have zero outbound
        connections to external IPs (verified from inside the container via docker exec ss).
        """
        conns = _container_established_connections(ollama_container["name"])
        external = [c for c in conns if c["is_external"]]
        assert external == [], \
            f"Ollama has unexpected external connections: {[c['remote'] for c in external]}"

    def test_mapped_port_is_accessible_from_host(self, ollama_container):
        """The host-mapped port can accept TCP connections."""
        host_port = ollama_container["port"]
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect(("127.0.0.1", host_port))
        finally:
            s.close()

    @requires_linux_docker
    def test_ollama_process_detected_by_sentinel_on_linux(self, ollama_container):
        """
        On Linux, container PIDs are visible to the host; sentinel must detect ollama.
        (Skipped on macOS where Docker Desktop uses a VM.)
        """
        found = get_llm_processes()
        matched = [p for p in found
                   if "ollama" in p.name.lower() or "ollama" in p.matched_pattern.lower()]
        assert matched, \
            f"Sentinel did not detect ollama. Found: {[(p.name, p.matched_pattern) for p in found]}"

    @requires_linux_docker
    def test_ollama_connections_via_psutil_on_linux(self, ollama_container):
        """
        On Linux, verify sentinel reads Ollama connections via psutil with no external IPs.
        """
        pids = _get_all_pids_in_container(ollama_container["name"])
        all_conns = []
        for pid in pids:
            all_conns.extend(get_connections_for_pid(pid))
        external = [c for c in all_conns if c.is_external]
        assert external == [], \
            f"Ollama (via psutil) has external connections: {[c.remote_addr for c in external]}"


# ---------------------------------------------------------------------------
# Tests: LocalAI via Docker
# ---------------------------------------------------------------------------

@requires_docker
class TestLocalAIViaDocker:
    def test_localai_container_is_running(self, localai_container):
        assert _container_is_running(localai_container["name"])

    def test_localai_http_api_is_reachable(self, localai_container):
        port = localai_container["port"]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/readyz", timeout=5) as resp:
            assert resp.status == 200

    def test_localai_listens_on_8080_inside_container(self, localai_container):
        ports = _container_listening_ports(localai_container["name"])
        assert 8080 in ports, \
            f"LocalAI not listening on 8080 inside container. Found: {ports}"

    def test_localai_has_no_external_connections(self, localai_container):
        conns = _container_established_connections(localai_container["name"])
        external = [c for c in conns if c["is_external"]]
        assert external == [], \
            f"LocalAI has external connections: {[c['remote'] for c in external]}"

    @requires_linux_docker
    def test_localai_process_detected_by_sentinel_on_linux(self, localai_container):
        found = get_llm_processes()
        matched = [p for p in found
                   if "localai" in p.name.lower() or "local-ai" in p.matched_pattern.lower()
                   or "localai" in p.matched_pattern.lower()]
        assert matched, \
            f"Sentinel did not detect LocalAI. Found: {[(p.name, p.matched_pattern) for p in found]}"


# ---------------------------------------------------------------------------
# Tests: native subprocess (all platforms, no Docker)
# ---------------------------------------------------------------------------

class TestNativeProcessDetection:
    def test_mock_ollama_detected_by_sentinel(self, mock_ollama_process):
        """
        A real subprocess with 'ollama' in its cmdline must be found by get_llm_processes().
        This test works on all platforms since it uses a native host process.
        """
        found = get_llm_processes()
        pid = mock_ollama_process["pid"]
        matched = [p for p in found if p.pid == pid]
        assert matched, \
            f"Process {pid} with 'ollama' in cmdline was not detected. " \
            f"All found: {[(p.pid, p.name, p.cmdline) for p in found]}"

    def test_mock_ollama_connections_are_local(self, mock_ollama_process):
        """All connections from the mock process must be classified as local."""
        pid = mock_ollama_process["pid"]
        conns = get_connections_for_pid(pid)
        external = [c for c in conns if c.is_external]
        assert external == [], f"Mock process has external connections: {external}"

    def test_mock_process_is_alive(self, mock_ollama_process):
        assert psutil.pid_exists(mock_ollama_process["pid"])


# ---------------------------------------------------------------------------
# Tests: external connection detection (simulated, no Docker)
# ---------------------------------------------------------------------------

class TestExternalConnectionDetection:
    def test_detects_external_tcp_connection(self):
        """
        Open a real TCP connection to an external IP and verify the sentinel
        classifies it as external. Skipped if the network blocks outbound TCP/53.
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            sock.connect(("8.8.8.8", 53))
        except (socket.timeout, OSError) as e:
            pytest.skip(f"Cannot reach 8.8.8.8:53: {e}")

        try:
            pid = psutil.Process().pid
            conns = get_connections_for_pid(pid)
            external = [c for c in conns
                        if c.remote_ip == "8.8.8.8" and c.remote_port == 53]
            assert external, "Sentinel did not find the external connection to 8.8.8.8:53"
            assert external[0].is_external is True
        finally:
            sock.close()

    def test_local_loopback_not_flagged_as_external(self):
        """A loopback TCP connection must never be classified as external."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect(("127.0.0.1", port))

        try:
            pid = psutil.Process().pid
            conns = get_connections_for_pid(pid)
            to_server = [c for c in conns
                         if c.remote_ip == "127.0.0.1" and c.remote_port == port]
            assert to_server, "Loopback connection not found in connection list"
            assert to_server[0].is_external is False
        finally:
            client.close()
            server.close()

    def test_private_lan_not_flagged_as_external(self):
        """Connections to RFC1918 addresses must not be flagged."""
        for ip in ["192.168.1.1", "10.0.0.1", "172.16.0.1"]:
            assert _is_external(ip) is False, f"{ip} should be local"
