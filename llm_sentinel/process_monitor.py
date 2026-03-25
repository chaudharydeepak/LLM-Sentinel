"""
Detects and tracks LLM-related processes running on the machine.
"""

import psutil
from dataclasses import dataclass, field
from typing import Optional

# Substring patterns matched against "process_name + cmdline" (lowercase)
LLM_PROCESS_PATTERNS = [
    "ollama",
    "llamafile",
    "llama-server",
    "llama-cpp",
    "llama_cpp",
    "lmstudio",
    "lm studio",
    "text-generation-webui",
    "text_generation_webui",
    "koboldcpp",
    "localai",
    "local-ai",
    "gpt4all",
    "stable-diffusion",
    "automatic1111",
    "comfyui",
    "mlc_llm",
    "mlc-llm",
    "torchserve",
    "tritonserver",
]

# Exact process-name matches (checked against proc.name only, case-insensitive)
LLM_EXACT_NAMES = {
    "llm",
    "jan",
    "vllm",
    "mistral",
    "whisper",
    "kobold",
}


@dataclass
class LLMProcess:
    pid: int
    name: str
    exe: str
    cmdline: str
    username: str
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    status: str = "unknown"
    matched_pattern: str = ""
    connections: list = field(default_factory=list)

    def __hash__(self):
        return hash(self.pid)


def _matches_llm_pattern(proc_name: str, cmdline: str) -> Optional[str]:
    """Return the matched pattern if the process looks like an LLM process."""
    name_lower = proc_name.lower()
    combined = f"{proc_name} {cmdline}".lower()

    # Exact process name match (avoids substring false positives on short words)
    if name_lower in LLM_EXACT_NAMES:
        return name_lower

    # Substring match against full name+cmdline for longer, distinctive patterns
    for pattern in LLM_PROCESS_PATTERNS:
        if pattern in combined:
            return pattern

    return None


def get_llm_processes() -> list[LLMProcess]:
    """Scan all running processes and return those that look like LLM runtimes."""
    found = []

    for proc in psutil.process_iter(
        ["pid", "name", "exe", "cmdline", "username", "status", "cpu_percent", "memory_info"]
    ):
        try:
            info = proc.info
            name = info.get("name") or ""
            exe = info.get("exe") or ""
            cmdline_list = info.get("cmdline") or []
            cmdline = " ".join(cmdline_list)
            username = info.get("username") or ""
            status = info.get("status") or "unknown"

            matched = _matches_llm_pattern(name, cmdline)
            if not matched:
                matched = _matches_llm_pattern(exe, "")

            if matched:
                mem = info.get("memory_info")
                memory_mb = (mem.rss / 1024 / 1024) if mem else 0.0

                found.append(
                    LLMProcess(
                        pid=info["pid"],
                        name=name,
                        exe=exe,
                        cmdline=cmdline[:120],
                        username=username,
                        cpu_percent=info.get("cpu_percent") or 0.0,
                        memory_mb=round(memory_mb, 1),
                        status=status,
                        matched_pattern=matched,
                    )
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return found
