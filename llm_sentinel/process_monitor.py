"""
Detects and tracks LLM-related processes running on the machine.

Uses multi-signal scoring so that renamed or generically-named processes
(e.g. "python3", "electron", a re-named binary) are still caught reliably.

  Signal                         Score   Method
  ───────────────────────────────────────────────────────────────────────
  Open model weight file          +3     psutil open_files()
  ML inference library in memory  +2     psutil memory_maps()
  Known name / cmdline pattern    +2     name + cmdline substring match
  RSS > 2 GB                      +1     memory_info.rss

  Threshold to include: score >= 2

A Spring Boot app on any port scores 0.
A renamed llama.cpp binary with a .gguf open scores 3.
python3 running vllm scores 2 from the cmdline pattern alone.
"""

import os
import psutil
from dataclasses import dataclass, field
from typing import Optional

DETECTION_THRESHOLD = 2

# ── Model weight file signals ─────────────────────────────────────────────────

# Extensions unambiguous enough on their own
_MODEL_EXTS_STRONG = frozenset({
    ".gguf", ".ggml", ".safetensors",
    ".q4_0", ".q4_1", ".q5_0", ".q5_1", ".q8_0",
})
# These need a size check — .bin and .pt are too generic otherwise
_MODEL_EXTS_LARGE = frozenset({".bin", ".pt", ".pth"})
_MODEL_MIN_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB

# ── ML library signals ────────────────────────────────────────────────────────

_ML_LIB_FRAGMENTS = (
    "libcublas", "libcudart", "libcuda.",   # NVIDIA CUDA
    "libtorch", "torch/lib",                 # PyTorch
    "libonnxruntime",                         # ONNX Runtime
    "libtensorflow",                          # TensorFlow
    "metal_performance_shaders",              # Apple MPS / CoreML
    "ggml",                                   # llama.cpp core library
    "libvulkan",                              # Vulkan compute backends
)

# ── Name / cmdline patterns ───────────────────────────────────────────────────

# Substring patterns matched against "name + cmdline" (lowercased).
# Use specific, distinctive strings — avoid anything that could appear in
# unrelated tool names or file paths.
LLM_PROCESS_PATTERNS = [
    # llama.cpp family
    "ollama", "llamafile", "llama-server", "llama-cpp", "llama_cpp",
    # popular frontends / servers
    "lmstudio", "lm studio",
    "text-generation-webui", "text_generation_webui",
    "koboldcpp", "localai", "local-ai",
    "gpt4all", "comfyui", "automatic1111", "stable-diffusion",
    "mlc_llm", "mlc-llm", "torchserve", "tritonserver",
    # Python-module tools — appear in cmdline, not process name
    "vllm.entrypoints",               # python -m vllm.entrypoints.openai.api_server
    "open_webui", "openwebui",        # Open WebUI (Ollama frontend)
    "anythingllm", "anything-llm",
    "tabbyml", "tabby-ml",
    "invokeai", "invoke-ai",
    "aider",                          # Aider coding assistant
    "-m interpreter",                 # Open Interpreter: python -m interpreter
    "text-generation-inference",      # HuggingFace TGI
]

# Exact process-name matches — short names where substring match would cause
# false positives (e.g. "jan" would hit "/Users/jan/..." in any python path).
LLM_EXACT_NAMES = {
    "llm",
    "jan",
    "vllm",
    "mistral",
    "whisper",
    "kobold",
}


# ── Dataclass ─────────────────────────────────────────────────────────────────

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
    matched_pattern: str = ""            # primary match label (backward compat)
    detection_score: int = 0             # total score across all signals
    detection_reasons: list = field(default_factory=list)  # human-readable signal list
    connections: list = field(default_factory=list)

    def __hash__(self):
        return hash(self.pid)


# ── Signal scorers ────────────────────────────────────────────────────────────

def _matches_llm_pattern(proc_name: str, cmdline: str) -> Optional[str]:
    """Return the matched pattern string, or None.  Used as one scoring signal."""
    name_lower = proc_name.lower()
    combined = f"{proc_name} {cmdline}".lower()

    if name_lower in LLM_EXACT_NAMES:
        return name_lower

    for pattern in LLM_PROCESS_PATTERNS:
        if pattern in combined:
            return pattern

    return None


def _score_name_cmdline(name: str, exe: str, cmdline: str) -> tuple[int, str]:
    """Score +2 if name, exe, or cmdline matches a known LLM pattern."""
    matched = _matches_llm_pattern(name, cmdline)
    if not matched:
        matched = _matches_llm_pattern(exe, "")
    if matched:
        return 2, matched
    return 0, ""


def _score_open_model_files(proc: psutil.Process) -> tuple[int, str]:
    """Score +3 if the process has a model weight file open."""
    try:
        for f in proc.open_files():
            path = f.path
            ext = os.path.splitext(path)[1].lower()
            if ext in _MODEL_EXTS_STRONG:
                return 3, f"model:{os.path.basename(path)}"
            if ext in _MODEL_EXTS_LARGE:
                try:
                    if os.path.getsize(path) >= _MODEL_MIN_SIZE_BYTES:
                        return 3, f"model:{os.path.basename(path)}"
                except OSError:
                    pass
    except (psutil.AccessDenied, psutil.NoSuchProcess, OSError, NotImplementedError):
        pass
    return 0, ""


def _score_ml_libraries(proc: psutil.Process) -> tuple[int, str]:
    """Score +2 if ML inference libraries are mapped into the process address space."""
    try:
        for mmap in proc.memory_maps():
            path_lower = mmap.path.lower()
            for lib in _ML_LIB_FRAGMENTS:
                if lib in path_lower:
                    return 2, f"lib:{lib}"
    except (psutil.AccessDenied, psutil.NoSuchProcess,
            OSError, NotImplementedError, AttributeError):
        pass
    return 0, ""


# ── Main detector ─────────────────────────────────────────────────────────────

def get_llm_processes() -> list[LLMProcess]:
    """Scan all running processes and return those that score >= DETECTION_THRESHOLD."""
    found = []

    for proc in psutil.process_iter(
        ["pid", "name", "exe", "cmdline", "username", "status", "cpu_percent", "memory_info"]
    ):
        try:
            info = proc.info
            name     = info.get("name") or ""
            exe      = info.get("exe") or ""
            cmdline  = " ".join(info.get("cmdline") or [])
            username = info.get("username") or ""
            status   = info.get("status") or "unknown"
            mem      = info.get("memory_info")
            memory_mb = (mem.rss / 1024 / 1024) if mem else 0.0

            score = 0
            reasons: list[str] = []
            primary = ""

            # ── Cheap signals (always checked) ───────────────────────────────

            s, match = _score_name_cmdline(name, exe, cmdline)
            if s:
                score += s
                reasons.append(f"pattern:{match}")
                primary = match

            if memory_mb > 2048:
                score += 1
                reasons.append(f"rss:{memory_mb:.0f}MB")

            # ── Expensive signals ─────────────────────────────────────────────
            # Only run for processes with meaningful memory (LLMs always need
            # substantial RAM; this skips kernel threads, tiny daemons, etc.)
            # Also skip if name/cmdline already crossed the threshold — no need.

            if score < DETECTION_THRESHOLD and memory_mb >= 10.0:
                s, reason = _score_open_model_files(proc)
                if s:
                    score += s
                    reasons.append(reason)
                    if not primary:
                        primary = reason

            if score < DETECTION_THRESHOLD and memory_mb >= 10.0:
                s, reason = _score_ml_libraries(proc)
                if s:
                    score += s
                    reasons.append(reason)
                    if not primary:
                        primary = reason

            if score >= DETECTION_THRESHOLD:
                found.append(LLMProcess(
                    pid=info["pid"],
                    name=name,
                    exe=exe,
                    cmdline=cmdline[:120],
                    username=username,
                    cpu_percent=info.get("cpu_percent") or 0.0,
                    memory_mb=round(memory_mb, 1),
                    status=status,
                    matched_pattern=primary,
                    detection_score=score,
                    detection_reasons=reasons,
                ))

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue

    return found
