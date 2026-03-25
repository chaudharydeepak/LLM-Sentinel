"""
Tests for llm_sentinel.process_monitor — pattern matching and process detection.
"""

import pytest
from unittest.mock import MagicMock, patch

import psutil

from llm_sentinel.process_monitor import (
    LLMProcess,
    _matches_llm_pattern,
    get_llm_processes,
    LLM_PROCESS_PATTERNS,
    LLM_EXACT_NAMES,
)


# ---------------------------------------------------------------------------
# _matches_llm_pattern — known LLM processes SHOULD match
# ---------------------------------------------------------------------------

class TestPatternMatchPositives:
    @pytest.mark.parametrize("name,cmdline", [
        ("ollama", ""),
        ("ollama_llama_server", ""),
        ("ollama", "/usr/local/bin/ollama serve"),
        ("LM Studio", ""),
        ("lmstudio", ""),
        ("llamafile", "/home/user/llamafile --model mistral.gguf"),
        ("llama-server", ""),
        ("llama-cpp", ""),
        ("text-generation-webui", "python server.py"),
        ("python", "text_generation_webui --model llama"),
        ("koboldcpp", ""),
        ("localai", ""),
        ("local-ai", ""),
        ("gpt4all", ""),
        ("comfyui", ""),
        ("stable-diffusion", ""),
        ("automatic1111", ""),
        ("mlc_llm", ""),
        ("mlc-llm", ""),
        ("torchserve", ""),
        ("tritonserver", ""),
    ])
    def test_matches(self, name, cmdline):
        assert _matches_llm_pattern(name, cmdline) is not None, \
            f"Expected '{name}' with cmdline '{cmdline}' to match"

    @pytest.mark.parametrize("name", [
        "llm",
        "LLM",
        "jan",
        "JAN",
        "vllm",
        "VLLM",
        "mistral",
        "Mistral",
        "whisper",
        "Whisper",
        "kobold",
    ])
    def test_exact_name_matches(self, name):
        assert _matches_llm_pattern(name, "") is not None, \
            f"Expected exact name '{name}' to match"


# ---------------------------------------------------------------------------
# _matches_llm_pattern — non-LLM processes should NOT match
# ---------------------------------------------------------------------------

class TestPatternMatchNegatives:
    @pytest.mark.parametrize("name,cmdline", [
        # The false positive we fixed — "llm" as substring
        ("betaenrollmentagent", ""),
        ("betaenrollmentd", ""),
        ("enrollment-service", ""),
        # Other common macOS/Linux processes
        ("kernel_task", ""),
        ("WindowServer", ""),
        ("Safari", ""),
        ("python3", ""),
        ("python3", "manage.py runserver"),
        ("node", "server.js"),
        ("nginx", ""),
        ("postgres", ""),
        ("redis-server", ""),
        ("bash", ""),
        ("zsh", ""),
        ("launchd", ""),
        ("systemd", ""),
        ("Finder", ""),
        ("Dock", ""),
        ("loginwindow", ""),
        ("coreaudiod", ""),
        ("configd", ""),
        # processes with "jan" or "llm" in path but not as the process name
        ("python3", "/Users/jan/projects/myapp.py"),  # username in path shouldn't trigger
        ("java", "-jar myapp.jar"),
    ])
    def test_does_not_match(self, name, cmdline):
        assert _matches_llm_pattern(name, cmdline) is None, \
            f"'{name}' with cmdline '{cmdline}' should NOT match"

    def test_jan_in_path_does_not_match_if_process_not_jan(self):
        # "jan" appears in the path but the process itself is python3
        result = _matches_llm_pattern("python3", "/Users/jan/scripts/run.py")
        assert result is None

    def test_llm_substring_in_cmdline_does_not_match_if_process_not_llm(self):
        # "llm" appears in cmdline but process name is "python3"
        result = _matches_llm_pattern("python3", "run_enrollment_tool.py")
        assert result is None


# ---------------------------------------------------------------------------
# Pattern completeness
# ---------------------------------------------------------------------------

class TestPatternLists:
    def test_no_duplicates_in_patterns(self):
        assert len(LLM_PROCESS_PATTERNS) == len(set(LLM_PROCESS_PATTERNS))

    def test_no_duplicates_in_exact_names(self):
        assert len(LLM_EXACT_NAMES) == len(set(LLM_EXACT_NAMES))

    def test_exact_names_not_duplicated_in_patterns(self):
        # Exact names should not also appear in substring patterns to avoid double-matching
        for name in LLM_EXACT_NAMES:
            assert name not in LLM_PROCESS_PATTERNS, \
                f"'{name}' is in both LLM_EXACT_NAMES and LLM_PROCESS_PATTERNS"


# ---------------------------------------------------------------------------
# get_llm_processes — mocked psutil
# ---------------------------------------------------------------------------

def _make_proc_info(pid, name, exe="", cmdline=None, username="user",
                    status="running", cpu_percent=0.0, memory_rss=100 * 1024 * 1024):
    mem = MagicMock()
    mem.rss = memory_rss
    return {
        "pid": pid,
        "name": name,
        "exe": exe,
        "cmdline": cmdline or [name],
        "username": username,
        "status": status,
        "cpu_percent": cpu_percent,
        "memory_info": mem,
    }


class TestGetLlmProcesses:
    def test_detects_ollama(self):
        proc_info = _make_proc_info(1234, "ollama", exe="/usr/bin/ollama",
                                    cmdline=["ollama", "serve"])
        mock_proc = MagicMock()
        mock_proc.info = proc_info

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[mock_proc]):
            result = get_llm_processes()

        assert len(result) == 1
        assert result[0].pid == 1234
        assert result[0].name == "ollama"

    def test_detects_llamafile(self):
        proc_info = _make_proc_info(5678, "llamafile",
                                    cmdline=["./llamafile", "--model", "mistral.gguf"])
        mock_proc = MagicMock()
        mock_proc.info = proc_info

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[mock_proc]):
            result = get_llm_processes()

        assert len(result) == 1
        assert result[0].matched_pattern == "llamafile"

    def test_ignores_non_llm_processes(self):
        procs = []
        for name in ["bash", "nginx", "postgres", "Safari", "kernel_task"]:
            info = _make_proc_info(1, name)
            mock_proc = MagicMock()
            mock_proc.info = info
            procs.append(mock_proc)

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=procs):
            result = get_llm_processes()

        assert result == []

    def test_multiple_llm_processes_detected(self):
        procs = []
        for pid, name in [(1, "ollama"), (2, "llamafile"), (3, "vllm")]:
            info = _make_proc_info(pid, name)
            mock_proc = MagicMock()
            mock_proc.info = info
            procs.append(mock_proc)

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=procs):
            result = get_llm_processes()

        assert len(result) == 3
        names = {p.name for p in result}
        assert names == {"ollama", "llamafile", "vllm"}

    def test_handles_access_denied_gracefully(self):
        mock_proc = MagicMock()
        mock_proc.info = _make_proc_info(1, "ollama")
        # Simulate a second process that raises AccessDenied when iterated
        bad_proc = MagicMock()
        bad_proc.info = MagicMock(side_effect=psutil.AccessDenied(2))

        with patch("llm_sentinel.process_monitor.psutil.process_iter",
                   return_value=[mock_proc, bad_proc]):
            # Should not raise, should return the one accessible process
            result = get_llm_processes()

        assert len(result) == 1

    def test_handles_process_vanishing_gracefully(self):
        mock_proc = MagicMock()
        mock_proc.info = MagicMock(side_effect=psutil.NoSuchProcess(999))

        with patch("llm_sentinel.process_monitor.psutil.process_iter",
                   return_value=[mock_proc]):
            result = get_llm_processes()

        assert result == []

    def test_memory_converted_to_mb(self):
        info = _make_proc_info(1, "ollama", memory_rss=512 * 1024 * 1024)  # 512 MB
        mock_proc = MagicMock()
        mock_proc.info = info

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[mock_proc]):
            result = get_llm_processes()

        assert result[0].memory_mb == pytest.approx(512.0, abs=1)

    def test_cmdline_truncated_to_120_chars(self):
        long_cmd = ["ollama"] + ["--arg"] * 50
        info = _make_proc_info(1, "ollama", cmdline=long_cmd)
        mock_proc = MagicMock()
        mock_proc.info = info

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[mock_proc]):
            result = get_llm_processes()

        assert len(result[0].cmdline) <= 120

    def test_detects_via_exe_path_when_name_differs(self):
        # Some systems show the process name truncated; exe has the full path
        info = _make_proc_info(1, "ollama_llama_", exe="/usr/local/bin/ollama")
        mock_proc = MagicMock()
        mock_proc.info = info

        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[mock_proc]):
            result = get_llm_processes()

        assert len(result) == 1
