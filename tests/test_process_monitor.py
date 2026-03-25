"""
Tests for llm_sentinel.process_monitor — multi-signal scoring detection.
"""

import os
import pytest
from unittest.mock import MagicMock, patch, PropertyMock

import psutil

from llm_sentinel.process_monitor import (
    LLMProcess,
    _matches_llm_pattern,
    _score_name_cmdline,
    _score_open_model_files,
    _score_ml_libraries,
    get_llm_processes,
    LLM_PROCESS_PATTERNS,
    LLM_EXACT_NAMES,
    DETECTION_THRESHOLD,
    _MODEL_EXTS_STRONG,
    _MODEL_EXTS_LARGE,
    _MODEL_MIN_SIZE_BYTES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_proc_info(pid, name, exe="", cmdline=None, username="user",
                    status="running", cpu_percent=0.0,
                    memory_rss=100 * 1024 * 1024):
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


def _make_mock_proc(pid, name, exe="", cmdline=None,
                    memory_rss=100 * 1024 * 1024,
                    open_files=None, memory_maps=None):
    """Build a mock psutil.Process with configurable open_files / memory_maps."""
    mock = MagicMock()
    mock.info = _make_proc_info(pid, name, exe=exe, cmdline=cmdline,
                                memory_rss=memory_rss)
    mock.open_files.return_value = open_files or []
    mock.memory_maps.return_value = memory_maps or []
    return mock


def _make_open_file(path):
    f = MagicMock()
    f.path = path
    return f


def _make_mmap(path):
    m = MagicMock()
    m.path = path
    return m


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
        ("python3", "text_generation_webui --model llama"),
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
        # new patterns
        ("python3", "-m vllm.entrypoints.openai.api_server"),
        ("python3", "uvicorn open_webui.main:app"),
        ("python3", "uvicorn openwebui.main:app"),
        ("python3", "-m anythingllm"),
        ("node", "anythingllm/server.js"),
        ("python3", "invokeai --root /models"),
        ("python3", "invoke-ai --port 9090"),
        ("python3", "aider --model gpt-4o"),
        ("python3", "-m interpreter"),
        ("python3", "text-generation-inference --model-id mistral"),
        ("tabbyml", ""),
        ("python3", "tabby-ml serve"),
    ])
    def test_matches(self, name, cmdline):
        assert _matches_llm_pattern(name, cmdline) is not None, \
            f"Expected '{name}' with cmdline '{cmdline}' to match"

    @pytest.mark.parametrize("name", [
        "llm", "LLM", "jan", "JAN", "vllm", "VLLM",
        "mistral", "Mistral", "whisper", "Whisper", "kobold",
    ])
    def test_exact_name_matches(self, name):
        assert _matches_llm_pattern(name, "") is not None, \
            f"Expected exact name '{name}' to match"


# ---------------------------------------------------------------------------
# _matches_llm_pattern — non-LLM processes should NOT match
# ---------------------------------------------------------------------------

class TestPatternMatchNegatives:
    @pytest.mark.parametrize("name,cmdline", [
        ("betaenrollmentagent", ""),
        ("betaenrollmentd", ""),
        ("enrollment-service", ""),
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
        ("java", "-jar myapp.jar"),
        ("java", "-jar spring-boot-app.jar --server.port=8080"),
        # "jan" or "llm" appearing in paths/args but not as process name
        ("python3", "/Users/jan/projects/myapp.py"),
        ("python3", "run_enrollment_tool.py"),
        # interpreter as a generic word should not match alone
        ("python3", "interpreter_config.py"),
    ])
    def test_does_not_match(self, name, cmdline):
        assert _matches_llm_pattern(name, cmdline) is None, \
            f"'{name}' / '{cmdline}' should NOT match"

    def test_jan_in_path_does_not_match_if_process_not_jan(self):
        assert _matches_llm_pattern("python3", "/Users/jan/scripts/run.py") is None

    def test_llm_substring_in_cmdline_does_not_match_if_process_not_llm(self):
        assert _matches_llm_pattern("python3", "run_enrollment_tool.py") is None

    def test_spring_boot_on_port_8000_does_not_match(self):
        assert _matches_llm_pattern("java", "-jar app.jar --server.port=8000") is None

    def test_generic_python_server_does_not_match(self):
        assert _matches_llm_pattern("python3", "manage.py runserver 0.0.0.0:8000") is None


# ---------------------------------------------------------------------------
# Pattern list integrity
# ---------------------------------------------------------------------------

class TestPatternLists:
    def test_no_duplicates_in_patterns(self):
        assert len(LLM_PROCESS_PATTERNS) == len(set(LLM_PROCESS_PATTERNS))

    def test_no_duplicates_in_exact_names(self):
        assert len(LLM_EXACT_NAMES) == len(set(LLM_EXACT_NAMES))

    def test_exact_names_not_duplicated_in_patterns(self):
        for name in LLM_EXACT_NAMES:
            assert name not in LLM_PROCESS_PATTERNS, \
                f"'{name}' is in both LLM_EXACT_NAMES and LLM_PROCESS_PATTERNS"

    def test_detection_threshold_is_positive(self):
        assert DETECTION_THRESHOLD >= 1


# ---------------------------------------------------------------------------
# _score_name_cmdline
# ---------------------------------------------------------------------------

class TestScoreNameCmdline:
    def test_known_name_scores_2(self):
        score, match = _score_name_cmdline("ollama", "", "")
        assert score == 2
        assert match == "ollama"

    def test_exact_name_scores_2(self):
        score, match = _score_name_cmdline("vllm", "", "")
        assert score == 2

    def test_cmdline_match_scores_2(self):
        score, match = _score_name_cmdline("python3", "", "-m vllm.entrypoints.openai.api_server")
        assert score == 2

    def test_no_match_scores_0(self):
        score, match = _score_name_cmdline("java", "/usr/bin/java", "-jar spring.jar")
        assert score == 0
        assert match == ""

    def test_exe_path_match(self):
        score, match = _score_name_cmdline("runner", "/usr/local/bin/ollama", "")
        assert score == 2


# ---------------------------------------------------------------------------
# _score_open_model_files
# ---------------------------------------------------------------------------

class TestScoreOpenModelFiles:
    def test_gguf_file_scores_3(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file("/models/mistral.gguf")]
        score, reason = _score_open_model_files(proc)
        assert score == 3
        assert "mistral.gguf" in reason

    def test_safetensors_scores_3(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file("/models/model.safetensors")]
        score, reason = _score_open_model_files(proc)
        assert score == 3

    def test_ggml_scores_3(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file("/models/model.ggml")]
        score, reason = _score_open_model_files(proc)
        assert score == 3

    @pytest.mark.parametrize("ext", [".q4_0", ".q4_1", ".q5_0", ".q5_1", ".q8_0"])
    def test_quantized_ext_scores_3(self, ext):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file(f"/models/model{ext}")]
        score, _ = _score_open_model_files(proc)
        assert score == 3

    def test_large_bin_file_scores_3(self, tmp_path):
        big_file = tmp_path / "model.bin"
        big_file.write_bytes(b"\x00" * (_MODEL_MIN_SIZE_BYTES + 1))
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file(str(big_file))]
        score, reason = _score_open_model_files(proc)
        assert score == 3

    def test_small_bin_file_does_not_score(self, tmp_path):
        small_file = tmp_path / "config.bin"
        small_file.write_bytes(b"\x00" * 1024)
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file(str(small_file))]
        score, _ = _score_open_model_files(proc)
        assert score == 0

    def test_no_model_files_scores_0(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [
            _make_open_file("/usr/lib/libc.so"),
            _make_open_file("/etc/hosts"),
            _make_open_file("/var/log/app.log"),
        ]
        score, _ = _score_open_model_files(proc)
        assert score == 0

    def test_access_denied_returns_0(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.side_effect = psutil.AccessDenied(1)
        score, _ = _score_open_model_files(proc)
        assert score == 0

    def test_no_such_process_returns_0(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.side_effect = psutil.NoSuchProcess(1)
        score, _ = _score_open_model_files(proc)
        assert score == 0

    def test_empty_open_files_scores_0(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = []
        score, _ = _score_open_model_files(proc)
        assert score == 0

    def test_nonexistent_bin_path_does_not_raise(self):
        proc = MagicMock(spec=psutil.Process)
        proc.open_files.return_value = [_make_open_file("/nonexistent/model.bin")]
        # os.path.getsize will raise OSError — should be caught gracefully
        score, _ = _score_open_model_files(proc)
        assert score == 0


# ---------------------------------------------------------------------------
# _score_ml_libraries
# ---------------------------------------------------------------------------

class TestScoreMlLibraries:
    # Use plain MagicMock (not spec=psutil.Process) because memory_maps
    # availability varies by platform and psutil version.

    @pytest.mark.parametrize("lib_path", [
        "/usr/local/cuda/lib/libcublas.so",
        "/usr/local/cuda/lib/libcudart.so",
        "/usr/lib/libcuda.so.1",
        "/home/user/.venv/lib/libtorch.so",
        "/home/user/.venv/torch/lib/libtorch_cpu.so",
        "/home/user/.venv/lib/libonnxruntime.so",
        "/home/user/.venv/lib/libtensorflow.so",
        "/System/Library/Frameworks/MetalPerformanceShaders.framework/metal_performance_shaders",
        "/usr/local/lib/ggml/libggml.so",
    ])
    def test_ml_lib_scores_2(self, lib_path):
        proc = MagicMock()
        proc.memory_maps.return_value = [_make_mmap(lib_path)]
        score, reason = _score_ml_libraries(proc)
        assert score == 2, f"Expected score 2 for {lib_path}"

    def test_no_ml_libs_scores_0(self):
        proc = MagicMock()
        proc.memory_maps.return_value = [
            _make_mmap("/usr/lib/libc.so"),
            _make_mmap("/usr/lib/libssl.so"),
            _make_mmap("/usr/lib/jvm/libjvm.so"),
        ]
        score, _ = _score_ml_libraries(proc)
        assert score == 0

    def test_access_denied_returns_0(self):
        proc = MagicMock()
        proc.memory_maps.side_effect = psutil.AccessDenied(1)
        score, _ = _score_ml_libraries(proc)
        assert score == 0

    def test_not_implemented_returns_0(self):
        proc = MagicMock()
        proc.memory_maps.side_effect = NotImplementedError
        score, _ = _score_ml_libraries(proc)
        assert score == 0

    def test_empty_maps_scores_0(self):
        proc = MagicMock()
        proc.memory_maps.return_value = []
        score, _ = _score_ml_libraries(proc)
        assert score == 0


# ---------------------------------------------------------------------------
# get_llm_processes — integration via mocked psutil
# ---------------------------------------------------------------------------

class TestGetLlmProcesses:
    def test_detects_ollama_by_name(self):
        proc = _make_mock_proc(1234, "ollama", exe="/usr/bin/ollama",
                               cmdline=["ollama", "serve"])
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
        assert result[0].pid == 1234
        assert result[0].name == "ollama"

    def test_detects_llamafile_by_name(self):
        proc = _make_mock_proc(5678, "llamafile",
                               cmdline=["./llamafile", "--model", "mistral.gguf"])
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
        assert result[0].matched_pattern == "llamafile"

    def test_detects_vllm_via_cmdline(self):
        """python3 running vllm — process name gives no signal, cmdline does."""
        proc = _make_mock_proc(
            100, "python3",
            cmdline=["python3", "-m", "vllm.entrypoints.openai.api_server"],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
        assert "vllm.entrypoints" in result[0].matched_pattern

    def test_detects_open_webui_via_cmdline(self):
        proc = _make_mock_proc(
            101, "python3",
            cmdline=["uvicorn", "open_webui.main:app", "--port", "3000"],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1

    def test_detects_renamed_binary_via_model_file(self):
        """A binary renamed to 'worker' but with a .gguf open — should be caught."""
        proc = _make_mock_proc(
            200, "worker", exe="/opt/renamed/inference-worker",
            memory_rss=4 * 1024 * 1024 * 1024,  # 4 GB
            open_files=[_make_open_file("/models/llama3-8b.gguf")],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
        assert "llama3-8b.gguf" in result[0].matched_pattern
        assert result[0].detection_score >= 3

    def test_detects_electron_lmstudio_via_ml_libs(self):
        """Electron process with CUDA libs loaded — not obvious from name."""
        proc = _make_mock_proc(
            300, "electron", exe="/Applications/LM Studio.app/electron",
            memory_rss=2 * 1024 * 1024 * 1024,
            memory_maps=[_make_mmap("/usr/local/cuda/lib/libcublas.so")],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
        assert result[0].detection_score >= 2

    def test_spring_boot_on_port_8000_not_detected(self):
        """Java Spring Boot app — must never be flagged as an LLM."""
        proc = _make_mock_proc(
            400, "java", exe="/usr/bin/java",
            cmdline=["java", "-jar", "myapp.jar", "--server.port=8000"],
            memory_rss=512 * 1024 * 1024,
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert result == []

    def test_large_rss_alone_not_sufficient(self):
        """A process with 3 GB RSS but no other signals should not be detected."""
        proc = _make_mock_proc(
            500, "chrome", exe="/usr/bin/chrome",
            memory_rss=3 * 1024 * 1024 * 1024,
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert result == []

    def test_ignores_non_llm_processes(self):
        procs = []
        for name in ["bash", "nginx", "postgres", "Safari", "kernel_task"]:
            procs.append(_make_mock_proc(1, name, memory_rss=10 * 1024 * 1024))
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=procs):
            result = get_llm_processes()
        assert result == []

    def test_multiple_llm_processes_detected(self):
        procs = [
            _make_mock_proc(1, "ollama"),
            _make_mock_proc(2, "llamafile"),
            _make_mock_proc(3, "vllm"),
        ]
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=procs):
            result = get_llm_processes()
        assert len(result) == 3

    def test_detection_score_and_reasons_populated(self):
        proc = _make_mock_proc(1, "ollama")
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert result[0].detection_score >= DETECTION_THRESHOLD
        assert len(result[0].detection_reasons) > 0

    def test_combined_signals_accumulate_score(self):
        """Name match (+2) + model file (+3) should yield score >= 5."""
        proc = _make_mock_proc(
            1, "ollama",
            open_files=[_make_open_file("/models/llama3.gguf")],
            memory_rss=6 * 1024 * 1024 * 1024,
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        # open_files check is skipped once name score >= threshold,
        # but the name score alone is sufficient
        assert len(result) == 1
        assert result[0].detection_score >= 2

    def test_tiny_process_skips_expensive_checks(self):
        """A tiny process (< 10 MB) with no name match is skipped without checking files."""
        proc = _make_mock_proc(
            1, "worker",
            memory_rss=1 * 1024 * 1024,  # 1 MB
            open_files=[_make_open_file("/models/big.gguf")],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        # Below 10 MB threshold — expensive checks are not run, so NOT detected
        assert result == []

    def test_handles_access_denied_gracefully(self):
        good = _make_mock_proc(1, "ollama")
        bad = MagicMock()
        # Use PropertyMock so accessing bad.info raises, not calling it
        type(bad).info = PropertyMock(side_effect=psutil.AccessDenied(2))
        with patch("llm_sentinel.process_monitor.psutil.process_iter",
                   return_value=[good, bad]):
            result = get_llm_processes()
        assert len(result) == 1

    def test_handles_process_vanishing_gracefully(self):
        proc = MagicMock()
        type(proc).info = PropertyMock(side_effect=psutil.NoSuchProcess(999))
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert result == []

    def test_memory_converted_to_mb(self):
        proc = _make_mock_proc(1, "ollama", memory_rss=512 * 1024 * 1024)
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert result[0].memory_mb == pytest.approx(512.0, abs=1)

    def test_cmdline_truncated_to_120_chars(self):
        long_cmd = ["ollama"] + ["--arg"] * 50
        proc = _make_mock_proc(1, "ollama", cmdline=long_cmd)
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result[0].cmdline) <= 120

    def test_detects_via_exe_path_when_name_differs(self):
        proc = _make_mock_proc(1, "truncated_name", exe="/usr/local/bin/ollama")
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1

    def test_open_files_access_denied_does_not_prevent_name_detection(self):
        """If open_files raises, name-matched process is still detected."""
        proc = _make_mock_proc(1, "ollama")
        proc.open_files.side_effect = psutil.AccessDenied(1)
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1  # detected by name, open_files not needed

    def test_aider_detected_via_cmdline(self):
        proc = _make_mock_proc(
            1, "python3",
            cmdline=["python3", "-m", "aider", "--model", "gpt-4o"],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1

    def test_open_interpreter_detected_via_cmdline(self):
        proc = _make_mock_proc(
            1, "python3",
            cmdline=["python3", "-m", "interpreter"],
        )
        with patch("llm_sentinel.process_monitor.psutil.process_iter", return_value=[proc]):
            result = get_llm_processes()
        assert len(result) == 1
