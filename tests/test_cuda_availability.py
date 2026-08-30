import subprocess

import pytest

from src.startup import cuda_availability as cuda


@pytest.fixture(autouse=True)
def _clear_caches():
    """Todas las funciones relevantes usan @lru_cache(maxsize=1) — sin
    limpiarlo, el resultado del primer test que las llame quedaría pegado
    para el resto de la suite, sin importar qué mockee cada test después."""
    cuda._is_gpu_feature_enabled_by_installer.cache_clear()
    cuda._is_nvidia_gpu_hardware_present.cache_clear()
    cuda.is_nvidia_gpu_available.cache_clear()
    cuda._driver_max_cuda_version.cache_clear()
    yield
    cuda._is_gpu_feature_enabled_by_installer.cache_clear()
    cuda._is_nvidia_gpu_hardware_present.cache_clear()
    cuda.is_nvidia_gpu_available.cache_clear()
    cuda._driver_max_cuda_version.cache_clear()


# --- _is_gpu_feature_enabled_by_installer ---


def test_installer_feature_always_enabled_when_running_from_source(monkeypatch):
    monkeypatch.setattr("sys.frozen", False, raising=False)
    assert cuda._is_gpu_feature_enabled_by_installer() is True


def test_installer_feature_enabled_when_frozen_and_marker_present(monkeypatch):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr(cuda.os.path, "isfile", lambda path: True)
    assert cuda._is_gpu_feature_enabled_by_installer() is True


def test_installer_feature_disabled_when_frozen_and_marker_missing(monkeypatch):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr(cuda.os.path, "isfile", lambda path: False)
    assert cuda._is_gpu_feature_enabled_by_installer() is False


# --- _is_nvidia_gpu_hardware_present ---


def test_hardware_absent_when_nvidia_smi_not_found(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: None)
    assert cuda._is_nvidia_gpu_hardware_present() is False


def test_hardware_present_when_nvidia_smi_reports_a_gpu(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="RTX 4070\n"),
    )
    assert cuda._is_nvidia_gpu_hardware_present() is True


def test_hardware_absent_when_nvidia_smi_returns_empty_output(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout="   \n"),
    )
    assert cuda._is_nvidia_gpu_hardware_present() is False


def test_hardware_absent_when_nvidia_smi_fails(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=1, stdout=""),
    )
    assert cuda._is_nvidia_gpu_hardware_present() is False


def test_hardware_absent_when_subprocess_raises(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")

    def _raise(*a, **k):
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=5)

    monkeypatch.setattr(cuda.subprocess, "run", _raise)
    assert cuda._is_nvidia_gpu_hardware_present() is False


# --- is_nvidia_gpu_available ---


def test_gpu_available_requires_both_installer_flag_and_hardware(monkeypatch):
    monkeypatch.setattr(cuda, "_is_gpu_feature_enabled_by_installer", lambda: True)
    monkeypatch.setattr(cuda, "_is_nvidia_gpu_hardware_present", lambda: True)
    assert cuda.is_nvidia_gpu_available() is True


def test_gpu_unavailable_when_installer_disabled_it(monkeypatch):
    monkeypatch.setattr(cuda, "_is_gpu_feature_enabled_by_installer", lambda: False)
    monkeypatch.setattr(cuda, "_is_nvidia_gpu_hardware_present", lambda: True)
    assert cuda.is_nvidia_gpu_available() is False


def test_gpu_unavailable_when_no_hardware(monkeypatch):
    monkeypatch.setattr(cuda, "_is_gpu_feature_enabled_by_installer", lambda: True)
    monkeypatch.setattr(cuda, "_is_nvidia_gpu_hardware_present", lambda: False)
    assert cuda.is_nvidia_gpu_available() is False


# --- gpu_unavailability_reason ---


def test_unavailability_reason_mentions_installer_when_that_is_the_cause(monkeypatch):
    monkeypatch.setattr(cuda, "_is_gpu_feature_enabled_by_installer", lambda: False)
    reason = cuda.gpu_unavailability_reason()
    assert "instalación" in reason


def test_unavailability_reason_mentions_hardware_when_that_is_the_cause(monkeypatch):
    monkeypatch.setattr(cuda, "_is_gpu_feature_enabled_by_installer", lambda: True)
    reason = cuda.gpu_unavailability_reason()
    assert "GPU NVIDIA" in reason


# --- _driver_max_cuda_version ---


def test_driver_version_none_when_nvidia_smi_not_found(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: None)
    assert cuda._driver_max_cuda_version() is None


@pytest.mark.parametrize(
    "stdout_text,expected",
    [
        ("... CUDA Version: 12.6 ...", (12, 6)),
        ("... CUDA UMD Version: 13.3 ...", (13, 3)),
        ("sin ninguna version reconocible", None),
    ],
)
def test_driver_version_parsing(monkeypatch, stdout_text, expected):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")
    monkeypatch.setattr(
        cuda.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, returncode=0, stdout=stdout_text),
    )
    assert cuda._driver_max_cuda_version() == expected


def test_driver_version_none_when_subprocess_raises(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: r"C:\nvidia-smi.exe")

    def _raise(*a, **k):
        raise OSError("no se pudo ejecutar")

    monkeypatch.setattr(cuda.subprocess, "run", _raise)
    assert cuda._driver_max_cuda_version() is None


# --- check_driver_supports_cuda_runtime ---


def test_check_driver_does_nothing_when_version_unknown(monkeypatch):
    monkeypatch.setattr(cuda, "_driver_max_cuda_version", lambda: None)
    cuda.check_driver_supports_cuda_runtime()  # no debe lanzar


def test_check_driver_does_nothing_when_version_is_sufficient(monkeypatch):
    monkeypatch.setattr(cuda, "_driver_max_cuda_version", lambda: (12, 6))
    cuda.check_driver_supports_cuda_runtime()  # no debe lanzar


def test_check_driver_raises_when_version_is_insufficient(monkeypatch):
    monkeypatch.setattr(cuda, "_driver_max_cuda_version", lambda: (11, 8))
    with pytest.raises(RuntimeError, match="CUDA 11.8"):
        cuda.check_driver_supports_cuda_runtime()
