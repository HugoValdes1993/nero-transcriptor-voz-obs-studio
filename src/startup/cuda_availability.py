"""
Detecta si la aceleración CUDA se puede ofrecer en este equipo, para decidir
si el checkbox de "aceleración CUDA" de la GUI puede estar activo. Combina
dos cosas independientes:

1. Hardware: hay una GPU NVIDIA con driver instalado — se detecta vía
   `nvidia-smi`, NO vía torch.cuda.is_available(): torch, tal como queda
   instalado por requirements.txt, puede terminar siendo una build CPU-only
   sin que eso tenga relación con si la GPU y el driver de NVIDIA están
   realmente disponibles en la máquina — de hecho Silero VAD (el único uso
   de torch en este proyecto, ver voice_activity_detector.py) corre en CPU
   siempre. nvidia-smi es la forma estándar de confirmar el driver sin
   depender de qué build de torch haya quedado instalada.

2. Instalador: en el .exe empaquetado, el instalador de Inno Setup deja (o
   no) un archivo marcador junto al ejecutable según si el usuario marcó
   "Habilitar aceleración GPU (CUDA)" al instalar (tildado por default —
   ver installer/voice_transcriber.iss). Corriendo desde código fuente
   (python main.py) esto no aplica: siempre se considera habilitado.

Esta función NO confirma que las DLLs de cuBLAS/cuDNN ya estén descargadas
(esas se bajan bajo demanda, ver cuda_runtime_downloader.py y
cuda_dll_setup.ensure_cuda_runtime_downloaded_and_registered, llamada recién
al arrancar una transcripción con CUDA elegido) — no tiene sentido esconder
la opción por no haber descargado todavía algo que se puede descargar al
toque.

Autor: Nero
"""

import os
import re
import shutil
import subprocess
import sys
from functools import lru_cache

from src.logging_utils import ComponentLogger
from src.startup.app_paths import get_app_root

logger = ComponentLogger("CUDA Setup")

# Nombre del archivo que installer/voice_transcriber.iss copia junto al .exe
# SOLO si el usuario dejó tildada la tarea "Habilitar aceleración GPU"
# durante la instalación (tildada por default).
_GPU_FEATURE_MARKER_FILENAME = "gpu_enabled.marker"

# Mínimo de CUDA que soporta el runtime que se descarga (ver
# cuda_runtime_downloader.py, ancla a paquetes de la serie 12.x).
MINIMUM_REQUIRED_CUDA_VERSION = (12, 0)


@lru_cache(maxsize=1)
def _is_gpu_feature_enabled_by_installer() -> bool:
    if not getattr(sys, "frozen", False):
        return True  # corriendo desde código fuente, no hay instalador de por medio
    return os.path.isfile(os.path.join(get_app_root(), _GPU_FEATURE_MARKER_FILENAME))


@lru_cache(maxsize=1)
def _is_nvidia_gpu_hardware_present() -> bool:
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return False

    try:
        result = subprocess.run(
            [nvidia_smi_path, "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        logger.warning(
            f"No se pudo ejecutar nvidia-smi ({error}); se asume que no hay GPU NVIDIA."
        )
        return False

    return result.returncode == 0 and bool(result.stdout.strip())


@lru_cache(maxsize=1)
def is_nvidia_gpu_available() -> bool:
    return _is_gpu_feature_enabled_by_installer() and _is_nvidia_gpu_hardware_present()


def gpu_unavailability_reason() -> str:
    """Solo tiene sentido llamarla cuando is_nvidia_gpu_available() ya dio
    False — explica CUÁL de los dos motivos aplica, para el hint de la GUI
    (ver src/gui/config_window.py._build_cuda_acceleration_section)."""
    if not _is_gpu_feature_enabled_by_installer():
        return (
            'La aceleración GPU está desactivada en esta instalación (no se marcó '
            '"Habilitar aceleración GPU" al instalar).'
        )
    return "No se detectó una GPU NVIDIA compatible con CUDA en este equipo."


@lru_cache(maxsize=1)
def _driver_max_cuda_version() -> tuple[int, int] | None:
    """
    Best-effort: lee la versión máxima de CUDA que soporta el driver
    instalado, parseando el encabezado de texto de `nvidia-smi` — no existe
    un --query-gpu estructurado para esto. El label cambió entre versiones
    de driver ("CUDA Version: 12.6" en la mayoría, "CUDA UMD Version: 13.3"
    en drivers más nuevos), así que se prueban ambos patrones. None si no se
    pudo determinar (mejor no bloquear por una lectura ambigua que arriesgar
    un falso negativo).
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return None

    try:
        result = subprocess.run([nvidia_smi_path], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = re.search(r"CUDA(?: UMD)? Version:\s*(\d+)\.(\d+)", result.stdout)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def check_driver_supports_cuda_runtime():
    """
    Avisa ANTES de arrancar la descarga de ~1.3GB del runtime de CUDA (ver
    cuda_dll_setup.ensure_cuda_runtime_downloaded_and_registered) si el
    driver instalado es claramente insuficiente — para no hacer esperar al
    usuario la descarga entera solo para enterarse al final de que no le
    sirve. Levanta RuntimeError con un mensaje accionable si la versión
    detectada es menor a la mínima requerida; no hace nada si no se pudo
    determinar la versión (se sigue igual, la descarga real es la
    verificación definitiva).
    """
    driver_max_version = _driver_max_cuda_version()
    if driver_max_version is None:
        return

    if driver_max_version < MINIMUM_REQUIRED_CUDA_VERSION:
        required = ".".join(map(str, MINIMUM_REQUIRED_CUDA_VERSION))
        found = ".".join(map(str, driver_max_version))
        raise RuntimeError(
            f"El driver de NVIDIA instalado soporta como máximo CUDA {found}, pero esta "
            f"app necesita CUDA {required}+. Actualiza el driver en "
            f"https://www.nvidia.com/drivers y vuelve a intentar, o desactiva "
            f'"Usar aceleración CUDA" para seguir en CPU.'
        )
