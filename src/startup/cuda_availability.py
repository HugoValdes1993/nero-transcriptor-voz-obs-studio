"""
Detecta si hay una GPU NVIDIA con soporte CUDA disponible en este equipo, para
decidir si el checkbox de "aceleración CUDA" de la GUI puede estar activo.

Se detecta vía `nvidia-smi` (el driver de NVIDIA), NO vía
torch.cuda.is_available(): torch, tal como queda instalado por
requirements.txt, puede terminar siendo una build CPU-only (según pip/índice
usado) sin que eso tenga relación con si la GPU y el driver de NVIDIA están
realmente disponibles en la máquina — de hecho Silero VAD (el único uso de
torch en este proyecto, ver voice_activity_detector.py) corre en CPU siempre.
La aceleración CUDA que de verdad importa aquí es la de faster-whisper /
CTranslate2 (ver cuda_dll_setup.py), que solo necesita el driver de NVIDIA
instalado — nvidia-smi es la forma estándar de confirmar eso sin depender de
qué build de torch haya quedado instalada.

Autor: Nero
"""

import shutil
import subprocess
from functools import lru_cache

from src.logging_utils import ComponentLogger

logger = ComponentLogger("CUDA Setup")


@lru_cache(maxsize=1)
def is_nvidia_gpu_available() -> bool:
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
