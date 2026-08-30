"""
En Windows, CTranslate2 (backend de faster-whisper) busca cublas64_12.dll y las
DLLs de cuDNN en el PATH del proceso. Estas DLLs NO se empaquetan con la app
(pesan ~1.3GB en total) — se descargan bajo demanda la primera vez que el
usuario activa aceleración CUDA (ver cuda_runtime_downloader.py) a un caché
en %LOCALAPPDATA%, y este módulo las registra en el proceso actual con
os.add_dll_directory, sin tocar el PATH del sistema operativo.

Dos funciones con responsabilidades distintas:

- register_available_cuda_dll_directories(): solo ESCANEA el caché de
  cuda_runtime_downloader y registra lo que ya esté ahí — barato, sin red.
  Debe importarse/llamarse ANTES que faster_whisper (o cualquier módulo que
  dependa de él) — por eso main.py la llama como primera línea, antes que el
  resto del pipeline. Si el usuario nunca activó CUDA todavía, esto
  simplemente no encuentra nada y no hace nada (no es un error).

- ensure_cuda_runtime_downloaded_and_registered(): si el escaneo de arriba
  no encontró nada, dispara la descarga (puede tardar minutos) y vuelve a
  registrar. Se llama UNA sola vez, desde
  src/server/transcription_pipeline.TranscriptionPipeline.__init__, y SOLO
  si el dispositivo elegido es "cuda" — nunca al arrancar la app sin más.
"""

import os

from src.logging_utils import ComponentLogger
from src.startup.cuda_availability import check_driver_supports_cuda_runtime
from src.startup.cuda_runtime_downloader import (
    download_and_extract_cuda_runtime,
    get_cuda_runtime_cache_dir,
    is_cuda_runtime_cached,
)

logger = ComponentLogger("CUDA Setup")


def _register_dll_directories_under(root_directory: str) -> bool:
    """Registra con os.add_dll_directory cada subcarpeta de `root_directory`
    que tenga una carpeta bin/, y las antepone también al PATH del proceso.
    Devuelve True si registró al menos una."""
    if not os.path.isdir(root_directory):
        return False

    dll_directories_to_prepend_to_path = []

    for subfolder_name in sorted(os.listdir(root_directory)):
        bin_directory = os.path.join(root_directory, subfolder_name, "bin")
        if os.path.isdir(bin_directory):
            os.add_dll_directory(bin_directory)
            dll_directories_to_prepend_to_path.append(bin_directory)
            logger.info(f"Registrado: {bin_directory}")

    if not dll_directories_to_prepend_to_path:
        return False

    # CRÍTICO: el binario compilado (.pyd) de CTranslate2 resuelve sus DLLs
    # dependientes mediante enlace dinámico estándar de Windows en tiempo de
    # import, el cual solo consulta el PATH del proceso — NO la lista de
    # add_dll_directory (esa lista solo aplica a LoadLibraryEx explícitos con
    # flags específicos). Por eso hace falta anteponer también al PATH.
    os.environ["PATH"] = (
        os.pathsep.join(dll_directories_to_prepend_to_path) + os.pathsep + os.environ.get("PATH", "")
    )
    return True


def register_available_cuda_dll_directories():
    if os.name != "nt":
        return  # En Linux, estas DLLs se resuelven vía LD_LIBRARY_PATH automáticamente

    if _register_dll_directories_under(get_cuda_runtime_cache_dir()):
        logger.success("Carpetas de CUDA registradas y antepuestas al PATH del proceso.")


def ensure_cuda_runtime_downloaded_and_registered():
    """
    Llamar SOLO cuando get_whisper_device() ya resolvió "cuda" — descarga el
    runtime si hace falta (ver cuda_runtime_downloader.download_and_extract_cuda_runtime,
    que reporta progreso vía src/status_hub.notify_status) y lo registra.
    """
    if not is_cuda_runtime_cached():
        # Chequeo barato ANTES de arrancar una descarga de ~1.3GB — evita hacer
        # esperar al usuario toda la descarga para enterarse recién al final
        # de que su driver de NVIDIA es muy viejo (ver su docstring).
        check_driver_supports_cuda_runtime()
        logger.info("Runtime de CUDA no encontrado en caché — descargando (primera vez, ~1.3GB)...")
        download_and_extract_cuda_runtime()

    register_available_cuda_dll_directories()


# Se ejecuta al importar el módulo (ver main.py, que lo importa como primera
# línea) — mismo patrón que antes: registra lo que YA esté cacheado de una
# corrida anterior, sin descargar nada. Si el usuario nunca usó CUDA todavía,
# no encuentra nada y no hace nada.
register_available_cuda_dll_directories()
