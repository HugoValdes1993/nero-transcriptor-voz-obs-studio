"""
Descarga y extrae las DLLs de CUDA que faster-whisper/CTranslate2 necesitan
para correr en GPU (cuBLAS, cuDNN, CUDA runtime, NVRTC) — la primera vez que
el usuario activa "Usar aceleración CUDA", no como parte de la instalación.

Por qué: esas DLLs vienen de 4 paquetes pip (nvidia-cublas-cu12,
nvidia-cudnn-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12) que pesan
~1.3GB en total de descarga (confirmado con una corrida real de este
módulo) — la GPU misma no viene con ellas (eso es el driver de NVIDIA, ver
cuda_availability.py), son librerías aparte. En vez de empaquetarlas
siempre (duplicaría el instalador para cualquiera que no tenga GPU), se bajan
bajo demanda a un caché de usuario, igual que faster-whisper ya hace con el
modelo de Whisper.

Las 4 ruedas (.whl) de estos paquetes son "py3-none-win_amd64" — universales
para cualquier Python 3.x de 64 bits en Windows, no atadas a una versión de
CPython puntual — así que no hace falta resolver cuál wheel matchea el
intérprete actual. Un .whl es un .zip común: se descarga y se extraen
directo las carpetas "nvidia/<subpaquete>/bin/*" que cuda_dll_setup.py ya
sabe registrar, sin necesitar pip ni un venv.

Autor: Nero
"""

import hashlib
import json
import os
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass

from src.logging_utils import ComponentLogger
from src.status_hub import notify_status

logger = ComponentLogger("CUDA Setup")

PYPI_JSON_URL_TEMPLATE = "https://pypi.org/pypi/{package}/{version}/json"

# Versiones ancladas a mano (las que están probadas en el entorno de
# desarrollo de este proyecto — ver `pip show nvidia-cublas-cu12` etc.). Si
# se suben en el futuro, subir también CACHE_VERSION_TAG de abajo para que
# no se reuse por error un caché de una combinación vieja/incompatible.
_CUDA_RUNTIME_PACKAGES = {
    "nvidia-cublas-cu12": "12.9.2.10",
    "nvidia-cudnn-cu12": "9.24.0.43",
    "nvidia-cuda-runtime-cu12": "12.9.79",
    "nvidia-cuda-nvrtc-cu12": "12.9.86",
}

# Carpeta dentro de cada paquete que coincide con el nombre que
# cuda_dll_setup.py espera ver bajo "nvidia/" (ej. "nvidia-cublas-cu12" ->
# "nvidia/cublas/bin/*.dll").
_PACKAGE_TO_SUBFOLDER = {
    "nvidia-cublas-cu12": "cublas",
    "nvidia-cudnn-cu12": "cudnn",
    "nvidia-cuda-runtime-cu12": "cuda_runtime",
    "nvidia-cuda-nvrtc-cu12": "cuda_nvrtc",
}

CACHE_VERSION_TAG = "-".join(_CUDA_RUNTIME_PACKAGES.values())

_DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


@dataclass
class _WheelInfo:
    url: str
    sha256: str
    size: int


def get_cuda_runtime_cache_dir() -> str:
    """
    Carpeta donde se extraen las DLLs descargadas — en el perfil del
    usuario (%LOCALAPPDATA%), NO adentro de la carpeta de instalación: así
    funciona igual corriendo desde código fuente o desde el .exe empaquetado
    (que puede vivir en Program Files, sin permiso de escritura), y
    sobrevive a una desinstalación/reinstalación de la app. El tag de
    versión en la ruta evita reusar un caché de una combinación de versiones
    vieja si el día de mañana se actualizan los pins de arriba.
    """
    local_app_data = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(local_app_data, "VoiceTranscriber", "cuda-runtime", CACHE_VERSION_TAG)


def is_cuda_runtime_cached() -> bool:
    """True si las 4 subcarpetas ya están extraídas con su carpeta bin/."""
    cache_dir = get_cuda_runtime_cache_dir()
    return all(
        os.path.isdir(os.path.join(cache_dir, subfolder, "bin"))
        for subfolder in _PACKAGE_TO_SUBFOLDER.values()
    )


def _fetch_wheel_info(package_name: str, version: str) -> _WheelInfo:
    url = PYPI_JSON_URL_TEMPLATE.format(package=package_name, version=version)
    request = urllib.request.Request(url, headers={"User-Agent": "voice-transcriber-cuda-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            metadata = json.load(response)
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(
            f"No se pudo consultar PyPI para {package_name}=={version} "
            f"(¿hay conexión a internet?): {error}"
        ) from error

    wheel_entry = next(
        (
            entry
            for entry in metadata["urls"]
            if entry["packagetype"] == "bdist_wheel" and entry["filename"].endswith("-win_amd64.whl")
        ),
        None,
    )
    if wheel_entry is None:
        raise RuntimeError(
            f"No se encontró una rueda win_amd64 para {package_name}=={version} en PyPI."
        )

    return _WheelInfo(
        url=wheel_entry["url"],
        sha256=wheel_entry["digests"]["sha256"],
        size=wheel_entry["size"],
    )


def _download_wheel_to(
    wheel_info: _WheelInfo,
    destination_path: str,
    progress_label: str,
    bytes_downloaded_before: int,
    total_size: int,
) -> int:
    """
    Descarga en streaming a `destination_path` + ".part", verificando el
    sha256 al final — solo se renombra al nombre definitivo si matchea, para
    que un .part a medio bajar nunca se confunda con uno completo.

    `bytes_downloaded_before`/`total_size` son sobre el total de los 4
    paquetes (no solo este archivo) — así el progreso reportado a la GUI
    (ver status_hub.notify_status) es una barra continua de 0 a 1 en vez de
    reiniciar a 0% cuatro veces. Devuelve el total acumulado de bytes ya
    bajados (para pasárselo a la siguiente llamada).
    """
    part_path = destination_path + ".part"
    sha256 = hashlib.sha256()
    bytes_downloaded_this_file = 0

    request = urllib.request.Request(wheel_info.url, headers={"User-Agent": "voice-transcriber-cuda-downloader"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response, open(part_path, "wb") as part_file:
            while True:
                chunk = response.read(_DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                part_file.write(chunk)
                sha256.update(chunk)
                bytes_downloaded_this_file += len(chunk)

                total_downloaded = bytes_downloaded_before + bytes_downloaded_this_file
                downloaded_mb = total_downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                notify_status(
                    f"Descargando runtime de CUDA — {progress_label}: "
                    f"{downloaded_mb:.0f}/{total_mb:.0f} MB...",
                    progress=(total_downloaded / total_size) if total_size else None,
                )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        if os.path.exists(part_path):
            os.remove(part_path)
        raise RuntimeError(f"Falló la descarga de {progress_label}: {error}") from error

    if sha256.hexdigest() != wheel_info.sha256:
        os.remove(part_path)
        raise RuntimeError(
            f"El archivo descargado de {progress_label} no coincide con el checksum esperado "
            f"(descarga corrupta o interrumpida) — reintentá."
        )

    os.replace(part_path, destination_path)
    return bytes_downloaded_before + bytes_downloaded_this_file


def _extract_bin_directory(wheel_path: str, subfolder_name: str, cache_dir: str):
    """Extrae solo las entradas 'nvidia/<subfolder_name>/bin/*' de la rueda
    (un .whl es un .zip común) directo a cache_dir/<subfolder_name>/bin/."""
    prefix = f"nvidia/{subfolder_name}/bin/"
    destination_bin_dir = os.path.join(cache_dir, subfolder_name, "bin")
    os.makedirs(destination_bin_dir, exist_ok=True)

    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            if not member.startswith(prefix) or member.endswith("/"):
                continue
            file_name = member[len(prefix):]
            with archive.open(member) as source, open(
                os.path.join(destination_bin_dir, file_name), "wb"
            ) as target:
                target.write(source.read())


def download_and_extract_cuda_runtime():
    """
    Descarga los 4 paquetes ancla en _CUDA_RUNTIME_PACKAGES y deja sus DLLs
    listas en get_cuda_runtime_cache_dir(). Idempotente: si is_cuda_runtime_cached()
    ya es True no hace falta llamar a esto, pero llamarlo de nuevo no rompe
    nada (vuelve a descargar y sobrescribe).
    """
    cache_dir = get_cuda_runtime_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    package_names = list(_CUDA_RUNTIME_PACKAGES.keys())

    logger.info("Resolviendo descargas del runtime de CUDA...")
    notify_status("Resolviendo descargas del runtime de CUDA...", progress=0.0)
    wheel_infos = {
        package_name: _fetch_wheel_info(package_name, _CUDA_RUNTIME_PACKAGES[package_name])
        for package_name in package_names
    }
    # Se resuelven los 4 tamaños ANTES de descargar nada para poder mostrar
    # una barra de progreso continua de 0 a 1 sobre el total combinado, en
    # vez de que se reinicie a 0% cuatro veces (una por paquete).
    total_size = sum(info.size for info in wheel_infos.values())
    bytes_downloaded_so_far = 0

    for index, package_name in enumerate(package_names, start=1):
        version = _CUDA_RUNTIME_PACKAGES[package_name]
        subfolder_name = _PACKAGE_TO_SUBFOLDER[package_name]
        progress_label = f"{package_name} ({index}/{len(package_names)})"
        wheel_info = wheel_infos[package_name]

        wheel_path = os.path.join(cache_dir, f"{package_name}-{version}.whl")
        bytes_downloaded_so_far = _download_wheel_to(
            wheel_info, wheel_path, progress_label, bytes_downloaded_so_far, total_size
        )
        _extract_bin_directory(wheel_path, subfolder_name, cache_dir)
        os.remove(wheel_path)
        logger.success(f"{package_name} listo.")

    notify_status("")
