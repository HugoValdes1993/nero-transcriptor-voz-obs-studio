"""
En Windows, CTranslate2 (backend de faster-whisper) busca cublas64_12.dll y las
DLLs de cuDNN en el PATH del proceso. Instalar el CUDA Toolkit completo de NVIDIA
es una opción, pero es pesado e innecesario: los paquetes pip nvidia-cublas-cu12
y nvidia-cudnn-cu12 ya traen esas DLLs. Este módulo las registra en el proceso
actual con os.add_dll_directory, sin tocar el PATH del sistema operativo.

Debe importarse ANTES que faster_whisper (o cualquier módulo que dependa de él)
— por eso main.py lo importa como primera línea, antes que el resto del pipeline.
"""

import os
import sys

from src.logging_utils import ComponentLogger

logger = ComponentLogger("CUDA Setup")


def register_cuda_dll_directories():
    if sys.platform != "win32":
        return  # En Linux, estas DLLs se resuelven vía LD_LIBRARY_PATH automáticamente

    try:
        import nvidia
    except ImportError:
        logger.warning(
            "No se encontró el paquete 'nvidia'. Instalar con: "
            "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12"
        )
        return

    # 'nvidia' es un namespace package (sin __init__.py), por lo que no tiene
    # __file__; hay que usar __path__, que sí existe para namespace packages.
    nvidia_search_paths = list(getattr(nvidia, "__path__", []))
    if not nvidia_search_paths:
        logger.warning("El paquete 'nvidia' no expone rutas de búsqueda (__path__ vacío).")
        return

    nvidia_packages_root_directory = nvidia_search_paths[0]

    # cublas64_12.dll depende en cadena de otras DLLs de CUDA runtime
    # (nvrtc, cuda_runtime, etc.), cada una en su propio subpaquete pip.
    # Registramos TODAS las carpetas bin bajo nvidia/*, no solo cublas/cudnn,
    # para que Windows pueda resolver toda la cadena de dependencias.
    registered_any_directory = False
    dll_directories_to_prepend_to_path = []

    for subpackage_name in sorted(os.listdir(nvidia_packages_root_directory)):
        subpackage_directory = os.path.join(nvidia_packages_root_directory, subpackage_name)
        if not os.path.isdir(subpackage_directory):
            continue

        subpackage_bin_directory = os.path.join(subpackage_directory, "bin")
        if os.path.isdir(subpackage_bin_directory):
            os.add_dll_directory(subpackage_bin_directory)
            dll_directories_to_prepend_to_path.append(subpackage_bin_directory)
            logger.info(f"Registrado: {subpackage_bin_directory}")
            registered_any_directory = True
        else:
            logger.info(
                f"'{subpackage_name}' no tiene carpeta 'bin' "
                f"(contenido: {os.listdir(subpackage_directory)})"
            )

    if not registered_any_directory:
        logger.warning("No se registró ninguna carpeta bin de CUDA.")
        return

    # CRÍTICO: el binario compilado (.pyd) de CTranslate2 resuelve sus DLLs
    # dependientes mediante enlace dinámico estándar de Windows en tiempo de
    # import, el cual solo consulta el PATH del proceso — NO la lista de
    # add_dll_directory (esa lista solo aplica a LoadLibraryEx explícitos con
    # flags específicos). Por eso hace falta anteponer también al PATH.
    os.environ["PATH"] = (
        os.pathsep.join(dll_directories_to_prepend_to_path) + os.pathsep + os.environ.get("PATH", "")
    )
    logger.success("Carpetas de CUDA registradas y antepuestas al PATH del proceso.")


register_cuda_dll_directories()
