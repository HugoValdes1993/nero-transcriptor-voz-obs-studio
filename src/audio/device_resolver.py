"""
Resuelve un dispositivo de entrada de audio a partir de su NOMBRE (no de su
índice numérico crudo).

Por qué esto es necesario: un mismo micrófono físico suele aparecer varias
veces en `sounddevice.query_devices()`, una entrada por cada host API que
Windows expone para ese hardware (MME, Windows DirectSound, Windows WASAPI,
Windows WDM-KS). Elegir por índice numérico es frágil por dos razones:

  1. El índice puede cambiar entre reinicios de Windows si se conecta o
     desconecta cualquier otro dispositivo de audio del sistema.
  2. Si por casualidad el índice elegido corresponde a la entrada WDM-KS del
     dispositivo en lugar de la WASAPI, el stream se abre accediendo al
     hardware directamente sin pasar por el mezclador de audio de Windows.
     Esto bloquea el dispositivo para cualquier otra aplicación (ej. OBS),
     que lo sigue viendo "conectado" pero deja de recibir nivel de audio,
     aunque el código pida explícitamente modo compartido (`exclusive=False`)
     — esa opción de shared/exclusive es específica de WASAPI y no tiene
     ningún efecto sobre un stream WDM-KS.

Por eso este resolver siempre prioriza, entre todas las entradas que
coincidan con el nombre buscado, la que tenga host API WASAPI.
"""

import sounddevice as sd

from src.logging_utils import ComponentLogger

logger = ComponentLogger("DeviceResolver")

# Orden de preferencia de host APIs si WASAPI no está disponible para el
# dispositivo buscado. WDM-KS no aparece en esta lista a propósito: nunca es
# una opción elegible (ver resolve_input_device_candidates), es el que causa
# el bloqueo de exclusividad descrito arriba.
HOST_API_PREFERENCE_ORDER = [
    "Windows WASAPI",
    "MME",
    "Windows DirectSound",
]

WDM_KS_HOST_API_NAME = "Windows WDM-KS"


class InputDeviceNotFoundError(Exception):
    pass


def _host_api_name(device_info: dict) -> str:
    return sd.query_hostapis(device_info["hostapi"])["name"]


def find_matching_input_devices(name_substring: str) -> list[dict]:
    """
    Retorna todas las entradas de entrada de audio cuyo nombre contiene
    `name_substring` (case-insensitive), sin importar el host API.
    Cada dict incluye además la clave 'host_api_name' para inspección.
    """
    name_substring_lower = name_substring.lower()
    matches = []
    for device_index, device_info in enumerate(sd.query_devices()):
        if device_info["max_input_channels"] <= 0:
            continue
        if name_substring_lower not in device_info["name"].lower():
            continue
        enriched_device_info = dict(device_info)
        enriched_device_info["index"] = device_index
        enriched_device_info["host_api_name"] = _host_api_name(device_info)
        matches.append(enriched_device_info)
    return matches


def resolve_input_device_candidates(
    name_substring: str, host_api_override: str | None = None
) -> list[dict]:
    """
    Busca todas las entradas de entrada que coincidan con `name_substring` y
    retorna una LISTA ordenada de mejor a peor candidata para abrir el
    stream (no una sola elegida). El llamador debe intentar abrir el stream
    con cada candidata en orden y quedarse con la primera que realmente
    arranque: algunos drivers (ej. ciertos auriculares Bluetooth
    "combinados", igual que el control DualSense) reportan la entrada WASAPI
    como disponible pero fallan recién al iniciar el stream con
    'PaErrorCode -9999: WdmSyncIoctl ...' — un error que solo se puede
    detectar intentándolo, no consultando sus propiedades de antemano.

    Si `host_api_override` está definido, la(s) entrada(s) cuyo host API
    coincida (substring, case-insensitive) con ese valor quedan primeras en
    la lista — sirve para saltar directo a un host API que ya se sabe que
    funciona para ese dispositivo puntual (ver comentario en
    config/settings.py) — pero el resto de las entradas seguras del mismo
    dispositivo se mantienen como fallback si incluso esa falla.

    Las entradas WDM-KS quedan descartadas SIEMPRE, incluso como único
    resultado disponible o vía host_api_override manual: acceden al
    hardware en modo exclusivo real por diseño, sin importar la config de
    exclusividad que se les pida, y bloquearían el dispositivo para
    cualquier otra app (ej. OBS) — bajo ninguna circunstancia se debe abrir
    un stream así.

    Lanza InputDeviceNotFoundError si no hay ninguna coincidencia segura
    (sin contar WDM-KS).
    """
    matching_devices = find_matching_input_devices(name_substring)

    if not matching_devices:
        available_names = sorted(
            {
                device_info["name"]
                for device_info in sd.query_devices()
                if device_info["max_input_channels"] > 0
            }
        )
        raise InputDeviceNotFoundError(
            f"Ningún dispositivo de entrada coincide con '{name_substring}'. "
            f"Dispositivos de entrada disponibles: {available_names}. "
            f"Corre scripts/list_audio_devices.py para ver el detalle completo."
        )

    safe_devices = [d for d in matching_devices if d["host_api_name"] != WDM_KS_HOST_API_NAME]
    if not safe_devices:
        raise InputDeviceNotFoundError(
            f"'{name_substring}' solo tiene disponible una entrada WDM-KS, que "
            f"accedería al hardware en modo exclusivo real y bloquearía el "
            f"dispositivo para otras apps (ej. OBS). No se puede usar de forma "
            f"segura — conecta el dispositivo por otra vía (ej. Bluetooth/USB "
            f"con driver WASAPI) o elige otro micrófono."
        )

    if host_api_override is not None and "wdm-ks" in host_api_override.lower():
        raise ValueError(
            "No se permite forzar el host API a WDM-KS: accede al hardware "
            "en modo exclusivo real y bloquearía el dispositivo para otras "
            "apps (ej. OBS)."
        )

    def preference_rank(device_info: dict) -> int:
        try:
            return HOST_API_PREFERENCE_ORDER.index(device_info["host_api_name"])
        except ValueError:
            return len(HOST_API_PREFERENCE_ORDER)  # host API desconocido: al final

    safe_devices.sort(key=preference_rank)

    if host_api_override is not None:
        host_api_override_lower = host_api_override.lower()
        overridden_first, rest = (
            [d for d in safe_devices if host_api_override_lower in d["host_api_name"].lower()],
            [d for d in safe_devices if host_api_override_lower not in d["host_api_name"].lower()],
        )
        if overridden_first:
            safe_devices = overridden_first + rest
            logger.info(
                f"Override manual de host API activo: se probará primero "
                f"'{overridden_first[0]['name']}' vía "
                f"{overridden_first[0]['host_api_name']} "
                f"(índice {overridden_first[0]['index']})"
            )
        else:
            logger.warning(
                f"'{name_substring}' no tiene ninguna entrada segura con host "
                f"API '{host_api_override}'; se ignora el override y se prueba "
                f"con el orden de preferencia normal."
            )

    if len(safe_devices) > 1:
        entries_description = ", ".join(
            f"{d['name']!r} vía {d['host_api_name']} (índice {d['index']})"
            for d in safe_devices
        )
        logger.info(
            f"'{name_substring}' coincide con {len(safe_devices)} entradas "
            f"seguras, se probarán en este orden si hace falta: "
            f"{entries_description}"
        )

    return safe_devices


def list_available_input_device_names() -> list[str]:
    """
    Nombres de dispositivos de entrada disponibles ahora mismo, para poblar
    el selector de micrófono de la GUI. Un mismo dispositivo físico aparece
    repetido por cada host API de Windows (ver docstring del módulo); aquí se
    devuelve un solo nombre por dispositivo, sin duplicar, en el mismo orden
    que reporta sounddevice.
    """
    seen_names = set()
    ordered_names = []
    for device_info in sd.query_devices():
        if device_info["max_input_channels"] <= 0:
            continue
        name = device_info["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        ordered_names.append(name)
    return ordered_names


def get_default_input_device_name() -> str | None:
    """Nombre del dispositivo de entrada que Windows tiene como predeterminado
    del sistema, o None si no hay ninguno configurado."""
    default_input_index = sd.default.device[0]
    if default_input_index is None or default_input_index < 0:
        return None
    return sd.query_devices(default_input_index, kind="input")["name"]


def resolve_input_device_candidates_or_default(
    name_substring: str, host_api_override: str | None = None
) -> tuple[list[dict], bool]:
    """
    Igual que resolve_input_device_candidates, pero en vez de lanzar
    InputDeviceNotFoundError cuando el dispositivo configurado no está
    disponible (ej. un micrófono USB o un control inalámbrico que se
    desconectó), cae a las candidatas del dispositivo de entrada por
    defecto del sistema operativo — así la app nunca deja de arrancar por
    esto.

    El default del sistema también queda sujeto a la misma regla de nunca
    usar WDM-KS (ver resolve_input_device_candidates): si su ÚNICA entrada
    disponible es WDM-KS, se propaga el error en vez de abrirlo en
    exclusivo.

    Retorna (candidatas, se_uso_fallback).
    """
    try:
        return resolve_input_device_candidates(name_substring, host_api_override), False
    except InputDeviceNotFoundError as error:
        default_input_index = sd.default.device[0]
        if default_input_index is None or default_input_index < 0:
            raise  # ni siquiera hay un default del sistema: no hay nada que hacer

        default_device_info = sd.query_devices(default_input_index, kind="input")
        try:
            default_candidates = resolve_input_device_candidates(default_device_info["name"])
        except InputDeviceNotFoundError as default_error:
            raise InputDeviceNotFoundError(
                f"{error} Tampoco se puede usar el dispositivo de entrada por "
                f"defecto del sistema ('{default_device_info['name']}'): "
                f"{default_error}"
            ) from error

        logger.warning(
            f"{error} Usando el dispositivo de entrada por defecto del "
            f"sistema en su lugar: '{default_device_info['name']}'."
        )
        return default_candidates, True
