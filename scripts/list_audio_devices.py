"""
Utilidad de diagnóstico: lista los dispositivos de entrada de audio
disponibles, agrupados por nombre, mostrando el host API de cada entrada.

Un mismo micrófono físico suele aparecer varias veces (una por host API:
MME, Windows DirectSound, Windows WASAPI, Windows WDM-KS). Elegir la
entrada equivocada (WDM-KS) hace que el dispositivo se abra en modo
exclusivo real y bloquee otras apps (ej. OBS) sin avisar: OBS lo sigue
viendo "conectado" pero el nivel de audio se queda en 0.

Usar el NOMBRE del dispositivo (marcado abajo como "recomendado") en
AUDIO_INPUT_DEVICE_NAME dentro de config/settings.py — no hace falta
elegir el índice a mano, src/audio/device_resolver.py ya prioriza WASAPI
automáticamente entre las entradas que compartan ese nombre.

Ejecutar con: python scripts/list_audio_devices.py (desde la raíz del proyecto)
"""

from collections import defaultdict

import sounddevice as sd

RECOMMENDED_HOST_API_ORDER = [
    "Windows WASAPI",
    "MME",
    "Windows DirectSound",
    "Windows WDM-KS",
]

print("Host APIs disponibles:")
for host_api_index, host_api_info in enumerate(sd.query_hostapis()):
    print(f"  [{host_api_index}] {host_api_info['name']}")

devices_by_name: dict[str, list[dict]] = defaultdict(list)
for device_index, device_info in enumerate(sd.query_devices()):
    if device_info["max_input_channels"] <= 0:
        continue
    host_api_name = sd.query_hostapis(device_info["hostapi"])["name"]
    devices_by_name[device_info["name"]].append(
        {"index": device_index, "host_api_name": host_api_name}
    )

print("\nDispositivos de entrada disponibles (agrupados por nombre):\n")
for device_name, entries in devices_by_name.items():
    entries_sorted = sorted(
        entries,
        key=lambda e: RECOMMENDED_HOST_API_ORDER.index(e["host_api_name"])
        if e["host_api_name"] in RECOMMENDED_HOST_API_ORDER
        else len(RECOMMENDED_HOST_API_ORDER),
    )
    best_host_api_name = entries_sorted[0]["host_api_name"]

    print(f"'{device_name}'")
    for entry in entries_sorted:
        is_recommended = entry["host_api_name"] == best_host_api_name
        marker = "  <- recomendado (usar este nombre en la config)" if is_recommended else ""
        warning = ""
        if entry["host_api_name"] == "Windows WDM-KS":
            warning = "  [ADVERTENCIA: modo exclusivo real, bloquea otras apps]"
        print(f"  [{entry['index']}] host API: {entry['host_api_name']}{marker}{warning}")
    print()

print(
    "Para configurar: copia el nombre del dispositivo (sin corchetes) en "
    "AUDIO_INPUT_DEVICE_NAME dentro de config/settings.py. No hace falta "
    "el índice numérico ni elegir el host API a mano — el resolver ya prioriza "
    "WASAPI automáticamente."
)
