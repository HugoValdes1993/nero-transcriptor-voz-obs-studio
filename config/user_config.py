"""
Persistencia de las opciones que el usuario cambia desde la GUI: dispositivo
de entrada de audio, aceleración CUDA, flujo de transcripción/traducción, y
estilo visual del overlay (posición, colores, tipografía).

config/settings.py define los defaults "de fábrica" del proyecto; este
módulo guarda ENCIMA las decisiones que el usuario toma en tiempo de
ejecución, en un JSON al lado del código (user_config.json en la raíz del
proyecto), para que:
  - sobrevivan a un reinicio de la app.
  - no haga falta editar config/settings.py a mano ni tener el código fuente
    a mano (pensado para cuando esto sea un .exe distribuible).

El resto del código debe leer estos valores a través de los getters de aquí
(nunca importando el constante de settings.py directo) para que un cambio
hecho desde la GUI se refleje la próxima vez que arranca el pipeline, sin
reiniciar el proceso — algo que no pasaría con un `from config.settings
import X`, porque ese import queda fijado al valor que tenía al momento de
cargarse el módulo.

Los setters SOLO actualizan la copia en memoria (_user_config); no escriben
a disco. El guardado en disco es una acción explícita del usuario —
save_user_config(), disparada por el botón "Guardar configuración" de la
GUI— para que quede claro cuándo un cambio queda persistido de verdad, en
vez de escribir el archivo en cada tecla/slider. Mientras tanto, el resto de
la app sigue viendo el valor más reciente igual (getters leen la memoria),
así que arrancar el pipeline sin haber guardado usa la selección actual sin
problema; lo único que se pierde si se cierra la app sin guardar es que la
PRÓXIMA vez que arranque no la recuerde.

Al importar este módulo, `_ensure_defaults_persisted` rellena en el JSON
cualquier parámetro que todavía no exista (primera ejecución: el archivo no
existe y quedan TODOS; ejecuciones posteriores a agregar una config nueva:
solo la que falta) con su valor default actual, y lo escribe a disco. Así
`user_config.json` siempre queda con parámetros explícitos y conocidos en
vez de depender de que el código recuerde el default cada vez que falta una
clave. Agregar una config nueva en el futuro es: sumarla a
`_TOP_LEVEL_DEFAULT_RESOLVERS` (o a OVERLAY_STYLE_DEFAULTS si es de estilo
del overlay) — se autocompleta sola la próxima vez que arranque la app,
tanto para usuarios nuevos como para los que ya tenían un JSON viejo.
"""

import json
import os

from config.settings import (
    AUDIO_INPUT_DEVICE_NAME as DEFAULT_AUDIO_INPUT_DEVICE_NAME,
    WHISPER_SOURCE_LANGUAGE as DEFAULT_WHISPER_SOURCE_LANGUAGE,
)
from src.logging_utils import ComponentLogger
from src.startup.app_paths import get_app_root
from src.startup.cuda_availability import is_nvidia_gpu_available

logger = ComponentLogger("UserConfig")

_USER_CONFIG_PATH = os.path.join(get_app_root(), "user_config.json")

# Los dos únicos flujos de transcripción/traducción soportados por la GUI
# (ver _build_translation_direction_section en config_window.py). No es una
# lista abierta de idiomas porque Argos Translate necesita el paquete
# instalado para cada par, y por ahora solo se distribuye/prueba con estos
# dos; agregar un tercer idioma implica sumar otra opción aquí y en la GUI.
TRANSLATION_DIRECTION_ES_TO_EN = "es_en"
TRANSLATION_DIRECTION_EN_TO_ES = "en_es"

# Si settings.py ya trae "en" como idioma de origen, el default de fábrica
# coherente es EN_TO_ES; para cualquier otro valor (incluido "es"), ES_TO_EN.
_DEFAULT_TRANSLATION_DIRECTION = (
    TRANSLATION_DIRECTION_EN_TO_ES
    if DEFAULT_WHISPER_SOURCE_LANGUAGE == "en"
    else TRANSLATION_DIRECTION_ES_TO_EN
)

# Estilo visual del overlay. El original y la traducción viven en dos
# páginas separadas (overlay/obs_overlay_original.html y
# overlay/obs_overlay_translated.html, cada una su propio Browser Source en
# OBS, posicionable de forma independiente), por eso las propiedades
# "generales" de aquí abajo son solo layout/fondo del cuadro — cada página
# las aplica sobre su única línea de texto, no hay nada compartido entre
# ambas más que este mismo look. Reproduce el look de fábrica original: si
# el usuario nunca tocó ningún control, el overlay se ve exactamente igual
# que antes de que existiera esta configuración.
# Efectos de aparición del texto FINAL de un subtítulo (ver
# src/gui/config_window.py._build_animation_row y el JS de
# overlay/obs_overlay_*.html). "none" reproduce el look de fábrica de
# siempre — el texto tentativo (parcial) nunca anima, solo el final.
SUBTITLE_ANIMATION_CHOICES = ("none", "fade", "bounce", "glitch")

OVERLAY_STYLE_DEFAULTS = {
    # General: layout y fondo del cuadro (aplica igual en ambas páginas).
    "bottom_offset_px": 40,
    "padding_px": 12,
    "border_radius_px": 10,
    "background_color": "#000000",
    "background_opacity": 0.55,
    "clear_delay_ms": 6000,
    # Texto original. width_px/height_px son el tamaño EXACTO de la zona de
    # texto (no un máximo relativo): así se puede hacer coincidir con la
    # sección del lienzo que el usuario ya tiene reservada en su escena de
    # OBS para los subtítulos, en vez de depender de un porcentaje relativo
    # al tamaño del Browser Source.
    "original_font_family": "Segoe UI",
    "original_font_size_px": 34,
    "original_text_color": "#ffffff",
    "original_text_opacity": 1.0,
    "original_font_weight": "700",
    "original_width_px": 800,
    "original_height_px": 2160,
    # Contorno de texto (-webkit-text-stroke). Ancho 0 = sin contorno, mismo
    # look de fábrica que antes de que existiera esta opción.
    "original_text_stroke_color": "#000000",
    "original_text_stroke_width_px": 0,
    "original_animation": "none",
    # Texto traducido.
    "translated_font_family": "Segoe UI",
    "translated_font_size_px": 26,
    "translated_text_color": "#ffffff",
    "translated_text_opacity": 1.0,
    "translated_font_weight": "700",
    "translated_width_px": 800,
    "translated_height_px": 2160,
    "translated_text_stroke_color": "#000000",
    "translated_text_stroke_width_px": 0,
    "translated_animation": "none",
}

# Looks completos preconfigurados, de fábrica (no se guardan en
# user_config.json, no se pueden borrar ni sobrescribir — ver
# is_built_in_style_preset). Cada uno es un override parcial sobre
# OVERLAY_STYLE_DEFAULTS, para no repetir las claves que no cambian.
BUILT_IN_STYLE_PRESETS = {
    "Clásico": dict(OVERLAY_STYLE_DEFAULTS),
    "Alto contraste": {
        **OVERLAY_STYLE_DEFAULTS,
        "background_color": "#000000",
        "background_opacity": 0.85,
        "original_text_color": "#ffffff",
        "original_text_stroke_color": "#000000",
        "original_text_stroke_width_px": 3,
        "translated_text_color": "#ffffff",
        "translated_text_stroke_color": "#000000",
        "translated_text_stroke_width_px": 3,
    },
    "Minimalista sin fondo": {
        **OVERLAY_STYLE_DEFAULTS,
        "background_opacity": 0.0,
        "original_text_stroke_color": "#000000",
        "original_text_stroke_width_px": 2,
        "translated_text_stroke_color": "#000000",
        "translated_text_stroke_width_px": 2,
    },
    "Neón": {
        **OVERLAY_STYLE_DEFAULTS,
        "background_color": "#000000",
        "background_opacity": 0.4,
        "original_text_color": "#39ff14",
        "original_text_stroke_color": "#003b00",
        "original_text_stroke_width_px": 1,
        "original_animation": "glitch",
        "translated_text_color": "#00e5ff",
        "translated_text_stroke_color": "#003b3b",
        "translated_text_stroke_width_px": 1,
        "translated_animation": "glitch",
    },
}


def _load_raw() -> dict:
    if not os.path.exists(_USER_CONFIG_PATH):
        return {}
    try:
        with open(_USER_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            return json.load(config_file)
    except (json.JSONDecodeError, OSError) as error:
        logger.warning(f"No se pudo leer '{_USER_CONFIG_PATH}' ({error}); se usan los defaults.")
        return {}


def _save_raw(data: dict):
    try:
        with open(_USER_CONFIG_PATH, "w", encoding="utf-8") as config_file:
            json.dump(data, config_file, indent=2, ensure_ascii=False)
    except OSError as error:
        logger.warning(f"No se pudo guardar '{_USER_CONFIG_PATH}' ({error}).")


# Resolvers (no constantes planas) porque algunos defaults se calculan en
# base al hardware/entorno real de la máquina (ej. CUDA) en vez de ser un
# valor fijo — se evalúan una sola vez, al completar el JSON por primera vez,
# y de ahí en adelante lo guardado en disco manda.
_TOP_LEVEL_DEFAULT_RESOLVERS = {
    "audio_input_device_name": lambda: DEFAULT_AUDIO_INPUT_DEVICE_NAME,
    "cuda_acceleration_enabled": is_nvidia_gpu_available,
    "translation_direction": lambda: _DEFAULT_TRANSLATION_DIRECTION,
}


def _ensure_defaults_persisted(config: dict) -> dict:
    """
    Rellena en `config` (in-place) cualquier parámetro de nivel superior o de
    overlay_style que todavía no exista, con su valor default actual, y
    guarda el resultado en disco si hubo algún cambio.
    """
    changed = False

    for key, resolve_default in _TOP_LEVEL_DEFAULT_RESOLVERS.items():
        if key not in config:
            config[key] = resolve_default()
            changed = True

    overlay_style = config.get("overlay_style", {})
    for key, default_value in OVERLAY_STYLE_DEFAULTS.items():
        if key not in overlay_style:
            overlay_style[key] = default_value
            changed = True
    config["overlay_style"] = overlay_style

    if changed:
        _save_raw(config)

    return config


# Se carga una sola vez en memoria al importar el módulo (completando
# cualquier default que falte, lo que sí se persiste de inmediato — ver
# _ensure_defaults_persisted). De ahí en adelante, los setters solo tocan
# esta copia en memoria: quedan aplicados al toque para el resto de la app
# (ej. el pipeline arranca con la última selección aunque no se haya
# guardado), pero el archivo en disco no cambia hasta que se llama
# explícitamente a save_user_config() — ver el botón "Guardar configuración"
# en config_window.py.
_user_config = _ensure_defaults_persisted(_load_raw())


def save_user_config():
    """Persiste en disco el estado actual en memoria. Disparado por el
    usuario desde la GUI (botón "Guardar configuración"); nada más en este
    módulo escribe el archivo a partir de un cambio de un setter."""
    _save_raw(_user_config)
    logger.info(f"Configuración guardada en '{_USER_CONFIG_PATH}'.")


def get_audio_input_device_name() -> str:
    return _user_config["audio_input_device_name"]


def set_audio_input_device_name(device_name: str):
    if _user_config.get("audio_input_device_name") == device_name:
        return
    _user_config["audio_input_device_name"] = device_name
    logger.info(f"Dispositivo de entrada seleccionado: '{device_name}'")


def get_cuda_acceleration_enabled() -> bool:
    return _user_config["cuda_acceleration_enabled"]


def set_cuda_acceleration_enabled(enabled: bool):
    if _user_config.get("cuda_acceleration_enabled") == enabled:
        return
    _user_config["cuda_acceleration_enabled"] = enabled
    logger.info(f"Aceleración CUDA {'activada' if enabled else 'desactivada'}.")


def get_translation_direction() -> str:
    return _user_config["translation_direction"]


def set_translation_direction(direction: str):
    if direction not in (TRANSLATION_DIRECTION_ES_TO_EN, TRANSLATION_DIRECTION_EN_TO_ES):
        raise ValueError(f"Flujo de traducción inválido: {direction!r}")
    if _user_config.get("translation_direction") == direction:
        return
    _user_config["translation_direction"] = direction
    logger.info(f"Flujo de transcripción/traducción seleccionado: '{direction}'")


def get_whisper_source_language() -> str:
    """Idioma que se habla frente al micrófono (el que transcribe Whisper)."""
    return "en" if get_translation_direction() == TRANSLATION_DIRECTION_EN_TO_ES else "es"


def get_translation_target_language() -> str:
    """Idioma al que se traduce el texto transcrito para el overlay."""
    return "es" if get_translation_direction() == TRANSLATION_DIRECTION_EN_TO_ES else "en"


def get_overlay_style() -> dict:
    return _user_config["overlay_style"]


def set_overlay_style_value(key: str, value):
    if key not in OVERLAY_STYLE_DEFAULTS:
        raise ValueError(f"Propiedad de estilo de overlay desconocida: {key!r}")

    overlay_style = _user_config["overlay_style"]
    if overlay_style.get(key) == value:
        return

    overlay_style[key] = value


def get_overlay_style_presets() -> dict:
    """Built-ins (ver BUILT_IN_STYLE_PRESETS) + los que el usuario haya
    guardado (persistidos bajo "overlay_style_presets" en user_config.json)."""
    return {**BUILT_IN_STYLE_PRESETS, **_user_config.get("overlay_style_presets", {})}


def is_built_in_style_preset(name: str) -> bool:
    return name in BUILT_IN_STYLE_PRESETS


def save_overlay_style_preset(name: str):
    """Guarda el overlay_style ACTUAL bajo `name`. Se guarda en memoria como
    el resto de los setters de este módulo — no queda en disco hasta el
    próximo save_user_config()."""
    if is_built_in_style_preset(name):
        raise ValueError(f"'{name}' es un preset de fábrica y no se puede sobrescribir.")
    _user_config.setdefault("overlay_style_presets", {})[name] = dict(_user_config["overlay_style"])
    logger.info(f"Preset de estilo guardado: '{name}'")


def delete_overlay_style_preset(name: str):
    if is_built_in_style_preset(name):
        raise ValueError(f"'{name}' es un preset de fábrica y no se puede eliminar.")
    _user_config.get("overlay_style_presets", {}).pop(name, None)
    logger.info(f"Preset de estilo eliminado: '{name}'")


def apply_overlay_style_preset(name: str) -> dict:
    """
    Reemplaza el overlay_style actual por el del preset `name`, mutando el
    dict EN VEZ de reemplazarlo — ConfigWindow._overlay_style y el
    OverlayServer del PipelineController guardan una referencia directa a
    este mismo dict (ver get_overlay_style, que ya devuelve la referencia
    viva en vez de una copia), así que reemplazar el objeto rompería esa
    referencia compartida.

    Rellena con OVERLAY_STYLE_DEFAULTS cualquier clave que el preset no
    tenga (ej. un preset guardado antes de agregar una propiedad nueva),
    mismo criterio defensivo que _ensure_defaults_persisted.
    """
    presets = get_overlay_style_presets()
    if name not in presets:
        raise ValueError(f"Preset de estilo desconocido: {name!r}")

    preset_values = presets[name]
    overlay_style = _user_config["overlay_style"]
    overlay_style.clear()
    for key, default_value in OVERLAY_STYLE_DEFAULTS.items():
        overlay_style[key] = preset_values.get(key, default_value)
    return overlay_style


def get_whisper_device() -> str:
    """
    Dispositivo efectivo ("cuda" o "cpu") para instanciar los modelos de
    Whisper. Se resuelve en tiempo real contra el hardware disponible ahora
    mismo en vez de confiar ciegamente en lo guardado en user_config.json:
    si ese JSON se generó en otra máquina con GPU (o la GPU de esta máquina
    dejó de estar disponible), igual hay que caer a CPU en vez de intentar
    abrir un modelo en CUDA y explotar al arrancar.
    """
    if not is_nvidia_gpu_available():
        return "cpu"
    return "cuda" if get_cuda_acceleration_enabled() else "cpu"
