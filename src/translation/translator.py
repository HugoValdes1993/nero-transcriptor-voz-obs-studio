"""
Traducción offline del texto ya transcrito, usando Argos Translate.

Argos Translate está construido sobre CTranslate2 (el mismo motor que usa
faster-whisper), así que corre 100% local sin depender de ninguna API
externa ni de internet en tiempo real. La primera vez que se pide un par de
idiomas (ej. es -> en) que todavía no está instalado, se descarga
automáticamente desde el índice público de Argos (requiere internet solo
esa vez; después queda instalado localmente, igual que los modelos de
Whisper y Silero VAD).

El import de argostranslate es LAZY (adentro de las funciones, no al tope
del módulo) a propósito: así, si TRANSLATION_ENABLED = False en
config/settings.py, nunca se intenta importar ni usar la librería, y un
usuario que no instaló `argostranslate` puede seguir usando el resto del
pipeline sin traducción sin que esto rompa nada.
"""

import os
import zlib

# Fuerza a Argos Translate a NO cuantizar en INT8 en CPU. Por defecto
# argostranslate.settings.compute_type es "auto", que en casi cualquier CPU
# moderna resuelve a "int8" (ver argostranslate/settings.py) — y hay
# antecedentes documentados de kernels INT8 de CTranslate2 en CPU que
# corrompen la salida en ciertas CPUs puntuales, con el mismo síntoma que un
# loop de repetición sin sentido en vez de un error limpio (ver
# https://github.com/argosopentech/argos-translate/issues/197 — se arregló
# recién en CTranslate2 2.10.1 para ESE caso puntual, pero el patrón
# "CPU + cuantización INT8 -> texto degenerado" no tiene garantía de estar
# erradicado para siempre en cualquier CPU). Argos Translate corre SIEMPRE en
# CPU (nunca se toca ARGOS_DEVICE_TYPE en este proyecto), sin relación con si
# Whisper usa GPU o no — este env var hay que fijarlo ANTES de importar
# argostranslate (argostranslate.settings lo lee una sola vez al importarse),
# por eso está a nivel de módulo. float32 no usa esos kernels; el costo de
# latencia es irrelevante acá porque la traducción corre una sola vez por
# frase final, no en tiempo real como Whisper.
os.environ.setdefault("ARGOS_COMPUTE_TYPE", "float32")

from config.settings import TRANSLATION_COMPRESSION_RATIO_THRESHOLD
from config.user_config import get_whisper_source_language, get_translation_target_language
from src.logging_utils import ComponentLogger
from src.status_hub import notify_status

logger = ComponentLogger("Translator")

# Guarda el par (origen, destino) para el que ya se verificó/instaló el
# paquete de Argos, no solo un booleano: el flujo de traducción se puede
# togglear desde la GUI entre una corrida del pipeline y la siguiente (ver
# config/user_config.get_translation_direction), así que un bool fijo
# dejaría pasar sin verificar un par nuevo que nunca se instaló.
_translation_package_ready_for_pair: tuple[str, str] | None = None


def _ensure_translation_package_installed(source_language: str, target_language: str):
    """
    Verifica que el paquete de traducción source_language -> target_language
    esté instalado; si no lo está, lo descarga e instala. Se cachea por par
    de idiomas para no repetir esta verificación (que implica una consulta a
    get_installed_languages) en cada llamada mientras el flujo no cambie.
    """
    global _translation_package_ready_for_pair
    requested_pair = (source_language, target_language)
    if _translation_package_ready_for_pair == requested_pair:
        return

    import argostranslate.package
    import argostranslate.translate

    installed_languages = argostranslate.translate.get_installed_languages()
    source_language_entry = next(
        (lang for lang in installed_languages if lang.code == source_language),
        None,
    )
    if source_language_entry is not None:
        already_installed = any(
            translation.to_lang.code == target_language
            for translation in source_language_entry.translations_from
        )
        if already_installed:
            _translation_package_ready_for_pair = requested_pair
            return

    logger.info(
        f"Paquete de traducción {source_language} -> "
        f"{target_language} no encontrado localmente. "
        f"Descargando (requiere internet, solo la primera vez)..."
    )
    # Este paso puede pasar bastante después de que la transcripción ya
    # está "Escuchando micrófono" (se dispara recién con la primera frase
    # que hay que traducir) — a diferencia de la carga del modelo de
    # Whisper (ver SpeechTranscriber), aquí SÍ hace falta el notify_status("")
    # de "ya terminó" al final, para no dejar este mensaje pisando
    # "Transcribiendo" para siempre. El try/finally lo asegura tanto si
    # esto termina bien como si falla (paquete inexistente, sin internet,
    # etc.) — hoy un fallo aquí termina matando todo el pipeline igual (ver
    # PipelineController._run), así que _reset_to_idle ya pisaría el
    # mensaje con el error de todos modos, pero no vale la pena dejar este
    # notify_status descalzado para el día que ese manejo de errores cambie.
    notify_status(f"Descargando paquete de traducción {source_language} → {target_language}...")
    try:
        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        matching_package = next(
            (
                pkg
                for pkg in available_packages
                if pkg.from_code == source_language
                and pkg.to_code == target_language
            ),
            None,
        )
        if matching_package is None:
            raise RuntimeError(
                f"No existe un paquete de Argos Translate para "
                f"{source_language} -> {target_language}. "
                f"Revisa los códigos de idioma disponibles en "
                f"https://www.argosopentech.com/argospm/index/"
            )

        argostranslate.package.install_from_path(matching_package.download())
        logger.success(f"Paquete {source_language} -> {target_language} instalado.")
    finally:
        notify_status("")

    _translation_package_ready_for_pair = requested_pair


def _is_degenerate_repetition(text: str) -> bool:
    """
    Detecta si `text` es un loop de repetición (ej. "mainstreammainstream"
    encadenado) — un fallo de decodificación observado en los modelos de
    Argos Translate/CTranslate2 con ciertos textos de entrada, sobre todo
    cuando el texto original ya viene con errores de la transcripción. A
    diferencia de faster-whisper (ver WHISPER_COMPRESSION_RATIO_THRESHOLD),
    argostranslate.translate.translate() no expone repetition_penalty ni
    no_repeat_ngram_size para evitar el loop en el momento de generarlo, así
    que se detecta después con el mismo criterio: texto repetitivo comprime
    mucho mejor que texto natural.

    Requiere un mínimo de longitud para que el ratio sea confiable — un
    texto corto y legítimo (ej. "sí") también comprime mal por puro overhead
    del formato, sin que eso indique un loop.
    """
    encoded_text = text.encode("utf-8")
    if len(encoded_text) < 32:
        return False

    compression_ratio = len(encoded_text) / len(zlib.compress(encoded_text))
    return compression_ratio > TRANSLATION_COMPRESSION_RATIO_THRESHOLD


def translate_text(original_text: str) -> str:
    """
    Traduce `original_text` según el flujo elegido en la GUI (ver
    config/user_config.get_whisper_source_language /
    get_translation_target_language). Retorna cadena vacía si
    `original_text` está vacío (evita instalar/consultar el paquete
    innecesariamente), si la traducción falla, o si el resultado parece un
    loop de repetición (ver _is_degenerate_repetition) — en ninguno de esos
    casos vale la pena tirar abajo todo el pipeline de transcripción por un
    problema puntual de esta frase.
    """
    if not original_text:
        return ""

    source_language = get_whisper_source_language()
    target_language = get_translation_target_language()

    _ensure_translation_package_installed(source_language, target_language)

    import argostranslate.translate

    try:
        translated_text = argostranslate.translate.translate(
            original_text, source_language, target_language
        )
    except Exception as error:
        logger.warning(f"Fallo al traducir {original_text!r}: {error}. Se omite la traducción.")
        return ""

    if _is_degenerate_repetition(translated_text):
        logger.warning(
            f"Traducción descartada por parecer un loop de repetición: {translated_text!r}"
        )
        return ""

    return translated_text
