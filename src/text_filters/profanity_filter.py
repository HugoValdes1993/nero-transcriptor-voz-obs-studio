"""
Censura palabras malsonantes en el texto que se transmite al overlay de OBS.

Se aplica SOLO al texto que se envía por WebSocket; la consola sigue
mostrando la transcripción completa sin censurar (ver
transcription_websocket_server.py), para poder debuggear el pipeline sin
perder información.

Estilo de censura: se conserva la primera letra de la palabra detectada y se
reemplaza el resto por tres asteriscos fijos (ej. "puta" -> "p***"),
sin importar la longitud original de la palabra.

Coincidencia por PALABRA COMPLETA (con límites de palabra), no por
substring: esto evita falsos positivos donde una palabra legítima contenga
la misma raíz que una palabra de la lista pero tenga un significado
completamente distinto.
"""

import re
import unicodedata

from config.settings import PROFANITY_WORDLIST

CENSOR_SUFFIX = "***"


def _strip_accents(text: str) -> str:
    normalized_text = unicodedata.normalize("NFD", text)
    return "".join(
        character
        for character in normalized_text
        if unicodedata.category(character) != "Mn"
    )


def _build_profanity_pattern(wordlist: list[str]) -> re.Pattern:
    """
    Compila un único patrón que matchea, como palabra completa (más
    cualquier sufijo alfabético, ej. "putas", "putazo"), cualquiera de las
    entradas de la lista, ignorando mayúsculas/minúsculas. El patrón se
    aplica sobre texto ya normalizado sin acentos (ver censor_text), así que
    las palabras de la lista también se normalizan aquí.
    """
    if not wordlist:
        # Patrón que nunca matchea nada, para no romper si la lista está vacía.
        return re.compile(r"(?!x)x")

    escaped_words = sorted(
        (re.escape(_strip_accents(word.lower())) for word in wordlist),
        key=len,
        reverse=True,  # entradas más largas primero: evita que una palabra
        # corta de la lista corte el match antes de tiempo dentro de una
        # palabra más larga también listada (ej. "puto" antes que "putos").
    )
    pattern_source = r"\b(?:" + "|".join(escaped_words) + r")\w*"
    return re.compile(pattern_source, re.IGNORECASE)


_PROFANITY_PATTERN = _build_profanity_pattern(PROFANITY_WORDLIST)


def censor_text(original_text: str) -> str:
    """
    Retorna una copia de `original_text` con las palabras de
    PROFANITY_WORDLIST censuradas como "primeraLetra***". El resto del
    texto se devuelve intacto, con sus acentos y mayúsculas originales.
    """
    accent_free_text = _strip_accents(original_text)

    # La detección de acentos no debería cambiar la longitud del texto en
    # los casos normales (texto ya en forma NFC, que es lo habitual). Si por
    # algún motivo cambiara, no arriesgamos cortar el texto en el lugar
    # equivocado: comparamos directamente sobre el texto original sin quitar
    # acentos (se pierde la insensibilidad a acentos en ese caso puntual,
    # pero nunca se corrompe el texto).
    search_text = accent_free_text if len(accent_free_text) == len(original_text) else original_text

    result_parts = []
    last_match_end = 0
    for match in _PROFANITY_PATTERN.finditer(search_text):
        match_start, match_end = match.span()
        result_parts.append(original_text[last_match_end:match_start])

        matched_original_word = original_text[match_start:match_end]
        if matched_original_word:
            result_parts.append(matched_original_word[0] + CENSOR_SUFFIX)

        last_match_end = match_end

    result_parts.append(original_text[last_match_end:])
    return "".join(result_parts)
