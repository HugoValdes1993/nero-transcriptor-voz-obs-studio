"""
Detecta si un texto transcrito coincide con alguna de las frases "enlatadas"
que Whisper suele alucinar cuando el audio es silencio o ruido (frases de su
propio dataset de entrenamiento, mayormente YouTube: "Suscríbete al canal",
créditos de subtítulos de Amara.org, etc.).

La comparación es insensible a mayúsculas, acentos y puntuación, y matchea
tanto si el texto ES una de esas frases como si la CONTIENE (Whisper a veces
las intercala con basura adicional).
"""

import re
import unicodedata

from config.settings import KNOWN_HALLUCINATION_PHRASES


def _normalize_for_comparison(text: str) -> str:
    accent_free_text = "".join(
        character
        for character in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(character) != "Mn"
    )
    # Colapsa cualquier cosa que no sea letra/número/espacio (puntuación,
    # signos de exclamación, etc.) para que "¡Suscríbete al canal!" matchee
    # igual que "suscribete al canal".
    only_alphanumeric_and_spaces = re.sub(r"[^a-z0-9\s]", " ", accent_free_text)
    return re.sub(r"\s+", " ", only_alphanumeric_and_spaces).strip()


_NORMALIZED_HALLUCINATION_PHRASES = [
    _normalize_for_comparison(phrase) for phrase in KNOWN_HALLUCINATION_PHRASES
]


def is_known_hallucination(candidate_text: str) -> bool:
    """
    Retorna True si `candidate_text` coincide con o contiene alguna de las
    frases de KNOWN_HALLUCINATION_PHRASES (normalizando mayúsculas, acentos
    y puntuación antes de comparar).
    """
    normalized_candidate = _normalize_for_comparison(candidate_text)
    if not normalized_candidate:
        return False

    for normalized_phrase in _NORMALIZED_HALLUCINATION_PHRASES:
        if normalized_phrase and normalized_phrase in normalized_candidate:
            return True
    return False
