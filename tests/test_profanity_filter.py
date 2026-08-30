import re

import pytest

from src.text_filters.profanity_filter import (
    CENSOR_SUFFIX,
    _build_profanity_pattern,
    censor_text,
)


def test_censor_text_masks_known_word():
    result = censor_text("Eres un puta idiota")
    assert "puta" not in result.lower()
    assert f"p{CENSOR_SUFFIX}" in result


def test_censor_text_preserves_original_case_of_first_letter():
    result = censor_text("PUTA cosa")
    assert result.startswith(f"P{CENSOR_SUFFIX}")


def test_censor_text_leaves_clean_text_untouched():
    clean_text = "Hola, este es un texto totalmente normal."
    assert censor_text(clean_text) == clean_text


def test_censor_text_empty_string_returns_empty_string():
    assert censor_text("") == ""


def test_censor_text_does_not_match_substring_mid_word():
    # "puta" es parte de "amputación" pero no en un límite de palabra —
    # no debe censurarse (evita falsos positivos).
    text = "El médico explicó la amputación con calma."
    assert censor_text(text) == text


def test_censor_text_matches_word_with_suffix():
    # El patrón captura sufijos alfabéticos de la palabra detectada
    # (ej. "putas", "putazo") a propósito — ver docstring del módulo.
    result = censor_text("son unas putas")
    assert f"p{CENSOR_SUFFIX}" in result
    assert "putas" not in result.lower()


def test_censor_text_is_accent_insensitive():
    # "coño" está en PROFANITY_WORDLIST; debe censurarse tanto con tilde
    # como sin ella.
    with_accent = censor_text("que coño pasa")
    without_accent = censor_text("que cono pasa")
    assert f"c{CENSOR_SUFFIX}" in with_accent
    assert f"c{CENSOR_SUFFIX}" in without_accent


def test_censor_text_censors_multiple_words_independently():
    result = censor_text("puta mierda")
    assert f"p{CENSOR_SUFFIX}" in result
    assert f"m{CENSOR_SUFFIX}" in result
    assert "puta" not in result.lower()
    assert "mierda" not in result.lower()


def test_censor_text_preserves_surrounding_text():
    result = censor_text("antes puta despues")
    assert result.startswith("antes ")
    assert result.endswith(" despues")


@pytest.mark.parametrize("wordlist", [[], None])
def test_build_profanity_pattern_empty_wordlist_never_matches(wordlist):
    pattern = _build_profanity_pattern(wordlist or [])
    assert pattern.search("cualquier texto sin nada raro") is None


def test_build_profanity_pattern_matches_only_at_word_boundary():
    pattern = _build_profanity_pattern(["cat"])
    assert pattern.search("a cat sat") is not None
    assert pattern.search("concatenate") is None


def test_build_profanity_pattern_is_case_insensitive():
    pattern = _build_profanity_pattern(["foo"])
    assert pattern.fullmatch("FOO") is not None
