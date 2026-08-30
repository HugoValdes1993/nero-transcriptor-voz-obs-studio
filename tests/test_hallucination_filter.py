from src.speech.hallucination_filter import is_known_hallucination


def test_detects_exact_known_phrase():
    assert is_known_hallucination("suscribete al canal") is True


def test_detects_phrase_with_different_case_and_accents():
    assert is_known_hallucination("¡SUSCRÍBETE AL CANAL!") is True


def test_detects_phrase_embedded_in_longer_text():
    assert is_known_hallucination("Bueno, gente, suscribete al canal por favor") is True


def test_returns_false_for_unrelated_text():
    assert is_known_hallucination("El clima hoy está soleado y agradable") is False


def test_returns_false_for_empty_string():
    assert is_known_hallucination("") is False


def test_returns_false_for_whitespace_only():
    assert is_known_hallucination("   ") is False


def test_punctuation_does_not_prevent_match():
    assert is_known_hallucination("gracias, por, ver, el, video.") is True
