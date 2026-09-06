import sys
import types

import pytest

from src.translation import translator


@pytest.fixture(autouse=True)
def _reset_package_cache():
    """_translation_package_ready_for_pair es un cache module-level (ver
    comentario en translator.py) — sin limpiarlo, un test que ya "instaló"
    un par de idiomas dejaría a los siguientes sin pasar por
    _ensure_translation_package_installed."""
    translator._translation_package_ready_for_pair = None
    yield
    translator._translation_package_ready_for_pair = None


@pytest.fixture
def stub_argostranslate(monkeypatch):
    """Reemplaza el módulo argostranslate.translate por un stub controlable,
    y fuerza _ensure_translation_package_installed a no hacer nada (el
    paquete ya está "instalado") para poder probar translate_text() en
    aislamiento de la instalación real de paquetes de Argos."""
    monkeypatch.setattr(translator, "_ensure_translation_package_installed", lambda *a, **k: None)
    monkeypatch.setattr(translator, "get_whisper_source_language", lambda: "es")
    monkeypatch.setattr(translator, "get_translation_target_language", lambda: "en")

    stub_module = types.SimpleNamespace(translate=lambda text, src, dst: "stub result")
    monkeypatch.setitem(sys.modules, "argostranslate.translate", stub_module)
    monkeypatch.setitem(sys.modules, "argostranslate", types.SimpleNamespace(translate=stub_module))
    return stub_module


# --- translate_text ---


def test_translate_text_empty_input_returns_empty_without_touching_argos(stub_argostranslate):
    assert translator.translate_text("") == ""


def test_translate_text_returns_translation_on_success(stub_argostranslate):
    stub_argostranslate.translate = lambda text, src, dst: "hello world"
    assert translator.translate_text("hola mundo") == "hello world"


def test_translate_text_returns_empty_when_argos_raises(stub_argostranslate):
    def _raise(text, src, dst):
        raise RuntimeError("modelo no disponible")

    stub_argostranslate.translate = _raise
    assert translator.translate_text("hola mundo") == ""


def test_translate_text_discards_repetition_loop(stub_argostranslate):
    stub_argostranslate.translate = lambda text, src, dst: "mainstream" * 20
    assert translator.translate_text("una frase cualquiera") == ""


def test_translate_text_keeps_normal_short_translation(stub_argostranslate):
    stub_argostranslate.translate = lambda text, src, dst: "sí"
    assert translator.translate_text("sí") == "sí"


# --- _is_degenerate_repetition ---


def test_short_text_never_flagged_as_repetition():
    assert translator._is_degenerate_repetition("no") is False


def test_natural_long_text_not_flagged_as_repetition():
    natural_text = "This is a normal sentence with varied words and no looping pattern at all."
    assert translator._is_degenerate_repetition(natural_text) is False


def test_repeated_word_flagged_as_repetition():
    assert translator._is_degenerate_repetition("mainstream" * 20) is True
