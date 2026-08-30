"""
IMPORTANTE: config/user_config.py lee y escribe user_config.json en la raíz
del repo al importarse (persiste la configuración real del usuario que corre
la app — ver el docstring del módulo). Todos los tests de aquí abajo usan el
fixture `isolated_config` para redirigir `_USER_CONFIG_PATH` a un archivo
temporal y reemplazar `_user_config` por un dict de prueba controlado, de
forma que NUNCA se lee ni se escribe el user_config.json real del usuario.
"""

import json

import pytest

from config import user_config


@pytest.fixture
def isolated_config(monkeypatch, tmp_path):
    config_path = tmp_path / "user_config.json"
    monkeypatch.setattr(user_config, "_USER_CONFIG_PATH", str(config_path))
    fresh_state = {
        "audio_input_device_name": "Micrófono de prueba",
        "cuda_acceleration_enabled": False,
        "translation_direction": user_config.TRANSLATION_DIRECTION_ES_TO_EN,
        "overlay_style": dict(user_config.OVERLAY_STYLE_DEFAULTS),
    }
    monkeypatch.setattr(user_config, "_user_config", fresh_state)
    return config_path


# --- audio_input_device_name ---


def test_set_and_get_audio_input_device_name(isolated_config):
    user_config.set_audio_input_device_name("Auriculares USB")
    assert user_config.get_audio_input_device_name() == "Auriculares USB"


def test_setting_same_audio_device_name_is_a_noop(isolated_config):
    user_config.set_audio_input_device_name("Micrófono de prueba")
    assert user_config.get_audio_input_device_name() == "Micrófono de prueba"


# --- cuda_acceleration_enabled ---


def test_set_and_get_cuda_acceleration_enabled(isolated_config):
    user_config.set_cuda_acceleration_enabled(True)
    assert user_config.get_cuda_acceleration_enabled() is True


# --- translation_direction ---


def test_set_and_get_translation_direction(isolated_config):
    user_config.set_translation_direction(user_config.TRANSLATION_DIRECTION_EN_TO_ES)
    assert user_config.get_translation_direction() == user_config.TRANSLATION_DIRECTION_EN_TO_ES


def test_set_invalid_translation_direction_raises(isolated_config):
    with pytest.raises(ValueError):
        user_config.set_translation_direction("fr_de")


@pytest.mark.parametrize(
    "direction,expected_source,expected_target",
    [
        (user_config.TRANSLATION_DIRECTION_ES_TO_EN, "es", "en"),
        (user_config.TRANSLATION_DIRECTION_EN_TO_ES, "en", "es"),
    ],
)
def test_source_and_target_language_follow_direction(
    isolated_config, direction, expected_source, expected_target
):
    user_config.set_translation_direction(direction)
    assert user_config.get_whisper_source_language() == expected_source
    assert user_config.get_translation_target_language() == expected_target


# --- overlay_style ---


def test_set_overlay_style_value_updates_get_overlay_style(isolated_config):
    user_config.set_overlay_style_value("original_font_size_px", 50)
    assert user_config.get_overlay_style()["original_font_size_px"] == 50


def test_set_overlay_style_value_rejects_unknown_key(isolated_config):
    with pytest.raises(ValueError):
        user_config.set_overlay_style_value("clave_inventada", 1)


# --- presets de estilo ---


def test_built_in_presets_are_listed_and_recognized(isolated_config):
    presets = user_config.get_overlay_style_presets()
    assert "Clásico" in presets
    assert user_config.is_built_in_style_preset("Clásico") is True
    assert user_config.is_built_in_style_preset("Mi preset") is False


def test_cannot_save_preset_with_built_in_name(isolated_config):
    with pytest.raises(ValueError):
        user_config.save_overlay_style_preset("Neón")


def test_save_and_list_custom_preset(isolated_config):
    user_config.set_overlay_style_value("original_font_size_px", 99)
    user_config.save_overlay_style_preset("Mi preset")

    presets = user_config.get_overlay_style_presets()
    assert presets["Mi preset"]["original_font_size_px"] == 99


def test_delete_custom_preset(isolated_config):
    user_config.save_overlay_style_preset("Temporal")
    user_config.delete_overlay_style_preset("Temporal")

    assert "Temporal" not in user_config.get_overlay_style_presets()


def test_cannot_delete_built_in_preset(isolated_config):
    with pytest.raises(ValueError):
        user_config.delete_overlay_style_preset("Clásico")


def test_apply_unknown_preset_raises(isolated_config):
    with pytest.raises(ValueError):
        user_config.apply_overlay_style_preset("no existe")


def test_apply_preset_mutates_same_dict_instance(isolated_config):
    # apply_overlay_style_preset debe mutar el dict de overlay_style EN VEZ
    # de reemplazarlo (ver su docstring): otros módulos guardan una
    # referencia directa a este mismo objeto.
    style_reference = user_config.get_overlay_style()

    user_config.apply_overlay_style_preset("Neón")

    assert style_reference is user_config.get_overlay_style()
    assert style_reference["original_text_color"] == "#39ff14"


def test_apply_preset_fills_missing_keys_with_defaults(isolated_config):
    user_config.save_overlay_style_preset("Incompleto")
    # Simula un preset viejo guardado antes de agregar una propiedad nueva.
    del user_config._user_config["overlay_style_presets"]["Incompleto"]["padding_px"]

    user_config.apply_overlay_style_preset("Incompleto")

    assert user_config.get_overlay_style()["padding_px"] == user_config.OVERLAY_STYLE_DEFAULTS["padding_px"]


# --- persistencia en disco ---


def test_save_user_config_writes_current_state_to_disk(isolated_config):
    user_config.set_audio_input_device_name("Mic guardado")
    user_config.save_user_config()

    with open(isolated_config, "r", encoding="utf-8") as config_file:
        saved = json.load(config_file)

    assert saved["audio_input_device_name"] == "Mic guardado"


def test_load_raw_reads_back_saved_content(isolated_config):
    user_config.set_translation_direction(user_config.TRANSLATION_DIRECTION_EN_TO_ES)
    user_config.save_user_config()

    reloaded = user_config._load_raw()
    assert reloaded["translation_direction"] == user_config.TRANSLATION_DIRECTION_EN_TO_ES


def test_load_raw_returns_empty_dict_when_file_does_not_exist(isolated_config):
    assert user_config._load_raw() == {}


def test_load_raw_returns_empty_dict_on_corrupt_json(isolated_config, tmp_path):
    isolated_config.write_text("{ esto no es json valido", encoding="utf-8")
    assert user_config._load_raw() == {}


# --- get_whisper_device ---


def test_whisper_device_is_cpu_when_no_gpu_available(isolated_config, monkeypatch):
    monkeypatch.setattr(user_config, "is_nvidia_gpu_available", lambda: False)
    user_config.set_cuda_acceleration_enabled(True)
    assert user_config.get_whisper_device() == "cpu"


def test_whisper_device_is_cuda_when_gpu_available_and_enabled(isolated_config, monkeypatch):
    monkeypatch.setattr(user_config, "is_nvidia_gpu_available", lambda: True)
    user_config.set_cuda_acceleration_enabled(True)
    assert user_config.get_whisper_device() == "cuda"


def test_whisper_device_is_cpu_when_gpu_available_but_disabled(isolated_config, monkeypatch):
    monkeypatch.setattr(user_config, "is_nvidia_gpu_available", lambda: True)
    user_config.set_cuda_acceleration_enabled(False)
    assert user_config.get_whisper_device() == "cpu"
