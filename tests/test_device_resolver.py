"""
Todos los tests de aquí mockean el módulo `sounddevice` (src.audio.
device_resolver.sd) — no dependen de hardware de audio real, así que corren
igual en una máquina sin micrófono ni en un runner de CI.
"""

import pytest

from src.audio import device_resolver

_HOST_APIS = {
    0: {"name": "Windows WASAPI"},
    1: {"name": "Windows WDM-KS"},
    2: {"name": "MME"},
}

_DEVICES = [
    {"name": "Mic A (WASAPI)", "max_input_channels": 2, "hostapi": 0},
    {"name": "Mic A (WDM-KS)", "max_input_channels": 2, "hostapi": 1},
    {"name": "Mic A (MME)", "max_input_channels": 2, "hostapi": 2},
    {"name": "System Default Mic", "max_input_channels": 1, "hostapi": 0},
    {"name": "Speaker Out", "max_input_channels": 0, "hostapi": 0},  # solo salida
]


def _fake_query_devices(index=None, kind=None):
    if index is None:
        return list(_DEVICES)
    return dict(_DEVICES[index])


@pytest.fixture
def fake_sounddevice(monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "query_devices", _fake_query_devices)
    monkeypatch.setattr(device_resolver.sd, "query_hostapis", lambda index: _HOST_APIS[index])
    return device_resolver.sd


# --- find_matching_input_devices ---


def test_finds_all_case_insensitive_matches(fake_sounddevice):
    matches = device_resolver.find_matching_input_devices("mic a")
    assert {m["host_api_name"] for m in matches} == {
        "Windows WASAPI",
        "Windows WDM-KS",
        "MME",
    }


def test_excludes_output_only_devices(fake_sounddevice):
    matches = device_resolver.find_matching_input_devices("speaker")
    assert matches == []


# --- resolve_input_device_candidates ---


def test_excludes_wdm_ks_and_prefers_wasapi(fake_sounddevice):
    candidates = device_resolver.resolve_input_device_candidates("Mic A")
    host_apis = [c["host_api_name"] for c in candidates]
    assert "Windows WDM-KS" not in host_apis
    assert host_apis[0] == "Windows WASAPI"
    assert host_apis[1] == "MME"


def test_raises_when_no_device_matches(fake_sounddevice):
    with pytest.raises(device_resolver.InputDeviceNotFoundError):
        device_resolver.resolve_input_device_candidates("dispositivo inexistente")


def test_raises_when_only_wdm_ks_available(fake_sounddevice, monkeypatch):
    only_wdm_ks = [{"name": "Solo WDMKS Device", "max_input_channels": 1, "hostapi": 1}]
    monkeypatch.setattr(device_resolver.sd, "query_devices", lambda **k: only_wdm_ks)
    with pytest.raises(device_resolver.InputDeviceNotFoundError, match="WDM-KS"):
        device_resolver.resolve_input_device_candidates("Solo WDMKS")


def test_rejects_wdm_ks_as_manual_override(fake_sounddevice):
    with pytest.raises(ValueError):
        device_resolver.resolve_input_device_candidates("Mic A", host_api_override="WDM-KS")


def test_host_api_override_moves_matching_device_first(fake_sounddevice):
    candidates = device_resolver.resolve_input_device_candidates(
        "Mic A", host_api_override="MME"
    )
    assert candidates[0]["host_api_name"] == "MME"


def test_unmatched_host_api_override_falls_back_to_default_order(fake_sounddevice):
    candidates = device_resolver.resolve_input_device_candidates(
        "Mic A", host_api_override="Bluetooth inexistente"
    )
    assert candidates[0]["host_api_name"] == "Windows WASAPI"


# --- list_available_input_device_names ---


def test_lists_unique_input_device_names_in_order(fake_sounddevice):
    names = device_resolver.list_available_input_device_names()
    assert names == ["Mic A (WASAPI)", "Mic A (WDM-KS)", "Mic A (MME)", "System Default Mic"]


# --- get_default_input_device_name ---


def test_default_device_name_none_when_no_default_configured(fake_sounddevice, monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [-1, 0]})())
    assert device_resolver.get_default_input_device_name() is None


def test_default_device_name_returns_configured_device(fake_sounddevice, monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [3, 0]})())
    assert device_resolver.get_default_input_device_name() == "System Default Mic"


# --- resolve_input_device_candidates_or_default ---


def test_or_default_returns_primary_candidates_without_fallback(fake_sounddevice, monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [3, 0]})())
    candidates, used_fallback = device_resolver.resolve_input_device_candidates_or_default("Mic A")
    assert used_fallback is False
    assert len(candidates) > 0


def test_or_default_falls_back_to_system_default(fake_sounddevice, monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [3, 0]})())
    candidates, used_fallback = device_resolver.resolve_input_device_candidates_or_default(
        "dispositivo inexistente"
    )
    assert used_fallback is True
    assert candidates[0]["name"] == "System Default Mic"


def test_or_default_reraises_when_no_system_default_either(fake_sounddevice, monkeypatch):
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [-1, 0]})())
    with pytest.raises(device_resolver.InputDeviceNotFoundError):
        device_resolver.resolve_input_device_candidates_or_default("dispositivo inexistente")


def test_or_default_raises_combined_error_when_default_also_unsafe(fake_sounddevice, monkeypatch):
    # El dispositivo default del sistema (índice 1) solo tiene una entrada
    # WDM-KS disponible: tampoco es una opción segura para usar como fallback.
    devices_with_unsafe_default = _DEVICES + [
        {"name": "Solo WDMKS Device", "max_input_channels": 1, "hostapi": 1}
    ]

    def _query_devices(index=None, kind=None):
        if index is not None:
            return dict(devices_with_unsafe_default[index])
        return list(devices_with_unsafe_default)

    monkeypatch.setattr(device_resolver.sd, "query_devices", _query_devices)
    monkeypatch.setattr(device_resolver.sd, "default", type("_D", (), {"device": [5, 0]})())

    with pytest.raises(device_resolver.InputDeviceNotFoundError, match="WDM-KS"):
        device_resolver.resolve_input_device_candidates_or_default("dispositivo inexistente")
