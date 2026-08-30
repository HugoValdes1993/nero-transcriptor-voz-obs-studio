import pytest

from src import status_hub


@pytest.fixture(autouse=True)
def _reset_listener():
    """El listener es un global a nivel de módulo — evita que un test deje
    un listener registrado que contamine el siguiente."""
    yield
    status_hub.set_status_listener(None)


def test_notify_status_without_listener_does_nothing():
    # No debe lanzar ninguna excepción aunque nadie esté escuchando.
    status_hub.notify_status("mensaje sin listener")


def test_registered_listener_receives_message_and_progress():
    received = []
    status_hub.set_status_listener(lambda message, progress: received.append((message, progress)))

    status_hub.notify_status("Descargando modelo...", 0.42)

    assert received == [("Descargando modelo...", 0.42)]


def test_progress_defaults_to_none():
    received = []
    status_hub.set_status_listener(lambda message, progress: received.append((message, progress)))

    status_hub.notify_status("Cargando...")

    assert received == [("Cargando...", None)]


def test_unregistering_listener_with_none_stops_notifications():
    received = []
    status_hub.set_status_listener(lambda message, progress: received.append(message))
    status_hub.set_status_listener(None)

    status_hub.notify_status("no debería llegar a nadie")

    assert received == []
