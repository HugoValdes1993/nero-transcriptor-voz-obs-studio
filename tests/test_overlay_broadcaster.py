"""
Tests con asyncio.run() sin depender de pytest-asyncio: cada test define un
cuerpo async y lo corre con asyncio.run(), evitando sumar una dependencia
extra solo para esta suite.
"""

import asyncio
import json

from src.server.overlay_broadcaster import OverlayBroadcaster


class FakeWebSocket:
    def __init__(self, fail_on_send: bool = False):
        self.sent_messages: list[str] = []
        self.fail_on_send = fail_on_send
        self._closed_event = asyncio.Event()

    async def send(self, message: str):
        if self.fail_on_send:
            raise ConnectionError("conexión cerrada abruptamente")
        self.sent_messages.append(message)

    async def wait_closed(self):
        await self._closed_event.wait()

    def close(self):
        self._closed_event.set()


def test_broadcast_with_no_clients_does_nothing():
    async def body():
        broadcaster = OverlayBroadcaster()
        await broadcaster.broadcast("hola", None, is_final=True, display_mode="original_only")

    asyncio.run(body())  # no debe lanzar ni bloquear


def test_broadcast_style_with_no_clients_does_nothing():
    async def body():
        broadcaster = OverlayBroadcaster()
        await broadcaster.broadcast_style({"background_color": "#000"})

    asyncio.run(body())


def test_register_client_sends_current_style_on_connect_then_unregisters():
    async def body():
        broadcaster = OverlayBroadcaster()
        broadcaster.set_current_style({"background_color": "#111111"})

        ws = FakeWebSocket()
        task = asyncio.create_task(broadcaster.register_client(ws))
        await asyncio.sleep(0)  # deja correr register_client hasta el await

        assert ws in broadcaster._connected_clients
        assert len(ws.sent_messages) == 1
        payload = json.loads(ws.sent_messages[0])
        assert payload == {"type": "style", "background_color": "#111111"}

        ws.close()
        await task

        assert ws not in broadcaster._connected_clients

    asyncio.run(body())


def test_register_client_without_current_style_sends_nothing_on_connect():
    async def body():
        broadcaster = OverlayBroadcaster()
        ws = FakeWebSocket()
        task = asyncio.create_task(broadcaster.register_client(ws))
        await asyncio.sleep(0)

        assert ws.sent_messages == []

        ws.close()
        await task

    asyncio.run(body())


def test_broadcast_sends_subtitle_payload_to_all_connected_clients():
    async def body():
        broadcaster = OverlayBroadcaster()
        ws_a, ws_b = FakeWebSocket(), FakeWebSocket()
        broadcaster._connected_clients.update({ws_a, ws_b})

        await broadcaster.broadcast(
            "hola mundo", "hello world", is_final=True, display_mode="both"
        )

        for ws in (ws_a, ws_b):
            assert len(ws.sent_messages) == 1
            payload = json.loads(ws.sent_messages[0])
            assert payload == {
                "type": "subtitle",
                "final": True,
                "original": "hola mundo",
                "translated": "hello world",
                "displayMode": "both",
            }

    asyncio.run(body())


def test_broadcast_swallows_send_errors_from_disconnected_clients():
    async def body():
        broadcaster = OverlayBroadcaster()
        healthy_ws = FakeWebSocket()
        broken_ws = FakeWebSocket(fail_on_send=True)
        broadcaster._connected_clients.update({healthy_ws, broken_ws})

        # No debe lanzar aunque un cliente falle al enviar.
        await broadcaster.broadcast("texto", None, is_final=False, display_mode="original_only")

        assert len(healthy_ws.sent_messages) == 1
        assert broken_ws.sent_messages == []

    asyncio.run(body())


def test_broadcast_style_sends_to_connected_clients():
    async def body():
        broadcaster = OverlayBroadcaster()
        ws = FakeWebSocket()
        broadcaster._connected_clients.add(ws)

        await broadcaster.broadcast_style({"background_opacity": 0.5})

        payload = json.loads(ws.sent_messages[0])
        assert payload == {"type": "style", "background_opacity": 0.5}

    asyncio.run(body())


def test_set_current_style_is_picked_up_by_next_connecting_client():
    async def body():
        broadcaster = OverlayBroadcaster()
        broadcaster.set_current_style({"padding_px": 20})

        ws = FakeWebSocket()
        task = asyncio.create_task(broadcaster.register_client(ws))
        await asyncio.sleep(0)

        payload = json.loads(ws.sent_messages[0])
        assert payload["padding_px"] == 20

        ws.close()
        await task

    asyncio.run(body())
