"""
Maneja las conexiones WebSocket de los overlays (el Browser Source de OBS)
y el envío de mensajes a todos los que estén conectados.

Es la única pieza del proyecto que sabe de WebSockets — el resto del
pipeline (TranscriptionPipeline) solo le pide a esta clase "envía este
texto", sin conocer los detalles de la conexión. Esto separa la
responsabilidad de "cómo se transcribe" de la de "cómo se entrega al
overlay" (principio de responsabilidad única).
"""

import asyncio
import json

from src.logging_utils import ComponentLogger

logger = ComponentLogger("Overlay")


class OverlayBroadcaster:
    def __init__(self):
        self._connected_clients = set()

        # Último estilo conocido (ver set_current_style): se reenvía a cada
        # overlay apenas se conecta, para que el Browser Source de OBS —que
        # se carga una sola vez y después reconecta solo cuando el servidor
        # vuelve a estar arriba— no se quede pegado con el estilo por
        # defecto hasta la próxima transcripción.
        self._current_style: dict | None = None

    def set_current_style(self, style: dict):
        """
        Actualiza el estilo que se manda a cada overlay que se conecte de
        aquí en más. Es sync (no manda nada por sí sola) porque se llama
        tanto al arrancar el pipeline como desde afuera del loop de asyncio
        (ver PipelineController.push_overlay_style) — quien quiera
        notificar a los overlays YA conectados debe llamar broadcast_style
        además de esto.
        """
        self._current_style = style

    async def register_client(self, websocket):
        """
        Callback que se le pasa a websockets.serve(): se ejecuta una vez
        por cada overlay que se conecta (normalmente, el Browser Source de
        OBS al cargar la escena).
        """
        self._connected_clients.add(websocket)
        logger.info(f"Overlay conectado ({len(self._connected_clients)} activo(s))")
        try:
            if self._current_style is not None:
                await websocket.send(json.dumps({"type": "style", **self._current_style}))
            await websocket.wait_closed()
        finally:
            self._connected_clients.discard(websocket)
            logger.info(f"Overlay desconectado ({len(self._connected_clients)} activo(s))")

    async def broadcast(
        self,
        original_text: str,
        translated_text: str | None,
        is_final: bool,
        display_mode: str,
    ):
        """
        Envía una transcripción (parcial o final) a todos los overlays
        conectados. Si no hay ninguno conectado, no hace nada (evita
        trabajo innecesario si todavía no se abrió OBS).
        """
        if not self._connected_clients:
            return

        message_payload = json.dumps(
            {
                "type": "subtitle",
                "final": is_final,
                "original": original_text,
                "translated": translated_text,
                "displayMode": display_mode,
            }
        )

        await asyncio.gather(
            *(client.send(message_payload) for client in self._connected_clients),
            return_exceptions=True,
        )

    async def broadcast_style(self, style: dict):
        """
        Empuja un cambio de estilo a los overlays YA conectados (ej. el
        usuario mueve un slider de la GUI mientras la transcripción está
        corriendo y OBS ya está abierto). No hace nada si no hay overlays
        conectados; set_current_style se encarga de que uno que se conecte
        más tarde reciba este mismo estilo igual.
        """
        if not self._connected_clients:
            return

        message_payload = json.dumps({"type": "style", **style})
        await asyncio.gather(
            *(client.send(message_payload) for client in self._connected_clients),
            return_exceptions=True,
        )
