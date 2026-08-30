"""
Servidor WebSocket del overlay, desacoplado del ciclo de vida de la
transcripción: vive desde que se abre la ventana de configuración hasta que
se cierra, no solo mientras el pipeline de Whisper está corriendo.

Por qué: así el Browser Source de OBS (y la vista previa de estilos que abre
config_window.py en el navegador) pueden conectarse y ver cambios de estilo
en vivo sin necesidad de arrancar la transcripción real — ver
PipelineController.start_overlay_server() / push_preview_subtitle().

Corre en su propio hilo con su propio event loop de asyncio (igual que antes
hacía PipelineController._run_pipeline), porque Tkinter necesita el hilo
principal para su mainloop.
"""

import asyncio
import threading

import websockets

from config.settings import WEBSOCKET_HOST, WEBSOCKET_PORT
from src.logging_utils import ComponentLogger
from src.server.overlay_broadcaster import OverlayBroadcaster

logger = ComponentLogger("Main")


class OverlayServer:
    def __init__(self):
        self.broadcaster = OverlayBroadcaster()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_event: asyncio.Event | None = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        """Se le pasa tal cual a TranscriptionPipeline, que ya lo recibe
        como dependencia inyectada (ver src/server/transcription_pipeline.py)
        para programar sus propios broadcasts sin conocer los detalles de
        este servidor."""
        return self._loop

    def start(self):
        """Bloquea brevemente hasta que el servidor ya está escuchando
        (ready.wait()), para que quien llame pueda asumir que push_* ya
        funciona apenas retorna."""
        if self.is_running:
            return
        ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(ready,), daemon=True)
        self._thread.start()
        ready.wait()

    def stop(self):
        if not self.is_running or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._stop_event.set)
        self._thread.join(timeout=5)

    def run_coroutine_threadsafe(self, coroutine):
        """Programa una coroutine en el loop de este servidor desde
        cualquier otro hilo (la GUI, o el hilo bloqueante del pipeline de
        transcripción). No hace nada si el servidor todavía no arrancó."""
        if self._loop is None:
            return
        asyncio.run_coroutine_threadsafe(coroutine, self._loop)

    def _run(self, ready: threading.Event):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._stop_event = asyncio.Event()
        try:
            self._loop.run_until_complete(self._serve_forever(ready))
        finally:
            self._loop.close()

    async def _serve_forever(self, ready: threading.Event):
        async with websockets.serve(self.broadcaster.register_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
            logger.success(f"Overlay disponible en ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
            ready.set()
            await self._stop_event.wait()
