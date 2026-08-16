"""
Arranca y detiene el pipeline completo (servidor WebSocket + transcripción)
en un hilo de fondo con su propio event loop de asyncio, separado del hilo
principal donde vive la GUI.

Por qué un hilo aparte: Tkinter necesita correr su mainloop en el hilo
principal, y el pipeline (audio bloqueante + asyncio.run) no puede compartir
ese mismo hilo. Esta clase es la que le permite a la ventana de
configuración (src/gui/config_window.py) iniciar y detener la transcripción
con el mismo botón, sin cerrar la ventana ni bloquearla mientras tanto.
"""

import asyncio
import threading
from typing import Callable, Optional

import websockets

from config.settings import WEBSOCKET_HOST, WEBSOCKET_PORT
from config.user_config import get_overlay_style
from src.logging_utils import ComponentLogger
from src.server.overlay_broadcaster import OverlayBroadcaster
from src.server.transcription_pipeline import TranscriptionPipeline

logger = ComponentLogger("Main")


class PipelineController:
    """
    start() dispara el hilo de fondo y retorna enseguida — no espera a que
    los modelos terminen de cargar. Cuando el pipeline termina (porque se
    llamó a stop() o porque algo falló), se invoca on_stopped(error) desde
    ESE hilo de fondo; quien lo pase debe reencolarlo al hilo de la GUI
    (ej. con root.after(0, ...)), porque Tkinter no es thread-safe.
    """

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._pipeline: Optional[TranscriptionPipeline] = None
        self._broadcaster: Optional[OverlayBroadcaster] = None
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

        # stop() puede llegar ANTES de que self._pipeline exista todavía —
        # TranscriptionPipeline() tarda varios segundos en construirse
        # (carga dos modelos de Whisper en CUDA), tiempo durante el cual el
        # botón "Detener" ya está habilitado en la GUI. Sin esta bandera,
        # un clic en esa ventana se perdía en silencio (stop() no hacía
        # nada porque self._pipeline todavía era None) y el pipeline
        # quedaba corriendo para siempre. _run_pipeline la revisa apenas
        # termina de construir el pipeline para no perder ese pedido.
        self._stop_requested = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, on_stopped: Callable[[Optional[Exception]], None]):
        if self.is_running:
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, args=(on_stopped,), daemon=True)
        self._thread.start()

    def stop(self):
        """No bloquea: solo le pide al pipeline que corte en su próxima
        iteración (ver TranscriptionPipeline.request_stop). on_stopped()
        se termina llamando solo cuando el hilo de fondo realmente termina,
        audio_capture.stop() incluido."""
        self._stop_requested.set()
        if self._pipeline is not None:
            self._pipeline.request_stop()

    def push_overlay_style(self, style: dict):
        """
        Empuja un cambio de estilo del overlay (ver src/gui/config_window.py)
        a los overlays ya conectados, sin esperar a que reconecten. Seguro
        de llamar aunque el pipeline no esté corriendo: en ese caso no hay
        nadie escuchando y no hace nada — el próximo arranque ya toma el
        estilo actualizado de user_config.json de todos modos.
        """
        if self._broadcaster is None or self._event_loop is None:
            return
        self._broadcaster.set_current_style(style)
        asyncio.run_coroutine_threadsafe(
            self._broadcaster.broadcast_style(style), self._event_loop
        )

    def _run(self, on_stopped: Callable[[Optional[Exception]], None]):
        error: Optional[Exception] = None
        try:
            asyncio.run(self._run_pipeline())
        except Exception as exc:
            logger.error(f"El pipeline se detuvo por un error: {exc}")
            error = exc
        finally:
            self._pipeline = None
            self._broadcaster = None
            self._event_loop = None
            on_stopped(error)

    async def _run_pipeline(self):
        event_loop = asyncio.get_running_loop()
        self._event_loop = event_loop
        self._broadcaster = OverlayBroadcaster()
        self._broadcaster.set_current_style(get_overlay_style())
        broadcaster = self._broadcaster
        self._pipeline = TranscriptionPipeline(broadcaster, event_loop)

        # Si stop() se llamó mientras se cargaban los modelos (ver comentario
        # en __init__), el pedido queda guardado en _stop_requested: hay que
        # aplicarlo ahora que el pipeline ya existe, o se perdería.
        if self._stop_requested.is_set():
            self._pipeline.request_stop()

        # El pipeline es bloqueante (audio + Whisper), así que corre en un
        # hilo aparte del executor por defecto mientras este loop de asyncio
        # atiende las conexiones WebSocket del overlay.
        pipeline_task = event_loop.run_in_executor(None, self._pipeline.run)

        async with websockets.serve(broadcaster.register_client, WEBSOCKET_HOST, WEBSOCKET_PORT):
            logger.success(f"Overlay disponible en ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
            await pipeline_task
