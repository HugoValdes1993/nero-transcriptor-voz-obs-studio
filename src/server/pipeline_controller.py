"""
Arranca y detiene la transcripción en un hilo de fondo, separado del hilo
principal donde vive la GUI. El servidor WebSocket del overlay (OverlayServer)
tiene su propio ciclo de vida, independiente de esto — ver
start_overlay_server()/stop_overlay_server(), llamados por
src/gui/config_window.py al abrir/cerrar la ventana, no al iniciar/detener
la transcripción.

Por qué un hilo aparte para la transcripción: sounddevice y faster-whisper
son bloqueantes, y Tkinter necesita correr su mainloop en el hilo principal.
Esta clase es la que le permite a la ventana de configuración
(src/gui/config_window.py) iniciar y detener la transcripción con el mismo
botón, sin cerrar la ventana ni bloquearla mientras tanto.
"""

import threading
from typing import Callable, Optional

from config.user_config import get_overlay_style
from src.logging_utils import ComponentLogger
from src.server.overlay_server import OverlayServer
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
        self._overlay_server = OverlayServer()
        self._thread: Optional[threading.Thread] = None
        self._pipeline: Optional[TranscriptionPipeline] = None

        # stop() puede llegar ANTES de que self._pipeline exista todavía —
        # TranscriptionPipeline() tarda varios segundos en construirse
        # (carga dos modelos de Whisper en CUDA), tiempo durante el cual el
        # botón "Detener" ya está habilitado en la GUI. Sin esta bandera,
        # un clic en esa ventana se perdía en silencio (stop() no hacía
        # nada porque self._pipeline todavía era None) y el pipeline
        # quedaba corriendo para siempre. _run revisa apenas termina de
        # construir el pipeline para no perder ese pedido.
        self._stop_requested = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start_overlay_server(self):
        """Levanta el servidor WebSocket del overlay. Se llama al abrir la
        ventana de configuración (ver ConfigWindow.__init__) — no depende de
        que la transcripción esté corriendo, para que el Browser Source de
        OBS y la vista previa de estilos puedan conectarse de entrada."""
        self._overlay_server.start()
        self._overlay_server.broadcaster.set_current_style(get_overlay_style())

    def stop_overlay_server(self):
        self._overlay_server.stop()

    def start(
        self,
        on_stopped: Callable[[Optional[Exception]], None],
        on_ready: Optional[Callable[[], None]] = None,
    ):
        if self.is_running:
            return
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, args=(on_stopped, on_ready), daemon=True)
        self._thread.start()

    def stop(self):
        """No bloquea: solo le pide al pipeline que corte en su próxima
        iteración (ver TranscriptionPipeline.request_stop). on_stopped()
        se termina llamando solo cuando el hilo de fondo realmente termina,
        audio_capture.stop() incluido."""
        self._stop_requested.set()
        if self._pipeline is not None:
            self._pipeline.request_stop()

    def join(self, timeout: Optional[float] = None):
        """
        Bloquea hasta que el hilo de la transcripción termina de verdad (o
        pasan `timeout` segundos) — a diferencia de stop(), que no espera.

        Hace falta al cerrar la ventana (ver
        ConfigWindow._on_close_requested), ANTES de llamar a
        stop_overlay_server(): TranscriptionPipeline.run() todavía puede
        estar programando un broadcast en el event loop del OverlayServer
        (vía asyncio.run_coroutine_threadsafe) hasta que su hilo termina
        de verdad; tirar abajo ese loop mientras tanto puede hacer que ese
        broadcast falle contra un loop ya cerrado.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def push_overlay_style(self, style: dict):
        """
        Empuja un cambio de estilo del overlay (ver src/gui/config_window.py)
        a los overlays ya conectados, sin esperar a que reconecten. Funciona
        aunque la transcripción no esté corriendo (el servidor del overlay
        vive independiente de eso — ver start_overlay_server).
        """
        self._overlay_server.broadcaster.set_current_style(style)
        self._overlay_server.run_coroutine_threadsafe(
            self._overlay_server.broadcaster.broadcast_style(style)
        )

    def push_preview_subtitle(self, original_text: str, translated_text: str):
        """
        Manda un subtítulo de muestra a los overlays conectados, para la
        vista previa de estilos (ver ConfigWindow._on_preview_toggled) —
        reusa el mismo mensaje que ya entienden los overlays para
        transcripciones reales, con displayMode "both" para que tanto el
        overlay original como el traducido tengan algo que mostrar sin
        importar el flujo de traducción configurado.
        """
        self._overlay_server.run_coroutine_threadsafe(
            self._overlay_server.broadcaster.broadcast(original_text, translated_text, True, "both")
        )

    def _run(
        self,
        on_stopped: Callable[[Optional[Exception]], None],
        on_ready: Optional[Callable[[], None]],
    ):
        error: Optional[Exception] = None
        try:
            self._pipeline = TranscriptionPipeline(
                self._overlay_server.broadcaster, self._overlay_server.loop, on_ready=on_ready
            )

            # Si stop() se llamó mientras se cargaban los modelos (ver
            # comentario en __init__), el pedido queda guardado en
            # _stop_requested: hay que aplicarlo ahora que el pipeline ya
            # existe, o se perdería.
            if self._stop_requested.is_set():
                self._pipeline.request_stop()

            self._pipeline.run()
        except Exception as exc:
            logger.error(f"El pipeline se detuvo por un error: {exc}")
            error = exc
        finally:
            self._pipeline = None
            on_stopped(error)
