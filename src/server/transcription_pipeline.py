"""
Orquesta el pipeline completo de subtítulos en tiempo real:

  micrófono -> detección de voz (VAD) -> segmentación de frases
      -> transcripción (parcial + final) -> traducción -> filtro de
      groserías -> broadcast al overlay de OBS

Esta clase es la única que conoce el orden completo del pipeline. Cada paso
individual (captura de audio, VAD, transcripción, traducción, filtros) vive
en su propio módulo y no sabe nada de los demás — TranscriptionPipeline es
la que los conecta, recibiendo el broadcaster como dependencia en vez de
crear su propio WebSocket (así se puede, por ejemplo, testear esta clase
con un broadcaster falso sin levantar un servidor real).

Autor: Nero
"""

import asyncio
import queue
import threading
from typing import Callable, Optional

from config.settings import (
    PARTIAL_TRANSCRIPTION_ENABLED,
    PARTIAL_WHISPER_MODEL_NAME,
    PARTIAL_WHISPER_COMPUTE_TYPE,
    PARTIAL_WHISPER_BEAM_SIZE,
    TRANSLATION_ENABLED,
    TRANSLATION_DISPLAY_MODE,
    TRANSLATE_PARTIALS,
    PROFANITY_FILTER_ENABLED,
    DEBUG_SAVE_UTTERANCE_WAV_ENABLED,
)
from config.user_config import get_whisper_device
from src.audio.stream_capture import AudioStreamCapture
from src.audio.debug_dump import save_utterance_debug_wav
from src.speech.voice_activity_detector import VoiceActivityDetector
from src.speech.utterance_segmenter import UtteranceSegmenter
from src.speech.speech_transcriber import SpeechTranscriber
from src.startup.cuda_dll_setup import ensure_cuda_runtime_downloaded_and_registered
from src.translation.translator import translate_text
from src.text_filters.profanity_filter import censor_text
from src.logging_utils import ComponentLogger

logger = ComponentLogger("Pipeline")


class TranscriptionPipeline:
    def __init__(
        self,
        broadcaster,
        event_loop: asyncio.AbstractEventLoop,
        on_ready: Optional[Callable[[], None]] = None,
    ):
        self._broadcaster = broadcaster
        self._event_loop = event_loop
        # Se llama una sola vez, apenas el micrófono ya está capturando de
        # verdad (ver run()) — es la señal real de "esto ya está andando",
        # en vez de que quien nos arrancó (PipelineController/ConfigWindow)
        # tenga que adivinar con un timer fijo cuánto tardan en cargarse los
        # modelos (que puede ser mucho más si hay que descargarlos la
        # primera vez — ver SpeechTranscriber._notify_model_loading).
        self._on_ready = on_ready

        # Las DLLs de CUDA se descargan bajo demanda (~1.3GB, ver
        # src/startup/cuda_runtime_downloader.py) — se resuelve UNA sola vez
        # aquí, antes de instanciar cualquier SpeechTranscriber (hay dos: final
        # y parcial, y ambos leerían "cuda" del mismo get_whisper_device()).
        if get_whisper_device() == "cuda":
            ensure_cuda_runtime_downloaded_and_registered()

        self._audio_capture = AudioStreamCapture()
        self._voice_activity_detector = VoiceActivityDetector()
        self._utterance_segmenter = UtteranceSegmenter()
        self._final_transcriber = SpeechTranscriber()
        self._partial_transcriber = self._build_partial_transcriber()

        # Evita reenviar al overlay el mismo texto parcial dos veces
        # seguidas; se resetea cada vez que una utterance cierra de verdad.
        self._last_broadcast_partial_text = ""

        # Permite pedirle a run() que corte el loop desde otro hilo (ej. el
        # botón "Detener transcripción" de la GUI) sin matar el proceso.
        self._stop_event = threading.Event()

    def request_stop(self):
        """Le pide al loop de run() que termine en su próxima iteración
        (hasta 1s de latencia, por el timeout de get_next_frame). Seguro de
        llamar desde otro hilo."""
        self._stop_event.set()

    @staticmethod
    def _build_partial_transcriber() -> SpeechTranscriber | None:
        """Instancia el transcriptor liviano para los parciales, o None si
        la transcripción parcial está desactivada en la config."""
        if not PARTIAL_TRANSCRIPTION_ENABLED:
            return None
        return SpeechTranscriber(
            model_name=PARTIAL_WHISPER_MODEL_NAME,
            compute_type=PARTIAL_WHISPER_COMPUTE_TYPE,
            beam_size=PARTIAL_WHISPER_BEAM_SIZE,
            condition_on_previous_text=False,
            apply_confidence_filters=False,
        )

    def run(self):
        """
        Loop bloqueante principal. Corre en un hilo aparte (sounddevice y
        faster-whisper son bloqueantes) mientras el hilo principal atiende
        el servidor WebSocket.
        """
        if self._stop_event.is_set():
            # Se pidió detener (ej. se cerró la ventana) mientras todavía se
            # estaban cargando/descargando los modelos en __init__ — eso
            # puede tardar minutos la primera vez. Ni abrir el micrófono ni
            # avisar on_ready tiene sentido aquí: quien pidió detener ya
            # puede estar esperando en join() (ver
            # PipelineController.join / ConfigWindow._on_close_requested),
            # o la ventana que recibiría on_ready ya ni existe.
            logger.info("Transcripción cancelada antes de arrancar (se pidió detener durante la carga de modelos).")
            return

        self._audio_capture.start()
        logger.success("Escuchando micrófono. Ctrl+C para detener.")
        if self._on_ready is not None:
            self._on_ready()

        try:
            while not self._stop_event.is_set():
                try:
                    audio_frame = self._audio_capture.get_next_frame(timeout_seconds=1.0)
                except queue.Empty:
                    continue

                self._process_frame(audio_frame)
        finally:
            self._audio_capture.stop()
            logger.info("Transcripción detenida.")

    def _process_frame(self, audio_frame):
        """Procesa un frame de audio: lo pasa por VAD y el segmentador, y
        decide si corresponde emitir un parcial o cerrar la utterance."""
        frame_has_speech = self._voice_activity_detector.frame_contains_speech(audio_frame)
        closed_utterance = self._utterance_segmenter.add_frame(audio_frame, frame_has_speech)

        if closed_utterance is None:
            self._maybe_emit_partial()
            return

        self._last_broadcast_partial_text = ""  # la frase que viene empieza sin parcial previo
        self._emit_final(closed_utterance)

    def _maybe_emit_partial(self):
        """Si corresponde (según PARTIAL_TRANSCRIPTION_INTERVAL_MS), transcribe
        y emite un adelanto tentativo del audio acumulado hasta ahora."""
        if self._partial_transcriber is None:
            return

        partial_audio = self._utterance_segmenter.maybe_get_partial_audio()
        if partial_audio is None:
            return

        partial_text = self._partial_transcriber.transcribe_utterance(partial_audio)
        if not partial_text or partial_text == self._last_broadcast_partial_text:
            return

        self._last_broadcast_partial_text = partial_text
        logger.transcript("Parcial", partial_text)

        translated_partial_text = None
        if TRANSLATION_ENABLED and TRANSLATE_PARTIALS:
            translated_partial_text = translate_text(partial_text)
            logger.transcript("Parcial traducido", translated_partial_text)

        overlay_text = censor_text(partial_text) if PROFANITY_FILTER_ENABLED else partial_text
        self._broadcast(overlay_text, translated_partial_text, is_final=False)

    def _emit_final(self, closed_utterance):
        """Transcribe con el modelo grande la utterance que acaba de cerrar
        por silencio, y la manda al overlay como texto confirmado."""
        if DEBUG_SAVE_UTTERANCE_WAV_ENABLED:
            debug_wav_path = save_utterance_debug_wav(closed_utterance)
            logger.info(f"Utterance de debug guardada en: {debug_wav_path}")

        transcribed_text = self._final_transcriber.transcribe_utterance(closed_utterance)
        if not transcribed_text:
            return

        logger.transcript("Transcripción", transcribed_text)

        translated_text = None
        if TRANSLATION_ENABLED:
            translated_text = translate_text(transcribed_text)
            logger.transcript("Traducción", translated_text)

        overlay_text = censor_text(transcribed_text) if PROFANITY_FILTER_ENABLED else transcribed_text
        self._broadcast(overlay_text, translated_text, is_final=True)

    def _broadcast(self, overlay_text: str, translated_text: str | None, is_final: bool):
        """Programa el envío al overlay en el event loop de asyncio, ya que
        este método corre desde el hilo bloqueante del pipeline, no desde
        el loop async donde vive el servidor WebSocket."""
        display_mode = TRANSLATION_DISPLAY_MODE if TRANSLATION_ENABLED else "original_only"
        asyncio.run_coroutine_threadsafe(
            self._broadcaster.broadcast(overlay_text, translated_text, is_final, display_mode),
            self._event_loop,
        )
