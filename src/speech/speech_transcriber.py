"""
Wrapper sobre faster-whisper (CTranslate2) para transcribir utterances de audio
ya segmentadas por el VoiceActivityDetector.

Ninguna de estas medidas elimina las alucinaciones al 100% (ningún modelo de
ASR lo garantiza), pero combinadas reducen su frecuencia drásticamente:

  1. Filtro de energía (RMS) ANTES de llamar a Whisper: descarta utterances
     casi silenciosas que Silero VAD marcó como voz por error.
  2. Filtros de confianza POR SEGMENTO que ya devuelve faster-whisper:
     no_speech_prob, avg_logprob, compression_ratio.
  3. Lista negra de frases "enlatadas" típicas del dataset de entrenamiento
     de Whisper (ver hallucination_filter.py), aplicada al texto final.
"""

import numpy as np
import faster_whisper
from faster_whisper import WhisperModel

from config.settings import (
    WHISPER_MODEL_NAME,
    WHISPER_COMPUTE_TYPE,
    WHISPER_BEAM_SIZE,
    WHISPER_VAD_FILTER,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_NO_SPEECH_PROB_THRESHOLD,
    WHISPER_AVG_LOGPROB_THRESHOLD,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    MIN_UTTERANCE_RMS_ENERGY,
)
from config.user_config import get_whisper_device, get_whisper_source_language
from src.speech.hallucination_filter import is_known_hallucination
from src.logging_utils import ComponentLogger
from src.status_hub import notify_status

logger = ComponentLogger("Transcriber")


class SpeechTranscriber:
    def __init__(
        self,
        model_name: str = WHISPER_MODEL_NAME,
        compute_type: str = WHISPER_COMPUTE_TYPE,
        beam_size: int = WHISPER_BEAM_SIZE,
        condition_on_previous_text: bool = WHISPER_CONDITION_ON_PREVIOUS_TEXT,
        apply_confidence_filters: bool = True,
    ):
        """
        Los parámetros con default reproducen el comportamiento de la
        transcripción final (large-v3-turbo). Para una instancia de
        parciales, se pasan explícitamente model_name/compute_type/beam_size
        más livianos y apply_confidence_filters=False (ver
        PARTIAL_WHISPER_* en config/settings.py) — los parciales son
        tentativos y descartables, no vale la pena filtrarlos tan agresivo
        como al texto final.
        """
        # Se resuelve en cada instancia (no como default de parámetro) porque
        # el checkbox de CUDA de la GUI se puede togglear entre una corrida
        # del pipeline y la siguiente, y get_whisper_device() ya contempla
        # el caso "no hay GPU NVIDIA disponible" cayendo a CPU.
        device = get_whisper_device()

        # int8_float16 (el default de fábrica en config/settings.py) requiere
        # GPU: CTranslate2 no lo soporta en CPU. Si CUDA está desactivado,
        # se cae a int8 automáticamente en vez de fallar al cargar el modelo.
        if device == "cpu" and "float16" in compute_type:
            logger.warning(
                f"CUDA desactivado: usando compute_type='int8' en CPU en vez de "
                f"'{compute_type}' (float16 requiere GPU)."
            )
            compute_type = "int8"

        self._notify_model_loading(model_name)
        try:
            self.whisper_model = WhisperModel(
                model_name,
                device=device,
                compute_type=compute_type,
            )
        except Exception as error:
            # El chequeo de check_driver_supports_cuda_runtime() (ver
            # cuda_dll_setup.ensure_cuda_runtime_downloaded_and_registered)
            # es best-effort — si igual llega hasta acá con un driver
            # insuficiente/incompatible, CTranslate2 tira un error de bajo
            # nivel bastante críptico. Se reemplaza por uno accionable.
            if device == "cuda":
                raise RuntimeError(
                    "No se pudo inicializar CUDA para cargar el modelo (driver de NVIDIA "
                    "desactualizado o incompatible). Actualizá el driver en "
                    "https://www.nvidia.com/drivers, o desactivá \"Usar aceleración CUDA\" "
                    f"en la configuración para seguir en CPU. Detalle: {error}"
                ) from error
            raise
        # No hace falta un notify_status("") de "ya terminó" acá: hay dos
        # instancias de SpeechTranscriber (final + parcial) y todavía falta
        # que arranque la captura de audio — quien decide cuándo mostrar
        # "Transcribiendo" de verdad es TranscriptionPipeline.run() a través
        # de on_ready (ver PipelineController.start), no acá.
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.apply_confidence_filters = apply_confidence_filters

    @staticmethod
    def _notify_model_loading(model_name: str):
        """
        Avisa a la GUI mientras se resuelve el modelo — puede tardar bastante
        si hay que descargarlo de Hugging Face (de unos cientos de MB a
        ~1.6GB para large-v3-turbo) en vez de cargarlo desde caché local.

        download_model(local_files_only=True) no descarga nada: si el modelo
        ya está en caché devuelve la ruta al toque; si no, tira una excepción
        (el tipo exacto depende de la versión de huggingface_hub, por eso se
        captura genérico) — con eso alcanza para distinguir "va a descargar"
        de "ya lo tengo, esto es solo carga rápida".
        """
        try:
            faster_whisper.download_model(model_name, local_files_only=True)
            message = f"Cargando modelo de transcripción '{model_name}'..."
        except Exception:
            message = (
                f"Descargando modelo de transcripción '{model_name}' "
                f"(primera vez, puede tardar varios minutos)..."
            )
        logger.info(message)
        notify_status(message)

    def transcribe_utterance(self, utterance_audio_float32: np.ndarray) -> str:
        """
        Recibe el buffer de audio completo de una utterance (float32, 16kHz mono)
        y retorna el texto transcrito, ya concatenado y limpio. Retorna cadena
        vacía si la utterance se descarta por cualquiera de los filtros
        anti-alucinación.
        """
        utterance_rms_energy = float(np.sqrt(np.mean(np.square(utterance_audio_float32))))
        if utterance_rms_energy < MIN_UTTERANCE_RMS_ENERGY:
            logger.info(
                f"Utterance descartada antes de transcribir "
                f"(RMS={utterance_rms_energy:.4f} < {MIN_UTTERANCE_RMS_ENERGY}): "
                f"audio casi silencioso, probable falso positivo de VAD"
            )
            return ""

        # Se lee en cada llamada (no en __init__) porque el flujo de
        # transcripción/traducción elegido en la GUI se resuelve en tiempo
        # real, igual que get_whisper_device() más arriba.
        segments, _ = self.whisper_model.transcribe(
            utterance_audio_float32,
            language=get_whisper_source_language(),
            beam_size=self.beam_size,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=self.condition_on_previous_text,
        )

        accepted_segment_texts = []
        for segment in segments:
            segment_text = segment.text.strip()
            if not segment_text:
                continue

            if self.apply_confidence_filters:
                if segment.no_speech_prob > WHISPER_NO_SPEECH_PROB_THRESHOLD:
                    logger.info(
                        f"Segmento descartado "
                        f"(no_speech_prob={segment.no_speech_prob:.2f}): {segment_text!r}"
                    )
                    continue

                if segment.avg_logprob < WHISPER_AVG_LOGPROB_THRESHOLD:
                    logger.info(
                        f"Segmento descartado "
                        f"(avg_logprob={segment.avg_logprob:.2f}): {segment_text!r}"
                    )
                    continue

                if segment.compression_ratio > WHISPER_COMPRESSION_RATIO_THRESHOLD:
                    logger.info(
                        f"Segmento descartado "
                        f"(compression_ratio={segment.compression_ratio:.2f}): {segment_text!r}"
                    )
                    continue

            accepted_segment_texts.append(segment_text)

        final_text = " ".join(accepted_segment_texts).strip()

        if final_text and is_known_hallucination(final_text):
            logger.info(
                f"Texto descartado por lista negra de alucinaciones conocidas: {final_text!r}"
            )
            return ""

        return final_text
