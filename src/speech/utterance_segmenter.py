"""
Recibe frames de audio uno por uno junto con su etiqueta de voz/silencio
(entregada por VoiceActivityDetector) y acumula una utterance completa.
Cierra la utterance cuando detecta suficiente silencio continuo después
de haber capturado voz, y la descarta si es demasiado corta (ruido/clicks).

IMPORTANTE sobre silencios breves DENTRO de una utterance: un frame dura
solo ~32ms (AUDIO_FRAME_SAMPLES a 16kHz), y Silero VAD lo clasifica frame
por frame sin ningún suavizado. Una consonante suave, una sibilante, o
cualquier caída breve de energía puede hacer que un frame puntual en medio
de una palabra se marque como "silencio" aunque la persona siga hablando.
Si ese frame se descartara del buffer directamente (como hacía una versión
anterior de este archivo), el audio le llega a Whisper con un corte literal
en medio de la palabra — la causa más probable de pérdida de palabras o
pedazos de palabras en la transcripción.

Por eso esta versión SIEMPRE guarda el audio de cualquier frame que llegue
mientras hay una utterance en curso, sea voz o silencio. Solo se recorta,
al momento de cerrar, la cola de silencio que efectivamente disparó el
cierre (los últimos SILENCE_FRAMES_TO_CLOSE_UTTERANCE frames) — eso sí es
silencio real de cierre y no debe mandarse a Whisper.
"""

import numpy as np

from config.settings import (
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_FRAME_SAMPLES,
    VAD_SILENCE_DURATION_MS_TO_CLOSE_UTTERANCE,
    VAD_MIN_SPEECH_DURATION_MS,
    PARTIAL_TRANSCRIPTION_INTERVAL_MS,
)

FRAME_DURATION_MS = (AUDIO_FRAME_SAMPLES / AUDIO_SAMPLE_RATE_HZ) * 1000
SILENCE_FRAMES_TO_CLOSE_UTTERANCE = int(
    VAD_SILENCE_DURATION_MS_TO_CLOSE_UTTERANCE / FRAME_DURATION_MS
)
MIN_SPEECH_FRAMES = int(VAD_MIN_SPEECH_DURATION_MS / FRAME_DURATION_MS)

# Cuántos frames de voz acumulada equivalen a PARTIAL_TRANSCRIPTION_INTERVAL_MS.
# Mínimo 1 para evitar un intervalo de 0 si el valor configurado es muy chico.
PARTIAL_INTERVAL_FRAMES = max(1, int(PARTIAL_TRANSCRIPTION_INTERVAL_MS / FRAME_DURATION_MS))


class UtteranceSegmenter:
    def __init__(self):
        # Incluye TODOS los frames desde que arrancó la utterance (voz y
        # silencios breves intercalados), no solo los marcados como voz.
        self.accumulated_frames: list[np.ndarray] = []
        # Cuenta aparte, SOLO de frames marcados como voz, para el filtro
        # de duración mínima (MIN_SPEECH_FRAMES) — así una utterance con
        # mucho silencio intercalado pero poca voz real sigue descartándose.
        self.speech_frame_count = 0
        self.consecutive_silence_frame_count = 0
        # Cantidad de frames que había acumulados la última vez que se emitió
        # (o se consultó) un parcial. Sirve para saber cuánto audio "nuevo"
        # se sumó desde entonces sin necesidad de un timestamp de reloj real.
        self.frame_count_at_last_partial_emit = 0

    def maybe_get_partial_audio(self) -> np.ndarray | None:
        """
        Retorna el audio acumulado de la utterance ABIERTA hasta este
        momento, si ya pasaron al menos PARTIAL_INTERVAL_FRAMES frames desde
        el último parcial devuelto. Retorna None si todavía no toca (no pasó
        suficiente audio nuevo) o si no hay ninguna utterance en curso.

        A diferencia de add_frame(), este método NUNCA cierra ni resetea la
        utterance: es solo una "foto" del buffer que sigue creciendo. Llamarlo
        más de una vez sin que se haya sumado audio nuevo devuelve None la
        segunda vez.
        """
        if not self.accumulated_frames:
            return None

        frames_since_last_partial = (
            len(self.accumulated_frames) - self.frame_count_at_last_partial_emit
        )
        if frames_since_last_partial < PARTIAL_INTERVAL_FRAMES:
            return None

        self.frame_count_at_last_partial_emit = len(self.accumulated_frames)
        return np.concatenate(self.accumulated_frames)

    def add_frame(self, audio_frame: np.ndarray, contains_speech: bool) -> np.ndarray | None:
        """
        Procesa un frame nuevo. Retorna el audio de la utterance completa
        (numpy array) cuando se cierra por silencio, o None si la utterance
        sigue abierta o fue descartada por ser demasiado corta.
        """
        if contains_speech:
            self.accumulated_frames.append(audio_frame)
            self.speech_frame_count += 1
            self.consecutive_silence_frame_count = 0
            return None

        if not self.accumulated_frames:
            return None  # Silencio sin voz previa acumulada: no hay nada que cerrar

        # Silencio en medio (o al final) de una utterance en curso: se
        # guarda igual el audio — ver el docstring del módulo sobre por qué
        # no se descarta aquí. Recién se decide si era silencio de cierre
        # real más abajo, cuando efectivamente se cierra la utterance.
        self.accumulated_frames.append(audio_frame)
        self.consecutive_silence_frame_count += 1

        if self.consecutive_silence_frame_count < SILENCE_FRAMES_TO_CLOSE_UTTERANCE:
            return None  # Todavía no hay suficiente silencio para cerrar la frase

        # Se cierra: separamos el silencio de cierre real (los últimos
        # SILENCE_FRAMES_TO_CLOSE_UTTERANCE frames, que fueron los que
        # dispararon el cierre) del resto del audio, que sí puede incluir
        # voz y silencios breves intercalados que queremos conservar.
        trailing_silence_frame_count = self.consecutive_silence_frame_count
        speech_and_gaps_frames = self.accumulated_frames[:-trailing_silence_frame_count]
        speech_frame_count = self.speech_frame_count
        self._reset()

        if speech_frame_count < MIN_SPEECH_FRAMES:
            return None  # Descarta utterances demasiado cortas (ruido, clicks)

        if not speech_and_gaps_frames:
            return None  # Toda la utterance terminó siendo silencio de cierre

        return np.concatenate(speech_and_gaps_frames)

    def _reset(self):
        self.accumulated_frames = []
        self.speech_frame_count = 0
        self.consecutive_silence_frame_count = 0
        self.frame_count_at_last_partial_emit = 0
