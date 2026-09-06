"""
Wrapper sobre Silero VAD para determinar, frame por frame, si el audio capturado
contiene voz. Se usa para segmentar utterances antes de enviarlas a Whisper,
evitando alucinaciones del modelo durante silencios o ruido de fondo.

El modelo de Silero es una RNN (LSTMCell, ver VADDecoderRNNJIT) que mantiene
estado interno ENTRE llamadas — no es sin memoria frame a frame. La propia
referencia oficial de Silero (utils.VADIterator, la clase de ejemplo que
provee el repo del modelo) resetea ese estado con model.reset_states() cada
vez que arranca un ciclo nuevo de detección, precisamente porque dejarlo
acumular indefinidamente hace que las probabilidades de voz se degraden con
el tiempo (típico en sesiones largas, ej. un stream de horas): la app
empieza escuchando bien y con el correr de los minutos necesita voz cada vez
más fuerte/clara para seguir detectándose como voz. Por eso reset() existe
acá y TranscriptionPipeline la llama al cerrarse cada utterance (ver
_process_frame) — el mismo punto en el que UtteranceSegmenter también
resetea su propio estado.
"""

import torch
import numpy as np

from config.settings import (
    AUDIO_SAMPLE_RATE_HZ,
    VAD_SPEECH_PROBABILITY_THRESHOLD,
)


class VoiceActivityDetector:
    def __init__(self):
        self.silero_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
        )
        self.silero_model.eval()

    def frame_contains_speech(self, audio_frame_float32: np.ndarray) -> bool:
        """
        Recibe un frame de audio en float32 (rango [-1.0, 1.0]) y retorna
        True si Silero VAD detecta voz por encima del umbral configurado.
        """
        audio_tensor = torch.from_numpy(audio_frame_float32)
        with torch.no_grad():
            speech_probability = self.silero_model(
                audio_tensor, AUDIO_SAMPLE_RATE_HZ
            ).item()
        return speech_probability >= VAD_SPEECH_PROBABILITY_THRESHOLD

    def reset(self):
        """Limpia el estado interno de la RNN (ver docstring del módulo).
        Se llama al cerrar cada utterance, para que el silencio/voz de la
        frase siguiente arranque sin arrastrar contexto de la anterior."""
        self.silero_model.reset_states()
