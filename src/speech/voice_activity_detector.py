"""
Wrapper sobre Silero VAD para determinar, frame por frame, si el audio capturado
contiene voz. Se usa para segmentar utterances antes de enviarlas a Whisper,
evitando alucinaciones del modelo durante silencios o ruido de fondo.
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
