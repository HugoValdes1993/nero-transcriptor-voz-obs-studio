"""
Utilidad de debug: guarda cada utterance (el mismo buffer float32 a 16kHz que
recibe SpeechTranscriber) como archivo .wav, para poder escucharlo y verificar
si el problema de precisión es del audio de entrada (ruido, volumen bajo,
procesamiento agresivo del driver) y no de Whisper en sí.

Activar/desactivar desde config/settings.py (DEBUG_SAVE_UTTERANCE_WAV_ENABLED).
Los archivos se guardan en la carpeta configurada en DEBUG_UTTERANCE_WAV_DIR,
uno por utterance, con timestamp en el nombre para poder correlacionarlos con
la línea impresa en consola.
"""

import os
import time
import wave

import numpy as np

from config.settings import AUDIO_SAMPLE_RATE_HZ, DEBUG_UTTERANCE_WAV_DIR


def save_utterance_debug_wav(utterance_audio_float32: np.ndarray) -> str:
    """
    Guarda el buffer de audio (float32, rango [-1.0, 1.0]) como .wav de 16
    bits PCM mono, en DEBUG_UTTERANCE_WAV_DIR. Retorna la ruta del archivo
    generado.
    """
    os.makedirs(DEBUG_UTTERANCE_WAV_DIR, exist_ok=True)

    timestamp_label = time.strftime("%Y%m%d_%H%M%S")
    unique_suffix = str(time.time_ns())[-6:]  # evita colisiones dentro del mismo segundo
    output_path = os.path.join(
        DEBUG_UTTERANCE_WAV_DIR, f"utterance_{timestamp_label}_{unique_suffix}.wav"
    )

    clipped_audio = np.clip(utterance_audio_float32, -1.0, 1.0)
    pcm16_audio = (clipped_audio * 32767.0).astype(np.int16)

    with wave.open(output_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)  # 16 bits
        wav_file.setframerate(AUDIO_SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm16_audio.tobytes())

    return output_path
