"""
Captura de audio del micrófono en tiempo real usando sounddevice.

Muchos dispositivos (ej. auriculares con micrófono de controles de consola)
no soportan 16kHz nativamente. Por eso se captura a la frecuencia nativa que
reporta el propio dispositivo y se resamplea en software a AUDIO_SAMPLE_RATE_HZ
(16kHz) antes de entregar los frames, que siempre salen en tamaño fijo
(AUDIO_FRAME_SAMPLES) en float32 mono, listos para el VoiceActivityDetector.
"""

import queue

import numpy as np
import sounddevice as sd
import torch
import torchaudio.functional as torchaudio_functional

from config.settings import (
    AUDIO_SAMPLE_RATE_HZ,
    AUDIO_FRAME_SAMPLES,
    AUDIO_INPUT_HOST_API_OVERRIDES,
)
from config.user_config import get_audio_input_device_name
from src.audio.device_resolver import resolve_input_device_candidates_or_default
from src.logging_utils import ComponentLogger

logger = ComponentLogger("AudioCapture")


class AudioStreamCapture:
    def __init__(self):
        self.audio_frame_queue: "queue.Queue[np.ndarray]" = queue.Queue()

        configured_device_name = get_audio_input_device_name()
        # El override de host API es específico del driver de cada
        # dispositivo (ver AUDIO_INPUT_HOST_API_OVERRIDES en
        # config/settings.py): solo se prueba primero si el micrófono
        # elegido en la GUI está en ese mapa; cualquier otro empieza por el
        # mejor host API que el resolver encuentre (WASAPI si está
        # disponible). Ver start() para por qué esto es una LISTA de
        # candidatas y no un solo dispositivo elegido de antemano.
        host_api_override = AUDIO_INPUT_HOST_API_OVERRIDES.get(configured_device_name)
        self._device_candidates, self._used_fallback_device = resolve_input_device_candidates_or_default(
            configured_device_name, host_api_override
        )

        self.input_device_index: int | None = None
        self.native_sample_rate_hz: int | None = None
        self.native_blocksize: int | None = None
        self.input_stream: sd.InputStream | None = None

        # Buffer de arrastre para compensar el redondeo del resampleo entre bloques
        self._resampled_carry_over = np.empty((0,), dtype=np.float32)

    def _on_audio_block_captured(self, indata, frames, time_info, status):
        if status:
            logger.warning(f"Estado del stream de audio: {status}")

        if self.native_sample_rate_hz == AUDIO_SAMPLE_RATE_HZ:
            resampled_block = indata[:, 0].copy()
        else:
            native_audio_tensor = torch.from_numpy(indata[:, 0].copy())
            resampled_tensor = torchaudio_functional.resample(
                native_audio_tensor,
                orig_freq=self.native_sample_rate_hz,
                new_freq=AUDIO_SAMPLE_RATE_HZ,
            )
            resampled_block = resampled_tensor.numpy()

        # Concatena con el arrastre del bloque anterior y corta en frames exactos
        # de AUDIO_FRAME_SAMPLES, guardando el remanente para el próximo callback.
        combined_audio = np.concatenate([self._resampled_carry_over, resampled_block])

        full_frame_count = len(combined_audio) // AUDIO_FRAME_SAMPLES
        usable_sample_count = full_frame_count * AUDIO_FRAME_SAMPLES

        for frame_start in range(0, usable_sample_count, AUDIO_FRAME_SAMPLES):
            frame = combined_audio[frame_start : frame_start + AUDIO_FRAME_SAMPLES]
            self.audio_frame_queue.put(frame)

        self._resampled_carry_over = combined_audio[usable_sample_count:]

    def start(self):
        """
        Prueba cada candidata (ver resolve_input_device_candidates_or_default
        en device_resolver.py) hasta que una abra Y arranque de verdad,
        quedándose con la primera que funcione. Hace falta este intento real
        porque algunos drivers (ej. ciertos auriculares Bluetooth
        "combinados", igual que el control DualSense) reportan la entrada
        WASAPI como disponible pero fallan recién al arrancar el stream con
        'PaErrorCode -9999: WdmSyncIoctl ...' — un error que solo aparece al
        intentarlo, no antes. Si eso pasa, se prueba la siguiente entrada
        segura del mismo dispositivo (nunca WDM-KS, ver device_resolver.py)
        en vez de tirar abajo todo el pipeline.
        """
        last_error: Exception | None = None

        for device_info in self._device_candidates:
            try:
                self._open_and_start(device_info)
                return
            except sd.PortAudioError as error:
                logger.warning(
                    f"No se pudo arrancar '{device_info['name']}' vía "
                    f"{device_info['host_api_name']} ({error}); probando la "
                    f"siguiente entrada segura disponible."
                )
                self._reset_stream_state()
                last_error = error

        raise RuntimeError(
            f"No se pudo abrir ninguna entrada de audio segura para el "
            f"micrófono configurado. Último error: {last_error}"
        ) from last_error

    def _open_and_start(self, device_info: dict):
        native_sample_rate_hz = int(device_info["default_samplerate"])
        # Bloque nativo con la misma duración que AUDIO_FRAME_SAMPLES a 16kHz
        native_blocksize = int(AUDIO_FRAME_SAMPLES * native_sample_rate_hz / AUDIO_SAMPLE_RATE_HZ)

        # Fuerza modo compartido en WASAPI: el modo exclusivo suele fallar con
        # drivers de dispositivos no estándar (ej. micrófono de auriculares
        # conectado vía un control inalámbrico).
        #
        # auto_convert=True evita un error conocido de PortAudio con drivers
        # de audio "combinados" (ej. el del DualSense, que comparte chip para
        # audio y otras funciones del control): al iniciar el stream, WASAPI
        # intenta negociar el formato de audio directamente con el driver
        # mediante una consulta de bajo nivel (WdmSyncIoctl) que ese driver
        # en particular no responde correctamente, aunque el dispositivo
        # funcione bien para todo lo demás (PaErrorCode -9999). Con
        # auto_convert=True, PortAudio hace la conversión de formato él mismo
        # en vez de depender de esa negociación con el driver — pero en
        # algunos drivers ni siquiera eso alcanza, de ahí el fallback en
        # start() a la siguiente entrada segura si esto igual falla.
        wasapi_extra_settings = None
        if "WASAPI" in device_info["host_api_name"]:
            wasapi_extra_settings = sd.WasapiSettings(exclusive=False, auto_convert=True)

        input_stream = sd.InputStream(
            samplerate=native_sample_rate_hz,
            blocksize=native_blocksize,
            device=device_info["index"],
            channels=1,
            dtype="float32",
            callback=self._on_audio_block_captured,
            extra_settings=wasapi_extra_settings,
        )

        # Se asignan ANTES de start(): el callback puede empezar a correr en
        # su propio hilo apenas arranca el stream, y necesita
        # self.native_sample_rate_hz ya seteado.
        self.input_device_index = device_info["index"]
        self.native_sample_rate_hz = native_sample_rate_hz
        self.native_blocksize = native_blocksize
        self.input_stream = input_stream

        input_stream.start()

        fallback_note = (
            " (fallback: el configurado no está disponible)" if self._used_fallback_device else ""
        )
        logger.success(
            f"Dispositivo '{device_info['name']}'{fallback_note} vía "
            f"{device_info['host_api_name']} capturando a "
            f"{native_sample_rate_hz} Hz, resampleando a {AUDIO_SAMPLE_RATE_HZ} Hz"
        )

    def _reset_stream_state(self):
        if self.input_stream is not None:
            try:
                self.input_stream.close()
            except sd.PortAudioError:
                pass
        self.input_device_index = None
        self.native_sample_rate_hz = None
        self.native_blocksize = None
        self.input_stream = None

    def stop(self):
        self.input_stream.stop()
        self.input_stream.close()

    def get_next_frame(self, timeout_seconds: float = 1.0) -> np.ndarray:
        return self.audio_frame_queue.get(timeout=timeout_seconds)
