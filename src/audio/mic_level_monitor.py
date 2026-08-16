"""
Monitor de nivel de audio del micrófono, usado por el botón "Probar
micrófono" de la ventana de configuración.

Abre un stream de solo lectura sobre el dispositivo de entrada configurado y
expone el nivel RMS más reciente, sin pasar por el resto del pipeline de
transcripción (resampleo a 16kHz, VAD, Whisper, etc. — nada de eso hace
falta solo para confirmar que el micrófono está entregando señal). Ver
src/audio/stream_capture.py para el stream real que sí usa todo eso.
"""

import threading

import numpy as np
import sounddevice as sd

from config.settings import AUDIO_INPUT_HOST_API_OVERRIDES
from config.user_config import get_audio_input_device_name
from src.audio.device_resolver import resolve_input_device_candidates_or_default


class MicLevelMonitor:
    """
    start() abre el stream; el callback de PortAudio corre en su propio
    hilo y actualiza el nivel bajo un lock. get_level() es seguro de llamar
    desde el hilo de la GUI en cualquier momento, incluso si el stream
    todavía no arrancó o ya se detuvo (devuelve 0.0).
    """

    def __init__(self):
        self._stream: sd.InputStream | None = None
        self._lock = threading.Lock()
        self._current_level = 0.0

    @property
    def is_active(self) -> bool:
        return self._stream is not None

    def start(self):
        """
        Prueba cada candidata (ver resolve_input_device_candidates_or_default
        en device_resolver.py) hasta que una abra Y arranque de verdad —
        mismo motivo que AudioStreamCapture.start() en stream_capture.py:
        algunos drivers reportan la entrada WASAPI como disponible pero
        fallan recién al arrancar el stream ('PaErrorCode -9999:
        WdmSyncIoctl ...'), un error que solo aparece al intentarlo.
        """
        if self.is_active:
            return

        configured_device_name = get_audio_input_device_name()
        # Mismo criterio que stream_capture.py: el override de host API es
        # específico del driver de cada dispositivo, no de cualquier
        # micrófono que el usuario haya elegido desde la GUI.
        host_api_override = AUDIO_INPUT_HOST_API_OVERRIDES.get(configured_device_name)
        device_candidates, _ = resolve_input_device_candidates_or_default(
            configured_device_name, host_api_override
        )

        last_error: Exception | None = None
        for device_info in device_candidates:
            # Mismo motivo que en stream_capture.py: modo compartido en
            # WASAPI para no bloquear el dispositivo ni chocar con drivers
            # no estándar.
            wasapi_extra_settings = None
            if "WASAPI" in device_info["host_api_name"]:
                wasapi_extra_settings = sd.WasapiSettings(exclusive=False, auto_convert=True)

            stream = sd.InputStream(
                samplerate=int(device_info["default_samplerate"]),
                device=device_info["index"],
                channels=1,
                dtype="float32",
                callback=self._on_audio_block,
                extra_settings=wasapi_extra_settings,
            )
            try:
                stream.start()
            except sd.PortAudioError as error:
                try:
                    stream.close()
                except sd.PortAudioError:
                    pass
                last_error = error
                continue

            self._stream = stream
            return

        raise RuntimeError(
            f"No se pudo abrir ninguna entrada de audio segura para el "
            f"micrófono configurado. Último error: {last_error}"
        ) from last_error

    def _on_audio_block(self, indata, frames, time_info, status):
        rms_level = float(np.sqrt(np.mean(np.square(indata[:, 0]))))
        with self._lock:
            self._current_level = rms_level

    def get_level(self) -> float:
        with self._lock:
            return self._current_level

    def stop(self):
        if self._stream is None:
            return
        self._stream.stop()
        self._stream.close()
        self._stream = None
        with self._lock:
            self._current_level = 0.0
