"""
Punto de entrada de la app.

Abre la ventana de configuración (src/gui/config_window.py), que se queda
abierta durante toda la vida de la app: desde ahí mismo se inicia y se
detiene el pipeline de transcripción (captura de micrófono -> detección de
voz -> segmentación de frases -> transcripción -> traducción -> filtro de
groserías -> broadcast al overlay de OBS vía WebSocket), sin cerrar la
ventana ni bloquearla mientras corre (ver src/server/pipeline_controller.py).
Cerrar la ventana termina el programa.

Ejecutar con: python main.py

Autor: Nero
"""

# Debe importarse ANTES que cualquier módulo que dependa de faster-whisper /
# CTranslate2 (ej. src.speech.speech_transcriber, vía PipelineController),
# para que las DLLs de CUDA se registren a tiempo en Windows. Ver el
# docstring de este módulo para el detalle completo del problema que resuelve.
from src.startup import cuda_dll_setup  # noqa: F401

from src.gui.config_window import ConfigWindow
from src.logging_utils import print_startup_banner

if __name__ == "__main__":
    print_startup_banner()
    app = ConfigWindow()
    app.mainloop()
