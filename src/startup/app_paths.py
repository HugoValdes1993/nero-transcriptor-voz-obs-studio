"""
Punto único para resolver "dónde vive la app en disco", sea corriendo desde
código fuente (python main.py) o empaquetada como .exe (ver build.spec).

Antes cada módulo que necesitaba esto (config/user_config.py,
src/gui/config_window.py) calculaba su propia ruta contando carpetas hacia
arriba desde su __file__ — frágil, duplicado, y además roto para el .exe: en
un build empaquetado __file__ apunta adentro del bundle de PyInstaller, no a
donde el usuario puso el ejecutable.

Autor: Nero
"""

import os
import sys


def get_app_root() -> str:
    """
    - Empaquetado (.exe, ver build.spec): carpeta que contiene el
      ejecutable. El build es 'onedir' a propósito, no 'onefile' — con
      'onefile' esta carpeta sería una carpeta temporal distinta en cada
      arranque (sys._MEIPASS), lo que rompería dos cosas: user_config.json
      (el usuario perdería su configuración guardada entre sesiones) y la
      ruta a overlay/*.html que el usuario pega en OBS como "Local file"
      (dejaría de existir apenas se cierra la app).
    - Código fuente: raíz del repo (voice-transcriber/), tres niveles arriba
      de este archivo (src/startup/app_paths.py -> src/startup -> src -> raíz).
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
