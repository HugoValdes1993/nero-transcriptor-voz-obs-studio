"""
Logging a consola para todo el proyecto.

Reemplaza los `print("[algo] mensaje")` sueltos que antes había repartidos
por cada archivo, por un formato único y legible:

    12:34:56 [INFO] AudioCapture  Dispositivo 'Auriculares' a 44100 Hz...
    12:34:59 [Transcripción] Hola, esto es una prueba

Cada componente crea su propio logger con `ComponentLogger("Nombre")`, así
que cualquier línea impresa se puede rastrear al módulo que la generó de un
vistazo. Los colores se desactivan solos si la salida no es una terminal
(por ejemplo, si se redirige a un archivo de log).
"""

import sys
from datetime import datetime


class _AnsiColor:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GRAY = "\033[90m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    MAGENTA = "\033[95m"


def _terminal_supports_color() -> bool:
    return sys.stdout.isatty()


class ComponentLogger:
    """
    Logger simple con un prefijo fijo (el nombre del componente), para que
    quede claro qué parte del pipeline generó cada línea de la consola.
    """

    def __init__(self, component_name: str):
        self.component_name = component_name
        self._color_enabled = _terminal_supports_color()

    def _colorize(self, text: str, color: str) -> str:
        if not self._color_enabled:
            return text
        return f"{color}{text}{_AnsiColor.RESET}"

    def _timestamp(self) -> str:
        return datetime.now().strftime("%H:%M:%S")

    def _print_line(self, level_label: str, level_color: str, message: str):
        timestamp = self._colorize(self._timestamp(), _AnsiColor.GRAY)
        level_tag = self._colorize(f"[{level_label}]", level_color)
        component_tag = self._colorize(self.component_name, _AnsiColor.CYAN)
        print(f"{timestamp} {level_tag} {component_tag}  {message}")

    def info(self, message: str):
        """Mensaje informativo normal (ej. 'Dispositivo detectado')."""
        self._print_line("INFO", _AnsiColor.CYAN, message)

    def success(self, message: str):
        """Confirmación de que algo terminó bien (ej. 'Escuchando micrófono')."""
        self._print_line(" OK ", _AnsiColor.GREEN, message)

    def warning(self, message: str):
        """Algo no bloqueante pero que vale la pena que el usuario note."""
        self._print_line("WARN", _AnsiColor.YELLOW, message)

    def error(self, message: str):
        """Algo que impide continuar con lo que se estaba haciendo."""
        self._print_line("ERROR", _AnsiColor.RED, message)

    def transcript(self, label: str, text: str):
        """
        Línea destacada para texto transcrito o traducido — se resalta en
        negrita/magenta para distinguirla fácilmente del resto de los logs
        de diagnóstico mientras el streamer mira la consola de reojo.
        """
        timestamp = self._colorize(self._timestamp(), _AnsiColor.GRAY)
        label_tag = self._colorize(f"[{label}]", _AnsiColor.MAGENTA + _AnsiColor.BOLD)
        print(f"{timestamp} {label_tag} {text}")


def print_startup_banner():
    """Firma del proyecto, impresa una vez al arrancar `python main.py`."""
    color_enabled = _terminal_supports_color()

    def colorize(text: str, color: str) -> str:
        if not color_enabled:
            return text
        return f"{color}{text}{_AnsiColor.RESET}"

    print(colorize("Voice Transcriber", _AnsiColor.CYAN + _AnsiColor.BOLD))
    print(colorize("  by Nero", _AnsiColor.GRAY))
    print()
