"""
Notificación de estados transitorios (ej. "descargando modelo...") desde
cualquier parte del pipeline hacia quien esté escuchando — en la práctica,
la ventana de configuración (ver src/gui/config_window.py), que muestra el
mensaje en su status_label mientras dura.

Separado de logging_utils.py porque el propósito acá no es imprimir en
consola sino avisarle en vivo a la GUI — aunque en la práctica casi siempre
se usa junto a un logger.info() con un mensaje parecido, para quien esté
mirando la consola en vez de la ventana.

Un callback global simple alcanza (no una cola ni varios suscriptores):
solo hay una ventana de configuración por proceso. Quien llama a
notify_status() puede correr en cualquier hilo (ej. el hilo de fondo del
pipeline de transcripción) — el listener es responsable de reencolar al
hilo de la GUI si hace falta (ver ConfigWindow._on_download_status).
"""

from typing import Callable, Optional

_listener: Optional[Callable[[str], None]] = None


def set_status_listener(listener: Optional[Callable[[str], None]]):
    """Registra quién recibe las notificaciones de aquí en más. Pasar None
    para des-registrar (ver ConfigWindow._on_close_requested)."""
    global _listener
    _listener = listener


def notify_status(message: str):
    """
    Avisa un estado transitorio. Una cadena vacía es la señal de "ya
    terminó lo que sea que se estaba avisando" — ver
    ConfigWindow._apply_download_status para cómo se interpreta.
    No hace nada si nadie se registró (ej. scripts sueltos, tests).
    """
    if _listener is not None:
        _listener(message)
