"""
Ventana principal de la app: configuración + control de arranque/detención
del pipeline (launcher).

A diferencia de un launcher clásico, esta ventana NO se cierra al iniciar la
transcripción: se queda abierta y el mismo botón pasa a decir "Detener
transcripción", para poder cortar el pipeline sin matar la app. Pensada para
eventualmente empaquetarse en un .exe distribuible a otros streamers, que no
deberían tener que editar config/settings.py a mano ni usar la consola.

La ventana nace centrada en la pantalla (ver _center_on_screen) y está
organizada como un tab panel (CTkTabview) con 3 pestañas: "Configuración
general" (dispositivo de entrada con botón de prueba de nivel en vivo,
flujo de transcripción/traducción, y aceleración por GPU), y "Estilos texto
original" / "Estilos texto traducido" (link para copiar el overlay
correspondiente + tipografía —fuente, tamaño, grosor y color/opacidad— y
ancho/alto exactos de la zona de texto, para poder hacerla coincidir con la
sección del lienzo que el usuario ya tiene reservada en su escena de OBS).

El guardado en disco es manual: los cambios se aplican en vivo (incluida la
vista previa por WebSocket en OBS si el pipeline está corriendo) apenas se
tocan, pero no se escriben en user_config.json hasta que se aprieta el botón
"Guardar configuración" del footer — ver save_user_config() en
config/user_config.py y _on_close_requested (avisa si hay cambios sin
guardar antes de cerrar la ventana).

Autor: Nero
"""

import os
import re
import subprocess
import tkinter
import webbrowser
import winreg
from pathlib import Path
from tkinter import colorchooser, font as tkfont, messagebox, simpledialog

import customtkinter as ctk

from config.settings import MIN_UTTERANCE_RMS_ENERGY
from config.user_config import (
    get_audio_input_device_name,
    set_audio_input_device_name,
    get_cuda_acceleration_enabled,
    set_cuda_acceleration_enabled,
    get_translation_direction,
    set_translation_direction,
    TRANSLATION_DIRECTION_ES_TO_EN,
    TRANSLATION_DIRECTION_EN_TO_ES,
    get_overlay_style,
    set_overlay_style_value,
    get_overlay_style_presets,
    is_built_in_style_preset,
    save_overlay_style_preset,
    delete_overlay_style_preset,
    apply_overlay_style_preset,
    save_user_config,
)
from src.audio.device_resolver import (
    get_default_input_device_name,
    list_available_input_device_names,
)
from src.audio.mic_level_monitor import MicLevelMonitor
from src.server.pipeline_controller import PipelineController
from src.startup.app_paths import get_app_root
from src.startup.cuda_availability import is_nvidia_gpu_available, gpu_unavailability_reason
from src.status_hub import set_status_listener

# Raíz de la app (repo en dev, carpeta del .exe empaquetado) — ver
# src/startup/app_paths.get_app_root para el porqué de por qué no se calcula
# acá mismo con __file__.
_PROJECT_ROOT = get_app_root()


def _parse_shell_open_command(command_template: str) -> list[str]:
    """Separa un comando del registro tipo `"C:\\ruta\\app.exe" "%1"` en
    tokens, sacando las comillas — regex simple en vez de shlex.split
    (pensado para bash) o subprocess.list2cmdline (arma comandos, no los
    parsea)."""
    tokens = re.findall(r'"[^"]*"|\S+', command_template)
    return [token.strip('"') for token in tokens]


def _resolve_default_browser_command() -> list[str] | None:
    """
    Resuelve el comando del navegador que Windows tiene configurado como
    predeterminado para abrir links http, leyendo el mismo registro que usa
    el panel "Aplicaciones predeterminadas". None si no se pudo leer (ej.
    nunca se eligió un navegador default explícito).
    """
    try:
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice",
        ) as key:
            prog_id, _ = winreg.QueryValueEx(key, "ProgId")
        with winreg.OpenKey(winreg.HKEY_CLASSES_ROOT, rf"{prog_id}\shell\open\command") as key:
            command_template, _ = winreg.QueryValueEx(key, "")
    except OSError:
        return None
    return _parse_shell_open_command(command_template)


def _open_url_in_browser(url: str):
    """
    Abre `url` con el navegador real que el usuario eligió como
    predeterminado, en vez de webbrowser.open()/os.startfile() a secas.

    Por qué: esos dos terminan resolviendo la URL vía la asociación de la
    EXTENSIÓN .html (Panel de control -> "Aplicaciones predeterminadas" la
    guarda aparte de la asociación del PROTOCOLO http) — en muchas
    instalaciones de Windows esa asociación de archivo sigue apuntando a
    Internet Explorer por default de fábrica, un binario retirado en
    Windows 11 que no abre ninguna ventana visible aunque "funcione" sin
    tirar error. La asociación del protocolo http, en cambio, sí apunta al
    navegador real (Edge/Chrome/Firefox/etc.), así que se resuelve esa en
    vez de dejar que Windows elija.
    """
    command = _resolve_default_browser_command()
    if command:
        args = [url if part == "%1" else part for part in command]
        if url not in args:
            args.append(url)
        try:
            subprocess.Popen(args)
            return
        except OSError:
            pass
    webbrowser.open(url)


class _FontFamilyPicker:
    """
    Campo de texto + lista desplegable para elegir una fuente instalada en el
    sistema (ver ConfigWindow._system_font_families), filtrable escribiendo.

    No usa CTkComboBox: su dropdown es un tkinter.Menu nativo de Windows, que
    no responde a la rueda del mouse — con cientos de fuentes instaladas,
    poder filtrar escribiendo pero no poder scrollear el resultado lo hacía
    prácticamente inutilizable. Este picker arma su propia lista desplegable
    (un Listbox dentro de un Toplevel sin bordes, posicionado debajo del
    campo) para poder scrollear con la rueda como cualquier lista normal.
    """

    MAX_VISIBLE_ROWS = 8
    POPUP_BG_COLOR = "#2b2b2b"
    POPUP_BORDER_COLOR = "#565b5e"
    POPUP_SELECT_COLOR = "#1f6aa5"

    def __init__(self, parent, all_values: list[str], initial_value: str, on_value_committed):
        self._all_values = all_values
        self._on_value_committed = on_value_committed
        self._last_committed_value = initial_value
        self._popup = None
        self._listbox = None

        self.entry = ctk.CTkEntry(parent)
        self.entry.insert(0, initial_value)
        self.entry.bind("<KeyRelease>", self._on_key_release)
        self.entry.bind("<Return>", self._on_return)
        self.entry.bind("<Escape>", lambda _event: self._close_popup())
        self.entry.bind("<FocusOut>", self._on_focus_out)

    def pack(self, **kwargs):
        self.entry.pack(**kwargs)

    def set(self, value: str):
        self.entry.delete(0, "end")
        self.entry.insert(0, value)

    def _matching_values(self) -> list[str]:
        typed = self.entry.get().strip()
        if not typed:
            return self._all_values
        return [name for name in self._all_values if typed.casefold() in name.casefold()]

    def _commit(self, value: str):
        self.set(value)
        if value != self._last_committed_value:
            self._last_committed_value = value
            self._on_value_committed(value)

    def _on_key_release(self, event):
        # Return/Escape/flechas ya tienen su propio binding — no deben
        # reabrir ni refiltrar la lista.
        if event.keysym in ("Return", "Escape", "Up", "Down"):
            return
        matches = self._matching_values()
        if matches:
            self._show_popup(matches)
        else:
            self._close_popup()

    def _on_return(self, _event=None):
        typed = self.entry.get().strip()
        match = next((name for name in self._all_values if name.casefold() == typed.casefold()), None)
        if match is not None:
            self._commit(match)
            self._close_popup()
        return "break"

    def _on_focus_out(self, _event=None):
        # Al hacer click en un ítem del Listbox, el foco sale del Entry antes
        # de que termine de procesarse el click — se posterga la validación
        # para no cerrar/revertir de encima de una selección en curso.
        self.entry.after(150, self._finalize_focus_out)

    def _finalize_focus_out(self):
        if self._popup is not None and self.entry.focus_get() is self._listbox:
            return
        self._close_popup()

        typed = self.entry.get().strip()
        match = next((name for name in self._all_values if name.casefold() == typed.casefold()), None)
        if match is None:
            # No coincide con ninguna fuente instalada (typo, o foco perdido
            # a medio escribir): se revierte en vez de aplicar un
            # font-family que el overlay no va a poder resolver.
            self.set(self._last_committed_value)
            return
        self._commit(match)

    def _create_popup(self):
        self._popup = tkinter.Toplevel(self.entry)
        self._popup.withdraw()
        self._popup.overrideredirect(True)
        self._popup.attributes("-topmost", True)

        frame = tkinter.Frame(
            self._popup,
            bg=self.POPUP_BG_COLOR,
            highlightthickness=1,
            highlightbackground=self.POPUP_BORDER_COLOR,
        )
        frame.pack(fill="both", expand=True)

        scrollbar = tkinter.Scrollbar(frame)
        scrollbar.pack(side="right", fill="y")

        self._listbox = tkinter.Listbox(
            frame,
            yscrollcommand=scrollbar.set,
            bg=self.POPUP_BG_COLOR,
            fg="white",
            selectbackground=self.POPUP_SELECT_COLOR,
            selectforeground="white",
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
            exportselection=False,
        )
        self._listbox.pack(side="left", fill="both", expand=True)
        scrollbar.configure(command=self._listbox.yview)

        # Se lee la posición del click directamente (nearest) en vez de
        # depender de <<ListboxSelect>>/foco: así la selección no queda
        # sujeta a la carrera con el FocusOut del Entry (ver _on_focus_out).
        self._listbox.bind("<Button-1>", self._on_listbox_clicked)
        self._listbox.bind("<MouseWheel>", self._on_mousewheel)

    def _on_listbox_clicked(self, event):
        index = self._listbox.nearest(event.y)
        if index < 0:
            return
        self._commit(self._listbox.get(index))
        self._close_popup()
        return "break"

    def _on_mousewheel(self, event):
        self._listbox.yview_scroll(int(-1 * (event.delta / 120)), "units")
        return "break"

    def _show_popup(self, matches: list[str]):
        if self._popup is None:
            self._create_popup()

        self._listbox.delete(0, "end")
        for name in matches:
            self._listbox.insert("end", name)
        self._listbox.configure(height=min(len(matches), self.MAX_VISIBLE_ROWS))

        self._popup.update_idletasks()
        width = self.entry.winfo_width()
        height = self._popup.winfo_reqheight()
        x = self.entry.winfo_rootx()
        y = self.entry.winfo_rooty() + self.entry.winfo_height()
        self._popup.geometry(f"{width}x{height}+{x}+{y}")
        self._popup.deiconify()
        self._popup.lift()

    def _close_popup(self):
        if self._popup is not None:
            self._popup.withdraw()


class ConfigWindow(ctk.CTk):
    WINDOW_TITLE = "Voice Transcriber — Configuración"
    WINDOW_SIZE = "600x680"

    TAB_GENERAL = "Configuración general"
    TAB_ORIGINAL_STYLE = "Estilos texto original"
    TAB_TRANSLATED_STYLE = "Estilos texto traducido"

    # Rutas a los dos overlays (ver overlay/*.html): cada uno se agrega como
    # Browser Source separado en OBS (ver README, sección "Integración con
    # OBS"), por eso cada pestaña de estilo tiene su propio botón de copiar.
    ORIGINAL_OVERLAY_FILE_PATH = os.path.join(_PROJECT_ROOT, "overlay", "obs_overlay_original.html")
    TRANSLATED_OVERLAY_FILE_PATH = os.path.join(_PROJECT_ROOT, "overlay", "obs_overlay_translated.html")

    COPY_LINK_BUTTON_TEXT = "📋 Copiar link del overlay"
    COPY_LINK_BUTTON_CONFIRMATION_TEXT = "✓ Copiado"
    COPY_LINK_CONFIRMATION_DURATION_MS = 1500

    SAVE_BUTTON_TEXT = "💾 Guardar configuración"
    SAVE_BUTTON_CONFIRMATION_TEXT = "✓ Configuración guardada"
    SAVE_CONFIRMATION_DURATION_MS = 1500

    # Tamaño de fuente: acotado a un máximo razonable para subtítulos (nada
    # impide que el usuario quiera letra grande, pero sin límite un desliz
    # del slider puede terminar en un tamaño inutilizable).
    FONT_SIZE_MIN_PX = 12
    FONT_SIZE_MAX_PX = 120

    # Ancho/alto de la zona de texto: rango generoso para cubrir desde una
    # esquina chica hasta un canvas 4K completo (3840x2160), ya que tienen
    # que poder coincidir con cualquier sección que el usuario haya
    # reservado en su lienzo de OBS.
    TEXT_ZONE_WIDTH_MIN_PX = 100
    TEXT_ZONE_WIDTH_MAX_PX = 3840
    TEXT_ZONE_HEIGHT_MIN_PX = 30
    TEXT_ZONE_HEIGHT_MAX_PX = 2160

    # Etiqueta legible -> valor real de la propiedad CSS font-weight. Un
    # subconjunto curado de los pesos estándar (100-900 de a 100) en vez de
    # los nueve completos: los intermedios raros (100/200/900) casi no se
    # distinguen entre sí en fuentes de sistema y solo suman ruido al
    # selector.
    FONT_WEIGHT_OPTIONS = {
        "Normal": "400",
        "Medium": "500",
        "Semi-negrita": "600",
        "Negrita": "700",
        "Extra negrita": "800",
    }

    # Etiqueta legible -> nombre interno del efecto (ver
    # SUBTITLE_ANIMATION_CHOICES en config/user_config.py y el JS de
    # overlay/obs_overlay_*.html, que es quien realmente sabe animar cada
    # uno). Se dispara solo cuando una frase queda transcripta en
    # definitiva, nunca en el texto tentativo.
    SUBTITLE_ANIMATION_OPTIONS = {
        "Ninguno": "none",
        "Fade": "fade",
        "Bounce": "bounce",
        "Glitch": "glitch",
    }

    # Rango del slider de opacidad del texto, en porcentaje (0-100) — se
    # guarda como fracción 0.0-1.0 (ver original_text_opacity/
    # translated_text_opacity en OVERLAY_STYLE_DEFAULTS), igual convención
    # que ya usa background_opacity.
    TEXT_OPACITY_MIN_PERCENT = 0
    TEXT_OPACITY_MAX_PERCENT = 100

    # Ancho del contorno de texto (-webkit-text-stroke). 0 = sin contorno;
    # un tope bajo alcanza porque a partir de unos pocos px el contorno
    # empieza a comerse los huecos de letras como "o"/"e" y se vuelve
    # ilegible en vez de ayudar.
    TEXT_STROKE_WIDTH_MIN_PX = 0
    TEXT_STROKE_WIDTH_MAX_PX = 6

    PREVIEW_BUTTON_START_TEXT = "👁 Vista previa"
    PREVIEW_BUTTON_STOP_TEXT = "⏹ Detener vista previa"
    # Texto de muestra que se manda al overlay mientras la vista previa está
    # activa — se reenvía cada PREVIEW_REFRESH_INTERVAL_MS (por debajo de
    # clear_delay_ms, 6000ms de fábrica) para que no desaparezca solo
    # mientras el usuario todavía está ajustando estilos.
    PREVIEW_SAMPLE_ORIGINAL_TEXT = "Así se ve tu subtítulo en vivo"
    PREVIEW_SAMPLE_TRANSLATED_TEXT = "This is how your live subtitle looks"
    PREVIEW_REFRESH_INTERVAL_MS = 2000

    STYLE_PRESET_PLACEHOLDER = "Elegir preset..."
    SAVE_PRESET_BUTTON_TEXT = "💾 Guardar como preset..."
    DELETE_PRESET_BUTTON_TEXT = "🗑 Eliminar preset"

    IDLE_BUTTON_TEXT = "Iniciar transcripción"
    STARTING_BUTTON_TEXT = "Iniciando..."
    RUNNING_BUTTON_TEXT = "Detener transcripción"
    STOPPING_BUTTON_TEXT = "Deteniendo..."

    IDLE_BUTTON_COLOR = "#1f6aa5"
    RUNNING_BUTTON_COLOR = "#a53f2a"

    IDLE_STATUS_COLOR = "gray60"
    RUNNING_STATUS_COLOR = "#3fa54a"
    ERROR_STATUS_COLOR = "#e05252"

    MIC_TEST_START_TEXT = "🎤 Probar micrófono"
    MIC_TEST_STOP_TEXT = "Detener prueba"

    # Cada cuánto se refresca la barra de nivel mientras la prueba está
    # activa. Puramente visual, no tiene relación con AUDIO_FRAME_SAMPLES.
    MIC_LEVEL_POLL_INTERVAL_MS = 80

    # Nivel RMS que se dibuja como barra llena (voz a volumen normal/alto).
    # No es un límite técnico, solo hace que la barra sea legible en vez de
    # quedarse casi vacía todo el tiempo (el RMS de voz normal ronda 0.05-0.2).
    MIC_LEVEL_METER_MAX_RMS = 0.3

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(self.WINDOW_TITLE)
        self.resizable(False, False)
        self._center_on_screen()

        self._pipeline_controller = PipelineController()
        self._mic_level_monitor = MicLevelMonitor()

        # El servidor WebSocket del overlay vive desde que se abre esta
        # ventana (ver stop_overlay_server en _on_close_requested), no solo
        # mientras la transcripción está corriendo — así la vista previa de
        # estilos (ver _on_preview_toggled) funciona sin arrancar Whisper.
        self._pipeline_controller.start_overlay_server()

        # Avisa en el status_label mientras se cargan/descargan modelos de
        # Whisper o paquetes de Argos Translate — ver
        # src/status_hub.py y _on_download_status.
        set_status_listener(self._on_download_status)

        # Fuentes instaladas en el sistema operativo (no una lista fija de
        # fábrica): incluye cualquier fuente que el usuario haya cargado a
        # mano en Windows, porque el Browser Source de OBS es Chromium (CEF)
        # y en Windows resuelve font-family contra las mismas fuentes que ve
        # Tk acá (GDI/DirectWrite) — lo que aparece en este selector se va a
        # poder aplicar en el overlay. Se calcula una sola vez porque las dos
        # pestañas de estilo (original/traducido) comparten el mismo listado.
        self._system_font_families = self._get_system_font_families()

        # Copia de trabajo en memoria del estilo del overlay: los controles
        # de las pestañas de estilo se construyen a partir de esto, y cada
        # cambio actualiza tanto esta copia como el JSON persistido (ver
        # _on_style_value_changed) para no tener que releer el archivo cada
        # vez que se mueve un slider o se edita un campo.
        self._overlay_style = get_overlay_style()

        # Registro de los campos numéricos de tamaño (ancho/alto de la zona
        # de texto), para poder forzar su validación antes de arrancar la
        # transcripción — ver _build_pixel_entry_row y _validate_style_inputs.
        self._pixel_size_entries = []

        # Botones de vista previa (uno por pestaña de estilo) y el id del
        # `after` pendiente de cada uno mientras está activa — ver
        # _build_preview_toggle / _on_preview_toggled. Se deshabilitan
        # mientras la transcripción real está corriendo (_request_start) y
        # se detienen todos al reconstruir las pestañas de estilo tras
        # aplicar un preset (_rebuild_style_tabs) o al cerrar la ventana.
        self._preview_toggle_buttons = []
        self._active_preview_after_ids = {}

        # El guardado en disco es manual (botón "Guardar configuración" en
        # el footer, ver save_user_config en config/user_config.py) — este
        # flag distingue "hay cambios sin guardar" para poder avisar antes
        # de cerrar la ventana y perderlos sin querer.
        self._has_unsaved_changes = False

        # Se pone en True al principio de _on_close_requested — los
        # callbacks que llegan desde el hilo de fondo del pipeline
        # (_handle_pipeline_ready, _handle_pipeline_stopped,
        # _on_download_status) lo chequean antes de tocar cualquier widget:
        # si la descarga de un modelo tarda más que el timeout de
        # PipelineController.join() en _on_close_requested, la ventana ya
        # puede estar destruida para cuando ese hilo por fin termina.
        self._is_closing = False

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close_requested)

    @staticmethod
    def _get_system_font_families() -> list[str]:
        """
        Lista ordenada de fuentes instaladas en el sistema, vía Tk (que en
        Windows lee las mismas fuentes registradas que usa GDI/DirectWrite).
        Se excluyen las fuentes "verticales" de Windows (prefijo "@", ej.
        "@MS Gothic") — son variantes pensadas para texto vertical japonés,
        no aportan nada como opción de subtítulo horizontal y solo duplican
        ruido en el selector.
        """
        families = {name for name in tkfont.families() if not name.startswith("@")}
        return sorted(families, key=str.casefold)

    def _center_on_screen(self):
        """Calcula la posición para que la ventana quede centrada en la
        pantalla, en vez de nacer en la esquina superior izquierda (default
        de Tkinter) o en la posición donde haya quedado la última ventana."""
        window_width, window_height = (int(value) for value in self.WINDOW_SIZE.split("x"))
        x = (self.winfo_screenwidth() - window_width) // 2
        y = (self.winfo_screenheight() - window_height) // 2
        self.geometry(f"{self.WINDOW_SIZE}+{x}+{y}")

    def _build_layout(self):
        self._build_header()
        self._build_content_area()
        self._build_footer()

    def _build_header(self):
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=24, pady=(18, 8))

        ctk.CTkLabel(
            header,
            text="Voice Transcriber",
            font=ctk.CTkFont(size=22, weight="bold"),
        ).pack(anchor="w")

        self.status_label = ctk.CTkLabel(
            header,
            text="● Detenido",
            font=ctk.CTkFont(size=13),
            text_color=self.IDLE_STATUS_COLOR,
        )
        self.status_label.pack(anchor="w")

        # Barra de progreso real para descargas donde se conoce el avance en
        # bytes (ver status_hub.notify_status y _apply_download_status) —
        # arranca oculta (sin .pack todavía) y solo se muestra mientras dura
        # una de esas descargas. No todas las notificaciones traen progreso
        # numérico (ej. la del modelo de Whisper no lo tiene), en esos casos
        # se muestra solo el texto del status_label, sin esta barra.
        self.download_progress_bar = ctk.CTkProgressBar(header)
        self.download_progress_bar.set(0)

    def _build_content_area(self):
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=24, pady=8)

        general_tab = self.tabview.add(self.TAB_GENERAL)
        original_style_tab = self.tabview.add(self.TAB_ORIGINAL_STYLE)
        translated_style_tab = self.tabview.add(self.TAB_TRANSLATED_STYLE)

        # Las tres pestañas usan un CTkScrollableFrame como contenido: la
        # ventana tiene tamaño fijo (resizable=False) y la cantidad de
        # controles por pestaña ya no entra siempre en el alto disponible,
        # así que cada una necesita poder scrollear por su cuenta.
        self._build_general_tab(self._build_scrollable_tab_content(general_tab))

        # Referencias guardadas para poder vaciar y repoblar estas dos
        # pestañas cuando se aplica un preset de estilo — ver
        # _rebuild_style_tabs.
        self._original_style_content = self._build_scrollable_tab_content(original_style_tab)
        self._translated_style_content = self._build_scrollable_tab_content(translated_style_tab)
        self._populate_style_tabs()

    def _populate_style_tabs(self):
        """
        Cuerpo compartido entre el build inicial (_build_content_area) y
        _rebuild_style_tabs (tras aplicar un preset) — arma el contenido de
        ambas pestañas de estilo a partir de self._overlay_style. El
        selector de presets vive una sola vez, al principio de la pestaña
        de texto original, porque aplica a las dos pestañas (y al fondo
        general) a la vez.
        """
        self._build_style_presets_section(self._original_style_content)
        self._build_overlay_link_section(self._original_style_content, self.ORIGINAL_OVERLAY_FILE_PATH)
        self._build_text_size_controls(
            self._original_style_content, prefix="original", tab_name=self.TAB_ORIGINAL_STYLE
        )

        self._build_overlay_link_section(self._translated_style_content, self.TRANSLATED_OVERLAY_FILE_PATH)
        self._build_text_size_controls(
            self._translated_style_content, prefix="translated", tab_name=self.TAB_TRANSLATED_STYLE
        )

    def _rebuild_style_tabs(self):
        """Vacía y reconstruye ambas pestañas de estilo para reflejar un
        cambio que tocó muchos valores a la vez (aplicar un preset) — los
        controles individuales (sliders, swatches) no tienen un mecanismo
        de re-sync propio, así que reconstruirlos desde cero es más simple
        y confiable que actualizar cada uno a mano."""
        self._stop_all_previews()
        self._pixel_size_entries = []
        self._preview_toggle_buttons = []

        for child in self._original_style_content.winfo_children():
            child.destroy()
        for child in self._translated_style_content.winfo_children():
            child.destroy()

        self._populate_style_tabs()

        # _build_preview_toggle siempre crea los botones habilitados — si la
        # transcripción real está corriendo (el selector de presets no se
        # deshabilita durante eso, a propósito, para poder cambiar de look
        # en vivo), hay que volver a aplicarles el mismo "disabled" que les
        # puso _request_start, o quedarían clickeables de nuevo y mandarían
        # subtítulos de muestra mezclados con la transcripción real.
        if self._pipeline_controller.is_running:
            for button in self._preview_toggle_buttons:
                button.configure(state="disabled")

    @staticmethod
    def _build_scrollable_tab_content(tab: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
        content = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        content.pack(fill="both", expand=True)
        return content

    def _build_general_tab(self, parent):
        self._build_audio_device_section(parent)
        self._build_translation_direction_section(parent)
        self._build_cuda_acceleration_section(parent)

    def _build_style_presets_section(self, parent):
        """
        Selector de presets de estilo (built-in + guardados por el usuario).
        Vive una sola vez, al principio de la pestaña de texto original,
        porque aplicar un preset pisa TODO el overlay_style (fondo general +
        texto original + texto traducido) de una — ver
        apply_overlay_style_preset en config/user_config.py.
        """
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=4, pady=(0, 14))

        ctk.CTkLabel(
            section,
            text="Preset de estilo (aplica a original, traducido y fondo)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        self.preset_menu = ctk.CTkOptionMenu(
            section, values=self._preset_menu_values(), command=self._on_preset_selected
        )
        self.preset_menu.set(self.STYLE_PRESET_PLACEHOLDER)
        self.preset_menu.pack(fill="x", pady=(6, 0))

        buttons_row = ctk.CTkFrame(section, fg_color="transparent")
        buttons_row.pack(fill="x", pady=(6, 0))

        ctk.CTkButton(
            buttons_row,
            text=self.SAVE_PRESET_BUTTON_TEXT,
            height=28,
            command=self._on_save_preset_clicked,
        ).pack(side="left")

        self.delete_preset_button = ctk.CTkButton(
            buttons_row,
            text=self.DELETE_PRESET_BUTTON_TEXT,
            height=28,
            fg_color="transparent",
            border_width=1,
            state="disabled",
            command=self._on_delete_preset_clicked,
        )
        self.delete_preset_button.pack(side="left", padx=(8, 0))

    def _preset_menu_values(self) -> list[str]:
        return [self.STYLE_PRESET_PLACEHOLDER] + list(get_overlay_style_presets().keys())

    def _on_preset_selected(self, name: str):
        if name == self.STYLE_PRESET_PLACEHOLDER:
            return

        self._overlay_style = apply_overlay_style_preset(name)
        self._has_unsaved_changes = True
        self._pipeline_controller.push_overlay_style(self._overlay_style)
        self._rebuild_style_tabs()

        # _rebuild_style_tabs reconstruye este mismo selector desde cero
        # (ver _build_style_presets_section) — hay que volver a seleccionar
        # el preset recién aplicado y reflejar si se puede borrar o no.
        self.preset_menu.set(name)
        self.delete_preset_button.configure(state="disabled" if is_built_in_style_preset(name) else "normal")

    def _on_save_preset_clicked(self):
        name = simpledialog.askstring(
            "Guardar preset", "Nombre del preset:", parent=self
        )
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if is_built_in_style_preset(name):
            messagebox.showerror(
                "Nombre inválido", f"'{name}' es un preset de fábrica y no se puede sobrescribir."
            )
            return

        save_overlay_style_preset(name)
        self._has_unsaved_changes = True
        self.preset_menu.configure(values=self._preset_menu_values())
        self.preset_menu.set(name)
        self.delete_preset_button.configure(state="normal")

    def _on_delete_preset_clicked(self):
        name = self.preset_menu.get()
        if name == self.STYLE_PRESET_PLACEHOLDER or is_built_in_style_preset(name):
            return
        if not messagebox.askyesno("Eliminar preset", f"¿Eliminar el preset '{name}'?"):
            return

        delete_overlay_style_preset(name)
        self._has_unsaved_changes = True
        self.preset_menu.configure(values=self._preset_menu_values())
        self.preset_menu.set(self.STYLE_PRESET_PLACEHOLDER)
        self.delete_preset_button.configure(state="disabled")

    def _build_overlay_link_section(self, parent, overlay_file_path: str):
        """
        Botón para copiar la ruta del overlay correspondiente (ver
        ORIGINAL_OVERLAY_FILE_PATH / TRANSLATED_OVERLAY_FILE_PATH), lista
        para pegar en el campo "Local file" del Browser Source de OBS sin
        tener que navegar la carpeta del proyecto a mano cada vez.
        """
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=4, pady=(6, 0))

        ctk.CTkLabel(
            section,
            text="Archivo para agregar como Browser Source en OBS",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        path_entry = ctk.CTkEntry(section)
        path_entry.insert(0, overlay_file_path)
        path_entry.configure(state="readonly")
        path_entry.pack(fill="x", pady=(4, 0))

        buttons_row = ctk.CTkFrame(section, fg_color="transparent")
        buttons_row.pack(fill="x", pady=(6, 0))

        copy_button = ctk.CTkButton(buttons_row, text=self.COPY_LINK_BUTTON_TEXT, width=180, height=28)
        copy_button.configure(
            command=lambda: self._on_copy_overlay_link_clicked(overlay_file_path, copy_button)
        )
        copy_button.pack(side="left")

        self._build_preview_toggle(buttons_row, overlay_file_path)

    def _build_preview_toggle(self, parent, overlay_file_path: str):
        """
        Botón que abre este overlay en el navegador por defecto y le manda
        texto de muestra (ver PipelineController.push_preview_subtitle) sin
        depender de que la transcripción esté corriendo — el servidor ya
        está arriba desde que se abrió esta ventana (ver
        start_overlay_server en __init__). Los cambios de estilo se ven al
        toque porque _on_style_value_changed ya empuja cada cambio en vivo.
        """
        button = ctk.CTkButton(
            parent,
            text=self.PREVIEW_BUTTON_START_TEXT,
            width=200,
            height=28,
            fg_color="transparent",
            border_width=1,
        )
        button.configure(command=lambda: self._on_preview_toggled(button, overlay_file_path))
        button.pack(side="left", padx=(8, 0))
        self._preview_toggle_buttons.append(button)

    def _on_preview_toggled(self, button, overlay_file_path: str):
        if button in self._active_preview_after_ids:
            self._stop_preview(button)
            return

        # Path.as_uri() arma la URI file:// bien formada (barras correctas
        # y espacios/caracteres especiales del path codificados como %20,
        # etc.); _open_url_in_browser evita que Windows la abra con
        # Internet Explorer vía la asociación de archivo .html en vez del
        # navegador real (ver su docstring).
        _open_url_in_browser(Path(overlay_file_path).as_uri())
        button.configure(text=self.PREVIEW_BUTTON_STOP_TEXT)
        self._run_preview_tick(button)

    def _run_preview_tick(self, button):
        self._pipeline_controller.push_preview_subtitle(
            self.PREVIEW_SAMPLE_ORIGINAL_TEXT, self.PREVIEW_SAMPLE_TRANSLATED_TEXT
        )
        after_id = self.after(self.PREVIEW_REFRESH_INTERVAL_MS, lambda: self._run_preview_tick(button))
        self._active_preview_after_ids[button] = after_id

    def _stop_preview(self, button):
        after_id = self._active_preview_after_ids.pop(button, None)
        if after_id is not None:
            self.after_cancel(after_id)
        # El botón puede haber sido destruido ya (ver _rebuild_style_tabs,
        # que llama a _stop_all_previews ANTES de destruir los widgets) —
        # solo tocarlo si sigue vivo.
        if button.winfo_exists():
            button.configure(text=self.PREVIEW_BUTTON_START_TEXT)

    def _stop_all_previews(self):
        for button in list(self._active_preview_after_ids.keys()):
            self._stop_preview(button)

    def _on_copy_overlay_link_clicked(self, overlay_file_path: str, button):
        self.clipboard_clear()
        self.clipboard_append(overlay_file_path)
        # Sin este update() el portapapeles puede quedar vacío si la ventana
        # pierde el foco enseguida (ej. el usuario alt-tabea a OBS al toque
        # para pegarlo) — Tkinter recién "compromete" el contenido copiado
        # cuando procesa este evento.
        self.update()

        button.configure(text=self.COPY_LINK_BUTTON_CONFIRMATION_TEXT)
        self.after(
            self.COPY_LINK_CONFIRMATION_DURATION_MS,
            lambda: button.configure(text=self.COPY_LINK_BUTTON_TEXT),
        )

    def _build_paired_row(self, parent, top_pady: int = 6, column_gap: int = 10):
        """
        Frame con dos columnas lado a lado, para agrupar dos controles
        relacionados (ej. fuente + grosor, ancho + alto) en una sola fila en
        vez de apilarlos uno debajo del otro — reduce el alto total de la
        pestaña sin sacrificar controles.
        """
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(top_pady, 0))
        left_column = ctk.CTkFrame(row, fg_color="transparent")
        left_column.pack(side="left", fill="both", expand=True, padx=(0, column_gap // 2))
        right_column = ctk.CTkFrame(row, fg_color="transparent")
        right_column.pack(side="left", fill="both", expand=True, padx=(column_gap // 2, 0))
        return left_column, right_column

    def _build_text_size_controls(self, parent, prefix: str, tab_name: str):
        """
        Tipografía (fuente, tamaño, grosor, color) y ancho/alto EXACTOS de la
        zona de texto — para poder hacerla coincidir con la sección del
        lienzo que el usuario ya tiene reservada en su escena de OBS, en
        vez de depender de un porcentaje relativo al tamaño del Browser
        Source. Fuente+grosor y ancho+alto van de a pares en una misma fila
        (ver _build_paired_row) porque son controles chicos y relacionados —
        apilarlos uno por fila era más alto de lo que necesitaban.
        """
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=4, pady=(12, 0))

        ctk.CTkLabel(
            section,
            text="Tipografía",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        font_family_column, font_weight_column = self._build_paired_row(section)
        self._build_font_family_row(font_family_column, f"{prefix}_font_family")
        self._build_font_weight_row(font_weight_column, f"{prefix}_font_weight")

        self._build_font_size_slider_row(section, f"{prefix}_font_size_px")
        self._build_text_color_controls(section, prefix)
        self._build_text_stroke_controls(section, prefix)
        self._build_animation_row(section, f"{prefix}_animation")

        width_column, height_column = self._build_paired_row(section)
        self._build_pixel_entry_row(
            width_column,
            "Ancho",
            f"{prefix}_width_px",
            self.TEXT_ZONE_WIDTH_MIN_PX,
            self.TEXT_ZONE_WIDTH_MAX_PX,
            tab_name,
        )
        self._build_pixel_entry_row(
            height_column,
            "Alto",
            f"{prefix}_height_px",
            self.TEXT_ZONE_HEIGHT_MIN_PX,
            self.TEXT_ZONE_HEIGHT_MAX_PX,
            tab_name,
        )

    def _build_font_size_slider_row(self, parent, style_key: str):
        current_value = self._overlay_style[style_key]

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))

        header_row = ctk.CTkFrame(row, fg_color="transparent")
        header_row.pack(fill="x")
        ctk.CTkLabel(
            header_row,
            text=f"Tamaño de fuente (máx. {self.FONT_SIZE_MAX_PX}px)",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        value_label = ctk.CTkLabel(
            header_row,
            text=f"{current_value}px",
            font=ctk.CTkFont(size=12),
            text_color=self.IDLE_STATUS_COLOR,
        )
        value_label.pack(side="right")

        def on_slider_moved(raw_value):
            value = int(round(raw_value))
            value_label.configure(text=f"{value}px")
            self._on_style_value_changed(style_key, value)

        slider = ctk.CTkSlider(
            row,
            from_=self.FONT_SIZE_MIN_PX,
            to=self.FONT_SIZE_MAX_PX,
            number_of_steps=self.FONT_SIZE_MAX_PX - self.FONT_SIZE_MIN_PX,
            command=on_slider_moved,
        )
        slider.set(current_value)
        slider.pack(fill="x", pady=(2, 0))

    def _build_font_family_row(self, parent, style_key: str):
        """
        Selector de fuente para ESTE overlay (original o traducido — cada
        pestaña llama a este método con su propio style_key, así que hay un
        selector independiente por pestaña). Ver _FontFamilyPicker para el
        motivo de no usar CTkComboBox acá.
        """
        current_value = self._overlay_style[style_key]
        all_fonts = self._system_font_families
        # Fallback defensivo: si el JSON tiene una fuente que ya no está
        # instalada en este equipo (ej. config copiada de otra máquina), se
        # agrega para no perderla silenciosamente al mostrar el selector.
        if current_value not in all_fonts:
            all_fonts = sorted(all_fonts + [current_value], key=str.casefold)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(
            row,
            text="Fuente",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        picker = _FontFamilyPicker(
            row,
            all_values=all_fonts,
            initial_value=current_value,
            on_value_committed=lambda value: self._on_style_value_changed(style_key, value),
        )
        picker.pack(fill="x", pady=(2, 0))

    def _build_text_color_controls(self, parent, prefix: str):
        """
        Color de fuente + opacidad. El picker es el diálogo de color nativo
        del sistema operativo (paleta + HEX + RGB personalizado) — esta app
        es de escritorio (customtkinter/Tkinter), no una página web, así que
        no existe un <input type="color"> de HTML para reusar aquí; el
        diálogo nativo de Windows es su equivalente más directo. Ese diálogo
        no soporta canal alfa (tampoco lo soporta el <input type="color">
        del navegador, la spec HTML5 no lo permite), por eso la opacidad va
        aparte, mismo patrón que ya usan background_color + background_opacity.
        """
        color_style_key = f"{prefix}_text_color"
        opacity_style_key = f"{prefix}_text_opacity"

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(
            row,
            text="Color de fuente",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        picker_row = ctk.CTkFrame(row, fg_color="transparent")
        picker_row.pack(fill="x", pady=(2, 0))

        swatch = ctk.CTkLabel(
            picker_row,
            text="",
            width=28,
            height=22,
            corner_radius=4,
            fg_color=self._overlay_style[color_style_key],
        )
        swatch.pack(side="left")

        hex_label = ctk.CTkLabel(
            picker_row,
            text=self._overlay_style[color_style_key],
            font=ctk.CTkFont(size=12),
        )
        hex_label.pack(side="left", padx=(8, 0))

        def on_pick_color():
            # askcolor devuelve ((r, g, b), "#rrggbb") o (None, None) si el
            # usuario cancela el diálogo.
            _, picked_hex = colorchooser.askcolor(
                color=self._overlay_style[color_style_key],
                title="Elegir color de fuente",
            )
            if picked_hex is None:
                return
            swatch.configure(fg_color=picked_hex)
            hex_label.configure(text=picked_hex)
            self._on_style_value_changed(color_style_key, picked_hex)

        ctk.CTkButton(
            picker_row,
            text="🎨 Elegir color",
            width=120,
            height=26,
            command=on_pick_color,
        ).pack(side="right")

        self._build_opacity_slider_row(row, opacity_style_key)

    def _build_text_stroke_controls(self, parent, prefix: str):
        """
        Contorno de texto (-webkit-text-stroke): mismo patrón de color que
        _build_text_color_controls, más un slider de ancho en vez de
        opacidad. Ancho 0 (default) = sin contorno, así que por defecto el
        overlay se ve exactamente igual que antes de que existiera esta
        opción.
        """
        color_style_key = f"{prefix}_text_stroke_color"
        width_style_key = f"{prefix}_text_stroke_width_px"

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(
            row,
            text="Contorno de texto",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        picker_row = ctk.CTkFrame(row, fg_color="transparent")
        picker_row.pack(fill="x", pady=(2, 0))

        swatch = ctk.CTkLabel(
            picker_row,
            text="",
            width=28,
            height=22,
            corner_radius=4,
            fg_color=self._overlay_style[color_style_key],
        )
        swatch.pack(side="left")

        hex_label = ctk.CTkLabel(
            picker_row,
            text=self._overlay_style[color_style_key],
            font=ctk.CTkFont(size=12),
        )
        hex_label.pack(side="left", padx=(8, 0))

        def on_pick_color():
            _, picked_hex = colorchooser.askcolor(
                color=self._overlay_style[color_style_key],
                title="Elegir color de contorno",
            )
            if picked_hex is None:
                return
            swatch.configure(fg_color=picked_hex)
            hex_label.configure(text=picked_hex)
            self._on_style_value_changed(color_style_key, picked_hex)

        ctk.CTkButton(
            picker_row,
            text="🎨 Elegir color",
            width=120,
            height=26,
            command=on_pick_color,
        ).pack(side="right")

        current_width = self._overlay_style[width_style_key]

        width_header_row = ctk.CTkFrame(row, fg_color="transparent")
        width_header_row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(
            width_header_row,
            text=f"Ancho (0 = sin contorno, máx. {self.TEXT_STROKE_WIDTH_MAX_PX}px)",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        width_value_label = ctk.CTkLabel(
            width_header_row,
            text=f"{current_width}px",
            font=ctk.CTkFont(size=12),
            text_color=self.IDLE_STATUS_COLOR,
        )
        width_value_label.pack(side="right")

        def on_width_slider_moved(raw_value):
            value = int(round(raw_value))
            width_value_label.configure(text=f"{value}px")
            self._on_style_value_changed(width_style_key, value)

        width_slider = ctk.CTkSlider(
            row,
            from_=self.TEXT_STROKE_WIDTH_MIN_PX,
            to=self.TEXT_STROKE_WIDTH_MAX_PX,
            number_of_steps=self.TEXT_STROKE_WIDTH_MAX_PX - self.TEXT_STROKE_WIDTH_MIN_PX,
            command=on_width_slider_moved,
        )
        width_slider.set(current_width)
        width_slider.pack(fill="x", pady=(2, 0))

    def _build_opacity_slider_row(self, parent, style_key: str):
        current_percent = round(self._overlay_style[style_key] * 100)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))

        header_row = ctk.CTkFrame(row, fg_color="transparent")
        header_row.pack(fill="x")
        ctk.CTkLabel(
            header_row,
            text="Opacidad",
            font=ctk.CTkFont(size=12),
        ).pack(side="left")
        value_label = ctk.CTkLabel(
            header_row,
            text=f"{current_percent}%",
            font=ctk.CTkFont(size=12),
            text_color=self.IDLE_STATUS_COLOR,
        )
        value_label.pack(side="right")

        def on_slider_moved(raw_value):
            percent = int(round(raw_value))
            value_label.configure(text=f"{percent}%")
            self._on_style_value_changed(style_key, round(percent / 100, 2))

        slider = ctk.CTkSlider(
            row,
            from_=self.TEXT_OPACITY_MIN_PERCENT,
            to=self.TEXT_OPACITY_MAX_PERCENT,
            number_of_steps=self.TEXT_OPACITY_MAX_PERCENT - self.TEXT_OPACITY_MIN_PERCENT,
            command=on_slider_moved,
        )
        slider.set(current_percent)
        slider.pack(fill="x", pady=(2, 0))

    def _build_font_weight_row(self, parent, style_key: str):
        current_value = self._overlay_style[style_key]
        label_by_weight = {weight: label for label, weight in self.FONT_WEIGHT_OPTIONS.items()}
        # Fallback defensivo: los únicos valores posibles hoy son los del
        # propio dict (nadie más escribe esta clave), pero si en algún
        # momento queda un valor viejo/manual que no matchea ninguna
        # etiqueta, mejor mostrar la primera opción que romper el selector.
        current_label = label_by_weight.get(current_value, next(iter(self.FONT_WEIGHT_OPTIONS)))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(
            row,
            text="Grosor",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        menu = ctk.CTkOptionMenu(
            row,
            values=list(self.FONT_WEIGHT_OPTIONS.keys()),
            command=lambda selected_label: self._on_style_value_changed(
                style_key, self.FONT_WEIGHT_OPTIONS[selected_label]
            ),
        )
        menu.set(current_label)
        menu.pack(fill="x", pady=(2, 0))

    def _build_animation_row(self, parent, style_key: str):
        """Mismo patrón que _build_font_weight_row — selector de efecto de
        aparición (ver SUBTITLE_ANIMATION_OPTIONS)."""
        current_value = self._overlay_style[style_key]
        label_by_animation = {value: label for label, value in self.SUBTITLE_ANIMATION_OPTIONS.items()}
        current_label = label_by_animation.get(current_value, next(iter(self.SUBTITLE_ANIMATION_OPTIONS)))

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(
            row,
            text="Efecto de aparición (solo en texto final)",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        menu = ctk.CTkOptionMenu(
            row,
            values=list(self.SUBTITLE_ANIMATION_OPTIONS.keys()),
            command=lambda selected_label: self._on_style_value_changed(
                style_key, self.SUBTITLE_ANIMATION_OPTIONS[selected_label]
            ),
        )
        menu.set(current_label)
        menu.pack(fill="x", pady=(2, 0))

    def _build_pixel_entry_row(
        self, parent, label_text: str, style_key: str, min_value: int, max_value: int, tab_name: str
    ):
        current_value = self._overlay_style[style_key]

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(
            row,
            text=f"{label_text} ({min_value}-{max_value}px)",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        entry = ctk.CTkEntry(row)
        entry.insert(0, str(current_value))
        entry.pack(fill="x", pady=(4, 0))

        error_label = ctk.CTkLabel(
            row,
            text="",
            font=ctk.CTkFont(size=10),
            text_color=self.ERROR_STATUS_COLOR,
            anchor="w",
        )
        error_label.pack(anchor="w")

        def commit_value(_event=None) -> bool:
            """Retorna True si el valor del campo es válido (y ya quedó
            guardado); False si no, dejando el mensaje de error visible."""
            raw_text = entry.get().strip()
            try:
                value = int(raw_text)
            except ValueError:
                error_label.configure(text="Ingresa un número entero.")
                return False
            if not (min_value <= value <= max_value):
                error_label.configure(text=f"Debe estar entre {min_value} y {max_value}.")
                return False
            error_label.configure(text="")
            self._on_style_value_changed(style_key, value)
            return True

        # <Return> aplica el valor sin perder el foco; <FocusOut> lo aplica
        # también si el usuario simplemente hace clic en otro control sin
        # apretar Enter. Ninguno de los dos alcanza para bloquear el inicio
        # de la transcripción por sí solo: un CTkButton no siempre le saca
        # el foco al Entry con un solo clic, así que un valor inválido
        # puede quedar sin commitear y sin disparar <FocusOut> — por eso
        # _validate_style_inputs() vuelve a forzar el commit de todos estos
        # campos antes de arrancar (ver _request_start).
        entry.bind("<Return>", commit_value)
        entry.bind("<FocusOut>", commit_value)

        self._pixel_size_entries.append((tab_name, entry, commit_value))

    def _on_style_value_changed(self, style_key: str, value):
        self._overlay_style[style_key] = value
        set_overlay_style_value(style_key, value)
        self._has_unsaved_changes = True
        # Si la transcripción ya está corriendo y OBS conectado, el cambio
        # se ve al toque; si no, push_overlay_style() no hace nada y el
        # próximo arranque igual toma el valor recién elegido (persistido o
        # no — set_overlay_style_value ya actualizó la memoria).
        self._pipeline_controller.push_overlay_style(self._overlay_style)

    def _validate_style_inputs(self) -> bool:
        """
        Fuerza el commit de todos los campos de tamaño de la zona de texto
        antes de arrancar la transcripción. Si alguno queda inválido, cambia
        a esa pestaña, le pone el foco, y cancela el arranque — de otra
        forma sería posible iniciar con un valor a medio escribir que nunca
        llegó a guardarse (ver el comentario en _build_pixel_entry_row).
        """
        first_invalid_entry = None
        first_invalid_tab_name = None

        for tab_name, entry, commit_value in self._pixel_size_entries:
            if not commit_value() and first_invalid_entry is None:
                first_invalid_entry = entry
                first_invalid_tab_name = tab_name

        if first_invalid_entry is None:
            return True

        self.tabview.set(first_invalid_tab_name)
        first_invalid_entry.focus_set()
        self.status_label.configure(
            text="● Hay un valor de tamaño inválido en los estilos del overlay",
            text_color=self.ERROR_STATUS_COLOR,
        )
        return False

    def _build_audio_device_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="Dispositivo de entrada (micrófono)",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        device_names = list_available_input_device_names()
        saved_device_name = get_audio_input_device_name()
        unavailable_hint = ""

        if saved_device_name in device_names:
            initial_value = saved_device_name
        else:
            # Lo guardado no está disponible ahora mismo (desconectado, o
            # nunca se conectó en esta máquina): se muestra el default del
            # sistema como selección, SIN pisar lo guardado — si el usuario
            # no toca el selector, el próximo arranque vuelve a intentar con
            # el dispositivo original (ver device_resolver con fallback).
            initial_value = get_default_input_device_name()
            if initial_value and initial_value not in device_names:
                device_names.insert(0, initial_value)
            unavailable_hint = f"'{saved_device_name}' no está disponible ahora"
            unavailable_hint += (
                f" — usando '{initial_value}'." if initial_value else " y no hay default del sistema."
            )

        self.audio_device_menu = ctk.CTkOptionMenu(
            section,
            values=device_names or ["(sin dispositivos detectados)"],
            command=self._on_audio_device_selected,
        )
        self.audio_device_menu.set(initial_value or "(sin dispositivos detectados)")
        self.audio_device_menu.pack(fill="x", pady=(6, 0))

        self.audio_device_hint_label = ctk.CTkLabel(
            section,
            text=unavailable_hint,
            font=ctk.CTkFont(size=11),
            text_color=self.ERROR_STATUS_COLOR,
            anchor="w",
            justify="left",
        )
        self.audio_device_hint_label.pack(anchor="w", pady=(4, 0))

        self._build_mic_test_controls(section)

    def _on_audio_device_selected(self, selected_device_name: str):
        set_audio_input_device_name(selected_device_name)
        self._has_unsaved_changes = True
        self.audio_device_hint_label.configure(text="")
        # Si la prueba está corriendo, hay que reabrirla sobre el dispositivo
        # recién elegido — si no, seguiría escuchando el anterior.
        if self._mic_level_monitor.is_active:
            self._stop_mic_test()
            self._start_mic_test()

    def _build_mic_test_controls(self, parent):
        """
        Visualizador de testeo: deja confirmar que el micrófono elegido
        realmente está entregando audio (nivel en vivo + mensaje), sin
        necesidad de arrancar la transcripción completa para comprobarlo.
        """
        test_row = ctk.CTkFrame(parent, fg_color="transparent")
        test_row.pack(fill="x", pady=(6, 0))

        self.mic_test_button = ctk.CTkButton(
            test_row,
            text=self.MIC_TEST_START_TEXT,
            width=160,
            height=28,
            command=self._on_mic_test_toggled,
        )
        self.mic_test_button.pack(side="left")

        self.mic_level_status_label = ctk.CTkLabel(
            test_row,
            text="",
            font=ctk.CTkFont(size=11),
            text_color=self.IDLE_STATUS_COLOR,
            anchor="w",
            justify="left",
        )
        self.mic_level_status_label.pack(side="left", padx=(10, 0))

        self.mic_level_bar = ctk.CTkProgressBar(parent)
        self.mic_level_bar.set(0)
        self.mic_level_bar.pack(fill="x", pady=(6, 0))

    def _on_mic_test_toggled(self):
        if self._mic_level_monitor.is_active:
            self._stop_mic_test()
        else:
            self._start_mic_test()

    def _start_mic_test(self):
        try:
            self._mic_level_monitor.start()
        except Exception as error:
            self.mic_level_status_label.configure(
                text=f"No se pudo abrir el micrófono: {error}",
                text_color=self.ERROR_STATUS_COLOR,
            )
            return

        self.mic_test_button.configure(text=self.MIC_TEST_STOP_TEXT, fg_color=self.RUNNING_BUTTON_COLOR)
        self.mic_level_status_label.configure(
            text="Escuchando… habla frente al micrófono", text_color=self.IDLE_STATUS_COLOR
        )
        self._poll_mic_level()

    def _stop_mic_test(self):
        self._mic_level_monitor.stop()
        self.mic_test_button.configure(text=self.MIC_TEST_START_TEXT, fg_color=self.IDLE_BUTTON_COLOR)
        self.mic_level_status_label.configure(text="", text_color=self.IDLE_STATUS_COLOR)
        self.mic_level_bar.set(0)

    def _poll_mic_level(self):
        # Se autodetiene solo: si stop() ya se llamó (botón, cambio de
        # dispositivo, cierre de ventana, inicio del pipeline), no hay que
        # reprogramar el próximo tick.
        if not self._mic_level_monitor.is_active:
            return

        level = self._mic_level_monitor.get_level()
        self.mic_level_bar.set(min(1.0, level / self.MIC_LEVEL_METER_MAX_RMS))

        # Mismo umbral que usa el pipeline real para descartar utterances
        # casi silenciosas (ver MIN_UTTERANCE_RMS_ENERGY en config/settings.py):
        # si aquí se ve "detectando audio", la transcripción real también lo
        # va a captar como voz.
        if level >= MIN_UTTERANCE_RMS_ENERGY:
            self.mic_level_status_label.configure(
                text="✓ Detectando audio", text_color=self.RUNNING_STATUS_COLOR
            )
        else:
            self.mic_level_status_label.configure(
                text="Escuchando… habla frente al micrófono", text_color=self.IDLE_STATUS_COLOR
            )

        self.after(self.MIC_LEVEL_POLL_INTERVAL_MS, self._poll_mic_level)

    def _build_translation_direction_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="Flujo de transcripción / traducción",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        # Mapeo explícito etiqueta legible <-> código interno, en vez de usar
        # el código como texto del selector: así el nombre de la opción no
        # depende de cómo se llame la constante en config/user_config.py.
        self._translation_direction_by_label = {
            "Español → Inglés": TRANSLATION_DIRECTION_ES_TO_EN,
            "Inglés → Español": TRANSLATION_DIRECTION_EN_TO_ES,
        }
        label_by_direction = {
            direction: label for label, direction in self._translation_direction_by_label.items()
        }
        initial_label = label_by_direction[get_translation_direction()]

        self.translation_direction_menu = ctk.CTkOptionMenu(
            section,
            values=list(self._translation_direction_by_label.keys()),
            command=self._on_translation_direction_selected,
        )
        self.translation_direction_menu.set(initial_label)
        self.translation_direction_menu.pack(fill="x", pady=(6, 0))

        ctk.CTkLabel(
            section,
            text=(
                "Idioma que se habla frente al micrófono y a qué idioma se "
                "traduce el subtítulo en el overlay."
            ),
            font=ctk.CTkFont(size=11),
            text_color=self.IDLE_STATUS_COLOR,
            anchor="w",
            justify="left",
            wraplength=480,
        ).pack(anchor="w", pady=(4, 0))

    def _on_translation_direction_selected(self, selected_label: str):
        set_translation_direction(self._translation_direction_by_label[selected_label])
        self._has_unsaved_changes = True

    def _build_cuda_acceleration_section(self, parent):
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            section,
            text="Aceleración por GPU",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).pack(anchor="w")

        # Se detecta una sola vez al abrir la ventana: si no hay GPU NVIDIA,
        # el checkbox queda inhabilitado y forzado a "no marcado" — no tiene
        # sentido ofrecer una opción que va a fallar al arrancar el pipeline
        # (ver config/user_config.get_whisper_device, que además fuerza CPU
        # en tiempo real por las dudas de que esto quede desactualizado).
        gpu_available = is_nvidia_gpu_available()

        self.cuda_checkbox_var = ctk.BooleanVar(
            value=get_cuda_acceleration_enabled() if gpu_available else False
        )
        self.cuda_checkbox = ctk.CTkCheckBox(
            section,
            text="Usar aceleración CUDA",
            variable=self.cuda_checkbox_var,
            command=self._on_cuda_acceleration_toggled,
            state="normal" if gpu_available else "disabled",
        )
        self.cuda_checkbox.pack(anchor="w", pady=(6, 0))

        if gpu_available:
            hint_text = (
                "Recomendado: la transcripción corre en la GPU y es mucho más "
                "rápida. Desactívala solo si tienes problemas con los drivers de CUDA."
            )
            hint_color = self.IDLE_STATUS_COLOR
        else:
            hint_text = f"{gpu_unavailability_reason()} La transcripción va a correr en CPU (más lenta)."
            hint_color = self.ERROR_STATUS_COLOR

        self.cuda_hint_label = ctk.CTkLabel(
            section,
            text=hint_text,
            font=ctk.CTkFont(size=11),
            text_color=hint_color,
            anchor="w",
            justify="left",
            wraplength=480,
        )
        self.cuda_hint_label.pack(anchor="w", pady=(4, 0))

    def _on_cuda_acceleration_toggled(self):
        set_cuda_acceleration_enabled(self.cuda_checkbox_var.get())
        self._has_unsaved_changes = True

    def _build_footer(self):
        footer = ctk.CTkFrame(self, fg_color="transparent")
        footer.pack(fill="x", padx=24, pady=(10, 20))

        self.save_button = ctk.CTkButton(
            footer,
            text=self.SAVE_BUTTON_TEXT,
            height=32,
            fg_color="transparent",
            border_width=1,
            command=self._on_save_clicked,
        )
        self.save_button.pack(fill="x", pady=(0, 8))

        self.start_stop_button = ctk.CTkButton(
            footer,
            text=self.IDLE_BUTTON_TEXT,
            height=40,
            fg_color=self.IDLE_BUTTON_COLOR,
            command=self._on_start_stop_clicked,
        )
        self.start_stop_button.pack(fill="x")

    def _on_save_clicked(self):
        # Los campos de tamaño pueden tener un valor tipeado que todavía no
        # se commiteó (ver _build_pixel_entry_row) — se fuerza igual que
        # antes de arrancar la transcripción, para no guardar a medio
        # escribir ni perder silenciosamente lo que se ve en pantalla.
        if not self._validate_style_inputs():
            return

        save_user_config()
        self._has_unsaved_changes = False

        original_text = self.save_button.cget("text")
        self.save_button.configure(text=self.SAVE_BUTTON_CONFIRMATION_TEXT)
        self.after(
            self.SAVE_CONFIRMATION_DURATION_MS,
            lambda: self.save_button.configure(text=original_text),
        )

    def _on_start_stop_clicked(self):
        if self._pipeline_controller.is_running:
            self._request_stop()
        else:
            self._request_start()

    def _request_start(self):
        if not self._validate_style_inputs():
            return

        # El stream de prueba y el del pipeline no pueden compartir el mismo
        # dispositivo de forma confiable (algunos drivers solo permiten un
        # cliente en modo exclusivo) — se corta la prueba antes de arrancar.
        if self._mic_level_monitor.is_active:
            self._stop_mic_test()
        self.mic_test_button.configure(state="disabled")

        # La vista previa manda subtítulos de muestra al mismo overlay que
        # va a usar la transcripción real — se corta para no mezclar ambos.
        self._stop_all_previews()
        for button in self._preview_toggle_buttons:
            button.configure(state="disabled")

        self.start_stop_button.configure(text=self.STARTING_BUTTON_TEXT, state="disabled")
        self.status_label.configure(
            text="● Cargando modelos… (puede descargar la primera vez)",
            text_color=self.IDLE_STATUS_COLOR,
        )
        self._pipeline_controller.start(
            on_stopped=self._handle_pipeline_stopped, on_ready=self._handle_pipeline_ready
        )

    def _handle_pipeline_ready(self):
        """
        Se llama desde el hilo de fondo del pipeline (ver
        TranscriptionPipeline.run vía on_ready) justo cuando el micrófono ya
        está capturando de verdad — los modelos ya terminaron de cargarse
        (y de descargarse, si hacía falta). Antes esto se adivinaba con un
        timer fijo (ASSUME_RUNNING_AFTER_MS) que quedaba corto si el primer
        arranque tenía que descargar un modelo grande de Hugging Face.
        """
        if self._is_closing:
            return
        self.after(0, self._mark_running)

    def _mark_running(self):
        # Si ya se detuvo (ej. falló al resolver el micrófono), el callback
        # on_stopped ya se encargó de resetear el botón — no pisarlo aquí.
        if not self._pipeline_controller.is_running:
            return
        self.start_stop_button.configure(
            text=self.RUNNING_BUTTON_TEXT,
            fg_color=self.RUNNING_BUTTON_COLOR,
            state="normal",
        )
        self.status_label.configure(text="● Transcribiendo", text_color=self.RUNNING_STATUS_COLOR)

    def _on_download_status(self, message: str, progress: float | None = None):
        """
        Listener registrado en status_hub (ver __init__) — lo puede llamar
        el hilo de fondo del pipeline (carga de modelos de Whisper,
        instalación de paquetes de Argos Translate, descarga del runtime de
        CUDA), nunca el hilo de la GUI, así que hay que reencolar con
        after(0, ...) antes de tocar cualquier widget de Tkinter.
        """
        if self._is_closing:
            return
        self.after(0, lambda: self._apply_download_status(message, progress))

    def _apply_download_status(self, message: str, progress: float | None = None):
        if message:
            self.status_label.configure(text=f"● {message}", text_color=self.IDLE_STATUS_COLOR)
            if progress is not None:
                self.download_progress_bar.set(progress)
                self.download_progress_bar.pack(fill="x", pady=(4, 0))
            else:
                self.download_progress_bar.pack_forget()
            return

        self.download_progress_bar.pack_forget()

        # "" = ya terminó lo que se estaba avisando (ver
        # status_hub.notify_status) — se restaura "Transcribiendo" en vez de
        # dejar el mensaje transitorio pisado para siempre. La carga de
        # modelos de Whisper no manda este "ya terminó" (ver
        # SpeechTranscriber._notify_model_loading): ahí quien restaura el
        # estado correcto es _handle_pipeline_ready. Esto es para el caso de
        # una descarga que puede pasar DESPUÉS, con la transcripción ya
        # corriendo (ej. instalar un paquete de traducción nuevo).
        #
        # Se chequea el TEXTO del botón, no is_running: el hilo de fondo
        # sigue vivo (is_running=True) durante toda la instalación del
        # paquete, incluso después de que el usuario ya apretó "Detener" —
        # si acá se restaurara "Transcribiendo" con solo mirar is_running,
        # se pisaría el "Deteniendo…" que _request_stop ya puso, justo antes
        # de que _reset_to_idle termine de poner "Detenido" un instante
        # después. Solo hay que restaurar si de verdad seguimos en el
        # estado "Transcribiendo" (nadie pidió detener ni cerrar mientras
        # tanto).
        if self.start_stop_button.cget("text") == self.RUNNING_BUTTON_TEXT:
            self.status_label.configure(text="● Transcribiendo", text_color=self.RUNNING_STATUS_COLOR)

    def _request_stop(self):
        self.start_stop_button.configure(text=self.STOPPING_BUTTON_TEXT, state="disabled")
        self.status_label.configure(text="● Deteniendo…", text_color=self.IDLE_STATUS_COLOR)
        self._pipeline_controller.stop()

    def _handle_pipeline_stopped(self, error: Exception | None):
        # Se llama desde el hilo de fondo del pipeline: hay que reencolar al
        # hilo de la GUI antes de tocar cualquier widget de Tkinter.
        if self._is_closing:
            return
        self.after(0, lambda: self._reset_to_idle(error))

    def _reset_to_idle(self, error: Exception | None):
        self.start_stop_button.configure(
            text=self.IDLE_BUTTON_TEXT,
            fg_color=self.IDLE_BUTTON_COLOR,
            state="normal",
        )
        self.mic_test_button.configure(state="normal")
        for button in self._preview_toggle_buttons:
            button.configure(state="normal")
        if error is not None:
            self.status_label.configure(text=f"● Error: {error}", text_color=self.ERROR_STATUS_COLOR)
        else:
            self.status_label.configure(text="● Detenido", text_color=self.IDLE_STATUS_COLOR)

    def _on_close_requested(self):
        if self._has_unsaved_changes and not messagebox.askyesno(
            "Cambios sin guardar",
            "Hay cambios de configuración sin guardar. ¿Salir de todas formas sin guardarlos?",
            icon="warning",
        ):
            return

        self._is_closing = True
        self._stop_all_previews()
        set_status_listener(None)
        if self._mic_level_monitor.is_active:
            self._mic_level_monitor.stop()
        if self._pipeline_controller.is_running:
            self._pipeline_controller.stop()
            # Espera a que el hilo de la transcripción termine de verdad
            # ANTES de apagar el servidor del overlay — si no, el pipeline
            # podría estar a mitad de programar un broadcast en un event
            # loop que se está por cerrar (ver PipelineController.join).
            self._pipeline_controller.join(timeout=5)
        self._pipeline_controller.stop_overlay_server()
        self.destroy()
