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
from tkinter import colorchooser, messagebox

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
    save_user_config,
)
from src.audio.device_resolver import (
    get_default_input_device_name,
    list_available_input_device_names,
)
from src.audio.mic_level_monitor import MicLevelMonitor
from src.server.pipeline_controller import PipelineController
from src.startup.cuda_availability import is_nvidia_gpu_available

# Ruta absoluta a la raíz del proyecto (voice-transcriber/), tres niveles
# arriba de este archivo (src/gui/config_window.py -> src/gui -> src -> raíz).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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

    # Fuentes preinstaladas de fábrica en Windows 10/11 (no dependen de que
    # el usuario tenga nada extra instalado, algo importante pensando en el
    # .exe distribuible): cubren sans-serif, serif, monoespaciada y un par
    # de opciones "de impacto" típicas de subtítulos/streaming.
    FONT_FAMILY_OPTIONS = [
        "Segoe UI",
        "Arial",
        "Calibri",
        "Verdana",
        "Tahoma",
        "Trebuchet MS",
        "Georgia",
        "Times New Roman",
        "Consolas",
        "Comic Sans MS",
        "Impact",
    ]

    # Rango del slider de opacidad del texto, en porcentaje (0-100) — se
    # guarda como fracción 0.0-1.0 (ver original_text_opacity/
    # translated_text_opacity en OVERLAY_STYLE_DEFAULTS), igual convención
    # que ya usa background_opacity.
    TEXT_OPACITY_MIN_PERCENT = 0
    TEXT_OPACITY_MAX_PERCENT = 100

    IDLE_BUTTON_TEXT = "Iniciar transcripción"
    STARTING_BUTTON_TEXT = "Iniciando..."
    RUNNING_BUTTON_TEXT = "Detener transcripción"
    STOPPING_BUTTON_TEXT = "Deteniendo..."

    IDLE_BUTTON_COLOR = "#1f6aa5"
    RUNNING_BUTTON_COLOR = "#a53f2a"

    IDLE_STATUS_COLOR = "gray60"
    RUNNING_STATUS_COLOR = "#3fa54a"
    ERROR_STATUS_COLOR = "#e05252"

    # Tiempo que se le da al pipeline para arrancar antes de asumir que ya
    # está corriendo (no hay una señal explícita de "modelos cargados";
    # los logs de consola siguen siendo la fuente de verdad detallada).
    ASSUME_RUNNING_AFTER_MS = 1500

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

        # El guardado en disco es manual (botón "Guardar configuración" en
        # el footer, ver save_user_config en config/user_config.py) — este
        # flag distingue "hay cambios sin guardar" para poder avisar antes
        # de cerrar la ventana y perderlos sin querer.
        self._has_unsaved_changes = False

        self._build_layout()
        self.protocol("WM_DELETE_WINDOW", self._on_close_requested)

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

        # El resto de los controles de estilo (color, etc.) se agrega en una
        # etapa siguiente — por ahora cada pestaña tiene el link al overlay
        # que le corresponde y los controles de tipografía/tamaño.
        original_style_content = self._build_scrollable_tab_content(original_style_tab)
        self._build_overlay_link_section(original_style_content, self.ORIGINAL_OVERLAY_FILE_PATH)
        self._build_text_size_controls(
            original_style_content, prefix="original", tab_name=self.TAB_ORIGINAL_STYLE
        )

        translated_style_content = self._build_scrollable_tab_content(translated_style_tab)
        self._build_overlay_link_section(translated_style_content, self.TRANSLATED_OVERLAY_FILE_PATH)
        self._build_text_size_controls(
            translated_style_content, prefix="translated", tab_name=self.TAB_TRANSLATED_STYLE
        )

    @staticmethod
    def _build_scrollable_tab_content(tab: ctk.CTkFrame) -> ctk.CTkScrollableFrame:
        content = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        content.pack(fill="both", expand=True)
        return content

    def _build_general_tab(self, parent):
        self._build_audio_device_section(parent)
        self._build_translation_direction_section(parent)
        self._build_cuda_acceleration_section(parent)

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

        copy_button = ctk.CTkButton(section, text=self.COPY_LINK_BUTTON_TEXT, width=180, height=28)
        copy_button.configure(
            command=lambda: self._on_copy_overlay_link_clicked(overlay_file_path, copy_button)
        )
        copy_button.pack(anchor="w", pady=(6, 0))

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
        current_value = self._overlay_style[style_key]
        # Fallback defensivo: si el JSON tiene una fuente que no está en la
        # lista curada (ej. editada a mano), se agrega al final en vez de
        # perderla silenciosamente al mostrar el selector.
        options = list(self.FONT_FAMILY_OPTIONS)
        if current_value not in options:
            options.append(current_value)

        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x")
        ctk.CTkLabel(
            row,
            text="Fuente",
            font=ctk.CTkFont(size=12),
        ).pack(anchor="w")

        menu = ctk.CTkOptionMenu(
            row,
            values=options,
            command=lambda selected: self._on_style_value_changed(style_key, selected),
        )
        menu.set(current_value)
        menu.pack(fill="x", pady=(2, 0))

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
            hint_text = (
                "No se detectó una GPU NVIDIA compatible con CUDA en este equipo "
                "— esta opción no está disponible y la transcripción va a correr "
                "en CPU (más lenta)."
            )
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

        self.start_stop_button.configure(text=self.STARTING_BUTTON_TEXT, state="disabled")
        self.status_label.configure(text="● Iniciando…", text_color=self.IDLE_STATUS_COLOR)
        self._pipeline_controller.start(on_stopped=self._handle_pipeline_stopped)
        self.after(self.ASSUME_RUNNING_AFTER_MS, self._mark_running_if_still_alive)

    def _mark_running_if_still_alive(self):
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

    def _request_stop(self):
        self.start_stop_button.configure(text=self.STOPPING_BUTTON_TEXT, state="disabled")
        self.status_label.configure(text="● Deteniendo…", text_color=self.IDLE_STATUS_COLOR)
        self._pipeline_controller.stop()

    def _handle_pipeline_stopped(self, error: Exception | None):
        # Se llama desde el hilo de fondo del pipeline: hay que reencolar al
        # hilo de la GUI antes de tocar cualquier widget de Tkinter.
        self.after(0, lambda: self._reset_to_idle(error))

    def _reset_to_idle(self, error: Exception | None):
        self.start_stop_button.configure(
            text=self.IDLE_BUTTON_TEXT,
            fg_color=self.IDLE_BUTTON_COLOR,
            state="normal",
        )
        self.mic_test_button.configure(state="normal")
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

        if self._mic_level_monitor.is_active:
            self._mic_level_monitor.stop()
        if self._pipeline_controller.is_running:
            self._pipeline_controller.stop()
        self.destroy()
