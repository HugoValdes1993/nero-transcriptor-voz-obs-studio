"""
Configuración centralizada del pipeline de transcripción + traducción para OBS.
Ajustar estos valores según el hardware disponible y el caso de uso.

Autor: Nero
"""

# --- Audio ---
AUDIO_SAMPLE_RATE_HZ = 16000          # Requerido tanto por Silero VAD como por Whisper
AUDIO_FRAME_SAMPLES = 512             # Tamaño de frame esperado por Silero VAD a 16kHz

# Selección de dispositivo de entrada POR NOMBRE (no por índice numérico).
# Un mismo micrófono físico aparece varias veces en la lista de dispositivos,
# una entrada por cada host API de Windows (MME, DirectSound, WASAPI, WDM-KS).
# Buscar por nombre + audio_device_resolver.resolve_input_device_candidates()
# prioriza normalmente la entrada WASAPI entre esos duplicados, excluyendo
# siempre la entrada WDM-KS (que bloquea el dispositivo en modo exclusivo
# para otras apps, como OBS, sin importar la config de exclusividad que se
# le pida — nunca es una opción elegible, ver device_resolver.py). Alcanza
# con que este substring coincida con parte del nombre reportado por
# list_audio_devices.py; no hace falta el nombre exacto.
#
# Este valor es solo el default de fábrica. Una vez que el usuario elige un
# dispositivo desde el selector de la GUI, ese valor se guarda en
# user_config.json (ver config/user_config.py) y pasa a tener prioridad —
# stream_capture.py lee siempre desde ahí, no desde esta constante. Si el
# dispositivo elegido deja de estar disponible (se desconectó), se cae
# automáticamente al dispositivo de entrada por defecto del sistema
# operativo (ver device_resolver.resolve_input_device_candidates_or_default).
AUDIO_INPUT_DEVICE_NAME = "DualSense Wireless Controller"

# Overrides manuales de host API, por dispositivo. Normalmente no hace falta
# tocar esto: el resolver ya prioriza WASAPI, que da la mejor calidad de
# audio disponible (a diferencia de DirectSound, que pasa por el mezclador
# de Windows y puede degradar la precisión de Whisper). Un dispositivo entra
# en este mapa SOLO cuando su driver puntualmente falla con WASAPI — por
# ejemplo, algunos drivers de audio "combinados" (como el del DualSense) no
# exponen cierto nodo de control que WASAPI intenta consultar al iniciar el
# stream, lo que causa 'PaErrorCode -9999: WdmSyncIoctl ... ERROR_NOT_FOUND'.
# En ese caso, forzar DirectSound suele evitar el problema por completo.
#
# Como el dispositivo de entrada ahora se elige libremente desde la ventana
# de configuración (ver config/user_config.get_audio_input_device_name), este
# override tiene que quedar atado al dispositivo puntual que lo necesita, no
# aplicarse global a cualquiera que el usuario seleccione — de lo contrario
# un micrófono sin ese problema terminaría capturando por DirectSound sin
# necesidad, perdiendo calidad. Cualquier dispositivo que no esté en este
# mapa usa el host API que el resolver elija automáticamente (WASAPI si está
# disponible). Agrega una entrada aquí solo si confirmas que un dispositivo
# puntual falla o suena peor con WASAPI.
AUDIO_INPUT_HOST_API_OVERRIDES = {
    "DualSense Wireless Controller": "Windows DirectSound",
}

# --- Voice Activity Detection (Silero VAD) ---
VAD_SPEECH_PROBABILITY_THRESHOLD = 0.5   # Probabilidad mínima para considerar el frame como voz
VAD_SILENCE_DURATION_MS_TO_CLOSE_UTTERANCE = 600   # Silencio necesario para cerrar una frase
VAD_MIN_SPEECH_DURATION_MS = 250          # Descarta utterances demasiado cortas (ruido/clicks)

# --- Whisper (faster-whisper) ---
WHISPER_MODEL_NAME = "large-v3-turbo"     # Mejor precisión posible en GPU de 4-6 GB VRAM

# El dispositivo ("cuda" o "cpu") ya NO se fija aquí: se decide con el
# checkbox "Usar aceleración CUDA" de la ventana de configuración (guardado
# en user_config.json, ver config/user_config.get_whisper_device). Si no se
# detecta una GPU NVIDIA compatible, se fuerza CPU automáticamente sin
# importar lo que diga ese checkbox.
WHISPER_COMPUTE_TYPE = "int8_float16"     # Cuantización para caber en 4-6 GB de VRAM (GPU); en CPU se usa int8 automáticamente

# El idioma de origen (WHISPER_SOURCE_LANGUAGE, lo que se habla frente al
# micrófono) y el de destino de la traducción (TRANSLATION_TARGET_LANGUAGE,
# ver más abajo) tampoco se fijan aquí: se eligen como un solo flujo
# ("Español -> Inglés" o "Inglés -> Español") desde el selector de la
# ventana de configuración, guardado en user_config.json (ver
# config/user_config.get_whisper_source_language /
# get_translation_target_language). Este valor solo se usa como default de
# fábrica la primera vez que corre la app, antes de que exista una
# preferencia guardada.
WHISPER_SOURCE_LANGUAGE = "es"
WHISPER_BEAM_SIZE = 5                     # Mayor beam size = más precisión, más latencia
WHISPER_VAD_FILTER = False                # Ya filtramos con Silero antes de llegar aquí

# Evita que una alucinación en una utterance "contamine" el contexto de la
# siguiente (Whisper usa el texto previo como parte del prompt por defecto,
# lo que puede encadenar alucinaciones una tras otra).
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

# Filtros de confianza para descartar segmentos alucinados (frases inventadas
# como "Suscríbete al canal" o "Gracias por ver el video", típicas cuando el
# audio es casi silencio/ruido pero igual se le pide a Whisper que transcriba
# algo). Cada segmento que devuelve faster-whisper trae sus propias métricas
# de confianza; si no las superan, se descarta ese segmento en vez de
# mandarlo al overlay.
#   - no_speech_prob: probabilidad (0-1) de que el segmento NO tenga voz
#     real, según el propio modelo. Más alto que este umbral -> se descarta.
WHISPER_NO_SPEECH_PROB_THRESHOLD = 0.6
#   - avg_logprob: confianza promedio (log-probabilidad) del texto generado.
#     Más bajo (más negativo) que este umbral -> el modelo no está seguro de
#     lo que generó -> se descarta.
WHISPER_AVG_LOGPROB_THRESHOLD = -1.0
#   - compression_ratio: ratio de compresión del texto generado. Un valor
#     alto indica texto repetitivo o sin sentido (síntoma típico de
#     alucinación). Mismo umbral que usa Whisper de OpenAI internamente.
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

# Energía mínima (RMS, sobre audio float32 en [-1.0, 1.0]) para siquiera
# intentar transcribir una utterance. Si Silero VAD marca frames como "voz"
# por error (ruido de fondo, soplido del micrófono, etc.) pero el audio es
# casi silencio real, se descarta ANTES de llamar a Whisper — menos
# oportunidades de alucinar, y de paso se ahorra cómputo de GPU. Si ves que
# se está descartando audio que sí tenía voz real (hablando bajito), baja
# este valor.
MIN_UTTERANCE_RMS_ENERGY = 0.01

# Frases "enlatadas" que Whisper repite de su propio dataset de
# entrenamiento (mayormente YouTube) cuando el audio es silencio/ruido. Si el
# texto final coincide con alguna de estas (comparación insensible a
# mayúsculas/acentos/puntuación), se descarta directamente sin importar qué
# tan "segura" estuvo Whisper al generarlo. Agrega más si ves otras frases
# repetirse en tus propias alucinaciones.
KNOWN_HALLUCINATION_PHRASES = [
    "suscribete a mi canal",
    "suscribete al canal",
    "no olvides suscribirte",
    "no olvides suscribirte al canal",
    "dale like y suscribete",
    "activa la campanita",
    "gracias por ver el video",
    "gracias por ver este video",
    "gracias por ver",
    "gracias por ver hasta el final",
    "nos vemos en el proximo video",
    "hasta el proximo video",
    "sigueme en instagram",
    "subtitulos realizados por la comunidad de amara.org",
    "subtitulado por la comunidad de amara.org",
    "amara.org",
    "www.youtube.com",
    "suscribete",
]

# --- Transcripción parcial (feedback inmediato antes de cerrar la frase) ---
# Sistema de dos niveles: mientras la utterance sigue abierta (el usuario
# sigue hablando y todavía no hay suficiente silencio para cerrarla), se
# transcribe el buffer acumulado hasta el momento cada
# PARTIAL_TRANSCRIPTION_INTERVAL_MS y se envía al overlay marcado como
# {"final": false}. Cuando la utterance cierra por silencio real (el
# comportamiento de siempre), se transcribe el buffer completo con el
# modelo grande y se envía como {"final": true}, reemplazando en el overlay
# lo que se venía mostrando como parcial.
#
# Poner en False para volver al comportamiento anterior (solo texto final).
PARTIAL_TRANSCRIPTION_ENABLED = True

# Cada cuántos ms (de audio con voz acumulada) se dispara un nuevo parcial.
# Más bajo = feedback más inmediato pero más pasadas de Whisper por frase
# (más costo de GPU). 700ms es un punto de partida razonable.
PARTIAL_TRANSCRIPTION_INTERVAL_MS = 700

# Modelo separado y más liviano para los parciales: se transcribe el mismo
# audio varias veces por frase (una vez por intervalo, reprocesando desde
# el inicio de la utterance cada vez, porque faster-whisper no soporta
# continuar desde un estado interno), así que usar large-v3-turbo para todo
# dispararía el costo de GPU. Ambos modelos (este y WHISPER_MODEL_NAME)
# quedan cargados en VRAM al mismo tiempo — con int8_float16 en los dos
# normalmente entran juntos en 4-6GB, pero conviene confirmarlo en tu GPU
# puntual; si no entra, bajar PARTIAL_WHISPER_MODEL_NAME a "tiny".
#
# IMPORTANTE: usar un modelo multilingüe aquí, no una variante ".en" ni
# "distil-large-v3". Los modelos "distil-*" de Systran/HF son destilaciones
# entrenadas solo con audio en inglés — si WHISPER_SOURCE_LANGUAGE no es
# "en", el modelo igual va a intentar generar texto en inglés (no sabe
# transcribir en otros idiomas), lo cual parece traducción no pedida pero en
# realidad es el modelo equivocado para el idioma configurado.
PARTIAL_WHISPER_MODEL_NAME = "base"
PARTIAL_WHISPER_COMPUTE_TYPE = "int8_float16"
PARTIAL_WHISPER_BEAM_SIZE = 1  # velocidad por sobre precisión: es solo un adelanto tentativo

# Los parciales NO aplican los filtros estrictos de confianza que sí aplica
# la transcripción final (no_speech_prob, avg_logprob, compression_ratio):
# son descartables y reemplazables por diseño, así que se prioriza mostrar
# algo rápido en pantalla sobre filtrar agresivamente. Sí se sigue aplicando
# el filtro de energía RMS (MIN_UTTERANCE_RMS_ENERGY) y la lista negra de
# alucinaciones conocidas, porque son baratos y evitan basura obvia.

# --- Traducción (Argos Translate — offline, sin API externa) ---
# Se traduce el texto ya transcrito (idioma de origen -> idioma de destino,
# ver el flujo elegido en la GUI más arriba) antes de mandarlo al overlay.
# Argos Translate usa modelos basados en CTranslate2 (el mismo motor que
# faster-whisper) que corren 100% local; la primera vez que se usa un par de
# idiomas nuevo, se descarga el paquete correspondiente (requiere internet
# solo esa vez).
TRANSLATION_ENABLED = True

# Default de fábrica del idioma de destino, usado solo hasta que exista una
# preferencia guardada — ver el comentario de WHISPER_SOURCE_LANGUAGE más
# arriba. Paquetes de idioma disponibles en
# https://www.argosopentech.com/argospm/index/
TRANSLATION_TARGET_LANGUAGE = "en"

# Qué se muestra en el overlay:
#   "original_only"     -> solo el texto transcrito, sin traducir (como antes)
#   "translation_only"   -> solo la traducción
#   "both"                -> las dos líneas, original arriba y traducción abajo
TRANSLATION_DISPLAY_MODE = "both"

# Si además de traducir el texto final se traduce cada parcial. Da feedback
# de la traducción más inmediato, pero cada parcial dispara una traducción
# extra (además de la transcripción parcial que ya corre cada
# PARTIAL_TRANSCRIPTION_INTERVAL_MS) — más carga de CPU. Si notas que la
# app se atrasa, poner en False: los parciales se muestran sin traducir y
# solo el texto final llega traducido.
TRANSLATE_PARTIALS = True

# --- Servidor WebSocket (salida hacia el overlay de OBS) ---
WEBSOCKET_HOST = "localhost"
WEBSOCKET_PORT = 8765

# --- Filtro de malas palabras (solo aplica al overlay, no a la consola) ---
PROFANITY_FILTER_ENABLED = True

# Lista de palabras a censurar. Coincidencia por PALABRA COMPLETA (no por
# substring), insensible a mayúsculas/acentos. Cada entrada además matchea
# sus variantes con sufijos (ej. "puta" también captura "putas", "putazo",
# etc.) gracias a \w* en el patrón de profanity_filter.py — no hace falta
# listar cada inflexión a mano. Agrega o quita palabras según necesites;
# esta lista es un punto de partida, no pretende ser exhaustiva.
PROFANITY_WORDLIST = [
    "puta", "puto",
    "mierda",
    "cabron", "cabrona",
    "pendejo", "pendeja",
    "verga",
    "coño",
    "carajo",
    "gilipollas",
    "joder",
    "chingada", "chingado", "chingar",
    "hijueputa", "hijoputa",
    "marica", "maricon",
    "culero", "culera",
    "pinche",
    "concha",
    "boludo", "boluda",
    "forro",
    # --- Chilenismos ---
    "weon", "weona",
    "csm", "ctm",
    "conchetumadre", "conchetumare",
    "maraco",
    "cagada",
    "picurio", "picantera",
    "webon", "webona",
    "chucha",
    "cagar", "cagaste",
]

# --- Debug: guardar audio de cada utterance para diagnosticar problemas de
# precisión en la transcripción (ej. verificar si el driver del micrófono
# está aplicando ruido/AGC/supresión que degrada la calidad del audio antes
# de llegar a Whisper). Desactivar una vez resuelto el problema: escribir
# .wav en cada utterance agrega I/O innecesario en producción.
DEBUG_SAVE_UTTERANCE_WAV_ENABLED = False
DEBUG_UTTERANCE_WAV_DIR = "debug_utterances"