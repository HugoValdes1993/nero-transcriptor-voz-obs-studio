# Transcripción de voz en tiempo real para OBS

Pipeline local (sin costo, sin nube) que transcribe el micrófono en tiempo real
y transmite el texto a un overlay de OBS vía WebSocket.

Stack: `faster-whisper` (large-v3-turbo, CUDA, int8_float16) + `Silero VAD` + `sounddevice` + `websockets` + `Argos Translate`.

## Estructura del proyecto

```
voice-transcriber/
├── main.py                       # punto de entrada: python main.py
├── config/
│   └── settings.py               # TODA la configuración editable vive aquí
├── overlay/
│   ├── obs_overlay_original.html    # Browser Source con solo el texto original
│   └── obs_overlay_translated.html  # Browser Source con solo la traducción
├── scripts/
│   └── list_audio_devices.py     # utilidad de diagnóstico (dispositivos de audio)
└── src/
    ├── logging_utils.py          # logs a consola con formato consistente
    ├── startup/
    │   └── cuda_dll_setup.py     # registra las DLLs de CUDA en Windows
    ├── audio/
    │   ├── device_resolver.py    # elige el micrófono correcto por nombre
    │   ├── stream_capture.py     # captura y resamplea el audio del micrófono
    │   └── debug_dump.py         # guarda utterances como .wav para debug
    ├── speech/
    │   ├── voice_activity_detector.py  # Silero VAD: ¿este frame tiene voz?
    │   ├── utterance_segmenter.py      # agrupa frames en frases completas
    │   ├── speech_transcriber.py       # wrapper de faster-whisper
    │   └── hallucination_filter.py     # descarta frases "enlatadas" típicas
    ├── translation/
    │   └── translator.py         # traducción offline con Argos Translate
    ├── text_filters/
    │   └── profanity_filter.py   # censura groserías en el texto del overlay
    └── server/
        ├── overlay_broadcaster.py     # maneja las conexiones WebSocket
        └── transcription_pipeline.py  # orquesta todo el flujo de arriba
```

Cada módulo hace una sola cosa: `AudioStreamCapture` no sabe nada de Whisper,
`SpeechTranscriber` no sabe nada de WebSockets, `OverlayBroadcaster` no sabe
nada de audio. `TranscriptionPipeline` es la única pieza que conoce el orden
completo y conecta unos con otros.

## Requisitos

- Python 3.10+
- GPU NVIDIA con drivers CUDA instalados (probado para 4-6 GB de VRAM)
- Micrófono configurado como dispositivo de entrada del sistema

## Instalación

```bash
python3 -m venv venv
source venv/bin/activate        # En Windows: venv\Scripts\activate

pip install -r requirements.txt
```

La primera vez que se ejecute, `faster-whisper` descargará automáticamente
el modelo `large-v3-turbo` desde Hugging Face (requiere internet solo esa vez;
luego queda cacheado localmente). `torch.hub` descargará Silero VAD de la
misma forma, y Argos Translate el paquete de idioma si `TRANSLATION_ENABLED`
está activo.

## Ejecución

Desde la raíz del proyecto:

```bash
python main.py
```

Deberías ver:

```
12:34:56 [ OK ] AudioCapture  Dispositivo '...' capturando a 44100 Hz, resampleando a 16000 Hz
12:34:57 [ OK ] Main  Overlay disponible en ws://localhost:8765
12:34:58 [ OK ] Pipeline  Escuchando micrófono. Ctrl+C para detener.
```

Habla frente al micrófono y verás las transcripciones impresas en la consola
en tiempo real, con un log por componente (captura de audio, transcripción,
traducción, conexiones del overlay, etc.).

Para diagnosticar problemas de selección de micrófono:

```bash
python scripts/list_audio_devices.py
```

## Integración con OBS

El texto original y la traducción son **dos Browser Sources separadas**, para
poder ubicar cada una donde quieras en la escena (ej. original abajo,
traducción arriba, o en cualquier otra posición/tamaño) en vez de quedar
pegadas una debajo de la otra.

1. Agrega una fuente **Browser Source** (Fuente de navegador) y, en "Local
   file", selecciona `overlay/obs_overlay_original.html` de este proyecto.
2. Repetí el paso anterior con `overlay/obs_overlay_translated.html` para una
   segunda fuente (podés omitir este paso si no usás traducción).
3. Ajusta ancho/alto/posición de cada fuente por separado según tu escena.
4. Con `python main.py` corriendo, los subtítulos aparecerán automáticamente
   sobre la escena, cada línea en su propia fuente. Estas fuentes se
   configuran una sola vez: todos los ajustes de ahí en más se hacen desde la
   ventana de configuración de la app, no desde OBS.

Si `TRANSLATION_DISPLAY_MODE` (o el modo elegido en la GUI) no incluye una de
las dos líneas, esa fuente simplemente queda vacía — no hace falta quitarla
de la escena.

La ventana de configuración está organizada en pestañas: "Configuración
general" (dispositivo de entrada, flujo de transcripción/traducción,
aceleración por GPU) ya está resuelta de punta a punta. Las pestañas
"Estilos texto original" y "Estilos texto traducido" (tipografía, color,
tamaño por línea) están reservadas para una etapa siguiente.

## Transcripción parcial (feedback inmediato)

Por defecto, el overlay muestra texto tentativo (en itálica) mientras sigues
hablando, y lo reemplaza por el texto confirmado apenas la frase cierra por
silencio. Esto se controla con `PARTIAL_TRANSCRIPTION_ENABLED` en
`config/settings.py`.

Cómo funciona: cada `PARTIAL_TRANSCRIPTION_INTERVAL_MS` (700ms por defecto)
se transcribe el audio acumulado de la frase en curso con un modelo más
liviano (`PARTIAL_WHISPER_MODEL_NAME`, `base` por defecto — multilingüe) y se
envía al overlay como `{"final": false}`. Cuando la frase cierra de verdad,
se transcribe todo con `WHISPER_MODEL_NAME` (large-v3-turbo) y se envía como
`{"final": true}`, reemplazando el texto tentativo.

Esto implica tener **dos modelos cargados en VRAM al mismo tiempo**. Si te
quedas sin memoria en una GPU de 4GB, las opciones son (de menor a mayor
impacto en precisión de los parciales): bajar `PARTIAL_WHISPER_COMPUTE_TYPE`
a `"int8"`, cambiar `PARTIAL_WHISPER_MODEL_NAME` a `"tiny"`, o directamente
poner `PARTIAL_TRANSCRIPTION_ENABLED = False` para volver al comportamiento
original (solo texto final).

**Importante sobre el modelo elegido**: usar siempre un modelo multilingüe
aquí (`tiny`, `base`, `small`, etc.), nunca una variante `.en` ni
`distil-large-v3` — esos modelos "distil-*" están entrenados solo con audio
en inglés, y si `WHISPER_SOURCE_LANGUAGE` es otro idioma van a intentar
generar texto en inglés igual (parece una traducción no pedida, pero en
realidad es el modelo equivocado para el idioma configurado).

## Ajustes de precisión / latencia

Todo se controla desde `config/settings.py`:

- `WHISPER_MODEL_NAME`: el modelo usado para el texto final.
- `WHISPER_BEAM_SIZE`: subir a 8-10 mejora precisión a costa de latencia;
  bajar a 1-3 prioriza velocidad.
- `VAD_SILENCE_DURATION_MS_TO_CLOSE_UTTERANCE`: más alto = frases más largas
  y con más contexto (mejor precisión), pero más latencia antes de ver texto.
- El idioma de origen (Whisper): fijarlo evita que Whisper pierda tiempo
  detectando el idioma en cada utterance, y es requisito para que funcione
  la traducción (ver abajo). Se elige junto con el idioma de destino desde
  el selector "Flujo de transcripción / traducción" de la ventana de
  configuración, no editando `config/settings.py`.

## Traducción (Argos Translate, offline)

Con `TRANSLATION_ENABLED = True` en `config/settings.py`, cada transcripción
(parcial y final) se traduce localmente según el flujo elegido en el
selector "Flujo de transcripción / traducción" de la ventana de
configuración: "Español → Inglés" o "Inglés → Español". No depende de
ninguna API externa ni de internet salvo la primera vez, cuando se descarga
el paquete de idiomas correspondiente.

`TRANSLATION_DISPLAY_MODE` (en `config/settings.py`) controla qué se ve en
el overlay:
- `"original_only"`: solo el texto transcrito, sin traducir.
- `"translation_only"`: solo la traducción.
- `"both"`: las dos líneas, original arriba y traducción abajo (algo más chica).

`TRANSLATE_PARTIALS` decide si los parciales (el texto tentativo mientras
sigues hablando) también se traducen en el momento, o si solo se traduce el
texto final. Traducir parciales da feedback más inmediato pero suma carga
de CPU en cada intervalo — si notas que se atrasa, ponlo en `False`.

## Próximos pasos (no incluidos todavía)

- Persistencia de subtítulos en archivo `.srt` para VOD.
- Empaquetado como `.exe` distribuible para otros streamers.
- App de control (GUI) para elegir micrófono, tier de hardware y estilo del
  overlay sin editar `config/settings.py` a mano.

## Autor

Desarrollado por **Nero**.
