# Transcripción de voz en tiempo real para OBS

Pipeline local (sin costo, sin nube) que transcribe el micrófono en tiempo real
y transmite el texto a un overlay de OBS vía WebSocket.

Stack: `faster-whisper` (large-v3-turbo, CUDA, int8_float16) + `Silero VAD` + `sounddevice` + `websockets` + `Argos Translate`.

## Estructura del proyecto

```
voice-transcriber/
├── main.py                       # punto de entrada: python main.py
├── config/
│   ├── settings.py               # configuración "de fábrica" del proyecto
│   └── user_config.py            # configuración editable desde la GUI (persistida en user_config.json)
├── overlay/
│   ├── obs_overlay_original.html    # Browser Source con solo el texto original
│   └── obs_overlay_translated.html  # Browser Source con solo la traducción
├── scripts/
│   └── list_audio_devices.py     # utilidad de diagnóstico (dispositivos de audio)
└── src/
    ├── logging_utils.py          # logs a consola con formato consistente
    ├── startup/
    │   ├── cuda_dll_setup.py           # registra las DLLs de CUDA en Windows
    │   ├── cuda_runtime_downloader.py  # descarga esas DLLs bajo demanda (~1.3GB)
    │   ├── cuda_availability.py        # detecta si hay GPU NVIDIA disponible
    │   └── app_paths.py                # raíz de la app, en dev y empaquetada (.exe)
    ├── gui/
    │   └── config_window.py      # ventana de configuración + control de arranque/detención
    ├── audio/
    │   ├── device_resolver.py    # elige el micrófono correcto por nombre
    │   ├── stream_capture.py     # captura y resamplea el audio del micrófono
    │   ├── mic_level_monitor.py  # nivel de audio en vivo para "Probar micrófono"
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
        ├── overlay_server.py          # levanta el WebSocket del overlay (vive desde que se abre la GUI)
        ├── overlay_broadcaster.py     # conexiones y envío de mensajes a los overlays
        ├── pipeline_controller.py     # arranca/detiene la transcripción en un hilo aparte
        └── transcription_pipeline.py  # orquesta todo el flujo de arriba
```

Cada módulo hace una sola cosa: `AudioStreamCapture` no sabe nada de Whisper,
`SpeechTranscriber` no sabe nada de WebSockets, `OverlayBroadcaster` no sabe
nada de audio. `TranscriptionPipeline` es la única pieza que conoce el orden
completo y conecta unos con otros.

## Requisitos

- Python 3.10+
- Micrófono configurado como dispositivo de entrada del sistema
- GPU NVIDIA (probado para 4-6 GB de VRAM) — opcional pero muy recomendada
  para velocidad; sin ella la transcripción corre en CPU, más lenta. Los
  drivers de CUDA (cuBLAS/cuDNN, ~1.3GB) NO hace falta instalarlos a mano: la
  app los descarga sola la primera vez que activás "Usar aceleración CUDA"
  en la ventana de configuración (ver `src/startup/cuda_runtime_downloader.py`)
  — solo necesitás tener instalado el driver normal de NVIDIA.

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
aceleración por GPU) y "Estilos texto original" / "Estilos texto traducido"
(fuente —cualquiera instalada en el sistema, con buscador—, tamaño, color,
opacidad, y ancho/alto exactos de la zona de texto) ya están resueltas de
punta a punta.

## Personalización visual del overlay

Desde las pestañas "Estilos texto original" / "Estilos texto traducido" de la
ventana de configuración, además de tipografía/color/tamaño:

- **Presets de estilo**: un selector con 4 looks de fábrica ("Clásico", "Alto
  contraste", "Minimalista sin fondo", "Neón") más los que vos guardes con
  "💾 Guardar como preset...". Aplicar un preset pisa de una el fondo general
  y la tipografía de los dos textos (original y traducido); "🗑 Eliminar
  preset" borra los tuyos — los de fábrica no se pueden pisar ni eliminar.
- **Contorno de texto**: color y ancho (0-6px) independientes por línea, útil
  para mantener legibilidad sobre fondos claros o con `background_opacity`
  en 0.
- **Efectos de aparición**: "Ninguno" (default, sin cambios de comportamiento),
  "Fade", "Bounce" o "Glitch" (saltos de posición/color que se asientan en el
  texto normal) — se elige por separado para el texto original y el
  traducido, y solo se dispara con el texto FINAL de cada frase; el texto
  tentativo mientras seguís hablando nunca anima.
- **Vista previa en vivo**: el botón "👁 Vista previa" de cada pestaña abre
  ese overlay en tu navegador y le manda texto de muestra, sin necesidad de
  tener la transcripción corriendo ni el micrófono activo — el servidor
  WebSocket del overlay queda escuchando desde que se abre la ventana de
  configuración, no solo durante una transcripción real.

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

## Empaquetado como .exe

Para distribuir la app a otros streamers sin que necesiten Python ni instalar
dependencias a mano, se empaqueta con PyInstaller (ver `build.spec` para el
detalle de cada decisión — build "onedir", consola visible, qué paquetes
necesitan collect_all y por qué):

```bash
pip install -r requirements-build.txt
pyinstaller build.spec --clean
```

El resultado queda en `dist/VoiceTranscriber/` — esa carpeta COMPLETA es lo
que se distribuye, no solo el `.exe` suelto. `user_config.json` y los dos
overlays (`overlay/*.html`, para pegar en OBS como "Local file") quedan al
lado del `.exe` dentro de esa misma carpeta, y son estables entre sesiones —
por eso el build es "onedir" y no "onefile". Las DLLs de CUDA NO están acá
(ver "Requisitos" arriba): por eso este build pesa una fracción de lo que
pesaría con ellas adentro.

### Instalador y releases (GitHub Actions)

`installer/voice_transcriber.iss` (Inno Setup) arma un instalador de verdad
(acceso directo, desinstalador) a partir de ese mismo `dist/VoiceTranscriber/`.
Durante la instalación se puede elegir si habilitar la aceleración GPU
(tildado por default) — desmarcarla hace que la app nunca ofrezca el
checkbox de CUDA, sin importar el hardware (ver
`src/startup/cuda_availability.py`). No hace falta instalar Inno Setup a
mano: `.github/workflows/release.yml`
hace todo el proceso (build + instalador + checksum) solo, en un runner de
GitHub, y publica el resultado como GitHub Release (como borrador, para
revisar antes de publicarlo de verdad) apenas se pushea un tag `vX.Y.Z`:

```bash
git tag v1.0.0
git push origin v1.0.0
```

También se puede disparar el mismo build sin crear un release (para probar
que compila) desde la pestaña "Actions" de GitHub, con "Run workflow".

## Flujo de trabajo (GitFlow)

El repo sigue GitFlow, adaptado a `main` en vez de `master`:

- **`main`**: solo código de release. Cada versión publicada tiene su tag
  (`vX.Y.Z`) sobre un commit de `main` — nunca se tagea directo desde
  `develop` ni desde una rama de feature. El pipeline de release
  (`.github/workflows/release.yml`) tiene un job `verify-tag` que corta el
  build si el tag no es ancestro de `origin/main`, para que esto no dependa
  de acordarse hacerlo bien a mano.
- **`develop`**: rama de integración, donde se juntan las features
  terminadas antes de preparar una release.
- **`feature/<nombre>`**: sale de `develop`, vuelve a `develop` (merge o PR).
- **`release/<version>`**: sale de `develop` cuando se arma una versión
  (ajustes finales, changelog); se mergea a `main` (y de ahí se tagea) Y de
  vuelta a `develop`.
- **`hotfix/<nombre>`**: sale de `main` para un arreglo urgente sobre una
  versión ya publicada; se mergea a `main` (nuevo tag de patch) Y a
  `develop`.

Publicar una versión nueva, en resumen:

```bash
git checkout main
git merge --no-ff release/1.1.0   # o hotfix/lo-que-sea
git tag v1.1.0
git push origin main --tags
```

## Próximos pasos (no incluidos todavía)

- Persistencia de subtítulos en archivo `.srt` para VOD.

## Autor

Desarrollado por **Nero**.
