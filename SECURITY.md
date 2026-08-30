# Política de seguridad

## Alcance del proyecto

Voice Transcriber es una app de escritorio que corre 100% en la máquina del
usuario: no manda audio ni texto a ningún servidor propio ni de terceros. El
único componente de red es un WebSocket que se levanta en `localhost` para
que OBS (Browser Source) reciba el texto transcripto — no está pensado para
exponerse fuera de la propia PC.

Dos descargas que la app hace por su cuenta, fuera del instalador:

- El modelo de Whisper (`faster-whisper`), descargado por la librería misma.
- El runtime de CUDA (~1.3GB), solo si se activa aceleración GPU — ver
  `src/startup/cuda_runtime_downloader.py`.

## Versiones soportadas

Proyecto de un solo mantenedor, sin ramas de soporte a largo plazo: los
arreglos de seguridad se aplican únicamente a la última versión publicada en
[Releases](https://github.com/HugoValdes1993/nero-transcriptor-voz-obs-studio/releases).
Si se encuentra un problema en una versión vieja, conviene primero
comprobar si sigue presente en la última antes de reportarlo.

## Cómo verificar una descarga antes de instalarla

1. **Descargar el instalador solo desde la página oficial de
   [Releases](https://github.com/HugoValdes1993/nero-transcriptor-voz-obs-studio/releases)**
   de este repo — nunca desde un mirror, foro o link reenviado.

2. **Verificar el checksum SHA256.** Cada release incluye
   `VoiceTranscriberSetup.exe.sha256` junto al instalador. En PowerShell:

   ```powershell
   Get-FileHash VoiceTranscriberSetup.exe -Algorithm SHA256
   ```

   Comparar el resultado con el contenido de
   `VoiceTranscriberSetup.exe.sha256`. Si no coincide, la descarga está
   corrupta o fue alterada — no ejecutarla.

3. **Sobre la advertencia de Windows SmartScreen:** el instalador todavía no
   tiene firma de código (trámite en curso con SignPath para proyectos open
   source — ver el `TODO (SignPath)` en
   `.github/workflows/release.yml`). Windows va a mostrar "Windows protegió
   su PC" la primera vez; esto es esperable mientras no haya firma, y no
   reemplaza la verificación del checksum del punto anterior.

4. **Descarga del runtime de CUDA:** se hace por HTTPS directo desde PyPI
   (paquetes oficiales `nvidia-*-cu12`) y cada archivo se valida contra el
   SHA256 publicado por PyPI antes de usarse — si no coincide, se descarta y
   falla en vez de instalarse a medias (ver `_download_wheel_to` en
   `src/startup/cuda_runtime_downloader.py`).

5. **Build a partir del código fuente:** si se prefiere no confiar en el
   binario precompilado, se puede compilar siguiendo "Empaquetado como .exe"
   en el `README.md` — el pipeline de CI (`.github/workflows/release.yml`)
   es público y hace exactamente lo mismo que se haría a mano.

## Reportar una vulnerabilidad

Por favor **no abrir un issue público** para vulnerabilidades de seguridad.
Usar el reporte privado de GitHub:

**[Report a vulnerability](https://github.com/HugoValdes1993/nero-transcriptor-voz-obs-studio/security/advisories/new)**
(pestaña *Security* del repo → *Report a vulnerability*).

Incluir, si es posible:

- Versión afectada (o commit, si es desde código fuente).
- Pasos para reproducir el problema.
- Impacto esperado (qué podría hacer un atacante con esto).

Este es un proyecto personal mantenido por una sola persona, sin bug bounty:
no hay compensación económica por reportes, pero todo reporte serio se
revisa y se agradece. No hay un SLA formal de respuesta, pero el objetivo es
confirmar recepción en la primera semana.

## Fuera de alcance

- Vulnerabilidades en dependencias de terceros (`faster-whisper`,
  `ctranslate2`, `argos-translate`, etc.) — reportarlas directamente en el
  repo de esa dependencia. Si afectan específicamente cómo esta app las usa,
  igual son bienvenidas acá.
- Ataques que requieren acceso físico o admin previo a la máquina del
  usuario.
- El hecho de que el instalador no esté firmado todavía (es una limitación
  conocida, ver arriba, no una vulnerabilidad).

## Autor

Mantenido por **Nero**.
