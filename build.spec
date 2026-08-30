# -*- mode: python ; coding: utf-8 -*-
"""
Spec de PyInstaller para empaquetar la app como .exe distribuible a otros
streamers, sin que necesiten Python ni instalar dependencias a mano.

Generar el build con (requiere requirements-build.txt instalado):
    pyinstaller build.spec --clean

El resultado queda en dist/VoiceTranscriber/ (VoiceTranscriber.exe + todas
las DLLs/paquetes al lado) — esa carpeta completa es lo que se distribuye,
no solo el .exe suelto.

Decisiones que no son obvias mirando el resultado:

- Build 'onedir' (carpeta), NO 'onefile' (un solo .exe). Onefile se
  autoextrae a una carpeta temporal DISTINTA en cada arranque, lo que rompe
  dos cosas de esta app en particular: user_config.json (el usuario perdería
  la configuración guardada entre sesiones) y la ruta a overlay/*.html que
  el usuario pega en OBS como "Local file" (dejaría de existir apenas se
  cierra la app, invalidando el Browser Source ya configurado en su escena).
  Ver src/startup/app_paths.get_app_root, que asume esta carpeta estable.

- console=True (se ve una consola detrás de la ventana) a propósito: toda la
  app loguea a stdout vía src/logging_utils.ComponentLogger y nunca a un
  archivo — con --noconsole/windowed, Windows deja sys.stdout en None y el
  primer print() de la app (el banner de arranque) directamente explota. La
  consola además es la fuente de verdad detallada mientras corre el pipeline
  (ver comentarios en config_window.py), así que ocultarla le saca
  información al streamer, no soluciona nada.

- hiddenimports fuerza nvidia.cublas / nvidia.cudnn / nvidia.cuda_runtime /
  nvidia.cuda_nvrtc (los cuatro paquetes CUDA de requirements.txt): cada uno
  tiene un hook dedicado en pyinstaller-hooks-contrib que copia sus DLLs,
  pero un hook solo se dispara si PyInstaller detecta el import en su
  análisis estático — y src/startup/cuda_dll_setup.py hace `import nvidia`
  a secas y enumera subcarpetas en tiempo de ejecución (ver ese archivo),
  así que sin forzarlos acá esos cuatro paquetes quedarían afuera del build
  y la aceleración CUDA fallaría en el .exe aunque funcione en código fuente.

- collect_all para customtkinter/ctranslate2/faster_whisper/argostranslate:
  paquetes que cargan sus propias DLLs o assets (temas JSON de customtkinter,
  DLLs sueltas de CTranslate2) por fuera del grafo de imports de Python que
  el analizador de PyInstaller recorre solo, así que no alcanza con dejar
  que los detecte solo.

Autor: Nero
"""

import os

from PyInstaller.utils.hooks import collect_all

APP_NAME = "VoiceTranscriber"

hiddenimports = [
    "nvidia.cublas",
    "nvidia.cudnn",
    "nvidia.cuda_runtime",
    "nvidia.cuda_nvrtc",
]

datas = [
    (os.path.join("overlay", "obs_overlay_original.html"), "overlay"),
    (os.path.join("overlay", "obs_overlay_translated.html"), "overlay"),
]
binaries = []

for package_name in ("customtkinter", "ctranslate2", "faster_whisper", "argostranslate"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    # Por defecto (PyInstaller 6+) todo lo que no es el .exe queda adentro de
    # una subcarpeta _internal/ — así overlay/*.html terminaría en
    # dist/VoiceTranscriber/_internal/overlay/, no al lado del .exe como
    # asume get_app_root() (y como se le indica al usuario en el README/GUI
    # que pegue en OBS). contents_directory="." vuelve al layout plano de
    # antes de PyInstaller 6: todo queda directo en dist/VoiceTranscriber/.
    contents_directory=".",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_NAME,
)
