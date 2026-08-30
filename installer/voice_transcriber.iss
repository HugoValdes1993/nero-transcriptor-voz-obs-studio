; Script de Inno Setup para Voice Transcriber. Se compila desde la raíz del
; repo (ver .github/workflows/release.yml):
;
;   iscc installer\voice_transcriber.iss /DAppVersion=1.0.0
;
; AppVersion se pasa desde el tag de git (ver el workflow); "0.0.0" es solo
; un fallback para compilar a mano sin pasar nada. El resto de los valores
; son fijos: con CUDA descargándose bajo demanda (ver
; src/startup/cuda_runtime_downloader.py) hay un solo build (ver
; build.spec) — la distinción CPU/GPU queda a elección del usuario durante
; la instalación (ver la tarea "gpu_support" más abajo), no en dos
; instaladores separados.
;
; Autor: Nero

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

#define MyAppName "Voice Transcriber"
#define MyAppPublisher "Nero"
#define MyAppURL "https://github.com/HugoValdes1993/nero-transcriptor-voz-obs-studio"
#define MyAppExeName "VoiceTranscriber.exe"
#define DistDir "..\dist\VoiceTranscriber"

[Setup]
; GUID fijo: NO regenerar — es lo que le permite a Windows reconocer una
; instalación existente al reinstalar/actualizar en vez de crear una entrada
; duplicada en "Agregar o quitar programas".
AppId={{EB86955C-A8ED-477A-96B6-2F54E1CCC27A}
AppName={#MyAppName}
AppVersion={#AppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\VoiceTranscriber
DefaultGroupName=Voice Transcriber
; No pedimos privilegios de administrador: todo el contenido de DistDir
; (config, overlays) vive DENTRO de la carpeta de instalación (ver
; src/startup/app_paths.get_app_root), y el runtime de CUDA se descarga
; aparte a %LOCALAPPDATA% (ver src/startup/cuda_runtime_downloader.py) — no
; hace falta escribir en ningún lado que requiera elevación.
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=VoiceTranscriberSetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
DisableProgramGroupPage=yes

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
; Tildada por default (sin "Flags: unchecked"): la mayoría de quienes buscan
; este proyecto tienen GPU NVIDIA y la quieren usar. Desmarcarla hace que
; src/startup/cuda_availability.py nunca ofrezca el checkbox de CUDA en la
; app, sin importar el hardware — ver gpu_enabled.marker más abajo.
Name: "gpu_support"; Description: "Habilitar aceleración GPU (CUDA) — recomendado si tenés una placa NVIDIA"; GroupDescription: "Rendimiento:"

[Files]
; Todo el contenido de la carpeta onedir de PyInstaller (el .exe + todas las
; DLLs/paquetes al lado) — ver build.spec, que documenta por qué es "onedir"
; y no "onefile".
Source: "{#DistDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion
; Marcador de "GPU habilitada" — solo se copia si la tarea de arriba quedó
; tildada. Su sola presencia es lo que chequea cuda_availability.py.
Source: "gpu_enabled.marker"; DestDir: "{app}"; Tasks: gpu_support; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
