from src.logging_utils import ComponentLogger, print_startup_banner


def test_info_includes_component_name_and_message(capsys):
    logger = ComponentLogger("AudioCapture")
    logger.info("Dispositivo detectado")
    output = capsys.readouterr().out
    assert "AudioCapture" in output
    assert "Dispositivo detectado" in output
    assert "[INFO]" in output


def test_success_uses_ok_tag(capsys):
    logger = ComponentLogger("Main")
    logger.success("Escuchando micrófono")
    output = capsys.readouterr().out
    assert "[ OK ]" in output
    assert "Escuchando micrófono" in output


def test_warning_uses_warn_tag(capsys):
    logger = ComponentLogger("CUDA Setup")
    logger.warning("Driver desactualizado")
    output = capsys.readouterr().out
    assert "[WARN]" in output


def test_error_uses_error_tag(capsys):
    logger = ComponentLogger("Pipeline")
    logger.error("Fallo crítico")
    output = capsys.readouterr().out
    assert "[ERROR]" in output


def test_transcript_shows_label_and_text(capsys):
    logger = ComponentLogger("Main")
    logger.transcript("Transcripción", "Hola mundo")
    output = capsys.readouterr().out
    assert "[Transcripción]" in output
    assert "Hola mundo" in output


def test_color_disabled_when_stdout_is_not_a_tty(capsys):
    # Bajo pytest, sys.stdout está capturado (no es una terminal real), así
    # que _terminal_supports_color() debe dar False sin necesidad de mockear
    # nada — este es el caso real cuando el log se redirige a un archivo.
    logger = ComponentLogger("Main")
    assert logger._color_enabled is False

    logger.info("mensaje")
    output = capsys.readouterr().out
    assert "\033[" not in output


def test_color_enabled_wraps_text_in_ansi_codes(capsys):
    logger = ComponentLogger("Main")
    logger._color_enabled = True  # simula una terminal real

    logger.info("mensaje")
    output = capsys.readouterr().out
    assert "\033[" in output


def test_print_startup_banner_shows_project_name_and_author(capsys):
    print_startup_banner()
    output = capsys.readouterr().out
    assert "Voice Transcriber" in output
    assert "Nero" in output
