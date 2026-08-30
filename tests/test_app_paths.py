import os

from src.startup.app_paths import get_app_root


def test_source_mode_returns_repo_root(monkeypatch):
    monkeypatch.setattr("sys.frozen", False, raising=False)
    root = get_app_root()
    # main.py vive en la raíz del repo — confirma que la resolución de 3
    # niveles hacia arriba (src/startup/app_paths.py -> raíz) es correcta.
    assert os.path.isfile(os.path.join(root, "main.py"))


def test_frozen_mode_returns_executable_directory(monkeypatch):
    monkeypatch.setattr("sys.frozen", True, raising=False)
    monkeypatch.setattr("sys.executable", r"C:\Apps\VoiceTranscriber\VoiceTranscriber.exe")
    assert get_app_root() == r"C:\Apps\VoiceTranscriber"
