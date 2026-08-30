"""
Asegura que la raíz del repo esté en sys.path sin importar cómo se invoque
pytest (`pytest`, `python -m pytest`, o desde un IDE) — los módulos del
proyecto usan imports absolutos (`from config.settings import ...`,
`from src.logging_utils import ...`) que asumen la raíz del repo como
primer elemento de sys.path.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
