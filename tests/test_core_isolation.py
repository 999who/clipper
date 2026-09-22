"""Ядро не зависит от интерфейса и не тянет тяжёлые библиотеки при импорте."""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
UI_LIBRARIES = ["typer", "click", "rich", "textual"]
HEAVY_LIBRARIES = ["faster_whisper", "ctranslate2", "mediapipe", "cv2", "yt_dlp"]

CODE = """
import importlib, pkgutil, sys
import clipper.core as core
for module in pkgutil.iter_modules(core.__path__):
    importlib.import_module("clipper.core." + module.name)
print(",".join(name for name in {names!r} if name in sys.modules))
"""


def test_core_imports_no_ui_and_no_heavy_libraries():
    code = CODE.format(names=UI_LIBRARIES + HEAVY_LIBRARIES)
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert result.stdout.strip() == "", f"ядро импортирует: {result.stdout.strip()}"
