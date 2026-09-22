"""Открыть файл или папку программой по умолчанию (плеер, проводник)."""

import os
import subprocess
import sys
from pathlib import Path


def open_path(path: Path) -> None:
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 — только наши файлы из output/
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
