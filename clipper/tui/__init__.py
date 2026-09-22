"""TUI на Textual: `clipper tui`. Все действия — стрелками и Enter.

Интерфейс только вызывает ядро (clipper.core) — то же, что и CLI. Ядро о TUI
ничего не знает: прогресс приходит событиями, отмена — через CancelToken.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any


def run_tui(config_path: Path | None, overrides: dict[str, Any]) -> None:
    from clipper.tui.app import ClipperApp
    from clipper.tui.settings import SettingsStore

    store = SettingsStore(config_path, overrides)  # ошибка конфига — до запуска экрана
    logging.getLogger().addHandler(logging.NullHandler())  # предупреждения библиотек не рисуются поверх экрана
    log = _quiet_native_stderr(Path(store.config().paths.workdir))
    try:
        ClipperApp(store).run()
    finally:
        if log is not None:
            os.dup2(log[0], 2)
            os.close(log[0])


def _quiet_native_stderr(workdir: Path) -> tuple[int] | None:
    """Вывод нативных библиотек (CUDA, mediapipe) в stderr ломает экран — пишем его в файл."""
    try:
        workdir.mkdir(parents=True, exist_ok=True)
        saved = os.dup(2)
        sys.stderr.flush()
        with open(workdir / "clipper-tui.log", "ab") as file:
            os.dup2(file.fileno(), 2)
        return (saved,)
    except OSError:
        return None
