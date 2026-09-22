"""События прогресса — единственный канал, по которому ядро сообщает о ходе работы.

Ядро не печатает в терминал. Оно отправляет события в «приёмник» — любую функцию
вида `sink(event)`. CLI рисует события через rich, будущий TUI перешлёт их в свои
виджеты (из рабочего потока через `app.call_from_thread`). Ядро от этого не меняется.

Код ядра работает не с приёмником напрямую, а с `Reporter`:

    with reporter.stage("download", "Загрузка видео", total=size, unit="bytes") as st:
        ...
        st.update(done_bytes)
    reporter.warning("У видео нет heatmap")

Отмена: `CancelToken` общий для интерфейса и ядра. Интерфейс вызывает `cancel()`,
ядро проверяет его на каждом обновлении прогресса и между шагами и выбрасывает
`Cancelled`.
"""

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Union

from clipper.core.errors import Cancelled, ClipperError

# Единицы объёма работы — подсказка интерфейсу, как показывать прогресс.
Unit = Literal["", "bytes", "seconds"]


@dataclass(frozen=True)
class StageStarted:
    stage: str  # машинное имя этапа: "download", "transcribe", ...
    title: str  # текст для человека: "Загрузка видео"
    total: float | None = None  # объём работы; None — неизвестен
    unit: Unit = ""  # "" — штуки, "bytes" — байты, "seconds" — секунды медиа


@dataclass(frozen=True)
class StageProgress:
    stage: str
    done: float
    total: float | None = None
    message: str = ""


@dataclass(frozen=True)
class StageFinished:
    stage: str
    ok: bool = True
    message: str = ""  # итог («3 клипа») или причина сбоя
    elapsed: float = 0.0  # секунды


@dataclass(frozen=True)
class Message:
    level: Literal["info", "warning"]
    text: str


Event = Union[StageStarted, StageProgress, StageFinished, Message]
EventSink = Callable[[Event], None]


class CancelToken:
    """Флаг отмены, общий для интерфейса и ядра. Потокобезопасен."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        """Выбросить `Cancelled`, если операцию отменили."""
        if self._event.is_set():
            raise Cancelled()


class Stage:
    """Хэндл этапа внутри `Reporter.stage()`: обновляет прогресс и проверяет отмену."""

    # Чаще этого события прогресса не отправляются: интерфейсу хватает ~10 обновлений
    # в секунду, а yt-dlp и ffmpeg сообщают о прогрессе гораздо чаще.
    MIN_INTERVAL = 0.1

    def __init__(self, reporter: "Reporter", name: str, total: float | None) -> None:
        self._reporter = reporter
        self.name = name
        self.total = total
        self.done = 0.0
        self.result = ""  # итоговое сообщение для StageFinished
        self._last_emit = 0.0

    def update(self, done: float, total: float | None = None, message: str = "") -> None:
        self._reporter.cancel.check()
        if total is not None:
            self.total = total
        self.done = done
        now = time.monotonic()
        finished = self.total is not None and done >= self.total
        if finished or now - self._last_emit >= self.MIN_INTERVAL:
            self._last_emit = now
            self._reporter.emit(StageProgress(self.name, done, self.total, message))

    def advance(self, step: float = 1.0, message: str = "") -> None:
        self.update(self.done + step, message=message)


class Reporter:
    """Обёртка над приёмником событий для кода ядра. Без приёмника события отбрасываются."""

    def __init__(self, sink: EventSink | None = None, cancel: CancelToken | None = None) -> None:
        self._sink = sink
        self.cancel = cancel if cancel is not None else CancelToken()

    def emit(self, event: Event) -> None:
        if self._sink is not None:
            self._sink(event)

    def info(self, text: str) -> None:
        self.emit(Message("info", text))

    def warning(self, text: str) -> None:
        self.emit(Message("warning", text))

    @contextmanager
    def stage(self, name: str, title: str, total: float | None = None, unit: Unit = "") -> Iterator[Stage]:
        """Обернуть этап: событие начала, прогресс через хэндл, событие конца.

        Если внутри этапа случилось исключение (включая отмену), отправляется
        `StageFinished(ok=False)`, а исключение пробрасывается дальше.
        """
        self.cancel.check()
        handle = Stage(self, name, total)
        started = time.monotonic()
        self.emit(StageStarted(name, title, total, unit))
        try:
            yield handle
        except BaseException as exc:
            elapsed = time.monotonic() - started
            self.emit(StageFinished(name, ok=False, message=_describe(exc), elapsed=elapsed))
            raise
        elapsed = time.monotonic() - started
        self.emit(StageFinished(name, ok=True, message=handle.result, elapsed=elapsed))


def _describe(exc: BaseException) -> str:
    if isinstance(exc, (Cancelled, KeyboardInterrupt)):
        return "отменено"
    if isinstance(exc, ClipperError):
        return exc.message
    return f"{type(exc).__name__}: {exc}"
