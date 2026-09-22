"""Экран прогресса: задача ядра в фоновом потоке, её этапы и сообщения, отмена.

События `Reporter` приходят из потока задачи и передаются в виджеты через
`app.call_from_thread`. «Отмена» взводит `CancelToken` — тот же, что Ctrl+C в
CLI: ядро останавливается между шагами и прерывает ffmpeg.
"""

import contextlib
import traceback
from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, OptionList, ProgressBar, RichLog, Static
from textual.widgets.option_list import Option

from clipper.core.errors import Cancelled, ClipperError
from clipper.core.events import CancelToken, Event, Message, Reporter, StageFinished, StageProgress, StageStarted
from clipper.tui.widgets import CANCEL, MenuModal

Job = Callable[[Reporter], Any]


class StageRow(Horizontal):
    def __init__(self, title: str, total: float | None) -> None:
        super().__init__(classes="stage")
        self.title_text = title
        self.total = total

    def compose(self) -> ComposeResult:
        yield Label("… " + self.title_text, classes="stage-title")
        yield ProgressBar(total=self.total, show_eta=False, classes="stage-bar")
        yield Label("", classes="stage-note")

    def progress(self, done: float, total: float | None, message: str) -> None:
        bar = self.query_one(ProgressBar)
        bar.update(total=total if total else None, progress=done)
        if message:
            self.query_one(".stage-note", Label).update(message)

    def finish(self, ok: bool, message: str, elapsed: float) -> None:
        for widget in self.query(".stage-bar, .stage-note"):
            widget.remove()
        mark, style = ("✓", "green") if ok else ("✗", "red")
        text = Text.assemble((f"{mark} ", style), self.title_text)
        if message:
            text.append(f" — {message}")
        text.append(f"  ({elapsed:.1f} с)", style="dim")
        self.query_one(".stage-title", Label).update(text)


class ProgressScreen(Screen[Any]):
    """Выполняет `job(reporter)`. Результат возвращается через dismiss (None — ошибка или отмена)."""

    BINDINGS = [Binding("escape", "stop", "Отмена / назад")]

    def __init__(self, title: str, job: Job) -> None:
        super().__init__()
        self.title_text = title
        self.job = job
        self.token = CancelToken()
        self.rows: dict[str, StageRow] = {}
        self.warnings = 0
        self.running = True
        self.result: Any = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(self.title_text, classes="screen-title")
        yield VerticalScroll(id="stages")
        yield RichLog(id="log", wrap=True, markup=False, max_lines=500)
        yield Static("", id="error")
        yield OptionList(Option("Прервать", id="stop"), id="actions")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = self.title_text
        self.query_one("#actions", OptionList).focus()
        self.run_worker(self._work, thread=True, exclusive=True, name="job")

    # --- поток задачи ---

    def _work(self) -> None:
        reporter = Reporter(self._sink, self.token)
        try:
            result = self.job(reporter)
        except Cancelled:
            self._from_thread(self._failed, "Прервано.", None)
        except ClipperError as exc:
            self._from_thread(self._failed, exc.message, exc.hint)
        except Exception as exc:  # показать, а не уронить TUI
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self._from_thread(self._failed, f"Непредвиденная ошибка: {detail}", "Подробности — в CLI с флагом -v.")
        else:
            self._from_thread(self._done, result)

    def _sink(self, event: Event) -> None:
        self._from_thread(self.show_event, event)

    def _from_thread(self, callback: Callable[..., Any], *args: Any) -> None:
        with contextlib.suppress(RuntimeError):  # приложение уже закрыто
            self.app.call_from_thread(callback, *args)

    # --- экран ---

    def show_event(self, event: Event) -> None:
        if isinstance(event, StageStarted):
            row = StageRow(event.title, event.total)
            self.rows[event.stage] = row
            self.query_one("#stages", VerticalScroll).mount(row)
        elif isinstance(event, StageProgress) and event.stage in self.rows:
            self.rows[event.stage].progress(event.done, event.total, event.message)
        elif isinstance(event, StageFinished) and event.stage in self.rows:
            self.rows[event.stage].finish(event.ok, event.message, event.elapsed)
        elif isinstance(event, Message):
            log = self.query_one("#log", RichLog)
            if event.level == "warning":
                self.warnings += 1
                log.write(Text("Внимание: " + event.text, style="yellow"))
            else:
                log.write(Text(event.text))

    def _done(self, result: Any) -> None:
        self.running = False
        self.result = result
        if self.warnings == 0:
            self.dismiss(result)
            return
        self._set_action("next", "Дальше →")  # сначала дать прочитать предупреждения

    def _failed(self, message: str, hint: str | None) -> None:
        self.running = False
        text = Text.assemble(("Ошибка: ", "bold red"), message)
        if hint:
            text.append("\n→ " + hint, style="cyan")
        self.query_one("#error", Static).update(text)
        self._set_action("back", "← Назад")

    def _set_action(self, key: str, label: str) -> None:
        actions = self.query_one("#actions", OptionList)
        actions.set_options([Option(label, id=key)])
        actions.highlighted = 0
        actions.focus()

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.action_stop()

    def action_stop(self) -> None:
        if not self.running:
            self.dismiss(self.result)
            return
        question = MenuModal("Прервать работу?", [("yes", "Да, прервать"), ("no", "Нет, продолжить")])
        self.app.push_screen(question, self._confirm_stop)

    def _confirm_stop(self, answer: Any) -> None:
        if answer == "yes" and answer is not CANCEL:
            self.token.cancel()
            self.query_one("#log", RichLog).write(Text("Останавливаю…", style="yellow"))

    def cancel_now(self) -> None:
        """Выход из приложения во время работы."""
        self.token.cancel()
