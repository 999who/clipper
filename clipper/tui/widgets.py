"""Виджеты TUI: список настроек с выбором по Enter, окна выбора и ввода, полоса heatmap."""

from collections.abc import Callable
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from clipper.core.errors import ConfigError
from clipper.core.models import HeatPoint
from clipper.tui.settings import (
    Item,
    Section,
    Setting,
    SettingsStore,
    build_config,
    choices_for,
    edit_text,
    parse_input,
    show_value,
)

LABEL_WIDTH = 30
CANCEL = object()  # окно закрыли без выбора (None — законное значение)


def row(label: str, value: str = "", *, width: int = LABEL_WIDTH) -> Text:
    """Строка списка: подпись слева, значение справа."""
    text = Text(label.ljust(width))
    if value:
        text.append(value, style="bold cyan")
    return text


class SettingsList(OptionList):
    """Настройки списком: стрелки — выбрать строку, Enter — изменить. В конце — действия (▶ …)."""

    class Changed(Message):
        def __init__(self, key: str) -> None:
            super().__init__()
            self.key = key

    class Action(Message):
        def __init__(self, action: str) -> None:
            super().__init__()
            self.action = action

    class Help(Message):
        """Пояснение к выбранной строке (показывает экран)."""

        def __init__(self, text: str) -> None:
            super().__init__()
            self.text = text

    def __init__(
        self,
        store: SettingsStore,
        items: tuple[Item, ...],
        actions: list[tuple[str, str]],
        *,
        top: list[Setting] | None = None,
        values: dict[str, Any] | None = None,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id)
        self.store = store
        self.items = tuple(top or ()) + items
        self.actions = actions
        self.local = values if values is not None else {}  # настройки не из Config (ссылка на видео)
        self._settings = {item.key: item for item in self.items if isinstance(item, Setting)}

    def on_mount(self) -> None:
        self.rebuild()

    def current(self, key: str) -> Any:
        return self.local[key] if key in self.local else self.store.value(key)

    def rebuild(self) -> None:
        highlighted = self.highlighted
        options: list[Option | None] = []
        for item in self.items:
            if isinstance(item, Section):
                options.append(Option(Text(item.title, style="bold underline"), disabled=True))
            else:
                options.append(Option(self._prompt(item), id=f"set:{item.key}"))
        options.append(None)
        options += [Option(Text(label, style="bold"), id=f"act:{name}") for name, label in self.actions]
        self.set_options(options)
        if highlighted is not None and highlighted < self.option_count:
            self.highlighted = highlighted
        else:
            self.highlighted = next(i for i, o in enumerate(self.options) if not o.disabled)

    def _prompt(self, setting: Setting) -> Text:
        return row(setting.label, show_value(setting, self.current(setting.key), self.store.config()))

    def refresh_values(self) -> None:
        for key, setting in self._settings.items():
            self.replace_option_prompt(f"set:{key}", self._prompt(setting))

    @on(OptionList.OptionHighlighted)
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        event.stop()
        kind, _, name = (event.option.id or "").partition(":")
        text = self._settings[name].help if kind == "set" else "Enter — запустить."
        self.post_message(self.Help(text or "Enter — изменить."))

    @on(OptionList.OptionSelected)
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        option_id = event.option.id or ""
        kind, _, name = option_id.partition(":")
        if kind == "act":
            self.post_message(self.Action(name))
        elif kind == "set":
            self.edit(self._settings[name])

    def edit(self, setting: Setting) -> None:
        value = self.current(setting.key)
        if setting.kind == "bool":
            self.apply(setting, not value)
        elif setting.kind == "choice":
            choices = choices_for(setting, self.store.config())
            self.app.push_screen(ChoiceModal(setting.label, choices, value), lambda v: self._chosen(setting, v))
        else:

            def validate(text: str) -> str | None:
                return self._check(setting, parse_input(setting, text))

            modal = InputModal(setting.label, edit_text(setting, value), setting.help, validate)
            self.app.push_screen(modal, lambda text: self._typed(setting, text))

    def _chosen(self, setting: Setting, value: Any) -> None:
        if value is not CANCEL:
            self.apply(setting, value)

    def _typed(self, setting: Setting, text: Any) -> None:
        if text is not CANCEL:
            self.apply(setting, parse_input(setting, text))

    def _check(self, setting: Setting, value: Any) -> str | None:
        if setting.key in self.local:
            return None
        try:
            build_config(self.store.path, {**self.store.overrides, setting.key: value})
        except ConfigError as exc:
            return exc.message + (f"\n{exc.hint}" if exc.hint else "")
        return None

    def apply(self, setting: Setting, value: Any) -> None:
        if setting.key in self.local:
            self.local[setting.key] = value
        else:
            error = self.store.try_set(setting.key, value)
            if error:
                self.app.notify(error, title="Не подходит", severity="error", timeout=8)
                return
        self.refresh_values()
        self.post_message(self.Changed(setting.key))


class ChoiceModal(ModalScreen[Any]):
    """Выбор из вариантов: стрелки и Enter, Esc — отмена."""

    BINDINGS = [Binding("escape", "cancel", "Отмена")]

    def __init__(self, title: str, choices: tuple[tuple[Any, str], ...], current: Any) -> None:
        super().__init__()
        self.title_text = title
        self.choices = choices
        self.current = current

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            options = [Option(("● " if v == self.current else "  ") + label, id=str(i))
                       for i, (v, label) in enumerate(self.choices)]  # fmt: skip
            yield OptionList(*options, id="choices")

    def on_mount(self) -> None:
        options = self.query_one("#choices", OptionList)
        index = next((i for i, (v, _) in enumerate(self.choices) if v == self.current), 0)
        options.highlighted = index
        options.focus()

    @on(OptionList.OptionSelected, "#choices")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(self.choices[int(event.option.id or 0)][0])

    def action_cancel(self) -> None:
        self.dismiss(CANCEL)


class InputModal(ModalScreen[Any]):
    """Ввод значения: Enter — готово (с проверкой), Esc — отмена."""

    BINDINGS = [Binding("escape", "cancel", "Отмена")]

    def __init__(
        self, title: str, value: str, help: str = "", validate: Callable[[str], str | None] | None = None
    ) -> None:
        super().__init__()
        self.title_text = title
        self.value = value
        self.help = help
        self.validate = validate

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.title_text, classes="dialog-title")
            if self.help:
                yield Label(self.help, classes="dialog-help")
            yield Input(self.value, id="value")
            yield Label("", id="error", classes="dialog-error")
            yield Label("Enter — готово, Esc — отмена", classes="dialog-help")

    def on_mount(self) -> None:
        field = self.query_one("#value", Input)
        field.focus()
        field.select_all()  # ввод сразу заменяет старое значение

    @on(Input.Submitted, "#value")
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        error = self.validate(event.value) if self.validate else None
        if error:
            self.query_one("#error", Label).update(error)
            return
        self.dismiss(event.value)

    def action_cancel(self) -> None:
        self.dismiss(CANCEL)


class MenuModal(ModalScreen[Any]):
    """Вопрос с вариантами ответа (например, «Сохранить изменения?»)."""

    BINDINGS = [Binding("escape", "cancel", "Отмена")]

    def __init__(self, question: str, answers: list[tuple[str, str]]) -> None:
        super().__init__()
        self.question = question
        self.answers = answers

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Label(self.question, classes="dialog-title")
            yield OptionList(*(Option(label, id=key) for key, label in self.answers), id="answers")

    def on_mount(self) -> None:
        self.query_one("#answers", OptionList).focus()

    @on(OptionList.OptionSelected, "#answers")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(CANCEL)


# --- heatmap -----------------------------------------------------------------------------

BLOCKS = " ▁▂▃▄▅▆▇█"


def heatmap_text(
    points: list[HeatPoint] | None, duration: float, clips: list[tuple[float, float, bool]], width: int
) -> Text:
    """Две строки: график интереса и отметки клипов (▲ — включён, △ — выключен)."""
    width = max(10, width)
    text = Text()
    if points and duration > 0:
        cells = []
        for i in range(width):
            t = (i + 0.5) / width * duration
            value = next((p.value for p in points if p.start <= t < p.end), 0.0)
            cells.append(BLOCKS[min(len(BLOCKS) - 1, round(value * (len(BLOCKS) - 1)))])
        text.append("".join(cells), style="magenta")
    else:
        text.append("heatmap нет".center(width), style="dim")
    marks = [" "] * width
    for start, end, enabled in clips:
        if duration <= 0:
            break
        a = min(width - 1, int(start / duration * width))
        b = min(width - 1, max(a, int(end / duration * width)))
        for i in range(a, b + 1):
            if marks[i] != "▲":
                marks[i] = "▲" if enabled else "△"
    text.append("\n" + "".join(marks), style="bold yellow")
    return text


class Hint(Static):
    """Серая подсказка внизу экрана (например, та же команда для CLI)."""
