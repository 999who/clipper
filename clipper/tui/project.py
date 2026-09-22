"""Экран проекта: клипы таблицей, heatmap, правка границ, сохранение, переход к рендеру."""

import dataclasses
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import DataTable, Footer, Header, Label, OptionList, Static
from textual.widgets.option_list import Option

from clipper.core import pipeline
from clipper.core.errors import ClipperError
from clipper.core.models import Clip, Project, Transcript, load_source, load_transcript, parse_time, subtract_ranges
from clipper.core.render import output_dir
from clipper.tui.progress import ProgressScreen
from clipper.tui.settings import build_config
from clipper.tui.system import open_path
from clipper.tui.widgets import CANCEL, InputModal, MenuModal, heatmap_text, row

NUDGE = 1.0  # с: шаг сдвига границы
MIN_CLIP = 1.0  # с: короче клип не бывает


def clock(seconds: float) -> str:
    """1:02:03.4 или 24:41.3."""
    total = max(seconds, 0.0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    text = f"{int(minutes):02d}:{secs:04.1f}"
    return f"{int(hours)}:{text}" if hours >= 1 else text


def clip_text(clip: Clip, limit: int = 60) -> str:
    text = " ".join(w.text for w in clip.spoken_words())
    return text if len(text) <= limit else text[: limit - 1] + "…"


class ClipModal(ModalScreen[Any]):
    """Правка клипа: Enter на строке — изменить. «Готово» возвращает исправленный клип."""

    BINDINGS = [Binding("escape", "cancel", "Отмена")]

    def __init__(self, clip: Clip, duration: float) -> None:
        super().__init__()
        self.clip = clip
        self.duration = duration

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog wide"):
            yield Label("", id="clip-title", classes="dialog-title")
            yield OptionList(id="clip-options")
            yield Label("", id="clip-text", classes="dialog-help")
            yield Label("Enter — изменить, Esc — отмена", classes="dialog-help")

    def on_mount(self) -> None:
        self._refresh()
        self.query_one("#clip-options", OptionList).focus()

    def _refresh(self) -> None:
        clip = self.clip
        self.query_one("#clip-title", Label).update(
            f"Клип {clip.id}: {clock(clip.start)} – {clock(clip.end)}, {clip.duration:.1f} с"
        )
        options = [
            Option(row("Включён", "да" if clip.enabled else "нет"), id="enabled"),
            Option(row("Начало", clock(clip.start)), id="start"),
            Option(row("Конец", clock(clip.end)), id="end"),
            None,
            Option(f"Начало на {NUDGE:g} с раньше", id="start-"),
            Option(f"Начало на {NUDGE:g} с позже", id="start+"),
            Option(f"Конец на {NUDGE:g} с раньше", id="end-"),
            Option(f"Конец на {NUDGE:g} с позже", id="end+"),
            None,
            Option(Text("✓ Готово", style="bold"), id="done"),
        ]
        options_list = self.query_one("#clip-options", OptionList)
        highlighted = options_list.highlighted
        options_list.set_options(options)
        options_list.highlighted = highlighted if highlighted is not None else 0
        self.query_one("#clip-text", Label).update(clip_text(clip, 300) or "(слов в клипе нет)")

    def _bounds(self, start: float, end: float) -> str | None:
        if start < 0 or end > self.duration + 0.01:
            return f"Клип должен быть внутри видео: 0 – {clock(self.duration)}."
        if end - start < MIN_CLIP:
            return f"Клип должен быть не короче {MIN_CLIP:g} с."
        return None

    def _set(self, start: float, end: float) -> None:
        start, end = round(start, 3), round(end, 3)
        error = self._bounds(start, end)
        if error:
            self.app.notify(error, severity="error")
            return
        self.clip = dataclasses.replace(self.clip, start=start, end=end)
        self._refresh()

    @on(OptionList.OptionSelected, "#clip-options")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        key = event.option.id
        clip = self.clip
        if key == "enabled":
            self.clip = dataclasses.replace(clip, enabled=not clip.enabled)
            self._refresh()
        elif key in ("start", "end"):
            current = clip.start if key == "start" else clip.end

            def validate(text: str) -> str | None:
                try:
                    value = parse_time(text)
                except ValueError as exc:
                    return f"{exc}. Примеры: 24:41.3, 1:02:03, 90"
                return self._bounds(value, clip.end) if key == "start" else self._bounds(clip.start, value)

            modal = InputModal("Начало клипа" if key == "start" else "Конец клипа", clock(current),
                               "Время в видео: 24:41.3, 1:02:03 или секунды", validate)  # fmt: skip
            self.app.push_screen(modal, lambda text: self._typed(key, text))
        elif key == "start-":
            self._set(clip.start - NUDGE, clip.end)
        elif key == "start+":
            self._set(clip.start + NUDGE, clip.end)
        elif key == "end-":
            self._set(clip.start, clip.end - NUDGE)
        elif key == "end+":
            self._set(clip.start, clip.end + NUDGE)
        elif key == "done":
            self.dismiss(self.clip)

    def _typed(self, key: str, text: Any) -> None:
        if text is CANCEL:
            return
        value = parse_time(text)
        if key == "start":
            self._set(value, self.clip.end)
        else:
            self._set(self.clip.start, value)

    def action_cancel(self) -> None:
        self.dismiss(CANCEL)


class ProjectScreen(Screen[Any]):
    """Клипы проекта. Enter — изменить клип, пробел — включить/выключить."""

    BINDINGS = [
        Binding("space", "toggle", "Вкл/выкл"),
        Binding("s,ы", "save", "Сохранить"),
        Binding("r,к", "render", "Нарезать"),
        Binding("o,щ", "open_folder", "Папка"),
        Binding("c,с", "count", "Сколько клипов"),
        Binding("escape", "back", "Назад"),
    ]

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.project: Project | None = None
        self.heatmap = None
        self.transcript: Transcript | None = None
        self.dirty = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static("", id="info")
        yield Static("", id="heat")
        yield DataTable(id="clips", cursor_type="row", zebra_stripes=True)
        yield Static("", id="detail")
        yield OptionList(
            Option(Text("▶ Нарезать клипы…", style="bold"), id="render"),
            Option("Сколько клипов — найти заново…", id="count"),
            Option("Сохранить изменения", id="save"),
            Option("Открыть папку с клипами", id="folder"),
            Option("← Назад", id="back"),
            id="actions",
        )
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.project = pipeline.open_project(self.path)
        except ClipperError as exc:
            self.app.notify(exc.message, title="Проект не открывается", severity="error", timeout=10)
            self.app.pop_screen()
            return
        source = load_source(self.path.parent)
        self.heatmap = source.heatmap if source else None
        self.sub_title = self.project.source.title
        table = self.query_one("#clips", DataTable)
        table.add_columns("", "№", "Время", "Длина", "Почему", "Начало речи")
        self._fill()
        table.focus()

    # --- отображение ---

    def _fill(self) -> None:
        assert self.project is not None
        table = self.query_one("#clips", DataTable)
        cursor = table.cursor_row
        table.clear()
        for clip in self.project.clips:
            mark = Text("✓", style="bold green") if clip.enabled else Text("·", style="dim")
            table.add_row(mark, str(clip.id), f"{clock(clip.start)}–{clock(clip.end)}", f"{clip.duration:.0f} с",
                          clip.reason, clip_text(clip), key=str(clip.id))  # fmt: skip
        if self.project.clips:
            table.move_cursor(row=min(cursor, len(self.project.clips) - 1))
        self._info()

    def _info(self) -> None:
        assert self.project is not None
        clips = self.project.clips
        enabled = [c for c in clips if c.enabled]
        seconds = sum(c.duration for c in enabled)
        text = Text(f"Клипов: {len(clips)}, включено {len(enabled)} ({seconds:.0f} с)")
        text.append(f"   видео {clock(self.project.source.duration)}", style="dim")
        if self.dirty:
            text.append("   ● есть несохранённые изменения", style="bold yellow")
        self.query_one("#info", Static).update(text)
        self._heat()
        self._detail()

    def _heat(self) -> None:
        assert self.project is not None
        heat = self.query_one("#heat", Static)
        marks = [(c.start, c.end, c.enabled) for c in self.project.clips]
        heat.update(heatmap_text(self.heatmap, self.project.source.duration, marks, heat.size.width or 80))

    def on_resize(self) -> None:
        if self.project is not None:
            self._heat()

    def _detail(self) -> None:
        clip = self.selected()
        text = Text("")
        if clip is not None:
            text = Text.assemble((f"Клип {clip.id}: ", "bold"), clip_text(clip, 400) or "(слов нет)")
        self.query_one("#detail", Static).update(text)

    @on(DataTable.RowHighlighted, "#clips")
    def _highlighted(self, event: DataTable.RowHighlighted) -> None:
        self._detail()

    def selected(self) -> Clip | None:
        if self.project is None or not self.project.clips:
            return None
        table = self.query_one("#clips", DataTable)
        index = min(max(table.cursor_row, 0), len(self.project.clips) - 1)
        return self.project.clips[index]

    # --- правка ---

    @on(DataTable.RowSelected, "#clips")
    def _edit(self, event: DataTable.RowSelected) -> None:
        clip = self.selected()
        if clip is not None and self.project is not None:
            self.app.push_screen(ClipModal(clip, self.project.source.duration), self._edited)

    def _edited(self, clip: Any) -> None:
        if clip is CANCEL or self.project is None:
            return
        old = next(c for c in self.project.clips if c.id == clip.id)
        if (clip.start, clip.end) != (old.start, old.end):
            clip = self._with_words(clip)
        self._replace(clip)

    def _with_words(self, clip: Clip) -> Clip:
        """Слова для новых границ — из кэша распознавания (для субтитров)."""
        if self.transcript is None:
            self.transcript = load_transcript(self.path.parent)
        if self.transcript is None:
            return clip
        margin = self.app.store.config().transcribe.margin
        words = self.transcript.words_between(clip.start - margin, clip.end + margin)
        if subtract_ranges([(clip.start, clip.end)], self.transcript.ranges, min_length=0.5):
            self.app.notify(
                "Часть клипа не распознавалась — там не будет субтитров. "
                "Найдите моменты заново или распознайте кусок: clipper transcribe … --from … --to …",
                severity="warning",
                timeout=10,
            )
        return dataclasses.replace(clip, words=words)

    def _replace(self, clip: Clip) -> None:
        assert self.project is not None
        self.project.clips = [clip if c.id == clip.id else c for c in self.project.clips]
        self.dirty = True
        self._fill()

    def action_toggle(self) -> None:
        clip = self.selected()
        if clip is not None:
            self._replace(dataclasses.replace(clip, enabled=not clip.enabled))

    def action_save(self) -> bool:
        if self.project is None:
            return False
        try:
            pipeline.save_edited_project(self.path, self.project)
        except OSError as exc:
            self.app.notify(f"Не удалось сохранить: {exc}", severity="error")
            return False
        self.dirty = False
        self._info()
        self.app.notify("project.json сохранён (прежний — project.prev.json).")
        return True

    # --- переходы ---

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        {"render": self.action_render, "count": self.action_count, "save": self.action_save,
         "folder": self.action_open_folder, "back": self.action_back}[event.option.id or "back"]()  # fmt: skip

    def action_render(self) -> None:
        from clipper.tui.screens import RenderScreen

        if self.project is None:
            return
        if not any(c.enabled for c in self.project.clips):
            self.app.notify("Все клипы выключены — включите хотя бы один (пробел).", severity="warning")
            return
        if self.dirty and not self.action_save():  # рендер читает project.json с диска
            return
        self.app.push_screen(RenderScreen(self.path, sum(c.enabled for c in self.project.clips)))

    def action_count(self) -> None:
        """Найти моменты в том же видео заново, с другим числом клипов."""
        if self.project is None:
            return
        current = len(self.project.clips)

        def validate(text: str) -> str | None:
            try:
                count = int(text.strip())
            except ValueError:
                return "Введите целое число, например 10."
            return None if count >= 1 else "Нужен хотя бы 1 клип."

        help_text = (
            f"Сейчас клипов: {current}. Моменты найдутся заново в том же видео; уже распознанная речь "
            "берётся из кэша. Ручные правки клипов пропадут (прежний файл — project.prev.json)."
        )
        modal = InputModal("Сколько клипов найти", str(current), help_text, validate)
        self.app.push_screen(modal, self._recount)

    def _recount(self, text: Any) -> None:
        if text is CANCEL or self.project is None:
            return
        count = int(text.strip())
        self.app.store.try_set("select.clips", count)
        try:
            cfg = build_config(self.app.store.path, {**self.app.store.overrides, **self._same_search(count)})
        except ClipperError as exc:
            self.app.notify(exc.message, severity="error")
            return
        source = self.project.source.input
        job = ProgressScreen(f"Поиск моментов: {count}", lambda reporter: pipeline.analyze(source, cfg, reporter))
        self.app.push_screen(job, self._recounted)

    def _same_search(self, count: int) -> dict[str, Any]:
        """Тот же режим и ключевые слова, что у проекта; язык и verbatim — как в кэше распознавания."""
        assert self.project is not None
        overrides: dict[str, Any] = {"select.clips": count, "select.mode": self.project.mode}
        if self.project.keywords:
            overrides["select.keywords"] = list(self.project.keywords)
        transcript = load_transcript(self.path.parent)
        if transcript is not None and transcript.settings:
            overrides["transcribe.language"] = transcript.settings.get("language")
            overrides["transcribe.verbatim"] = bool(transcript.settings.get("verbatim"))
        return overrides

    def _recounted(self, result: Any) -> None:
        if not result:
            return
        _, path = result
        self.path = Path(path)
        self.project = pipeline.open_project(self.path)
        self.transcript = None
        self.dirty = False
        self._fill()
        self.app.notify(f"Клипов: {len(self.project.clips)}.")

    def action_open_folder(self) -> None:
        if self.project is None:
            return
        folder = output_dir(self.app.store.config(), self.project)
        if not folder.is_dir():
            self.app.notify("Клипов ещё нет — сначала нарежьте их.", severity="warning")
            return
        open_path(folder)

    def action_back(self) -> None:
        if not self.dirty:
            self.app.pop_screen()
            return
        question = MenuModal("Сохранить изменения в project.json?",
                             [("save", "Сохранить"), ("drop", "Не сохранять"), ("stay", "Остаться")])  # fmt: skip
        self.app.push_screen(question, self._leave)

    def _leave(self, answer: Any) -> None:
        if answer == "drop" or (answer == "save" and self.action_save()):
            self.app.pop_screen()
