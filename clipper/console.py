"""Вывод в терминал через rich: прогресс этапов, таблицы, ошибки.

Здесь только отображение. Логика живёт в clipper.core, а её события приходят
в ConsoleSink. Символы подобраны из базового набора шрифтов Windows (WGL4),
чтобы они не превращались в квадратики в старой консоли.
"""

import contextlib
import logging
import sys
from pathlib import Path
from types import TracebackType
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.markup import escape
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from clipper.core.doctor import CheckResult
from clipper.core.errors import ClipperError
from clipper.core.text import plural
from clipper.core.events import Event, Message, StageFinished, StageProgress, StageStarted
from clipper.core.render import RenderResult
from clipper.core.models import (
    Project,
    SRT_FILENAME,
    TRANSCRIPT_FILENAME,
    HeatPoint,
    SourceInfo,
    Transcript,
    heatmap_peak,
    heatmap_profile,
)

console = Console(highlight=False)


def setup_stdio() -> None:
    """UTF-8 для вывода, даже если его перенаправили в файл (иначе на Windows — cp1251)."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8", errors="replace")


def setup_logging(verbose: bool) -> None:
    """Технические логи ядра: с -v подробно, без -v — только предупреждения библиотек."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(name)s: %(message)s",
        handlers=[RichHandler(console=Console(stderr=True), show_time=False, show_path=False, markup=False)],
        force=True,
    )


# --- Прогресс -----------------------------------------------------------------------


class _AmountColumn(ProgressColumn):
    """«12.3/45.6 МБ», «01:23/05:00» или «2/5» — в зависимости от единиц этапа."""

    def render(self, task: Task) -> Text:
        unit = task.fields.get("unit", "")
        done, total = task.completed, task.total
        if unit == "bytes":
            text = f"{_mb(done)}/{_mb(total)} МБ" if total else f"{_mb(done)} МБ"
        elif unit == "seconds":
            text = f"{_clock(done)}/{_clock(total)}" if total else _clock(done)
        else:
            text = f"{done:g}/{total:g}" if total else ""
        return Text(text, style="progress.download")


class ConsoleSink:
    """Показывает события ядра в терминале. Использовать как контекстный менеджер."""

    def __init__(self, target: Console = console) -> None:
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            _AmountColumn(),
            TimeElapsedColumn(),
            TextColumn("[dim]{task.fields[message]}"),
            console=target,
            transient=True,
        )
        self._tasks: dict[str, TaskID] = {}
        self._titles: dict[str, str] = {}

    def __enter__(self) -> "ConsoleSink":
        self._progress.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._progress.stop()

    def __call__(self, event: Event) -> None:
        out = self._progress.console
        if isinstance(event, StageStarted):
            self._titles[event.stage] = event.title
            self._tasks[event.stage] = self._progress.add_task(
                escape(event.title), total=event.total, unit=event.unit, message=""
            )
        elif isinstance(event, StageProgress):
            task = self._tasks.get(event.stage)
            if task is not None:
                self._progress.update(task, completed=event.done, total=event.total, message=escape(event.message))
        elif isinstance(event, StageFinished):
            task = self._tasks.pop(event.stage, None)
            if task is not None:
                self._progress.remove_task(task)
            title = escape(self._titles.pop(event.stage, event.stage))
            if event.ok:
                result = f" — {escape(event.message)}" if event.message else ""
                out.print(f"[green]●[/] {title}{result} [dim]({_duration(event.elapsed)})[/]")
            else:
                reason = "отменено" if event.message == "отменено" else "ошибка"
                out.print(f"[red]●[/] {title} — {reason}")
        elif isinstance(event, Message):
            if event.level == "warning":
                out.print(f"[yellow]Внимание:[/] {escape(event.text)}")
            else:
                out.print(escape(event.text))


# --- Ошибки ---------------------------------------------------------------------------


def print_error(exc: ClipperError) -> None:
    console.print(f"[bold red]Ошибка:[/] {escape(exc.message)}")
    if exc.hint:
        console.print(f"[yellow]→[/] {_indent(escape(exc.hint))}")


# --- doctor ---------------------------------------------------------------------------

_STATUS = {
    "ok": "[green]ОК[/]",
    "warn": "[yellow]ВНИМАНИЕ[/]",
    "fail": "[red]ОШИБКА[/]",
    "info": "[dim]—[/]",
}


def print_doctor(results: list[CheckResult]) -> None:
    table = Table(show_header=True, header_style="bold", show_lines=False)
    table.add_column("Проверка", no_wrap=True)
    table.add_column("Статус", no_wrap=True)
    table.add_column("Подробности", overflow="fold")
    group = None
    for result in results:
        if result.group != group:
            if group is not None:
                table.add_section()
            table.add_row(f"[bold]{escape(result.group)}[/]", "", "")
            group = result.group
        table.add_row("  " + escape(result.name), _STATUS[result.status], escape(result.detail))
    console.print(table)

    todo = [r for r in results if r.hint and r.status != "ok"]
    if todo:
        console.print("\n[bold]Что сделать:[/]")
        for result in todo:
            console.print(f"  • [bold]{escape(result.name)}[/]: {_indent(escape(result.hint or ''), 4)}")

    fails = sum(r.status == "fail" for r in results)
    warns = sum(r.status == "warn" for r in results)
    console.print()
    if fails:
        extra = f", предупреждений: {warns}" if warns else ""
        console.print(f"[red]Проблем, которые мешают работе: {fails}[/]{extra}")
    elif warns:
        console.print(f"[yellow]Работать можно, но есть предупреждения: {warns}[/]")
    else:
        console.print("[green]Всё готово к работе.[/]")


# --- config ---------------------------------------------------------------------------


def print_config(config_yaml: str, path: Path | None, overridden: dict[str, Any]) -> None:
    if path is not None:
        console.print(f"Файл конфига: [bold]{escape(str(path.resolve()))}[/]")
    else:
        console.print(
            "Файл конфига не найден — используются значения по умолчанию. "
            "Создать файл с комментариями: [bold]clipper config --init[/]"
        )
    if overridden:
        console.print("Из командной строки: " + escape(", ".join(overridden)))
    console.print()
    console.print(Syntax(config_yaml, "yaml", theme="ansi_dark", background_color="default", word_wrap=True))


# --- download -------------------------------------------------------------------------


def print_source(source: SourceInfo, work_dir: Path) -> None:
    """Итог этапа загрузки: где видео, какое оно и мини-график heatmap."""
    audio = "есть звук" if source.has_audio else "[yellow]без звука[/]"
    fps = f"{source.fps:g} к/с" if source.fps else "? к/с"
    rows = [
        ("Видео", escape(source.video)),
        ("Название", escape(source.title)),
        ("Длительность", f"{_clock(source.duration)}   {source.width}×{source.height}, {fps}, {audio}"),
        ("Рабочая папка", escape(str(work_dir))),
    ]
    if source.heatmap:
        peak = heatmap_peak(source.heatmap)
        rows.append(
            (
                "Heatmap",
                f"[green]есть[/] — {len(source.heatmap)} точек, пик на {_clock(peak.start)}–{_clock(peak.end)}",
            )
        )
    elif source.kind == "file":
        rows.append(("Heatmap", "нет — у локального файла его не бывает; моменты выберет режим keywords"))
    elif source.kind == "url":
        rows.append(("Heatmap", "нет — он бывает только у видео YouTube; моменты выберет режим keywords"))
    else:
        rows.append(
            (
                "Heatmap",
                "[yellow]нет[/] — YouTube не показывает «Самые популярные фрагменты» для этого видео "
                "(обычно у новых или малопросматриваемых); моменты выберет режим keywords",
            )
        )
    console.print()
    for label, value in rows:
        console.print(f"[bold]{label + ':':<15}[/]{value}")
    if source.heatmap:
        console.print()
        width = max(20, min(len(source.heatmap), console.width - 4))
        for line in heatmap_chart(source.heatmap, source.duration, width):
            console.print(Text("  ") + line)


def heatmap_chart(points: list[HeatPoint], duration: float, width: int, height: int = 4) -> list[Text]:
    """Столбиковый график heatmap из символов █ и ▄ (есть во всех шрифтах Windows) + шкала времени."""
    profile = heatmap_profile(points, width, duration)
    top = max(profile) or 1.0
    levels = [0 if v <= 0 else max(1, round(v / top * height * 2)) for v in profile]
    peak_column = levels.index(max(levels))
    lines: list[Text] = []
    for row in range(height):
        floor = (height - 1 - row) * 2
        line = Text()
        for column, level in enumerate(levels):
            char = "█" if level >= floor + 2 else "▄" if level == floor + 1 else " "
            line.append(char, style="yellow" if column == peak_column else "cyan")
        lines.append(line)
    end = duration or points[-1].end  # подписи — по длительности видео, как в строке выше
    left, middle, right = _clock(0), _clock(end / 2), _clock(end)
    gap = max(width - len(left) - len(middle) - len(right), 2)
    axis = left + " " * (gap // 2) + middle + " " * (gap - gap // 2) + right
    lines.append(Text(axis, style="dim"))
    return lines


# --- transcribe -----------------------------------------------------------------------


def print_transcript(
    transcript: Transcript, work_dir: Path, requested: tuple[float, float], duration: float, preview: int = 6
) -> None:
    """Итог распознавания: язык, объём, файлы и первые фразы запрошенного отрезка."""
    start, end = requested
    whole = start <= 0.5 and end >= duration - 0.5
    segments = [s for s in transcript.segments if s.end > start and s.start < end]
    words = sum(len(s.words) for s in segments)
    span = "всё видео" if whole else f"{_clock(start)}–{_clock(end)}"
    rows = [
        ("Язык", escape(transcript.language or "?")),
        ("Отрезок", f"{span} ({_clock(end - start)})"),
        ("Распознано", f"{plural(len(segments), 'фраза', 'фразы', 'фраз')}, {plural(words, 'слово', 'слова', 'слов')}"),
        ("Кэш", escape(str(work_dir / TRANSCRIPT_FILENAME))),
        ("Субтитры", escape(str(work_dir / SRT_FILENAME))),
    ]
    console.print()
    for label, value in rows:
        console.print(f"[bold]{label + ':':<12}[/]{value}")
    if segments:
        console.print()
        for segment in segments[:preview]:
            console.print(f"  [dim]{_clock(segment.start)}[/]  {escape(segment.text)}")
        if len(segments) > preview:
            console.print(f"  [dim]… и ещё {plural(len(segments) - preview, 'фраза', 'фразы', 'фраз')}[/]")
    else:
        console.print("\n[yellow]Речь не найдена[/] — на этом отрезке тишина или только музыка.")
    console.print(
        "\n[dim]Проверка: откройте видео в VLC или MPC-HC и перетащите в окно файл transcript.srt — "
        "фразы должны совпадать с речью по времени.[/]"
    )


# --- analyze / render --------------------------------------------------------------------


def print_analysis(project: Project, path: Path, next_step: bool = True) -> None:
    """Таблица найденных клипов и путь к project.json."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", justify="right")
    table.add_column("Время", no_wrap=True)
    table.add_column("Длина", justify="right", no_wrap=True)
    table.add_column("Почему", overflow="fold")
    table.add_column("Начало речи", overflow="ellipsis", no_wrap=True, max_width=48)
    for clip in project.clips:
        words = " ".join(w.text for w in clip.spoken_words()[:10])
        table.add_row(
            str(clip.id),
            f"{_clock(clip.start)}–{_clock(clip.end)}",
            f"{clip.duration:.0f} с",
            escape(clip.reason),
            escape(words) or "[dim](без речи)[/]",
        )
    console.print()
    console.print(table)
    console.print(f"[bold]Проект:[/] {escape(str(path))}")
    if next_step:
        console.print(
            "[dim]Можно поправить границы (start/end), выключить клип (enabled: false) или исправить слова, "
            "затем: clipper render[/]"
        )


def print_render(results: list[RenderResult], folder: Path) -> None:
    """Итог рендера: какие клипы готовы, какие — нет и почему."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("id", justify="right")
    table.add_column("Файл")
    table.add_column("Длина", justify="right")
    cuts = any(r.removed > 0.05 for r in results)
    if cuts:
        table.add_column("Вырезано", justify="right")
    table.add_column("Статус", overflow="fold")
    for result in results:
        if result.path is not None:
            row = [str(result.clip_id), escape(result.path.name), f"{result.duration:.0f} с"]
            if cuts:
                fillers = f", паразиты: {result.fillers}" if result.fillers else ""
                row.append(f"{result.removed:.1f} с{fillers}" if result.removed > 0.05 else "—")
            table.add_row(*row, "[green]готово[/]")
        else:
            table.add_row(
                str(result.clip_id), "—", "—", *(["—"] if cuts else []), f"[red]ошибка:[/] {escape(result.error or '')}"
            )
    console.print()
    console.print(table)
    done = sum(r.path is not None for r in results)
    color = "green" if done == len(results) else "yellow"
    console.print(f"[{color}]Готово клипов: {done} из {len(results)}[/] — {escape(str(folder))}")


# --- Форматирование ------------------------------------------------------------------


def _mb(size: float | None) -> str:
    return f"{(size or 0) / 1_000_000:.1f}"


def _clock(seconds: float | None) -> str:
    total = int(seconds or 0)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} с"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes} мин {secs:02d} с"


def _indent(text: str, spaces: int = 2) -> str:
    return text.replace("\n", "\n" + " " * spaces)
