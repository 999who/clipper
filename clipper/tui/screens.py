"""Экраны TUI: главное меню, проекты, поиск моментов, рендер, результаты, проверка окружения."""

from datetime import datetime
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import DataTable, Footer, Header, Label, OptionList, Static
from textual.widgets.option_list import Option

from clipper.core import pipeline
from clipper.core.render import RenderResult
from clipper.tui.progress import ProgressScreen
from clipper.tui.settings import ANALYZE, RENDER, Setting, cli_command
from clipper.tui.system import open_path
from clipper.tui.widgets import Hint, SettingsList

SOURCE = Setting("source", "Ссылка на YouTube или файл", "text",
                 help="Вставьте ссылку или путь к видео (файл можно перетащить в окно терминала).")  # fmt: skip


def _quoted(text: str) -> str:
    return f'"{text}"' if text else '"…"'


def _show_help(screen: Screen[Any], event: "SettingsList.Help") -> None:
    screen.query_one("#help", Hint).update(Text(event.text, style="italic"))


# --- главное меню ---------------------------------------------------------------------------


class HomeScreen(Screen[Any]):
    BINDINGS = [Binding("q,й", "app.quit", "Выход")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(
            Text.assemble(
                ("clipper", "bold magenta"),
                (" — нарезка вертикальных клипов с субтитрами\n", "bold"),
                ("Стрелки — выбрать, Enter — открыть, Esc — назад, q — выход.", "dim"),
            ),
            id="intro",
        )
        yield OptionList(id="menu")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "главное меню"
        self._menu()
        self.query_one("#menu", OptionList).focus()

    def on_screen_resume(self) -> None:
        self._menu()

    def _menu(self) -> None:
        self.projects = pipeline.list_projects(self.app.store.config())
        options: list[Option | None] = [Option(Text("▶ Новое видео — найти моменты", style="bold"), id="new")]
        if self.projects:
            last = self.projects[0]
            options.append(Option(f"Продолжить: {last.title} — клипов {last.enabled} из {last.clips}", id="last"))
            options.append(Option(f"Все проекты ({len(self.projects)})", id="projects"))
        options += [None, Option("Проверка окружения", id="doctor"), Option("Выход", id="quit")]
        menu = self.query_one("#menu", OptionList)
        menu.set_options(options)
        menu.highlighted = 0

    @on(OptionList.OptionSelected, "#menu")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        from clipper.tui.project import ProjectScreen

        choice = event.option.id
        if choice == "new":
            self.app.push_screen(AnalyzeScreen())
        elif choice == "last":
            self.app.push_screen(ProjectScreen(self.projects[0].path))
        elif choice == "projects":
            self.app.push_screen(ProjectsScreen())
        elif choice == "doctor":
            self.app.push_screen(DoctorScreen())
        elif choice == "quit":
            self.app.exit()


class ProjectsScreen(Screen[Any]):
    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Проекты", classes="screen-title")
        yield OptionList(id="projects")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "проекты"
        self.projects = pipeline.list_projects(self.app.store.config())
        options = []
        for index, item in enumerate(self.projects):
            when = datetime.fromtimestamp(item.modified).strftime("%d.%m %H:%M")
            text = Text.assemble(
                (item.title, "bold"),
                (f"   клипов {item.enabled} из {item.clips}, {item.seconds:.0f} с   {when}", "dim"),
            )
            options.append(Option(text, id=str(index)))
        projects = self.query_one("#projects", OptionList)
        projects.set_options(options or [Option("Проектов пока нет", disabled=True)])
        projects.focus()

    @on(OptionList.OptionSelected, "#projects")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        from clipper.tui.project import ProjectScreen

        self.app.switch_screen(ProjectScreen(self.projects[int(event.option.id or 0)].path))


# --- поиск моментов ------------------------------------------------------------------------


class AnalyzeScreen(Screen[Any]):
    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, Any] = {"source": ""}

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Новое видео: где искать интересные моменты", classes="screen-title")
        yield SettingsList(self.app.store, ANALYZE, [("run", "▶ Найти моменты")], top=[SOURCE], values=self.values,
                           id="settings")  # fmt: skip
        yield Hint("", id="help")
        yield Hint("", id="command")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "поиск моментов"
        self.query_one("#settings", SettingsList).focus()
        self._command()

    def _command(self) -> None:
        store = self.app.store
        command = cli_command("analyze", ANALYZE, store.config(), store.base, (_quoted(self.values["source"]),))
        self.query_one("#command", Hint).update(Text("То же в командной строке: " + command, style="dim"))

    @on(SettingsList.Changed)
    def _changed(self, event: SettingsList.Changed) -> None:
        self._command()

    @on(SettingsList.Help)
    def _help(self, event: SettingsList.Help) -> None:
        _show_help(self, event)

    @on(SettingsList.Action)
    def _run(self, event: SettingsList.Action) -> None:
        source = (self.values.get("source") or "").strip()
        if not source:
            self.app.notify("Сначала укажите ссылку или файл (первая строка).", severity="warning")
            settings = self.query_one("#settings", SettingsList)
            settings.highlighted = 0
            return
        cfg = self.app.store.config()
        job = ProgressScreen(
            "Поиск моментов", lambda reporter: pipeline.analyze(source, cfg, reporter, output=cfg.paths.output)
        )
        self.app.push_screen(job, self._done)

    def _done(self, result: Any) -> None:
        if not result:
            return
        from clipper.tui.project import ProjectScreen

        _, path = result
        self.app.switch_screen(ProjectScreen(path))


# --- рендер -----------------------------------------------------------------------------------


class RenderScreen(Screen[Any]):
    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def __init__(self, project_path: Path, enabled: int) -> None:
        super().__init__()
        self.project_path = project_path
        self.enabled = enabled

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Как нарезать клипы", classes="screen-title")
        yield SettingsList(self.app.store, RENDER, [("render", f"▶ Нарезать клипы: {self.enabled}")], id="settings")
        yield Hint("", id="help")
        yield Hint("", id="command")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "рендер"
        self.query_one("#settings", SettingsList).focus()
        self._command()

    def _command(self) -> None:
        store = self.app.store
        extra = ("--project", self.project_path.parent.name)
        command = cli_command("render", RENDER, store.config(), store.base, extra)
        self.query_one("#command", Hint).update(Text("То же в командной строке: " + command, style="dim"))

    @on(SettingsList.Changed)
    def _changed(self, event: SettingsList.Changed) -> None:
        self._command()

    @on(SettingsList.Help)
    def _help(self, event: SettingsList.Help) -> None:
        _show_help(self, event)

    @on(SettingsList.Action)
    def _run(self, event: SettingsList.Action) -> None:
        cfg = self.app.store.config()
        path = str(self.project_path)
        job = ProgressScreen(
            "Нарезка клипов", lambda reporter: pipeline.render(cfg, reporter, path, output=cfg.paths.output)
        )
        self.app.push_screen(job, self._rendered)

    def _rendered(self, result: Any) -> None:
        if not result:
            return
        _, folder, results = result
        self.app.push_screen(ResultsScreen(folder, results), self._after_results)

    def _after_results(self, choice: Any) -> None:
        if choice == "home":
            while len(self.app.screen_stack) > 2:
                self.app.pop_screen()
        elif choice == "project":
            self.app.pop_screen()


class ResultsScreen(Screen[Any]):
    BINDINGS = [Binding("escape", "back", "К проекту")]

    def __init__(self, folder: Path, results: list[RenderResult]) -> None:
        super().__init__()
        self.folder = folder
        self.results = results

    def compose(self) -> ComposeResult:
        done = sum(r.path is not None for r in self.results)
        yield Header()
        yield Label(f"Готово клипов: {done} из {len(self.results)} — {self.folder}", classes="screen-title")
        yield DataTable(id="results", cursor_type="row", zebra_stripes=True)
        yield Label("Enter на клипе — открыть в плеере.", classes="dialog-help")
        yield OptionList(
            Option("Открыть папку с клипами", id="folder"),
            Option("← К проекту", id="project"),
            Option("В главное меню", id="home"),
            id="actions",
        )
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "результат"
        table = self.query_one("#results", DataTable)
        table.add_columns("№", "Файл", "Длина", "Кадр", "Вырезано", "Статус")
        for result in self.results:
            if result.path is not None:
                cut = f"{result.removed:.1f} с" if result.removed > 0.05 else "—"
                table.add_row(str(result.clip_id), result.path.name, f"{result.duration:.0f} с", result.frame or "—",
                              cut, Text("готово", style="green"), key=str(result.clip_id))  # fmt: skip
            else:
                table.add_row(str(result.clip_id), "—", "—", "—", "—", Text(f"ошибка: {result.error}", style="red"),
                              key=str(result.clip_id))  # fmt: skip
        table.focus()

    @on(DataTable.RowSelected, "#results")
    def _open(self, event: DataTable.RowSelected) -> None:
        result = self.results[event.cursor_row]
        if result.path is not None:
            open_path(result.path)

    @on(OptionList.OptionSelected, "#actions")
    def _action(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id == "folder":
            open_path(self.folder)
        else:
            self.dismiss(event.option.id)

    def action_back(self) -> None:
        self.dismiss("project")


# --- проверка окружения ----------------------------------------------------------------------

STATUS = {"ok": ("ОК", "green"), "warn": ("внимание", "yellow"), "fail": ("ошибка", "red"), "info": ("—", "dim")}


class DoctorScreen(Screen[Any]):
    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Проверка окружения", classes="screen-title")
        yield DataTable(id="checks", cursor_type="row", zebra_stripes=True)
        yield Static("Проверяю…", id="hints")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "clipper doctor"
        table = self.query_one("#checks", DataTable)
        table.add_columns("Группа", "Проверка", "Статус", "Подробности")
        table.focus()
        self._run()

    @work(thread=True, exclusive=True)
    def _run(self) -> None:
        from clipper.core.doctor import run_doctor

        results = run_doctor(config_path=self.app.store.path)
        self.app.call_from_thread(self._show, results)

    def _show(self, results: list[Any]) -> None:
        table = self.query_one("#checks", DataTable)
        hints = Text()
        for check in results:
            label, style = STATUS.get(check.status, (check.status, ""))
            table.add_row(check.group, check.name, Text(label, style=style), check.detail)
            if check.hint and check.status in ("warn", "fail"):
                hints.append(f"{check.name}: ", style="bold")
                hints.append(check.hint + "\n")
        problems = sum(c.status == "fail" for c in results)
        summary = (
            Text("Всё в порядке.\n", style="green") if not problems else Text(f"Проблем: {problems}\n", style="red")
        )
        self.query_one("#hints", Static).update(summary + hints)
