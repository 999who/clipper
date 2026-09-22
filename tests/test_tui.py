"""TUI: настройки, навигация стрелками и Enter, прогресс с отменой, правка проекта, рендер.

Приложение запускается без терминала (App.run_test), ядро подменяется там, где
нужны видео, сеть или видеокарта.
"""

import asyncio
import json
import threading
from pathlib import Path

import pytest

pytest.importorskip("textual")

from clipper.core import pipeline  # noqa: E402
from clipper.core.errors import ClipperError  # noqa: E402
from clipper.core.models import Clip, Project, SourceInfo, Word, save_project  # noqa: E402
from clipper.core.render import RenderResult  # noqa: E402
from clipper.tui import settings as st  # noqa: E402
from clipper.tui.app import ClipperApp  # noqa: E402
from clipper.tui.progress import ProgressScreen  # noqa: E402
from clipper.tui.project import ClipModal, ProjectScreen  # noqa: E402
from clipper.tui.screens import AnalyzeScreen, HomeScreen, RenderScreen, ResultsScreen  # noqa: E402
from clipper.tui.widgets import ChoiceModal, InputModal, heatmap_text  # noqa: E402

SIZE = (110, 40)


def run(coro):
    return asyncio.run(coro)


def make_store(tmp_path: Path, **overrides) -> st.SettingsStore:
    return st.SettingsStore(None, {"paths.workdir": str(tmp_path / "work"), **overrides})


def make_project(tmp_path: Path) -> Path:
    source = SourceInfo("vid", "file", "v.mp4", str(tmp_path / "v.mp4"), "Моё видео", 600.0, 1920, 1080, 30.0, True)
    clips = [
        Clip(1, 60.0, 90.0, reason="пик heatmap 1.00", words=[Word("Привет", 61.0, 61.5)]),
        Clip(2, 300.0, 330.0, reason="пик heatmap 0.70"),
    ]
    work = tmp_path / "work" / "vid"
    save_project(work, Project(source, "heatmap", [], "ru", "", clips))
    return work / "project.json"


# --- настройки (без экрана) ---------------------------------------------------------------


def test_store_validates_like_cli(tmp_path):
    store = make_store(tmp_path)
    assert store.try_set("select.clips", 12) is None and store.value("select.clips") == 12
    error = store.try_set("select.max_len", 5)
    assert error and "максимальная длина" in error
    assert store.value("select.max_len") == 60  # плохое значение не применилось
    assert store.try_set("reframe.aspect", "4:3") is not None


def test_cli_command_shows_only_changes(tmp_path):
    store = make_store(tmp_path)
    store.try_set("select.clips", 10)
    store.try_set("transcribe.verbatim", True)
    store.try_set("select.keywords", "да ладно, побед*")
    command = st.cli_command("analyze", st.ANALYZE, store.config(), store.base, ('"URL"',))
    assert command == 'clipper analyze "URL" --keywords "да ладно, побед*" --clips 10 --verbatim'
    store.try_set("audio.cut_pauses", True)
    store.try_set("reframe.detector", "mediapipe")
    render = st.cli_command("render", st.RENDER, store.config(), store.base)
    assert render == "clipper render --cut-pauses"  # detector в форме нет — в команде тоже


def test_values_for_display_and_input():
    lang = next(s for s in st.ANALYZE if isinstance(s, st.Setting) and s.key == "transcribe.language")
    assert st.show_value(lang, None) == "определять автоматически"
    clips = st.Setting("select.clips", "", "int")
    assert st.parse_input(clips, " 7 ") == 7
    keywords = st.Setting("select.keywords", "", "list")
    assert st.edit_text(keywords, ["а", "б"]) == "а, б" and st.parse_input(keywords, "  ") is None
    assert st.show_value(st.Setting("x", "", "float"), 0.6) == "0.6"


def test_heatmap_text_marks_clips():
    from clipper.core.models import HeatPoint

    points = [HeatPoint(i * 10.0, (i + 1) * 10.0, v) for i, v in enumerate([0.0, 0.5, 1.0, 0.2])]
    text = heatmap_text(points, 40.0, [(20.0, 30.0, True), (0.0, 5.0, False)], width=10).plain
    graph, marks = text.split("\n")
    assert graph == "  ▄▄▄██▂▂▂"
    assert marks == "△△   ▲▲▲  "


# --- экраны --------------------------------------------------------------------------------


def test_home_menu_and_analyze_form(tmp_path, monkeypatch):
    project_path = make_project(tmp_path)
    calls = {}

    def fake_analyze(source, cfg, reporter, **kwargs):
        with reporter.stage("transcribe", "Распознавание речи", total=2) as stage:
            stage.update(2)
            stage.result = "10 слов"
        calls["source"], calls["clips"] = source, cfg.select.clips
        return None, project_path

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)

    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, HomeScreen)
            menu = app.screen.query_one("#menu")
            assert "Продолжить: Моё видео" in str(menu.get_option("last").prompt)
            await pilot.press("enter")  # «Новое видео»
            assert isinstance(app.screen, AnalyzeScreen)

            await pilot.press("enter")  # ссылка → окно ввода
            assert isinstance(app.screen, InputModal)
            await pilot.press(*"D:/v.mp4", "enter")
            await pilot.press("down", "down", "down", "enter")  # «Сколько клипов»
            await pilot.press("1", "5", "enter")  # старое значение выделено — ввод его заменяет
            await pilot.pause()
            command = str(app.screen.query_one("#command").render())
            assert '"D:/v.mp4"' in command and "--clips 15" in command

            await pilot.press("down", "down", "down", "down", "down")  # к «▶ Найти моменты»
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ProjectScreen)
            assert calls == {"source": "D:/v.mp4", "clips": 15}

    run(scenario())


def test_input_modal_shows_validation_error(tmp_path):
    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            await pilot.press("enter", "down", "down", "down", "down", "enter")  # «Длина клипа от»
            assert isinstance(app.screen, InputModal)
            await pilot.press("9", "0", "enter")  # больше max_len
            await pilot.pause()
            assert isinstance(app.screen, InputModal)
            assert "максимальная длина" in str(app.screen.query_one("#error").render())
            await pilot.press("escape")
            assert app.store.value("select.min_len") == 20

    run(scenario())


def test_project_edit_toggle_and_save(tmp_path):
    project_path = make_project(tmp_path)

    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            app.push_screen(ProjectScreen(project_path))
            await pilot.pause()
            screen = app.screen
            assert "Клипов: 2, включено 2" in str(screen.query_one("#info").render())

            await pilot.press("down", "space")  # выключить клип 2
            await pilot.press("up", "enter")  # правка клипа 1
            assert isinstance(app.screen, ClipModal)
            await pilot.press("down", "down", "down", "enter")  # «Начало на 1 с раньше»
            await pilot.press("down", "down", "down", "down", "enter")  # «✓ Готово»
            await pilot.pause()
            assert screen.dirty and "несохранённые" in str(screen.query_one("#info").render())

            await pilot.press("s")
            await pilot.pause()
            assert not screen.dirty

    run(scenario())
    data = json.loads(project_path.read_text(encoding="utf-8"))
    first, second = data["clips"]
    assert first["start"] == "00:00:59.000" and second["enabled"] is False
    assert (project_path.parent / "project.prev.json").is_file()


def test_render_flow_and_results(tmp_path, monkeypatch):
    project_path = make_project(tmp_path)
    seen = {}

    def fake_render(cfg, reporter, project=None, only=None):
        seen["project"], seen["aspect"] = project, cfg.reframe.aspect
        reporter.warning("Паузы не найдены")
        folder = tmp_path / "out" / "vid"
        return (
            None,
            folder,
            [RenderResult(1, folder / "clip_01.mp4", duration=30.0), RenderResult(2, None, error="ffmpeg")],
        )

    monkeypatch.setattr(pipeline, "render", fake_render)

    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            app.push_screen(ProjectScreen(project_path))
            await pilot.pause()
            await pilot.press("r")
            await pilot.pause()
            assert isinstance(app.screen, RenderScreen)
            await pilot.press("down", "enter")  # «Формат»
            assert isinstance(app.screen, ChoiceModal)
            await pilot.press("down", "enter")  # 1:1
            await pilot.press("end", "enter")  # «▶ Нарезать клипы»
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert isinstance(app.screen, ProgressScreen)  # было предупреждение — ждём «Дальше»
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, ResultsScreen)
            table = app.screen.query_one("#results")
            assert table.row_count == 2
            await pilot.press("escape")  # к проекту
            await pilot.pause()
            assert isinstance(app.screen, ProjectScreen)

    run(scenario())
    assert seen == {"project": str(project_path), "aspect": "1:1"}


def test_progress_shows_errors_and_cancels(tmp_path):
    started = threading.Event()

    def failing(reporter):
        raise ClipperError("Видео не найдено", hint="Проверьте ссылку.")

    def long_job(reporter):
        started.set()
        while True:
            reporter.cancel.check()
            threading.Event().wait(0.01)

    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            results = []
            app.push_screen(ProgressScreen("Тест", failing), results.append)
            await app.workers.wait_for_complete()
            await pilot.pause()
            error = str(app.screen.query_one("#error").render())
            assert "Видео не найдено" in error and "Проверьте ссылку" in error
            await pilot.press("enter")  # «← Назад»
            await pilot.pause()
            assert results == [None]

            app.push_screen(ProgressScreen("Долго", long_job), results.append)
            await pilot.pause()
            started.wait(5)
            await pilot.press("escape")  # «Прервать работу?»
            await pilot.press("enter")  # «Да, прервать»
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert "Прервано" in str(app.screen.query_one("#error").render())

    run(scenario())


def test_cli_has_tui_command():
    from typer.testing import CliRunner

    from clipper import cli

    result = CliRunner().invoke(cli.app, ["tui", "--help"])
    assert result.exit_code == 0 and "стрелками и Enter" in result.output


def test_project_recount_reuses_search_and_transcript_settings(tmp_path, monkeypatch):
    from clipper.core.models import Transcript, save_transcript

    project_path = make_project(tmp_path)
    save_transcript(project_path.parent, Transcript("ru", {"language": "ru", "verbatim": True}, [], []))
    seen = {}

    def fake_analyze(source, cfg, reporter, **kwargs):
        seen.update(source=source, clips=cfg.select.clips, mode=cfg.select.mode, lang=cfg.transcribe.language,
                    verbatim=cfg.transcribe.verbatim)  # fmt: skip
        project = pipeline.open_project(project_path)
        project.clips = [Clip(i, i * 60.0, i * 60.0 + 30) for i in range(1, cfg.select.clips + 1)]
        save_project(project_path.parent, project)
        return project, project_path

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)

    async def scenario():
        app = ClipperApp(make_store(tmp_path))
        async with app.run_test(size=SIZE) as pilot:
            await pilot.pause()
            app.push_screen(ProjectScreen(project_path))
            await pilot.pause()
            screen = app.screen
            await pilot.press("c")
            assert isinstance(app.screen, InputModal)
            await pilot.press("0", "enter")  # не число клипов
            await pilot.pause()
            assert "хотя бы 1" in str(app.screen.query_one("#error").render())
            await pilot.press("backspace", "7", "enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen is screen
            assert "Клипов: 7" in str(screen.query_one("#info").render())
            assert app.store.value("select.clips") == 7

    run(scenario())
    assert seen == {"source": "v.mp4", "clips": 7, "mode": "heatmap", "lang": "ru", "verbatim": True}
