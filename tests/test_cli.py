import sys
from io import StringIO

import pytest
from rich.console import Console
from typer.testing import CliRunner

from clipper import cli
from clipper.console import ConsoleSink
from clipper.core.events import Reporter

runner = CliRunner()


def test_help_lists_commands():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.output
    assert "config" in result.output


def test_main_without_arguments_opens_tui(monkeypatch):
    import clipper.tui

    started = {}
    monkeypatch.setattr(clipper.tui, "run_tui", lambda path, overrides: started.update(path=path, overrides=overrides))
    monkeypatch.setattr(sys, "argv", ["clipper"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert started == {"path": None, "overrides": {}}


def test_config_shows_overrides():
    result = runner.invoke(cli.app, ["config", "--set", "select.clips=3", "--set", "reframe.aspect=1:1"])
    assert result.exit_code == 0, result.output
    assert "clips: 3" in result.output
    assert "aspect: '1:1'" in result.output


def test_config_error_is_friendly():
    result = runner.invoke(cli.app, ["config", "--set", "select.clipz=3"])
    assert result.exit_code == 1
    assert "select.clips" in result.output
    assert "Traceback" not in result.output


def test_config_init_creates_file_once(tmp_path):
    result = runner.invoke(cli.app, ["config", "--init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "clipper.yaml").is_file()
    again = runner.invoke(cli.app, ["config", "--init"])
    assert again.exit_code == 1
    assert "уже существует" in again.output


def test_config_reads_file_from_current_folder(tmp_path):
    (tmp_path / "clipper.yaml").write_text("select:\n  clips: 9\n", encoding="utf-8")
    result = runner.invoke(cli.app, ["config"])
    assert result.exit_code == 0, result.output
    assert "clips: 9" in result.output


def test_doctor_prints_report():
    result = runner.invoke(cli.app, ["doctor"])
    assert result.exit_code in (0, 1), result.output
    for row in ("Python", "ffmpeg", "yt-dlp", "Видеокарта", "Файл конфига"):
        assert row in result.output


def test_console_sink_renders_events():
    out = StringIO()
    with ConsoleSink(Console(file=out, width=100)) as sink:
        reporter = Reporter(sink)
        with reporter.stage("download", "Загрузка видео", total=2_000_000, unit="bytes") as stage:
            stage.update(1_000_000, message="половина")
            stage.result = "готово"
        reporter.warning("у видео нет heatmap")
    text = out.getvalue()
    assert "Загрузка видео — готово" in text
    assert "Внимание: у видео нет heatmap" in text


def fake_youtube_source(heatmap):
    from clipper.core.models import SourceInfo

    return SourceInfo(
        id="dQw4w9WgXcQ",
        kind="youtube",
        input="https://youtu.be/dQw4w9WgXcQ",
        video="C:/work/dQw4w9WgXcQ/source.mp4",
        title="Тестовое видео",
        duration=200.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        heatmap=heatmap,
    )


def test_print_source_shows_heatmap_chart(monkeypatch):
    from pathlib import Path

    from clipper import console as console_module
    from clipper.core.models import HeatPoint

    values = [0.2] * 100
    values[60] = 1.0  # пик на 2:00
    heatmap = [HeatPoint(i * 2.0, (i + 1) * 2.0, value) for i, value in enumerate(values)]
    out = StringIO()
    monkeypatch.setattr(console_module, "console", Console(file=out, width=120))
    console_module.print_source(fake_youtube_source(heatmap), Path("work/dQw4w9WgXcQ"))
    text = out.getvalue()
    assert "Длительность:  03:20   1920×1080, 30 к/с, есть звук" in text
    assert "пик на 02:00–02:02" in text
    chart = [line for line in text.splitlines() if "█" in line or "▄" in line]
    assert len(chart) == 4  # 4 строки столбиков
    assert "00:00" in text and "01:40" in text and "03:20" in text


def test_heatmap_chart_levels():
    from clipper.console import heatmap_chart
    from clipper.core.models import HeatPoint

    points = [HeatPoint(0, 1, 0.0), HeatPoint(1, 2, 0.5), HeatPoint(2, 3, 1.0), HeatPoint(3, 4, 0.1)]
    lines = [line.plain for line in heatmap_chart(points, 4.0, width=4, height=2)]
    assert lines[0] == "  █ "  # верхний ряд: только максимум
    assert lines[1] == " ██▄"  # 0.5 — полный нижний ряд, 0.1 — полблока
    assert lines[2].startswith("00:00") and lines[2].endswith("00:04")


def test_print_source_without_heatmap(monkeypatch):
    from pathlib import Path

    from clipper import console as console_module

    out = StringIO()
    monkeypatch.setattr(console_module, "console", Console(file=out, width=200))
    console_module.print_source(fake_youtube_source(None), Path("work/x"))
    assert "Самые популярные фрагменты" in out.getvalue()
    assert "keywords" in out.getvalue()
