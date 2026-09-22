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


def test_main_without_arguments_prints_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["clipper"])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert "doctor" in capsys.readouterr().out


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
