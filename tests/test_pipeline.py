"""Этап 3 целиком: analyze → project.json → render, на настоящем ffmpeg и фальшивом Whisper."""

import json
from pathlib import Path
from typing import NamedTuple

import pytest
from helpers import make_video, needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import pipeline
from clipper.core import render as render_module
from clipper.core import transcribe as tr
from clipper.core.config import load_config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import probe
from clipper.core.models import HeatPoint, SourceInfo, load_project, load_transcript

DURATION = 90.0


class FakeWord(NamedTuple):
    start: float
    end: float
    word: str
    probability: float = 0.9


class FakeSegment(NamedTuple):
    start: float
    end: float
    text: str
    words: list
    avg_logprob: float = -0.2
    no_speech_prob: float = 0.01


class FakeInfo(NamedTuple):
    language: str
    language_probability: float


class Speaker:
    """Фальшивый Whisper: предложения по 2 с из 4 слов («раз два три конец.»).

    `special` — {секунда от начала куска: слово}, чтобы вставить ключевые слова.
    """

    def __init__(self, special: dict[float, str] | None = None):
        self.special = special or {}
        self.seconds: list[float] = []

    def transcribe(self, audio, **kwargs):
        seconds = len(audio) / tr.SAMPLE_RATE
        self.seconds.append(round(seconds, 1))

        def segments():
            t = 0.0
            while t + 2 <= seconds:
                texts = ["раз", "два", "три", "конец."]
                words = [
                    FakeWord(t + 0.5 * i, t + 0.5 * i + 0.4, " " + self.special.get(round(t + 0.5 * i, 1), text))
                    for i, text in enumerate(texts)
                ]
                yield FakeSegment(t, t + 1.9, "".join(w.word for w in words), words)
                t += 2

        return segments(), FakeInfo("ru", 0.99)


def peak_heatmap(peaks: dict[int, float]) -> list[HeatPoint]:
    step = DURATION / 100
    return [HeatPoint(i * step, (i + 1) * step, peaks.get(i, 0.1)) for i in range(100)]


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Видео 90 с «с YouTube» с heatmap: пики на 20 и 70 с."""
    video = make_video(tmp_path / "work" / "vid" / "source.mp4", seconds=DURATION, size="320x240")
    source = SourceInfo(
        "vid", "youtube", "https://youtu.be/vid", str(video), "Тест", DURATION, 320, 240, 25.0, True,
        peak_heatmap({22: 1.0, 77: 0.8}),
    )  # fmt: skip
    monkeypatch.setattr(pipeline, "prepare_source", lambda text, cfg, reporter: source)
    speaker = Speaker()
    monkeypatch.setattr(tr, "load_whisper", lambda cfg, reporter: speaker)
    cfg = load_config(
        None,
        {"paths.workdir": str(tmp_path / "work"), "paths.output": str(tmp_path / "out"), "select.min_len": 10,
         "select.max_len": 20, "select.clips": 2, "render.encoder": "x264"},
    )  # fmt: skip
    return cfg, source, speaker


def at_word_boundary(t: float, words) -> bool:
    return not any(w.start + 1e-6 < t < w.end - 1e-6 for w in words)


@needs_ffmpeg
def test_analyze_by_heatmap(setup):
    cfg, source, speaker = setup
    project, path = pipeline.analyze("https://youtu.be/vid", cfg, Reporter())

    assert path == Path(cfg.paths.workdir).resolve() / "vid" / "project.json"
    assert [c.id for c in project.clips] == [1, 2]
    first, second = project.clips
    assert first.start < 20.25 < first.end and second.start < 69.75 < second.end
    for clip in project.clips:
        assert 10 <= clip.duration <= 20 + 1e-6
        assert at_word_boundary(clip.start, clip.words) and at_word_boundary(clip.end, clip.words)
        assert clip.words and clip.words[0].start < clip.start  # слова с запасом за границами
    # Распознавалось не всё видео, а окна вокруг пиков.
    assert sum(speaker.seconds) < DURATION * 0.7
    assert load_transcript(path.parent).ranges != [(0.0, DURATION)]
    assert load_project(path).clips[0].reason == "пик heatmap 1.00"


@needs_ffmpeg
def test_analyze_without_heatmap_suggests_keywords(setup, monkeypatch):
    cfg, source, _ = setup
    source.heatmap = None
    with pytest.raises(ClipperError) as err:
        pipeline.analyze("https://youtu.be/vid", cfg, Reporter())
    assert "--mode-select keywords" in err.value.hint


@needs_ffmpeg
def test_analyze_by_keywords_and_backup(setup, monkeypatch):
    cfg, source, _ = setup
    monkeypatch.setattr(tr, "load_whisper", lambda cfg, reporter: Speaker({50.0: "победа!"}))
    cfg.select.mode = "keywords"
    cfg.select.keywords = ["побед*"]
    project, path = pipeline.analyze("https://youtu.be/vid", cfg, Reporter())
    (clip,) = project.clips
    assert clip.start < 50 < clip.end
    assert clip.reason == "ключевые слова: «побед*»"
    assert load_transcript(path.parent).ranges == [(0.0, DURATION)]  # для keywords — всё видео

    cfg.select.keywords = ["нет такого слова"]
    with pytest.raises(ClipperError, match="не прозвучало"):
        pipeline.analyze("https://youtu.be/vid", cfg, Reporter())

    cfg.select.keywords = ["побед*"]
    events = []
    pipeline.analyze("https://youtu.be/vid", cfg, Reporter(events.append))
    assert (path.parent / "project.prev.json").is_file()  # ручные правки не теряются
    assert any("project.prev.json" in getattr(e, "text", "") for e in events)


@needs_ffmpeg
def test_render_clips_after_hand_edit(setup):
    cfg, source, _ = setup
    _, path = pipeline.analyze("https://youtu.be/vid", cfg, Reporter())
    data = json.loads(path.read_text(encoding="utf-8"))
    data["clips"][0]["start"], data["clips"][0]["end"] = "00:00:10.000", "00:00:14.000"
    data["clips"][1]["enabled"] = False
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    project, folder, results = pipeline.render(cfg, Reporter())
    assert folder == Path(cfg.paths.output).resolve() / "vid"
    (result,) = results  # второй клип выключен
    assert result.path == folder / "clip_01.mp4" and result.error is None
    media = probe(str(result.path), "ffprobe")
    assert media.duration == pytest.approx(4.0, abs=0.15)
    assert (media.width, media.height, media.has_audio, media.video_codec) == (1080, 1920, True, "h264")
    assert not list(folder.glob("*.part.mp4"))


@needs_ffmpeg
def test_render_selected_clips_and_isolated_failures(setup, monkeypatch):
    cfg, source, _ = setup
    pipeline.analyze("https://youtu.be/vid", cfg, Reporter())
    real = render_module.render_clip

    def flaky(clip, *args, **kwargs):
        if clip.id == 1:
            raise ClipperError("ffmpeg упал")
        return real(clip, *args, **kwargs)

    monkeypatch.setattr(render_module, "render_clip", flaky)
    _, folder, results = pipeline.render(cfg, Reporter())
    assert [(r.clip_id, r.error) for r in results] == [(1, "ffmpeg упал"), (2, None)]
    assert (folder / "clip_02.mp4").is_file()

    with pytest.raises(ClipperError, match=r"нет клипов с id \[7\]"):
        pipeline.render(cfg, Reporter(), only={7})


def test_find_project(tmp_path):
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    with pytest.raises(ClipperError, match="нет ни одного проекта"):
        pipeline.find_project(cfg)
    for name in ("old", "new"):
        (tmp_path / "work" / name).mkdir(parents=True)
        (tmp_path / "work" / name / "project.json").write_text("{}", encoding="utf-8")
    import os

    os.utime(tmp_path / "work" / "old" / "project.json", (1, 1))
    assert pipeline.find_project(cfg).parent.name == "new"  # самый свежий
    assert pipeline.find_project(cfg, "old").parent.name == "old"  # по id
    assert pipeline.find_project(cfg, str(tmp_path / "work" / "old")).parent.name == "old"  # по папке
    with pytest.raises(ClipperError, match="не найден"):
        pipeline.find_project(cfg, "missing")


def test_broken_project_error_mentions_backup(tmp_path):
    path = tmp_path / "project.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ClipperError) as err:
        pipeline.open_project(path)
    assert "строка 1" in err.value.message and "project.prev.json" in err.value.hint


@needs_ffmpeg
def test_cli_analyze_render_run(setup, monkeypatch):
    from clipper import console as console_module

    monkeypatch.setattr(console_module.console, "width", 200)  # таблица без переносов
    cfg, _, _ = setup
    runner = CliRunner()
    common = ["--set", f"paths.workdir={cfg.paths.workdir}", "--set", f"paths.output={cfg.paths.output}"]
    result = runner.invoke(
        cli.app, ["analyze", "https://youtu.be/vid", "--clips", "1", "--min-len", "10", "--max-len", "20", *common]
    )
    assert result.exit_code == 0, result.output
    assert "пик heatmap 1.00" in result.output and "project.json" in result.output

    result = runner.invoke(cli.app, ["render", "--clip", "1", "--encoder", "x264", *common])
    assert result.exit_code == 0, result.output
    assert "Готово клипов: 1 из 1" in result.output

    bad = runner.invoke(cli.app, ["render", "--clip", "один", *common])
    assert bad.exit_code == 1 and "номера клипов" in bad.output

    result = runner.invoke(
        cli.app, ["run", "https://youtu.be/vid", "--clips", "2", "--min-len", "10", "--max-len", "20", *common]
    )
    assert result.exit_code == 0, result.output
    assert "Готово клипов: 2 из 2" in result.output
