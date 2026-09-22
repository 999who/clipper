"""Этап 2. Настоящую модель в тестах не запускаем: её заменяет FakeWhisper
с тем же интерфейсом, что у faster_whisper.WhisperModel.transcribe."""

import wave
from pathlib import Path
from typing import NamedTuple

import pytest
from helpers import make_video, needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import transcribe as tr
from clipper.core.config import load_config
from clipper.core.errors import Cancelled, ClipperError, DependencyError
from clipper.core.events import Reporter, StageProgress
from clipper.core.ffmpeg import run_ffmpeg
from clipper.core.models import (
    Segment,
    SourceInfo,
    Transcript,
    Word,
    format_time,
    load_transcript,
    merge_ranges,
    parse_time,
    subtract_ranges,
)


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


class FakeWhisper:
    """Каждые 2 секунды звука — фраза « раз два» (время от начала переданного куска)."""

    def __init__(self, extra: list[FakeSegment] | None = None):
        self.calls: list[tuple[float, dict]] = []
        self.extra = extra or []

    def transcribe(self, audio, **kwargs):
        seconds = len(audio) / tr.SAMPLE_RATE
        self.calls.append((round(seconds, 2), kwargs))

        def segments():
            t = 0.0
            while t + 2 <= seconds + 1e-6:
                words = [FakeWord(t + 0.1, t + 0.6, " раз"), FakeWord(t + 0.8, t + 1.5, " два,")]
                yield FakeSegment(t, t + 2, " раз два,", words)
                t += 2
            yield from self.extra

        return segments(), FakeInfo("ru", 0.98)


def factory(engine: FakeWhisper):
    return lambda cfg, reporter: engine


@pytest.fixture
def video_source(tmp_path):
    video = make_video(tmp_path / "in.mp4", seconds=6)
    source = SourceInfo("in", "file", str(video), str(video), "in", 6.0, 320, 240, 25.0, True)
    work = tmp_path / "work" / "in"
    return source, work


# --- время и отрезки -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("90", 90), ("1:30", 90), ("1:02:03.5", 3723.5), ("00:02:03,400", 123.4), (12.5, 12.5), ("0", 0)],
)
def test_parse_time(text, seconds):
    assert parse_time(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", ["", "abc", "1:70", "-5", "1:2:3:4", "1::2"])
def test_parse_time_errors(text):
    with pytest.raises(ValueError):
        parse_time(text)


def test_format_time():
    assert format_time(3723.5) == "01:02:03.500"
    assert format_time(0.0004) == "00:00:00.000"


def test_merge_and_subtract_ranges():
    assert merge_ranges([(5, 8), (0, 2), (1, 3), (8, 9), (4, 4)]) == [(0, 3), (5, 9)]
    assert merge_ranges([(0, 2), (2.5, 3)], gap=1) == [(0, 3)]
    assert subtract_ranges([(0, 10)], [(2, 4), (6, 7)]) == [(0, 2), (4, 6), (7, 10)]
    assert subtract_ranges([(0, 10)], [(0, 9.5)], min_length=1) == []
    assert subtract_ranges([(0, 3), (5, 8)], []) == [(0, 3), (5, 8)]


def test_transcript_round_trip(tmp_path):
    transcript = Transcript(
        language="ru",
        settings={"model": "large-v3"},
        ranges=[(0.0, 4.0)],
        segments=[Segment(0.1, 1.5, "Привет, мир", (Word("Привет,", 0.1, 0.6, 0.9), Word("мир", 0.7, 1.5, 0.8)))],
    )
    assert Transcript.from_dict(transcript.to_dict()) == transcript
    assert [w.text for w in transcript.words_between(0.65, 2)] == ["мир"]


# --- распознавание ------------------------------------------------------------------------


@needs_ffmpeg
def test_whole_video_with_word_timestamps(video_source):
    source, work = video_source
    engine = FakeWhisper()
    events = []
    result = tr.transcribe_source(source, work, load_config(), Reporter(events.append), engine_factory=factory(engine))

    assert result.language == "ru"
    assert result.ranges == [(0.0, 6.0)]
    assert [s.text for s in result.segments] == ["раз два,"] * 3
    assert result.words[2] == Word("раз", 2.1, 2.6, 0.9)  # время исходного видео
    assert engine.calls[0][1]["word_timestamps"] is True
    assert engine.calls[0][1]["language"] is None  # автоопределение
    assert engine.calls[0][1]["vad_filter"] is True
    assert load_transcript(work) == result
    assert (work / "audio16k.wav").is_file()
    assert any(isinstance(e, StageProgress) and e.stage == "transcribe" for e in events)
    assert any("Язык речи: ru" in getattr(e, "text", "") for e in events)


@needs_ffmpeg
def test_only_missing_ranges_are_transcribed(video_source):
    source, work = video_source
    cfg = load_config(None, {"transcribe.language": "ru"})
    engine = FakeWhisper()
    tr.transcribe_source(source, work, cfg, Reporter(), [(0, 2)], engine_factory=factory(engine))
    result = tr.transcribe_source(source, work, cfg, Reporter(), [(1, 6)], engine_factory=factory(engine))
    assert [seconds for seconds, _ in engine.calls] == [2.0, 4.0]  # второй раз — только 2…6
    assert engine.calls[1][1]["language"] == "ru"
    assert result.ranges == [(0.0, 6.0)]
    assert [s.start for s in result.segments] == [0.0, 2.0, 4.0]

    def must_not_load(cfg, reporter):
        raise AssertionError("модель не нужна: всё уже в кэше")

    cached = tr.transcribe_source(source, work, cfg, Reporter(), [(0, 6)], engine_factory=must_not_load)
    assert cached == result


@needs_ffmpeg
def test_changed_settings_or_force_retranscribe(video_source):
    source, work = video_source
    engine = FakeWhisper()
    tr.transcribe_source(source, work, load_config(), Reporter(), engine_factory=factory(engine))
    tr.transcribe_source(
        source, work, load_config(None, {"transcribe.language": "en"}), Reporter(), engine_factory=factory(engine)
    )
    tr.transcribe_source(
        source, work, load_config(None, {"transcribe.language": "en"}), Reporter(), force=True,
        engine_factory=factory(engine),
    )  # fmt: skip
    assert len(engine.calls) == 3


@needs_ffmpeg
def test_hallucinations_are_dropped(video_source):
    source, work = video_source
    fake_words = [FakeWord(5.0, 5.5, " Субтитры")]
    engine = FakeWhisper(
        extra=[
            FakeSegment(4.0, 5.0, " Субтитры сделал DimaTorzok", [FakeWord(4.0, 4.5, " Субтитры")]),
            FakeSegment(5.0, 5.9, " Продолжение следует...", fake_words, avg_logprob=-1.2),
            FakeSegment(5.0, 5.9, " Спасибо за просмотр!", fake_words, avg_logprob=-0.1, no_speech_prob=0.01),
        ]
    )
    result = tr.transcribe_source(source, work, load_config(), Reporter(), [(0, 6)], engine_factory=factory(engine))
    texts = [s.text for s in result.segments]
    assert "Субтитры сделал DimaTorzok" not in texts
    assert "Продолжение следует..." not in texts  # неуверенно — выброшено
    assert "Спасибо за просмотр!" in texts  # уверенно — это настоящая фраза


def test_source_without_audio(tmp_path):
    source = SourceInfo("x", "file", "x.mp4", "x.mp4", "x", 5.0, 320, 240, 25.0, False)
    with pytest.raises(ClipperError, match="нет звуковой дорожки"):
        tr.transcribe_source(source, tmp_path, load_config(), Reporter())


def test_convert_segment_offsets_clamps_and_orders():
    raw = FakeSegment(
        0.0,
        5.0,
        "  раз   два  ",
        [FakeWord(0.5, 1.0, " раз"), FakeWord(0.4, 0.9, " два"), FakeWord(2.0, 9.0, " три"), FakeWord(3, 3, "  ")],
    )
    segment = tr.convert_segment(raw, offset=100.0, range_end=4.0 + 100.0)
    assert segment.text == "раз два"
    assert [w.text for w in segment.words] == ["раз", "два", "три"]
    assert segment.words[1].start == 100.5  # не раньше предыдущего слова
    assert segment.words[2].end == 104.0  # не дальше конца отрезка
    assert tr.convert_segment(FakeSegment(0, 1, "", []), 0, 10) is None


def test_write_srt(tmp_path):
    transcript = Transcript(
        "ru", {}, [(0, 4000)], [Segment(3723.5, 3725.25, "Привет", (Word("Привет", 3723.5, 3725.25),))]
    )
    path = tr.write_srt(transcript, tmp_path / "t.srt")
    raw = path.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM — кириллица в плеерах Windows
    assert raw.decode("utf-8-sig") == "1\r\n01:02:03,500 --> 01:02:05,250\r\nПривет\r\n"  # одинаково на любой ОС


@pytest.mark.parametrize(
    ("error", "kind", "hint_part"),
    [
        (RuntimeError("CUDA failed with error out of memory"), ClipperError, "int8_float16"),
        (RuntimeError("Library cublas64_12.dll is not found or cannot be loaded"), DependencyError, "nvidia-cublas"),
        (RuntimeError("Could not load library cudnn_ops64_9.dll"), DependencyError, "nvidia-cudnn"),
        (RuntimeError("CUDA driver version is insufficient"), DependencyError, "драйвер"),
    ],
)
def test_whisper_errors(error, kind, hint_part):
    result = tr.whisper_error(error)
    assert isinstance(result, kind)
    assert hint_part in result.hint


def test_model_is_not_downloaded_here(tmp_path):
    pytest.importorskip("faster_whisper")
    assert tr.model_is_downloaded(str(tmp_path)) is False


def test_vad_model_works_on_this_python():
    """Silero VAD (onnxruntime) входит в faster-whisper — проверяем, что он запускается."""
    pytest.importorskip("faster_whisper")
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps

    assert get_speech_timestamps(np.zeros(tr.SAMPLE_RATE, dtype=np.float32)) == []


# --- звук и ffmpeg ----------------------------------------------------------------------


@needs_ffmpeg
def test_extract_audio_once_and_read_slices(video_source):
    source, work = video_source
    events = []
    path = tr.extract_audio(source, work, Reporter(events.append))
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
    assert tr.audio_duration(path) == pytest.approx(6.0, abs=0.1)
    mtime = path.stat().st_mtime_ns
    assert tr.extract_audio(source, work, Reporter()) == path
    assert path.stat().st_mtime_ns == mtime  # второй раз не извлекается
    audio = tr.read_audio(path, 1.0, 3.5)
    assert len(audio) == 40000
    assert 0.1 < float(abs(audio).max()) <= 1.0  # синус из тестового видео
    assert not list(work.glob("*.part.wav"))


@needs_ffmpeg
def test_run_ffmpeg_progress_cancel_and_errors(tmp_path):
    import shutil

    ffmpeg = shutil.which("ffmpeg")
    positions = []
    args = ["-f", "lavfi", "-i", "sine=duration=3", "-f", "null", "-"]
    run_ffmpeg(ffmpeg, args, on_progress=positions.append)
    assert positions and positions[-1] == pytest.approx(3.0, abs=0.1)

    reporter = Reporter()
    reporter.cancel.cancel()
    with pytest.raises(Cancelled):
        run_ffmpeg(ffmpeg, ["-re", "-f", "lavfi", "-i", "sine=duration=30", "-f", "null", "-"], cancel=reporter.cancel)

    log = tmp_path / "clipper.log"
    with pytest.raises(ClipperError) as err:
        run_ffmpeg(ffmpeg, ["-i", str(tmp_path / "нет.mp4"), "-f", "null", "-"], log_path=log)
    assert "нет.mp4" in err.value.message
    assert "нет.mp4" in log.read_text(encoding="utf-8")


# --- CLI ----------------------------------------------------------------------------


@needs_ffmpeg
def test_cli_transcribe(tmp_path, monkeypatch):
    make_video(tmp_path / "talk.mp4", seconds=6)
    monkeypatch.setattr(tr, "load_whisper", factory(FakeWhisper()))
    runner = CliRunner()
    result = runner.invoke(cli.app, ["transcribe", str(tmp_path / "talk.mp4"), "--from", "0:02", "--to", "6"])
    assert result.exit_code == 0, result.output
    assert "Язык:" in result.output and "ru" in result.output
    assert "2 фразы, 4 слова" in result.output
    srt = Path("work/talk/transcript.srt").read_text(encoding="utf-8-sig")
    assert "00:00:02,000 --> 00:00:04,000" in srt

    bad = runner.invoke(cli.app, ["transcribe", str(tmp_path / "talk.mp4"), "--from", "5", "--to", "1:70"])
    assert bad.exit_code == 1 and "--to" in bad.output
    backwards = runner.invoke(cli.app, ["transcribe", str(tmp_path / "talk.mp4"), "--from", "5", "--to", "3"])
    assert backwards.exit_code == 1 and "позже" in backwards.output


@pytest.mark.parametrize(
    ("count", "text"),
    [(1, "1 слово"), (3, "3 слова"), (5, "5 слов"), (11, "11 слов"), (21, "21 слово"), (112, "112 слов")],
)
def test_plural(count, text):
    from clipper.core.text import plural

    assert plural(count, "слово", "слова", "слов") == text


def test_huggingface_messages_are_silenced():
    import logging

    tr._quiet_huggingface()
    assert logging.getLogger("huggingface_hub").getEffectiveLevel() == logging.ERROR
