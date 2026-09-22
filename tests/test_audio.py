"""Этап 4: паузы, слова-паразиты, Timeline и рендер с вырезками (настоящий ffmpeg)."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import audio
from clipper.core import transcribe as tr
from clipper.core.config import load_config
from clipper.core.events import Reporter
from clipper.core.models import Clip, Project, SourceInfo, Transcript, Word, join_hyphenated, save_project
from clipper.core.render import cut_filter, render_project
from clipper.core.timeline import Timeline

# --- Timeline -----------------------------------------------------------------------


def test_timeline_maps_time_and_words():
    tl = Timeline.with_cuts(10.0, 20.0, [(12.0, 13.0), (15.0, 17.0), (19.95, 20.0)])
    assert tl.pieces == ((10.0, 12.0), (13.0, 15.0), (17.0, 19.95))
    assert tl.duration == pytest.approx(6.95)
    assert tl.removed == pytest.approx(3.05)
    assert not tl.is_whole
    assert tl.to_output(11.0) == pytest.approx(1.0)
    assert tl.to_output(12.5) == pytest.approx(2.0)  # внутри вырезки → начало следующего куска
    assert tl.to_output(18.0) == pytest.approx(5.0)
    assert tl.to_output(25.0) == pytest.approx(6.95)
    words = [Word("до", 9.0, 9.5), Word("раз", 11.0, 11.5), Word("эм", 15.5, 16.0), Word("два", 17.2, 17.6)]
    assert [(w.text, w.start, w.end) for w in tl.map_words(words)] == [("раз", 1.0, 1.5), ("два", 4.2, 4.6)]
    assert tl.relative_pieces() == [(0.0, 2.0), (3.0, 5.0), (7.0, 9.95)]
    assert Timeline.whole(5, 9).is_whole


def test_timeline_drops_tiny_pieces_between_cuts():
    tl = Timeline.with_cuts(0.0, 10.0, [(2.0, 5.0), (5.05, 8.0)])
    assert tl.pieces == ((0.0, 2.0), (8.0, 10.0))


# --- паузы --------------------------------------------------------------------------


def test_shrink_pauses_keeps_a_bit_of_silence():
    cuts = audio.shrink_pauses([(2.0, 3.0), (5.0, 5.4), (0.0, 1.0), (9.2, 10.0)], min_pause=0.6, start=0.0, end=10.0)
    k = audio.PAUSE_KEEP
    assert cuts == [(0.0, round(1.0 - k, 3)), (round(2.0 + k, 3), round(3.0 - k, 3)), (round(9.2 + k, 3), 10.0)]


def test_protect_words_and_word_gaps():
    words = [Word("тихо", 2.4, 2.8), Word("слово", 5.0, 5.5)]
    assert audio.protect_words([(2.0, 3.0)], words) == [(2.0, 2.4), (2.8, 3.0)]
    assert audio.word_gaps(words, 1.0, 6.0) == [(1.0, 2.4), (2.8, 5.0), (5.5, 6.0)]


def test_words_mode_keeps_long_gaps():
    # 1.5…2.5 — пауза; 3.0…9.0 — долго без слов (смех, игра): не пауза.
    words = [Word("раз", 1.0, 1.5), Word("два", 2.5, 3.0), Word("три", 9.0, 9.5)]
    clip = Clip(1, 1.0, 9.5, words=words)
    cfg = load_config(None, {"audio.cut_pauses": True, "audio.pause_detect": "words"})
    k = audio.PAUSE_KEEP
    assert audio.plan_edit(clip, cfg).pauses == [(round(1.5 + k, 3), round(2.5 - k, 3))]
    everything = load_config(None, {"audio.cut_pauses": True, "audio.pause_detect": "words", "audio.max_gap": 0})
    assert len(audio.plan_edit(clip, everything).pauses) == 2


def test_window_min_peak_and_hint():
    np = pytest.importorskip("numpy")
    rate = 8000
    loud = np.full(rate, 0.5, dtype=np.float32)
    quiet = np.full(rate, 0.01, dtype=np.float32)  # −40 дБ
    samples = np.concatenate([loud, quiet, loud])
    assert audio.window_min_peak(samples, rate, 0.6) == pytest.approx(-40.0, abs=0.1)
    assert audio.window_min_peak(samples[:100], rate, 0.6) is None

    hint = audio.pause_hint({2: -27.4, 1: -40.6}, -45)
    assert "клипы 1, 2" in hint and "самое тихое место: -41 дБ" in hint
    assert "--silence-db -36 " in hint
    noisy = audio.pause_hint({1: -21.6}, -35)  # показываем -22 — и порог считаем от -22
    assert "место: -22 дБ" in noisy and "лучше искать паузы по словам" in noisy and "--silence-db -17" in noisy


def test_join_hyphenated_words_and_text():
    words = [Word("Ха", 1.0, 1.2, 0.9), Word("-ха", 1.2, 1.4, 0.8), Word("-ха.", 1.4, 1.6), Word("кто", 2.0, 2.2)]
    words += [Word("-то", 2.2, 2.4), Word("-", 2.5, 2.6), Word("вот.", 2.7, 3.0), Word("-нет", 3.1, 3.3)]
    joined = join_hyphenated(words)
    assert [w.text for w in joined] == ["Ха-ха-ха.", "кто-то", "-", "вот.", "-нет"]
    assert (joined[0].start, joined[0].end, joined[0].prob) == (1.0, 1.6, 0.8)

    data = {"segments": [{"start": 1, "end": 2, "text": "Ха -ха -ха. Да - нет", "words": [
        ["Ха", 1.0, 1.2, 1.0], ["-ха", 1.2, 1.4, 1.0]]}]}  # fmt: skip
    (segment,) = Transcript.from_dict(data).segments
    assert segment.text == "Ха-ха-ха. Да - нет"
    assert [w.text for w in segment.words] == ["Ха-ха"]


def test_parse_silences():
    log = """[silencedetect @ 0x1] silence_start: 1.5
[silencedetect @ 0x1] silence_end: 2.75 | silence_duration: 1.25
size=N/A time=00:00:05.00
[silencedetect @ 0x1] silence_start: 4.2"""
    assert audio.parse_silences(log, offset=100.0, length=5.0) == [(101.5, 102.75), (104.2, 105.0)]


# --- паразиты -----------------------------------------------------------------------


@pytest.mark.parametrize(("text", "key"), [("Э-э-э,", "э"), ("Эмм...", "эм"), ("НУ", "ну"), ("Ммм", "м")])
def test_filler_key(text, key):
    assert audio.filler_key(text) == key


def test_filler_cuts_words_and_phrases_without_touching_neighbours():
    words = [
        Word("Ну,", 0.0, 0.3),
        Word("это", 0.35, 0.6),
        Word("как", 1.0, 1.2),
        Word("бы", 1.21, 1.4),
        Word("нуль", 2.0, 2.4),
        Word("Э-э", 3.0, 3.5),
        Word("вот", 3.52, 3.8),
    ]
    cuts, removed = audio.filler_cuts(words, ["ну", "как бы", "ээ"], start=0.0, end=10.0)
    assert [w.text for w in removed] == ["Ну,", "как", "бы", "Э-э"]
    assert cuts[0] == (0.0, 0.34)  # до соседнего слова «это» (0.35) не дотягивается
    assert cuts[1] == pytest.approx((0.96, 1.44))
    assert cuts[2] == pytest.approx((2.96, 3.52))  # следующее слово начинается в 3.52


def test_plan_edit_combinations():
    words = [Word("эм", 1.0, 1.3), Word("раз", 1.4, 1.8), Word("два", 4.0, 4.4)]
    clip = Clip(1, 0.5, 6.0, words=words)
    nothing = audio.plan_edit(clip, load_config())
    assert nothing.timeline.is_whole and not nothing.pauses

    cfg = load_config(None, {"audio.cut_pauses": True, "audio.remove_fillers": True, "audio.pause_detect": "words"})
    edit = audio.plan_edit(clip, cfg)
    assert [w.text for w in edit.fillers] == ["эм"]
    assert edit.pauses  # промежуток 1.8…4.0 и края клипа
    assert edit.timeline.map_words(words)[0].text == "раз"

    everything_silent = Clip(2, 0.0, 5.0, words=[])
    kept = audio.plan_edit(everything_silent, cfg)
    assert kept.timeline.is_whole  # вырезать весь клип нельзя — оставляем как есть


def test_cut_filter_graph():
    graph = cut_filter([(0.0, 1.5), (2.0, 4.0)], audio=True)
    assert graph.startswith("[0:v:0]split=2[vi0][vi1];[0:a:0]asplit=2[ai0][ai1];")
    assert "[vi1]trim=start=2.000:end=4.000,setpts=PTS-STARTPTS[v1]" in graph
    assert "afade=t=out:st=1.990:d=0.01" in graph
    assert graph.endswith("[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]")
    single = cut_filter([(0.5, 3.0)], audio=False)
    assert single == "[0:v:0]trim=start=0.500:end=3.000,setpts=PTS-STARTPTS[v0];[v0]concat=n=1:v=1:a=0[v]"


def test_verbatim_prompt_and_cache_key():
    plain = load_config()
    verbatim = load_config(None, {"transcribe.verbatim": True})
    assert "initial_prompt" not in tr.whisper_options(plain, "ru")
    assert tr.whisper_options(verbatim, "ru")["initial_prompt"].startswith("Эм")
    assert tr.whisper_options(verbatim, "en")["initial_prompt"].startswith("Umm")
    assert tr.transcript_settings(plain) != tr.transcript_settings(verbatim)  # кэш распознается заново


# --- настоящий ffmpeg -------------------------------------------------------------------


def make_talk(path: Path, seconds: int = 10, silence: tuple[float, float] = (3.0, 5.0)) -> Path:
    """Видео со звуком, в котором на `silence` полная тишина."""
    a, b = silence
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc=size=320x240:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds},volume=enable='between(t,{a},{b})':volume=0",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


def stream_durations(path: Path) -> tuple[float, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    streams = {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}
    return streams["video"], streams["audio"]


@needs_ffmpeg
def test_detect_silences_on_real_audio(tmp_path):
    video = make_talk(tmp_path / "talk.mp4")
    silences = audio.detect_silences(shutil.which("ffmpeg"), video, 1.0, 9.0, silence_db=-35, min_pause=0.6)
    ((a, b),) = silences
    assert a == pytest.approx(3.0, abs=0.1) and b == pytest.approx(5.0, abs=0.1)
    assert audio.quietest_level(shutil.which("ffmpeg"), video, 1.0, 9.0, 0.6) < -80  # там полная тишина
    assert audio.quietest_level(shutil.which("ffmpeg"), video, 6.0, 9.0, 0.6) > -20  # синус без пауз


def project_for(video: Path, clips: list[Clip]) -> Project:
    source = SourceInfo("talk", "file", str(video), str(video), "talk", 10.0, 320, 240, 25.0, True)
    return Project(source, "keywords", [], "ru", "", clips)


@needs_ffmpeg
def test_render_cuts_pause_and_filler_in_sync(tmp_path):
    video = make_talk(tmp_path / "talk.mp4")
    words = [Word("раз", 1.2, 1.6), Word("два", 2.0, 2.5), Word("ээ", 6.0, 6.5), Word("три", 6.6, 7.0)]
    project = project_for(video, [Clip(1, 1.0, 9.0, words=words)])
    cfg = load_config(
        None,
        {"audio.cut_pauses": True, "audio.remove_fillers": True, "render.encoder": "x264",
         "paths.output": str(tmp_path / "out")},
    )  # fmt: skip
    (result,) = render_project(project, tmp_path, cfg, Reporter())

    assert result.error is None
    assert result.fillers == 1
    # Тишина 3…5 с минус запасы по краям, «ээ» с запасами — порядка 2,3 с.
    assert result.removed == pytest.approx(2.0 - 2 * audio.PAUSE_KEEP + 0.54, abs=0.1)
    video_len, audio_len = stream_durations(result.path)
    assert video_len == pytest.approx(result.duration, abs=0.1)
    assert audio_len == pytest.approx(video_len, abs=0.08)  # звук и видео не разъехались


@needs_ffmpeg
def test_render_words_mode_and_cli_flags(tmp_path, monkeypatch):
    from clipper import console as console_module

    monkeypatch.setattr(console_module.console, "width", 200)
    video = make_talk(tmp_path / "talk.mp4", silence=(20, 21))  # по громкости пауз нет — только по словам
    words = [Word("раз", 1.0, 1.5), Word("два", 4.0, 4.5)]
    work = tmp_path / "work" / "talk"
    save_project(work, project_for(video, [Clip(1, 0.8, 6.0, words=words)]))

    runner = CliRunner()
    common = ["--set", f"paths.workdir={tmp_path / 'work'}", "--set", f"paths.output={tmp_path / 'out'}"]
    result = runner.invoke(cli.app, ["render", "--cut-pauses", "--pause-detect", "words", "--encoder", "x264", *common])
    assert result.exit_code == 0, result.output
    assert "Вырезано" in result.output and "паузы −" in result.output
    video_len, _ = stream_durations(tmp_path / "out" / "talk" / "clip_01.mp4")
    assert video_len < 5.2 - 2.0  # вырезан промежуток 1.5…4.0 и хвост 4.5…6.0

    plain = runner.invoke(cli.app, ["render", "--encoder", "x264", *common])
    assert plain.exit_code == 0 and "Вырезано" not in plain.output

    # По громкости пауз нет (синус без тишины) — подсказка, какой порог поставить.
    volume = runner.invoke(cli.app, ["render", "--cut-pauses", "--encoder", "x264", *common])
    assert volume.exit_code == 0, volume.output
    assert "Паузы тише -35 дБ не найдены (клип 1)" in volume.output
    assert "--pause-detect words" in volume.output
    assert "Вырезано" not in volume.output


@needs_ffmpeg
def test_render_warns_when_clip_gets_shorter_than_min_len(tmp_path):
    video = make_talk(tmp_path / "talk.mp4")
    project = project_for(video, [Clip(1, 1.0, 9.0, words=[Word("раз", 1.2, 1.6), Word("два", 6.0, 6.5)])])
    cfg = load_config(
        None,
        {"audio.cut_pauses": True, "render.encoder": "x264", "select.min_len": 7,
         "paths.output": str(tmp_path / "out")},
    )  # fmt: skip
    messages = []
    reporter = Reporter(sink=messages.append)
    (result,) = render_project(project, tmp_path, cfg, reporter)
    assert result.error is None and result.duration < 7
    assert any("короче min_len (7 с)" in getattr(m, "text", "") for m in messages)
