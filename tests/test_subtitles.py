"""Этап 5: субтитры — текст, строки, тайминг, .ass, стиль и наложение настоящим ffmpeg."""

import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import render as render_module
from clipper.core import subtitles as subs
from clipper.core.config import load_config
from clipper.core.errors import ConfigError
from clipper.core.events import Reporter
from clipper.core.models import Clip, Project, SourceInfo, Word, save_project
from clipper.core.render import render_project

STYLE = subs.Style()


def texts(groups):
    return [" ".join(c.text for c in group) for group in groups]


# --- текст и строки ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "shown"),
    [("как-то,", "КАК-ТО"), ("«Привет»", "ПРИВЕТ"), ("—", ""), ("...", ""), ("50%", "50%"), ("Ёлки!", "ЁЛКИ"),
     ("-ха.", "ХА"), ("{\\b1}жирно", "B1ЖИРНО")],
)  # fmt: skip
def test_display_text(raw, shown):
    assert subs.display_text(raw, STYLE) == shown


def test_display_text_without_caps_and_stripping_keeps_ass_safe():
    style = subs.Style(uppercase=False, strip_punctuation=False)
    assert subs.display_text("Привет, {мир}\\", style) == "Привет, (мир)/"


def test_grouping_by_words_sentences_pauses_and_length():
    words = [
        Word("Саня", 0.0, 0.3), Word("схватился.", 0.3, 0.8),  # конец предложения
        Word("Ха", 1.0, 1.1), Word("-ха", 1.1, 1.2), Word("-ха.", 1.2, 1.3),  # склеится в одно слово
        Word("я", 1.4, 1.5), Word("сейчас", 1.5, 1.8), Word("пойду", 1.8, 2.0),
        Word("за", 2.0, 2.1),  # не больше 3 слов
        Word("дровами", 2.1, 2.5), Word("—", 2.5, 2.6),  # тире не показывается, но заканчивает фразу
        Word("потом", 2.7, 3.0), Word("вернусь", 4.0, 4.5),  # пауза 1 с
        Word("достопримечательности", 5.0, 5.5), Word("рядом", 5.5, 5.9),  # длинное слово — отдельно
    ]  # fmt: skip
    groups = subs.group_captions(subs.captions(words, STYLE), STYLE)
    assert texts(groups) == [
        "САНЯ СХВАТИЛСЯ", "ХА-ХА-ХА", "Я СЕЙЧАС ПОЙДУ", "ЗА ДРОВАМИ", "ПОТОМ", "ВЕРНУСЬ",
        "ДОСТОПРИМЕЧАТЕЛЬНОСТИ", "РЯДОМ",
    ]  # fmt: skip
    one = subs.group_captions(subs.captions(words[:2], STYLE), STYLE, max_words=1)
    assert texts(one) == ["САНЯ", "СХВАТИЛСЯ"]


def test_caption_events_timing():
    items = [
        subs.Caption("РАЗ", 1.0, 1.3), subs.Caption("ДВА", 1.4, 1.8),
        subs.Caption("ТРИ", 1.9, 2.2),  # через 0.1 с — строка не гаснет до «ТРИ»
        subs.Caption("ЧЕТЫРЕ", 4.0, 4.4),  # через 1.8 с — строка держится hold и гаснет
    ]  # fmt: skip
    events = subs.caption_events([items[:2], items[2:3], items[3:]], STYLE, duration=4.5)
    spans = [(start, end, group[index].text) for start, end, group, index in events]
    assert spans == [(1.0, 1.4, "РАЗ"), (1.4, 1.9, "ДВА"), (1.9, 2.5, "ТРИ"), (4.0, 4.5, "ЧЕТЫРЕ")]


# --- .ass ------------------------------------------------------------------------------


def test_build_ass_vertical_frame():
    words = [Word("Саня", 0.5, 0.9), Word("схватился.", 0.95, 1.6)]
    ass = subs.build_ass(words, 1080, 1920, STYLE, duration=3.0)
    assert ass.info["PlayResX"] == "1080" and ass.info["PlayResY"] == "1920"
    style = ass.styles["Default"]
    assert style.fontname == "Montserrat Black" and style.fontsize == 130 and style.outline == 5
    first, second = ass.events
    assert (first.start, first.end, second.start, second.end) == (500, 950, 950, 1900)
    assert first.text.startswith("{\\an5\\pos(540,1382)}{\\c&H00E6FF&\\fscx90\\fscy90\\t(0,75,\\fscx112\\fscy112)")
    assert first.text.endswith("САНЯ{\\r} СХВАТИЛСЯ")
    assert second.text.startswith("{\\an5\\pos(540,1382)}САНЯ {\\c&H00E6FF&")
    text = ass.to_string("ass")
    assert "Style: Default,Montserrat Black,130" in text


def test_build_ass_scales_to_frame_and_fits_long_words():
    ass = subs.build_ass([Word("достопримечательности", 0.0, 1.0)], 1920, 1080, STYLE, duration=2.0)
    assert ass.styles["Default"].fontsize == pytest.approx(130 * 1080 / 1920, abs=0.1)
    (event,) = ass.events
    assert "\\pos(960,778)" in event.text
    assert "\\fs" in event.text  # длинное слово уменьшено, чтобы влезть по ширине
    no_pop = subs.build_ass([Word("да", 0.0, 1.0)], 1080, 1920, subs.Style(pop_duration=0), duration=2.0)
    assert "\\t(" not in no_pop.events[0].text


def test_ass_color():
    assert subs.ass_color("#FFE600") == "&H00E6FF&"
    assert subs.ass_color("12ab34") == "&H34AB12&"


# --- стиль ------------------------------------------------------------------------------


def test_builtin_capcut_matches_defaults():
    style, path = subs.load_style("capcut")
    assert style == subs.Style()
    assert path.name == "capcut.yaml"
    fonts = subs.font_files(style, path)
    assert any(f.name == "Montserrat-Black.ttf" for f in fonts)
    assert (subs.FONTS_DIR / "OFL.txt").is_file()


def test_user_style_overrides_only_listed_keys(tmp_path):
    (tmp_path / "fonts").mkdir()
    (tmp_path / "fonts" / "My.ttf").write_bytes(b"font")
    path = tmp_path / "my.yaml"
    path.write_text('size: 100\nhighlight: "#00FF00"\nfont_file: fonts/My.ttf\n', encoding="utf-8")
    style, _ = subs.load_style(str(path))
    assert (style.size, style.highlight, style.color) == (100, "#00FF00", "#FFFFFF")
    files = subs.font_files(style, path)
    assert files[-1] == tmp_path / "fonts" / "My.ttf"
    dest = subs.prepare_fonts(tmp_path / "tmp" / "fonts", files)
    assert (dest / "My.ttf").read_bytes() == b"font"


@pytest.mark.parametrize(
    ("content", "message", "hint"),
    [
        ("highlight: #00FF00\n", "highlight — цвет не указан", "в кавычках"),
        ("color: red\n", "непонятный цвет", "#RRGGBB"),
        ("colour: '#FFFFFF'\n", "неизвестный параметр «colour»", "«color»"),
        ("position: 1.5\n", "position", "верх кадра"),
        ("size: большой\n", "size", None),
    ],
)
def test_style_errors(tmp_path, content, message, hint):
    path = tmp_path / "bad.yaml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        subs.load_style(str(path))
    assert message in info.value.message
    if hint:
        assert hint in (info.value.hint or "") + info.value.message


def test_unknown_style_name_lists_builtin():
    with pytest.raises(ConfigError) as info:
        subs.load_style("tiktok")
    assert "capcut" in (info.value.hint or "")


# --- настоящий ffmpeg -------------------------------------------------------------------


def _has_libass() -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True).stdout
    return " subtitles " in out


needs_libass = pytest.mark.skipif(not _has_libass(), reason="нужен ffmpeg с libass")


def black_video(path: Path, seconds: int = 6, size: str = "540x960") -> Path:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=black:size={size}:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=duration={seconds}", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


def frame_colors(video: Path, at: float, size: tuple[int, int] = (1080, 1920)) -> tuple[int, int]:
    """(белых, жёлтых) пикселей в кадре на `at` секунде."""
    np = pytest.importorskip("numpy")
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", str(video), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    rgb = np.frombuffer(raw, np.uint8).reshape(size[1], size[0], 3).astype(int)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    white = int(((r > 200) & (g > 200) & (b > 200)).sum())
    yellow = int(((r > 200) & (g > 170) & (b < 90)).sum())
    return white, yellow


def project_for(video: Path, clips: list[Clip], size=(540, 960)) -> Project:
    source = SourceInfo("talk", "file", str(video), str(video), "talk", 6.0, size[0], size[1], 25.0, True)
    return Project(source, "keywords", [], "ru", "", clips)


@needs_ffmpeg
@needs_libass
def test_render_burns_subtitles_in_time_after_cuts(tmp_path):
    video = black_video(tmp_path / "talk.mp4")
    words = [Word("раз", 0.5, 1.0), Word("два", 1.0, 1.5), Word("три", 4.0, 4.5)]
    project = project_for(video, [Clip(1, 0.0, 5.5, words=words)])
    cfg = load_config(
        None,
        {"audio.cut_pauses": True, "audio.pause_detect": "words", "render.encoder": "x264",
         "paths.output": str(tmp_path / "out")},
    )  # fmt: skip
    work = tmp_path / "work"
    (result,) = render_project(project, work, cfg, Reporter())
    assert result.error is None and result.subtitles
    assert (work / "tmp" / "clip_01.ass").is_file()
    assert (work / "tmp" / "fonts" / "Montserrat-Black.ttf").is_file()

    # Пауза 1.5…4.0 вырезана (остаётся по 0.12 с), «три» начинается примерно на 1.74 с клипа.
    white, yellow = frame_colors(result.path, 0.2)
    assert white < 50 and yellow < 50  # до первого слова экран пуст
    white, yellow = frame_colors(result.path, 0.75)
    assert yellow > 300 and white > 300  # «РАЗ» жёлтое, «ДВА» белое
    white, yellow = frame_colors(result.path, 2.0)
    assert yellow > 300  # «ТРИ» уже на экране — субтитры сдвинулись вместе с вырезкой


@needs_ffmpeg
@needs_libass
def test_cli_no_subs_and_missing_libass(tmp_path, monkeypatch):
    video = black_video(tmp_path / "talk.mp4", seconds=3)
    work = tmp_path / "work" / "talk"
    save_project(work, project_for(video, [Clip(1, 0.0, 3.0, words=[Word("раз", 0.5, 2.5)])]))
    runner = CliRunner()
    common = ["--encoder", "x264", "--set", f"paths.workdir={tmp_path / 'work'}",
              "--set", f"paths.output={tmp_path / 'out'}"]  # fmt: skip
    clip = tmp_path / "out" / "talk" / "clip_01.mp4"

    result = runner.invoke(cli.app, ["render", "--no-subs", *common])
    assert result.exit_code == 0, result.output
    assert frame_colors(clip, 1.0) == (0, 0)

    result = runner.invoke(cli.app, ["render", "--max-words", "1", *common])
    assert result.exit_code == 0, result.output
    assert frame_colors(clip, 1.0)[1] > 300

    result = runner.invoke(cli.app, ["render", "--style", "нет-такого", *common])
    assert result.exit_code != 0 and "Нет стиля субтитров" in result.output

    monkeypatch.setattr(render_module, "list_filters", lambda ffmpeg: frozenset())
    result = runner.invoke(cli.app, ["render", *common])
    assert result.exit_code == 0, result.output
    assert "нет libass" in result.output
    assert frame_colors(clip, 1.0) == (0, 0)


def test_publish_when_target_is_locked(tmp_path, monkeypatch):
    """Windows: clip_01.mp4 открыт в плеере — рендер не теряется и не роняет остальные клипы."""
    monkeypatch.setattr(render_module, "REPLACE_DELAY", 0)
    real_replace = render_module.os.replace
    target = tmp_path / "clip_01.mp4"
    target.write_bytes(b"old")
    calls = []

    def locked(src, dst):
        calls.append(Path(dst).name)
        if Path(dst) == target:
            raise PermissionError(13, "Access is denied")
        real_replace(src, dst)

    monkeypatch.setattr(render_module.os, "replace", locked)
    part = tmp_path / "clip_01.part.mp4"
    part.write_bytes(b"new")
    saved = render_module.publish(part, target)
    assert saved == tmp_path / "clip_01.new.mp4" and saved.read_bytes() == b"new"
    assert target.read_bytes() == b"old" and not part.exists()
    assert calls.count("clip_01.mp4") == render_module.REPLACE_ATTEMPTS

    monkeypatch.setattr(render_module.os, "replace", real_replace)  # плеер закрыли
    part.write_bytes(b"newer")
    assert render_module.publish(part, target) == target
    assert target.read_bytes() == b"newer" and not saved.exists()


@needs_ffmpeg
def test_render_reports_locked_file_per_clip(tmp_path, monkeypatch):
    monkeypatch.setattr(render_module, "REPLACE_DELAY", 0)
    video = black_video(tmp_path / "talk.mp4", seconds=3)
    project = project_for(video, [Clip(1, 0.0, 1.5), Clip(2, 1.5, 3.0)])
    cfg = load_config(
        None, {"render.encoder": "x264", "subtitles.enabled": False, "paths.output": str(tmp_path / "out")}
    )
    real_replace = render_module.os.replace

    def locked(src, dst):
        if Path(dst).name.startswith("clip_01"):
            raise PermissionError(13, "Access is denied", str(dst))
        real_replace(src, dst)

    monkeypatch.setattr(render_module.os, "replace", locked)
    messages = []
    first, second = render_project(project, tmp_path / "work", cfg, Reporter(messages.append))
    assert first.path is None and "занят другой программой" in first.error
    assert second.error is None and second.path.name == "clip_02.mp4"
    assert not list((tmp_path / "out" / "talk").glob("*.part.mp4"))
