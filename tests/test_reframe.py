"""Этап 6: геометрия кадра, траектория за лицом, stream, calibrate и рендер 1080×1920."""

import json
import subprocess
from pathlib import Path

import pytest
from helpers import needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import facetrack as ft
from clipper.core import render as render_module
from clipper.core.config import LayoutConfig, Rect, load_config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.models import Clip, Project, SourceInfo, Word, save_project
from clipper.core.reframe import (
    Box,
    crop_size,
    fit_size,
    layout_for,
    stream_graph,
    stream_plan,
    video_graph,
    video_plan,
)
from clipper.core.render import render_project
from clipper.core.timeline import Timeline

# --- геометрия ------------------------------------------------------------------------


def plan_for(aspect="9:16", crop="face", background="blur", size=(1920, 1080)):
    cfg = load_config(None, {"reframe.aspect": aspect, "reframe.crop": crop, "reframe.background": background})
    return video_plan(cfg, *size)


def test_crop_and_fit_sizes():
    assert crop_size(1920, 1080, 9 / 16) == (608, 1080)
    assert crop_size(1920, 1080, 1.0) == (1080, 1080)
    assert crop_size(1080, 1920, 9 / 16) == (1080, 1920)
    assert crop_size(1080, 1920, 1.0) == (1080, 1080)  # вертикальное видео, квадрат — по ширине
    assert crop_size(1279, 719, 9 / 16) == (404, 718)  # нечётные размеры → чётное окно
    assert fit_size(1920, 1080) == (1080, 608)
    assert fit_size(1080, 1080) == (1080, 1080)
    assert fit_size(720, 1280) == (1080, 1920)


def test_video_plans():
    vertical = plan_for()
    assert (vertical.crop_w, vertical.crop_h, vertical.fg_w, vertical.fg_h) == (608, 1080, 1080, 1920)
    assert vertical.fills_frame and vertical.track
    square = plan_for("1:1")
    assert (square.crop_w, square.fg_w, square.fg_h, square.fills_frame) == (1080, 1080, 1080, False)
    original = plan_for("original")
    assert (original.crop_w, original.crop_h, original.fg_h, original.track) == (1920, 1080, 608, False)
    assert not plan_for(crop="center").track
    assert not plan_for(size=(1080, 1920)).track  # вертикальное видео 9:16 — двигать нечего


def test_box_at_clamps_and_keeps_even():
    plan = plan_for()
    assert plan.center_box() == Box(656, 0, 608, 1080)
    assert plan.box_at(0, 540) == Box(0, 0, 608, 1080)
    assert plan.box_at(1919, 540) == Box(1312, 0, 608, 1080)
    assert plan.box_at(1001, 540).x % 2 == 0


def test_video_graphs():
    assert video_graph(plan_for(), "[in]", "[out]", "clip_01.cmd") == (
        "[in]sendcmd=f=clip_01.cmd,crop@face=w=608:h=1080:x=656:y=0,scale=1080:1920:flags=lanczos,setsar=1[out]"
    )
    black = video_graph(plan_for("original", background="black"), "[in]", "[out]")
    assert black.endswith("scale=1080:608:flags=lanczos,setsar=1,pad=1080:1920:0:656:black[out]")
    blur = video_graph(plan_for("1:1"), "[in]", "[out]")
    assert blur.startswith("[in]split=2[fgsrc][bgsrc];") and "boxblur" in blur
    assert blur.endswith("[bg][fg]overlay=0:420[out]")


def test_stream_plan_scales_preset_to_real_frame():
    layout = LayoutConfig(webcam=Rect(1560, 810, 360, 270), webcam_zone=0.33, source_size=(1920, 1080))
    plan = stream_plan(layout, 1280, 720)  # видео 720p, координаты сняты на 1080p
    assert plan.webcam == Box(1040, 540, 240, 180)
    assert plan.top_h + plan.bottom_h == 1920 and plan.top_h % 2 == 0 and plan.top_h == 634
    assert plan.game.h == 720 and plan.game.x == (1280 - plan.game.w) // 2 // 2 * 2
    left = stream_plan(LayoutConfig(webcam=Rect(0, 0, 100, 100), game_crop="left"), 1920, 1080)
    right = stream_plan(LayoutConfig(webcam=Rect(0, 0, 100, 100), game_crop="right"), 1920, 1080)
    assert left.game.x == 0 and right.game.x + right.game.w == 1920
    outside = stream_plan(LayoutConfig(webcam=Rect(1800, 1000, 400, 300)), 1920, 1080)
    assert outside.webcam.x + outside.webcam.w <= 1920 and outside.webcam.y + outside.webcam.h <= 1080
    graph = stream_graph(plan, "[in]", "[out]")
    assert "[cam]crop=240:180:1040:540," in graph and graph.endswith("[top][bottom]vstack=inputs=2[out]")


def test_layout_errors():
    with pytest.raises(ClipperError, match="нужен пресет"):
        layout_for(load_config(None, {"reframe.mode": "stream"}))
    with pytest.raises(ClipperError, match="пресет «cam» не найден"):  # ловит уже проверка конфига
        load_config(None, {"reframe.mode": "stream", "reframe.layout": "cam"})


def test_timeline_to_source_inverts_to_output():
    tl = Timeline.with_cuts(10.0, 20.0, [(12.0, 13.0), (15.0, 17.0)])
    for t in (10.0, 11.5, 12.9, 14.0, 17.5, 20.0):
        if not any(a < t < b for a, b in [(12.0, 13.0), (15.0, 17.0)]):
            assert tl.to_source(tl.to_output(t)) == pytest.approx(t)
    assert tl.to_source(2.0) == pytest.approx(12.0)  # стык кусков: конец первого
    assert tl.to_source(2.01) == pytest.approx(13.01)
    assert tl.to_source(100) == 20.0


# --- траектория -----------------------------------------------------------------------


def face_at(cx, cy=0.5, size=0.2, score=0.9):
    return ft.Face(cx - size / 2 * 0.56, cy - size / 2, size * 0.56, size, score)


def samples_for(xs, cuts=(), fps=5):
    return [ft.Sample(round(i / fps, 3), () if x is None else (face_at(x),), i in cuts) for i, x in enumerate(xs)]


def test_small_movements_do_not_move_the_window():
    plan = plan_for()
    wobble = [0.5, 0.51, 0.49, 0.52, 0.5, 0.48, 0.51] * 3  # ±40 px — внутри мёртвой зоны (73 px)
    path, coverage = ft.camera_path(samples_for(wobble), plan)
    assert coverage == 1.0
    assert len({round(x) for x, _ in path}) == 1


def test_big_move_is_a_smooth_pan_and_scene_cut_jumps():
    plan = plan_for()
    xs = [0.3] * 10 + [0.7] * 10
    path, _ = ft.camera_path(samples_for(xs), plan)
    x = [p[0] for p in path]
    assert x[0] == pytest.approx(0.3 * 1920) and x[-1] == pytest.approx(0.7 * 1920)
    assert all(b >= a for a, b in zip(x, x[1:], strict=False))  # монотонно, без рывков назад
    assert max(b - a for a, b in zip(x, x[1:], strict=False)) < 0.4 * 1920 / 3  # проезд растянут на несколько кадров

    cut, _ = ft.camera_path(samples_for(xs, cuts={10}), plan)
    assert cut[9][0] == pytest.approx(0.3 * 1920) and cut[10][0] == pytest.approx(0.7 * 1920)


def test_lost_face_holds_then_returns_to_center_and_low_coverage_is_center():
    plan = plan_for()
    xs = [0.2] * 10 + [None] * 20
    path, coverage = ft.camera_path(samples_for(xs), plan)
    assert coverage == pytest.approx(1 / 3)
    assert path[10][0] == pytest.approx(0.2 * 1920)  # лицо только что пропало — окно ждёт на месте
    assert path[-1][0] == pytest.approx(960)  # ушло к центру

    rare, coverage = ft.camera_path(samples_for([0.2] + [None] * 19), plan)
    assert coverage == 0.05 and all(p == (960, 540) for p in rare)


def test_main_face_is_largest_but_sticky():
    big, small = face_at(0.2, size=0.3), face_at(0.8, size=0.2)
    samples = [ft.Sample(0.0, (big, small)), ft.Sample(0.2, (face_at(0.21, size=0.25), face_at(0.8, size=0.3)))]
    chosen = ft.choose_faces(samples)
    assert chosen[0] is big
    assert chosen[1].cx == pytest.approx(0.21)  # прежнее лицо не намного меньше — остаётся главным


def test_crop_commands_follow_output_time_after_cuts():
    plan = plan_for()
    samples = samples_for([0.3] * 10 + [0.7] * 10, cuts={10})  # сцена меняется на 2.0 с исходника
    path, _ = ft.camera_path(samples, plan)
    timeline = Timeline.with_cuts(0.0, 4.0, [(1.0, 1.5)])  # вырезано 0.5 с до смены сцены
    text = ft.crop_commands(samples, path, plan, timeline, fps=10)
    lines = text.strip().splitlines()
    assert lines[0] == f"0.000 crop@face x {plan.box_at(0.3 * 1920, 540).x}, crop@face y 0;"
    jump = [line for line in lines if f"x {plan.box_at(0.7 * 1920, 540).x}," in line][0]
    assert float(jump.split()[0]) == pytest.approx(1.5 - 0.05)  # 2.0 с исходника = 1.5 с клипа


def test_tiles_overlaps_and_scene_change():
    tiles = ft.square_tiles(960, 540)
    assert all(t[2] == 270 for t in tiles)
    assert {t[0] for t in tiles} >= {0, 690} and {t[1] for t in tiles} == {0, 135, 270}  # шаг — полплитки
    a, b = ft.Face(0.1, 0.1, 0.2, 0.2, 0.9), ft.Face(0.11, 0.1, 0.2, 0.2, 0.8)
    assert set(ft.suppress_overlaps([b, a, ft.Face(0.6, 0.6, 0.1, 0.1)])) == {a, ft.Face(0.6, 0.6, 0.1, 0.1)}

    np = pytest.importorskip("numpy")
    red1 = np.zeros((18, 32, 3), np.int16)
    red1[..., 0], red1[..., 1], red1[..., 2] = 179, 200, 200
    red2 = red1.copy()
    red2[..., 0] = 1
    assert ft.scene_change(red1, red2) < 1  # оттенок замкнут в круг: 179 и 1 — почти одно и то же


def test_samples_cache_roundtrip(tmp_path):
    path = tmp_path / "faces.json"
    key = {"version": ft.CACHE_VERSION, "start": 1.0}
    samples = [ft.Sample(1.0, (ft.Face(0.1, 0.2, 0.3, 0.4, 0.9),), False), ft.Sample(1.2, (), True)]
    ft.save_samples(path, key, samples)
    assert ft.load_samples(path, key) == samples
    assert ft.load_samples(path, {**key, "start": 2.0}) is None
    json.loads(path.read_text(encoding="utf-8"))


def test_yunet_detector_loads():
    pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    detector = ft.make_detector("yunet")
    assert detector.detect(np.zeros((360, 640, 3), np.uint8)) == []


# --- настоящий ffmpeg -------------------------------------------------------------------


def probe_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
         "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    w, h = out.strip().split(",")
    return int(w), int(h)


def square_video(path: Path, seconds: int = 4) -> Path:
    """Серый кадр 640×360, белый квадрат едет слева направо (x = 40 → 520)."""
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"color=gray:size=640x360:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"color=white:size=80x80:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=duration={seconds}",
         "-filter_complex", f"[0][1]overlay=x='40+480*t/{seconds}':y=140[v]", "-map", "[v]", "-map", "2:a",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


class SquareDetector:
    """«Детектор лиц» для тестов: белый квадрат на сером фоне."""

    def detect(self, bgr):
        mask = bgr[..., 0] > 240
        if not mask.any():
            return []
        h, w = mask.shape
        ys, xs = mask.nonzero()
        return [ft.Face(xs.min() / w, ys.min() / h, (xs.max() - xs.min()) / w, (ys.max() - ys.min()) / h, 1.0)]


def white_center(path: Path, at: float) -> float | None:
    """Где по горизонтали белое в кадре 1080×1920 (доля ширины)."""
    np = pytest.importorskip("numpy")
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-ss", f"{at}", "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    frame = np.frombuffer(raw, np.uint8).reshape(1920, 1080)
    xs = (frame[400:1400] > 240).nonzero()[1]
    return float(xs.mean()) / 1080 if len(xs) else None


def project_for(video: Path, clips: list[Clip], size=(640, 360)) -> Project:
    source = SourceInfo("sq", "file", str(video), str(video), "sq", 4.0, size[0], size[1], 25.0, True)
    return Project(source, "keywords", [], "ru", "", clips)


@needs_ffmpeg
def test_render_follows_the_face_through_cuts(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    monkeypatch.setattr(ft, "make_detector", lambda kind: SquareDetector())
    video = square_video(tmp_path / "sq.mp4")
    words = [Word("раз", 0.2, 0.8), Word("два", 2.6, 3.4)]
    project = project_for(video, [Clip(1, 0.0, 4.0, words=words)])
    cfg = load_config(
        None,
        {"audio.cut_pauses": True, "audio.pause_detect": "words", "subtitles.enabled": False,
         "render.encoder": "x264", "paths.output": str(tmp_path / "out")},
    )  # fmt: skip
    work = tmp_path / "work"
    (result,) = render_project(project, work, cfg, Reporter())
    assert result.error is None and result.frame == "лицо в 100% кадров"
    assert probe_size(result.path) == (1080, 1920)
    assert (work / "tmp" / "clip_01.cmd").is_file() and (work / "tmp" / "faces_01.json").is_file()
    # Квадрат всё время в середине кадра, хотя в исходнике он проехал через весь кадр
    # и кусок 0.92…2.48 с вырезан (время траектории — после вырезок).
    for at in (0.3, result.duration - 0.3):
        assert white_center(result.path, at) == pytest.approx(0.5, abs=0.12)

    # Второй рендер берёт найденные лица из кэша — детектор не нужен.
    monkeypatch.setattr(ft, "make_detector", lambda kind: pytest.fail("кэш не сработал"))
    (again,) = render_project(project, work, cfg, Reporter())
    assert again.error is None


@needs_ffmpeg
def test_render_modes_and_detector_failure(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    video = square_video(tmp_path / "sq.mp4", seconds=2)
    project = project_for(video, [Clip(1, 0.0, 2.0)])

    def broken(kind):
        raise ClipperError("mediapipe не импортируется", hint="pip install mediapipe")

    monkeypatch.setattr(ft, "make_detector", broken)
    messages = []
    base = {"render.encoder": "x264", "paths.output": str(tmp_path / "out")}
    (result,) = render_project(project, tmp_path / "w1", load_config(None, base), Reporter(messages.append))
    assert result.error is None and probe_size(result.path) == (1080, 1920)
    assert any("Слежение за лицом не работает" in getattr(m, "text", "") for m in messages)

    for aspect, background in (("1:1", "blur"), ("original", "black")):
        cfg = load_config(None, {**base, "reframe.aspect": aspect, "reframe.background": background,
                                 "reframe.crop": "center"})  # fmt: skip
        (result,) = render_project(project, tmp_path / "w2", cfg, Reporter())
        assert result.error is None and probe_size(result.path) == (1080, 1920)


@needs_ffmpeg
def test_stream_render_and_calibrate_via_cli(tmp_path):
    pytest.importorskip("cv2")
    video = square_video(tmp_path / "sq.mp4", seconds=2)
    (tmp_path / "clipper.yaml").write_text(
        "paths: {workdir: work, output: out}\n"
        "layouts:\n  cam:\n    source_size: [640, 360]\n    webcam: {x: 20, y: 120, width: 160, height: 120}\n",
        encoding="utf-8",
    )
    save_project(tmp_path / "work" / "sq", project_for(video, [Clip(1, 0.0, 2.0, words=[Word("да", 0.2, 1.0)])]))
    runner = CliRunner()

    result = runner.invoke(cli.app, ["render", "--layout", "cam", "--encoder", "x264"])
    assert result.exit_code == 0, result.output
    assert "стрим — пресет cam" in result.output
    assert probe_size(tmp_path / "out" / "sq" / "clip_01.mp4") == (1080, 1920)

    result = runner.invoke(cli.app, ["render", "--layout", "nope", "--encoder", "x264"])
    assert result.exit_code != 0 and "пресет «nope» не найден" in result.output

    result = runner.invoke(cli.app, ["calibrate", str(video), "--at", "1"])
    assert result.exit_code == 0, result.output
    assert "source_size: [640, 360]" in result.output
    import cv2

    (png,) = (tmp_path / "work").glob("*/calibrate.png")  # у файла без source.json своя папка: sq-2
    frame = cv2.imread(str(png))
    assert frame.shape == (360, 640, 3)

    result = runner.invoke(cli.app, ["calibrate", str(video), "--at", "1", "--layout", "cam"])
    assert result.exit_code == 0, result.output
    preview = cv2.imread(str(png.with_name("calibrate_cam.png")))
    assert preview.shape == (1920, 1080, 3)

    result = runner.invoke(cli.app, ["calibrate", str(video), "--at", "10"])
    assert result.exit_code != 0 and "за концом видео" in result.output


def test_render_module_uses_1080x1920_for_subtitles():
    assert (render_module.OUT_W, render_module.OUT_H) == (1080, 1920)
