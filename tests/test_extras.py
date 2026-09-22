"""Этап 7: обложки, склейка клипов, --out."""

import subprocess
from pathlib import Path

import pytest
from helpers import needs_ffmpeg
from typer.testing import CliRunner

from clipper import cli
from clipper.core import extras
from clipper.core.config import load_config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.models import Clip, Project, SourceInfo, Word, save_project
from clipper.core.render import render_project


def test_candidate_times():
    assert extras.candidate_times(10.0, count=4) == [2.0, 4.0, 6.0, 8.0]
    assert extras.candidate_times(0.0) == [0.0]


def test_pick_frame_prefers_sharp_bright_frames_without_captions():
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(1)
    flat = np.full((480, 270), 120, np.uint8)
    sharp = rng.integers(40, 200, (480, 270)).astype(np.uint8)
    sharper = rng.integers(0, 255, (480, 270)).astype(np.uint8)
    dark = (sharper // 20).astype(np.uint8)
    assert extras.sharpness(sharp) > extras.sharpness(flat)

    frames, times = [flat, sharp, sharper, dark], [1.0, 2.0, 3.0, 4.0]
    assert extras.pick_frame(frames, times, captions=[]) == 2
    assert extras.pick_frame(frames, times, captions=[(2.5, 3.5)]) == 1  # на 3.0 с — субтитры
    assert extras.pick_frame(frames, times, captions=[(0.0, 5.0)]) == 2  # субтитры везде — просто самый резкий
    assert extras.pick_frame([dark], [1.0], captions=[]) == 0  # выбирать не из чего


def test_sharpness_ignores_subtitle_zone():
    np = pytest.importorskip("numpy")
    frame = np.full((480, 270), 120, np.uint8)
    frame[340:400] = np.random.default_rng(2).integers(0, 255, (60, 270))  # «субтитры» в нижней трети
    assert extras.sharpness(frame) == 0.0


def test_concat_list_escapes_quotes(tmp_path):
    text = extras.concat_list([tmp_path / "clip_01.mp4", tmp_path / "it's.mp4"])
    lines = text.splitlines()
    assert lines[0] == "ffconcat version 1.0"
    assert lines[1] == f"file '{(tmp_path / 'clip_01.mp4').as_posix()}'"
    assert lines[2].endswith("it'\\''s.mp4'")


def test_caption_intervals(tmp_path):
    from clipper.core.subtitles import Style, build_ass, write_ass

    ass = write_ass(tmp_path / "c.ass", build_ass([Word("раз", 0.5, 1.0)], 1080, 1920, Style(), duration=3.0))
    assert extras.caption_intervals(ass) == [(0.5, 1.3)]
    assert extras.caption_intervals(None) == []


# --- настоящий ffmpeg -------------------------------------------------------------------


def talk_video(path: Path, seconds: int = 6) -> Path:
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=640x360:rate=25:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=duration={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
        check=True,
    )  # fmt: skip
    return path


def durations(path: Path) -> dict[str, float]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,duration", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    return {kind: float(value) for kind, value in (line.split(",") for line in out.split())}


def image_size(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    w, h = out.strip().split(",")
    return int(w), int(h)


def make_project(video: Path, clips: list[Clip]) -> Project:
    source = SourceInfo("talk", "file", str(video), str(video), "talk", 6.0, 640, 360, 25.0, True)
    return Project(source, "keywords", [], "ru", "", clips)


@needs_ffmpeg
def test_cli_concat_thumbnails_and_out(tmp_path):
    video = talk_video(tmp_path / "talk.mp4")
    clips = [Clip(1, 0.0, 2.0, words=[Word("раз", 0.2, 1.0)]), Clip(2, 2.0, 3.5), Clip(3, 4.0, 6.0, enabled=False)]
    save_project(tmp_path / "work" / "talk", make_project(video, clips))
    out = tmp_path / "Мои клипы"  # кириллица и пробел в пути
    runner = CliRunner()
    common = ["render", "--encoder", "x264", "--crop", "center", "--out", str(out),
              "--set", f"paths.workdir={tmp_path / 'work'}"]  # fmt: skip

    result = runner.invoke(cli.app, [*common, "--concat"])
    assert result.exit_code == 0, result.output
    folder = out / "talk"
    assert sorted(p.name for p in folder.iterdir()) == ["all_clips.mp4", "clip_01.jpg", "clip_01.mp4",
                                                        "clip_02.jpg", "clip_02.mp4"]  # fmt: skip
    assert "Склейка 2 клипов" in result.output and "Обложки" in result.output
    assert image_size(folder / "clip_01.jpg") == (1080, 1920)
    joined = durations(folder / "all_clips.mp4")
    assert joined["video"] == pytest.approx(3.5, abs=0.1)
    assert joined["audio"] == pytest.approx(joined["video"], abs=0.08)

    for path in folder.iterdir():
        path.unlink()
    result = runner.invoke(cli.app, [*common, "--no-thumbnails", "--no-concat"])
    assert result.exit_code == 0, result.output
    assert sorted(p.name for p in folder.iterdir()) == ["clip_01.mp4", "clip_02.mp4"]


@needs_ffmpeg
def test_concat_needs_two_clips_and_thumbnail_errors_do_not_fail_clip(tmp_path, monkeypatch):
    video = talk_video(tmp_path / "talk.mp4", seconds=2)
    project = make_project(video, [Clip(1, 0.0, 2.0)])
    cfg = load_config(
        None,
        {"render.encoder": "x264", "render.concat": True, "reframe.crop": "center",
         "paths.output": str(tmp_path / "out")},
    )  # fmt: skip

    def broken(*args, **kwargs):
        raise ClipperError("кадры не читаются")

    monkeypatch.setattr(extras, "make_thumbnail", broken)
    messages = []
    (result,) = render_project(project, tmp_path / "work", cfg, Reporter(messages.append))
    texts = [getattr(m, "text", "") for m in messages]
    assert result.error is None and result.thumbnail is None
    assert any("Обложка для clip_01.mp4 не получилась: кадры не читаются" in t for t in texts)
    assert any("хотя бы два готовых клипа" in t for t in texts)
    assert not (tmp_path / "out" / "talk" / extras.CONCAT_NAME).exists()
