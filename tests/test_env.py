import os
import shutil
import sys
from datetime import date
from types import SimpleNamespace

import pytest

from clipper.core import env
from clipper.core.errors import DependencyError
from clipper.core.ffmpeg import failure_reason, list_filters, parse_encoders, parse_filters

FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  A = Audio input/output
  | = Source or sink filter
 ... ass               V->V       Render ASS subtitles onto input video using the libass library.
 ..C crop              V->V       Crop the input video.
 ... color             |->V       Provide an uniformly colored input.
 .S. vstack            N->V       Stack video inputs vertically.
"""

ENCODERS = """Encoders:
 V..... = Video
 A..... = Audio
 .F.... = Frame-level multithreading
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 V....D h264_nvenc           NVIDIA NVENC H.264 encoder (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""

NVENC_FAILURE = """[h264_nvenc @ 0x55ce413efe80] Cannot load nvcuda.dll
[vost#0:0/h264_nvenc @ 0x55ce413efac0] Error while opening encoder - maybe incorrect parameters.
Error while filtering: Operation not permitted
[out#0/null @ 0x55ce413ee7c0] Nothing was written into output file.
"""


def test_parse_ffmpeg_listings():
    assert parse_filters(FILTERS) == {"ass", "crop", "color", "vstack"}
    assert parse_encoders(ENCODERS) == {"libx264", "h264_nvenc", "aac"}


def test_failure_reason_prefers_the_encoder_line():
    assert failure_reason(NVENC_FAILURE, "h264_nvenc", 255) == "Cannot load nvcuda.dll"
    assert failure_reason("", "h264_nvenc", 1) == "код выхода 1"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="нет ffmpeg")
def test_real_ffmpeg_has_the_filters_we_need():
    filters = list_filters(shutil.which("ffmpeg"))
    assert {"crop", "scale", "vstack", "silencedetect"} <= filters


def test_ytdlp_age_days():
    assert env.ytdlp_age_days("2026.08.19", today=date(2026, 9, 22)) == 34
    assert env.ytdlp_age_days("2026.08.19.232012", today=date(2026, 8, 20)) == 1  # nightly
    assert env.ytdlp_age_days("не версия") is None


def test_nvidia_library_dirs(tmp_path):
    sub = "bin" if sys.platform == "win32" else "lib"
    (tmp_path / "nvidia" / "cublas" / sub).mkdir(parents=True)
    (tmp_path / "nvidia" / "cudnn" / sub).mkdir(parents=True)
    (tmp_path / "nvidia" / "no_libs").mkdir()
    dirs = env.nvidia_library_dirs([str(tmp_path / "nvidia")])
    assert [d.parent.name for d in dirs] == ["cublas", "cudnn"]
    assert env.nvidia_library_dirs([]) == []


@pytest.mark.skipif(sys.platform == "win32", reason="на Windows функция меняет поиск DLL процесса")
def test_setup_cuda_dlls_does_nothing_outside_windows():
    assert env.setup_cuda_dlls() == []


def test_require_ffmpeg_explains_how_to_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    with pytest.raises(DependencyError) as err:
        env.require_ffmpeg()
    assert "ffmpeg" in err.value.message
    assert err.value.hint


def test_nvidia_gpus_parses_nvidia_smi(monkeypatch):
    output = "NVIDIA GeForce RTX 3060, 12288, 1100, 560.94\n"
    monkeypatch.setattr(env.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(
        env, "run_command", lambda args, timeout=30: SimpleNamespace(returncode=0, stdout=output, stderr="")
    )
    (gpu,) = env.nvidia_gpus()
    assert gpu.name == "NVIDIA GeForce RTX 3060"
    assert gpu.memory_total_mb == 12288
    assert gpu.memory_free_mb == 11188


def make_fake_ffmpeg(folder, libass: bool, name: str = "ffmpeg"):
    """Фальшивый ffmpeg: печатает версию и конфигурацию сборки на любые аргументы."""
    folder.mkdir(parents=True, exist_ok=True)
    flag = "--enable-libass" if libass else "--disable-libass"
    lines = [f"{name} version fake-{folder.name}", f"  configuration: --enable-gpl {flag}"]
    if sys.platform == "win32":
        path = folder / f"{name}.bat"
        path.write_text("@echo off\r\n" + "".join(f"echo {line}\r\n" for line in lines), encoding="ascii")
    else:
        path = folder / name
        path.write_text("#!/bin/sh\n" + "".join(f"echo '{line}'\n" for line in lines), encoding="ascii")
        path.chmod(0o755)
    return path


def test_ffmpeg_with_libass_wins_over_earlier_build_without_it(tmp_path, monkeypatch):
    old = make_fake_ffmpeg(tmp_path / "ffmpeg-2026-09-21" / "bin", libass=False)
    make_fake_ffmpeg(tmp_path / "ffmpeg-2026-09-21" / "bin", libass=False, name="ffprobe")
    new = make_fake_ffmpeg(tmp_path / "winget" / "links", libass=True)
    make_fake_ffmpeg(tmp_path / "winget" / "links", libass=True, name="ffprobe")
    make_fake_ffmpeg(tmp_path, libass=False)  # в текущей папке: не в PATH, брать нельзя
    search_path = os.pathsep.join([str(old.parent), str(new.parent)])
    monkeypatch.setenv("PATH", search_path)

    candidates = env.ffmpeg_candidates()
    assert [(c.path, c.libass) for c in candidates] == [(str(old), False), (str(new), True)]

    tool = env.require_ffmpeg()
    assert tool.path == str(new)
    assert tool.libass is True
    assert tool.version == "fake-links"
    assert env.find_ffprobe(tool).path == str(new.parent / new.name.replace("ffmpeg", "ffprobe"))


def test_ffmpeg_first_in_path_is_kept_when_libass_is_nowhere(tmp_path, monkeypatch):
    first = make_fake_ffmpeg(tmp_path / "a", libass=False)
    make_fake_ffmpeg(tmp_path / "b", libass=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path / "a"), str(tmp_path / "b")]))
    tool = env.find_ffmpeg()
    assert tool.path == str(first)
    assert tool.libass is False


def test_choose_ffmpeg_keeps_path_order_among_libass_builds():
    candidates = [
        env.FfmpegCandidate("/a/ffmpeg", False),
        env.FfmpegCandidate("/b/ffmpeg", None),
        env.FfmpegCandidate("/c/ffmpeg", True),
        env.FfmpegCandidate("/d/ffmpeg", True),
    ]
    assert env.choose_ffmpeg(candidates).path == "/c/ffmpeg"
    assert env.choose_ffmpeg(candidates[:2]).path == "/a/ffmpeg"
    assert env.choose_ffmpeg([]) is None


def test_buildconf_parsing():
    assert env.buildconf_has_libass("  configuration:\n    --enable-gpl\n    --enable-libass\n")
    assert not env.buildconf_has_libass("  configuration:\n    --enable-gpl\n")
