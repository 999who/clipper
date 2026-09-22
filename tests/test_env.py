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


def test_require_ffmpeg_explains_how_to_install(monkeypatch):
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
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
