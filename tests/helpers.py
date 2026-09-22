"""Общие помощники тестов."""

import shutil
import subprocess
from pathlib import Path

import pytest

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="нет ffmpeg/ffprobe"
)


def make_video(path: Path, seconds: float = 2, size: str = "320x240", audio: bool = True) -> Path:
    """Короткое тестовое видео, сгенерированное ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate=25:duration={seconds}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(args, check=True)
    return path
