"""Этап 7: обложки клипов и склейка всех клипов в один ролик.

- **Обложка** `clip_NN.jpg` (1080×1920) — самый резкий из 8 кадров готового
  клипа (равномерно по 10–90 % длины). Резкость — дисперсия лапласиана в
  верхних 60 % кадра, чтобы на выбор не влияли субтитры. Кадры без субтитров
  на экране в приоритете. Слишком тёмные и засвеченные кадры (затемнения,
  вспышки) не берутся.
- **Склейка** `all_clips.mp4` — клипы этого рендера по порядку проекта, без
  перекодирования (concat demuxer, `-c copy`). Все клипы одного рендера
  закодированы одинаково, поэтому склеиваются без потерь и за секунды.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from clipper.core.errors import ClipperError
from clipper.core.events import CancelToken
from clipper.core.ffmpeg import run_ffmpeg

THUMB_CANDIDATES = 8
PREVIEW_W, PREVIEW_H = 270, 480  # кадры-кандидаты в уменьшенном виде
SHARP_REGION = 0.6  # резкость считается по верхним 60 % кадра (ниже — субтитры)
DARK, BRIGHT = 16, 240  # средняя яркость вне этих границ — затемнение или вспышка
CONCAT_NAME = "all_clips.mp4"


# --- обложка ---------------------------------------------------------------------------


def candidate_times(duration: float, count: int = THUMB_CANDIDATES) -> list[float]:
    """Моменты-кандидаты: равномерно по 10–90 % длины клипа."""
    if duration <= 0:
        return [0.0]
    start, span = 0.1 * duration, 0.8 * duration
    return [round(start + span * (i + 0.5) / count, 3) for i in range(count)]


def sharpness(gray: Any) -> float:
    """Дисперсия дискретного лапласиана по верхней части кадра (numpy, без OpenCV)."""
    import numpy as np

    top = gray[: max(3, int(gray.shape[0] * SHARP_REGION))].astype(np.float32)
    lap = 4 * top[1:-1, 1:-1] - top[:-2, 1:-1] - top[2:, 1:-1] - top[1:-1, :-2] - top[1:-1, 2:]
    return float(lap.var())


def pick_frame(frames: list[Any], times: list[float], captions: list[tuple[float, float]]) -> int:
    """Номер лучшего кадра: резкий, не тёмный, по возможности без субтитров на экране."""

    def usable(i: int) -> bool:
        return DARK <= float(frames[i].mean()) <= BRIGHT

    def clean(i: int) -> bool:
        return not any(a <= times[i] < b for a, b in captions)

    indices = list(range(len(frames)))
    for group in ([i for i in indices if usable(i) and clean(i)], [i for i in indices if usable(i)], indices):
        if group:
            return max(group, key=lambda i: sharpness(frames[i]))
    return 0


def caption_intervals(ass: Path | None) -> list[tuple[float, float]]:
    """Когда на экране субтитры (по .ass клипа), в секундах."""
    if ass is None or not ass.is_file():
        return []
    import pysubs2

    try:
        subs = pysubs2.load(str(ass), encoding="utf-8")
    except Exception:  # обложка не стоит ошибки рендера
        return []
    return [(event.start / 1000, event.end / 1000) for event in subs if not event.is_comment]


def read_frames(ffmpeg: str, video: Path, times: list[float], cancel: CancelToken | None = None) -> list[Any]:
    """Уменьшенные кадры (оттенки серого) в заданные моменты."""
    import numpy as np

    frames = []
    size = PREVIEW_W * PREVIEW_H
    for at in times:
        if cancel is not None:
            cancel.check()
        args = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(video),
                "-frames:v", "1", "-vf", f"scale={PREVIEW_W}:{PREVIEW_H}",
                "-f", "rawvideo", "-pix_fmt", "gray", "-"]  # fmt: skip
        try:
            data = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=60).stdout
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ClipperError(f"Не удалось прочитать кадр для обложки: {exc}") from None
        if len(data) >= size:
            frames.append(np.frombuffer(data[:size], np.uint8).reshape(PREVIEW_H, PREVIEW_W))
    return frames


def make_thumbnail(
    ffmpeg: str,
    video: Path,
    duration: float,
    target: Path,
    *,
    ass: Path | None = None,
    log_path: Path | None = None,
    cancel: CancelToken | None = None,
) -> tuple[Path, float]:
    """Сохранить обложку клипа в target (.jpg). Возвращает путь и момент кадра."""
    from clipper.core.render import publish

    times = candidate_times(duration)
    frames = read_frames(ffmpeg, video, times, cancel)
    if not frames:
        raise ClipperError("Не удалось прочитать ни одного кадра для обложки.")
    times = times[: len(frames)]
    at = times[pick_frame(frames, times, caption_intervals(ass))]
    tmp = target.with_name(target.stem + ".part.jpg")
    args = ["-ss", f"{at:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "2", "-update", "1", str(tmp)]
    try:
        run_ffmpeg(ffmpeg, args, cancel=cancel, log_path=log_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return publish(tmp, target), at


# --- склейка ---------------------------------------------------------------------------


def concat_list(paths: list[Path]) -> str:
    """Список для concat demuxer: file '…' с экранированием одинарных кавычек."""
    lines = ["ffconcat version 1.0"]
    for path in paths:
        quoted = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{quoted}'")
    return "\n".join(lines) + "\n"


def concat_clips(
    ffmpeg: str,
    paths: list[Path],
    target: Path,
    list_dir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
    log_path: Path | None = None,
    cancel: CancelToken | None = None,
) -> Path:
    """Склеить клипы в target без перекодирования."""
    from clipper.core.render import publish

    list_dir.mkdir(parents=True, exist_ok=True)
    listing = list_dir / "concat.txt"
    listing.write_text(concat_list(paths), encoding="utf-8")
    tmp = target.with_name(target.stem + ".part.mp4")
    args = ["-f", "concat", "-safe", "0", "-i", str(listing), "-map", "0:v:0", "-map", "0:a:0?", "-c", "copy",
            "-movflags", "+faststart", str(tmp)]  # fmt: skip
    try:
        run_ffmpeg(ffmpeg, args, on_progress=on_progress, cancel=cancel, log_path=log_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return publish(tmp, target)
