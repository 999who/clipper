"""Запуск ffmpeg и ffprobe.

Сейчас здесь служебные запросы (фильтры, кодировщики, проверка NVENC) и чтение
параметров видео через ffprobe. Запуск рендера с прогрессом и отменой появится
вместе с этапом нарезки.
"""

import json
import re
from dataclasses import dataclass
from functools import cache
from typing import Any

from clipper.core.env import run_command
from clipper.core.errors import ClipperError


@dataclass(frozen=True)
class MediaInfo:
    duration: float  # с
    width: int  # как видит зритель (с учётом поворота)
    height: int
    fps: float
    video_codec: str
    has_audio: bool


def probe(path: str, ffprobe: str) -> MediaInfo:
    """Параметры видеофайла через ffprobe."""
    args = [ffprobe, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        result = run_command(args, timeout=60)
    except OSError as exc:
        raise ClipperError(f"ffprobe не запустился: {exc}") from None
    if result.returncode != 0:
        lines = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        reason = lines[-1] if lines else f"код выхода {result.returncode}"
        raise ClipperError(f"Не удалось прочитать видео {path}: {reason}", hint="Файл повреждён или это не видео.")
    try:
        data = json.loads(result.stdout)
    except ValueError:
        raise ClipperError(f"ffprobe вернул непонятный ответ для {path}") from None
    return parse_probe(data, path)


def parse_probe(data: dict[str, Any], path: str = "") -> MediaInfo:
    """Разобрать JSON от `ffprobe -show_format -show_streams`."""
    streams = data.get("streams") or []
    video = next(
        (s for s in streams if s.get("codec_type") == "video" and not (s.get("disposition") or {}).get("attached_pic")),
        None,
    )
    if video is None:
        raise ClipperError(f"В файле нет видеодорожки: {path}", hint="Нужен видеофайл, а не только звук или картинка.")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if abs(_rotation(video)) % 180 == 90:
        width, height = height, width
    duration = _float((data.get("format") or {}).get("duration")) or _float(video.get("duration"))
    if not duration:
        raise ClipperError(f"Не удалось узнать длительность видео: {path}")
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return MediaInfo(
        duration=duration,
        width=width,
        height=height,
        fps=round(fps, 3),
        video_codec=str(video.get("codec_name") or "?"),
        has_audio=any(s.get("codec_type") == "audio" for s in streams),
    )


def _rotation(stream: dict[str, Any]) -> int:
    for side_data in stream.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(_float(side_data["rotation"]) or 0)
    return int(_float((stream.get("tags") or {}).get("rotate")) or 0)


def _rate(value: Any) -> float:
    """«30000/1001» → 29.97; «0/0» и мусор → 0."""
    if not isinstance(value, str) or "/" not in value:
        return _float(value) or 0.0
    num, _, den = value.partition("/")
    numerator, denominator = _float(num), _float(den)
    return numerator / denominator if numerator and denominator else 0.0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_FILTER_LINE = re.compile(r"^\s*[T.][S.][C.]\s+(\S+)\s+\S*->\S*", re.M)
_ENCODER_LINE = re.compile(r"^\s*[VAS][F.][S.][X.][B.][D.]\s+(\S+)", re.M)


def parse_filters(listing: str) -> frozenset[str]:
    """Имена фильтров из вывода `ffmpeg -filters`."""
    return frozenset(_FILTER_LINE.findall(listing))


def parse_encoders(listing: str) -> frozenset[str]:
    """Имена кодировщиков из вывода `ffmpeg -encoders` (строки легенды пропускаются)."""
    return frozenset(name for name in _ENCODER_LINE.findall(listing) if name != "=")


@cache
def list_filters(ffmpeg: str) -> frozenset[str]:
    """Фильтры, собранные в этой сборке ffmpeg."""
    return parse_filters(run_command([ffmpeg, "-hide_banner", "-filters"], timeout=20).stdout)


@cache
def list_encoders(ffmpeg: str) -> frozenset[str]:
    """Кодировщики, собранные в этой сборке ffmpeg."""
    return parse_encoders(run_command([ffmpeg, "-hide_banner", "-encoders"], timeout=20).stdout)


def encoder_works(ffmpeg: str, encoder: str) -> tuple[bool, str]:
    """Закодировать несколько чёрных кадров, чтобы узнать, работает ли кодировщик.

    Наличие кодировщика в сборке ещё ничего не гарантирует: NVENC требует
    видеокарту NVIDIA и свежий драйвер. Возвращает (работает, причина сбоя).
    """
    args = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:r=25:d=0.2",
        "-c:v", encoder, "-f", "null", "-",
    ]  # fmt: skip
    try:
        result = run_command(args, timeout=30)
    except OSError as exc:
        return False, str(exc)
    if result.returncode == 0:
        return True, ""
    return False, failure_reason(result.stderr, encoder, result.returncode)


def failure_reason(stderr: str, encoder: str, returncode: int) -> str:
    """Самая полезная строка из ошибки ffmpeg.

    Причину обычно называет первая строка от самого кодировщика:
    «[h264_nvenc @ 0x…] Cannot load nvcuda.dll», «Driver does not support…».
    Последние строки («Nothing was written into output file») — лишь следствие.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    own = [line for line in lines if line.startswith(f"[{encoder} @")]
    reason = (own or lines or [f"код выхода {returncode}"])[0]
    return re.sub(r"^\[[^\]]*\]\s*", "", reason)
