"""Запуск ffmpeg и ffprobe.

Сейчас здесь служебные запросы (фильтры, кодировщики, проверка NVENC) и чтение
параметров видео через ffprobe. Запуск рендера с прогрессом и отменой появится
вместе с этапом нарезки.
"""

import collections
import json
import logging
import re
import subprocess
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

from clipper.core.env import run_command
from clipper.core.errors import ClipperError
from clipper.core.events import CancelToken

log = logging.getLogger(__name__)


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


# Флаги фильтра: «TSC» в ffmpeg до 7.x, «TS» в новых (флаг команд убрали).
_FILTER_LINE = re.compile(r"^\s*[T.][S.][C.]?\s+(\S+)\s+\S*->\S*", re.M)
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


def run_ffmpeg(
    ffmpeg: str,
    args: Sequence[str],
    *,
    on_progress: Callable[[float], None] | None = None,
    cancel: CancelToken | None = None,
    log_path: Path | None = None,
    cwd: Path | None = None,
) -> None:
    """Запустить ffmpeg с прогрессом и отменой.

    `on_progress(секунды)` получает позицию обработки из `-progress pipe:1`.
    При отмене (или любом исключении из on_progress) процесс ffmpeg завершается.
    Команда и вывод ошибок дописываются в `log_path` (work/<id>/clipper.log).
    """
    command = [ffmpeg, "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-nostats", "-progress", "pipe:1"]
    command += list(args)
    log.debug("ffmpeg: %s", subprocess.list2cmdline(command))
    tail: collections.deque[str] = collections.deque(maxlen=30)
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=cwd,
    )
    # stderr читаем в отдельном потоке, иначе ffmpeg может встать на заполненном буфере.
    reader = threading.Thread(target=lambda: tail.extend(process.stderr or []), daemon=True)
    reader.start()
    try:
        for line in process.stdout or []:
            key, _, value = line.strip().partition("=")
            if key == "out_time_us" and value.isdigit() and on_progress is not None:
                on_progress(int(value) / 1_000_000)
            if cancel is not None:
                cancel.check()
    except BaseException:
        process.kill()
        process.wait()
        raise
    finally:
        reader.join(timeout=5)
    code = process.wait()
    errors = [line.rstrip() for line in tail if line.strip()]
    if log_path is not None:
        _append_log(log_path, command, code, errors)
    if code != 0:
        # Последние строки ошибки без префиксов «[in#0 @ 0x…]», без повторов.
        cleaned = list(dict.fromkeys(re.sub(r"^\[[^\]]*\]\s*", "", line) for line in errors[-4:]))
        reason = " / ".join(cleaned) if cleaned else f"код выхода {code}"
        raise ClipperError(f"ffmpeg завершился с ошибкой: {reason}", hint=_log_hint(log_path))


def _append_log(log_path: Path, command: list[str], code: int, errors: list[str]) -> None:
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as file:
            file.write(f"$ {subprocess.list2cmdline(command)}\n")
            for line in errors:
                file.write(f"  {line}\n")
            file.write(f"  -> код выхода {code}\n\n")
    except OSError:
        log.debug("не удалось записать лог %s", log_path)


def _log_hint(log_path: Path | None) -> str | None:
    return f"Полная команда и вывод ffmpeg — в {log_path}" if log_path else None
