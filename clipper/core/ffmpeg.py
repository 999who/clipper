"""Запуск ffmpeg и ffprobe.

Пока здесь только служебные запросы: список фильтров и кодировщиков, проверка
NVENC. Запуск рендера с прогрессом и отменой появится вместе с этапом нарезки.
"""

import re
from functools import cache

from clipper.core.env import run_command

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
