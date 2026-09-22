"""Данные, которые этапы передают друг другу и сохраняют на диск.

Здесь только структуры и их (де)сериализация — без обращения к сети и ffmpeg.
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

SOURCE_FILENAME = "source.json"  # наш итог этапа 1 в work/<id>/
INFO_FILENAME = "info.json"  # полные метаданные yt-dlp в work/<id>/


@dataclass(frozen=True)
class HeatPoint:
    """Один отрезок графика «Самые популярные фрагменты» YouTube."""

    start: float  # с
    end: float  # с
    value: float  # 0…1; 1 — самое пересматриваемое место видео


@dataclass
class SourceInfo:
    """Исходное видео: где лежит, что это за файл и есть ли у него heatmap."""

    id: str  # имя рабочей папки work/<id>/
    kind: Literal["youtube", "url", "file"]
    input: str  # что передал пользователь: ссылка или путь
    video: str  # абсолютный путь к видеофайлу
    title: str
    duration: float  # с
    width: int
    height: int
    fps: float
    has_audio: bool
    heatmap: list[HeatPoint] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceInfo":
        values = dict(data)
        heatmap = values.get("heatmap")
        values["heatmap"] = None if heatmap is None else [HeatPoint(**point) for point in heatmap]
        return cls(**values)


# --- heatmap ------------------------------------------------------------------------


def parse_heatmap(raw: Any) -> list[HeatPoint] | None:
    """Heatmap из данных yt-dlp: [{start_time, end_time, value}, …] → список HeatPoint.

    Битые точки пропускаются, value ограничивается диапазоном 0…1, точки
    сортируются по времени. Если пригодных точек нет — None (heatmap нет).
    """
    if not isinstance(raw, list):
        return None
    points: list[HeatPoint] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        start = _number(item.get("start_time"))
        end = _number(item.get("end_time"))
        value = _number(item.get("value"))
        if start is None or end is None or value is None or end <= start or start < 0:
            continue
        points.append(HeatPoint(start, end, min(max(value, 0.0), 1.0)))
    points.sort(key=lambda point: point.start)
    return points or None


def heatmap_profile(points: list[HeatPoint], bins: int, duration: float | None = None) -> list[float]:
    """Сжать или растянуть heatmap до `bins` столбиков (для мини-графика).

    Значение столбика — максимум точек, которые его задевают: так узкий пик не
    теряется при сжатии. Участки без точек — 0.
    """
    if bins <= 0 or not points:
        return [0.0] * max(bins, 0)
    end = max(duration or 0.0, points[-1].end)
    width = end / bins
    profile = [0.0] * bins
    for point in points:
        first = min(int(point.start / width), bins - 1)
        last = min(max(math.ceil(point.end / width - 1e-9) - 1, first), bins - 1)
        for index in range(first, last + 1):
            profile[index] = max(profile[index], point.value)
    return profile


def heatmap_peak(points: list[HeatPoint]) -> HeatPoint:
    """Самая пересматриваемая точка (при равенстве — более ранняя)."""
    return max(points, key=lambda point: (point.value, -point.start))


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


# --- файлы --------------------------------------------------------------------------


def write_json_atomic(path: Path, data: Any) -> None:
    """Записать JSON так, чтобы сбой посреди записи не оставил битый файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=2)
            file.write("\n")
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_source(work_dir: Path, source: SourceInfo) -> Path:
    path = work_dir / SOURCE_FILENAME
    write_json_atomic(path, source.to_dict())
    return path


def load_source(work_dir: Path) -> SourceInfo | None:
    """Прочитать work/<id>/source.json. None — если файла нет или он испорчен."""
    path = work_dir / SOURCE_FILENAME
    try:
        return SourceInfo.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError, KeyError):
        return None
