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


def write_json_atomic(path: Path, data: Any, indent: int | None = 2) -> None:
    """Записать JSON так, чтобы сбой посреди записи не оставил битый файл."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=indent)
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


# --- распознавание ------------------------------------------------------------------

TRANSCRIPT_FILENAME = "transcript.json"  # кэш распознавания в work/<id>/
SRT_FILENAME = "transcript.srt"  # тот же текст субтитрами — проверить в плеере


@dataclass(frozen=True)
class Word:
    text: str  # как распознал Whisper, с пунктуацией: «смотри,»
    start: float  # с от начала исходного видео
    end: float
    prob: float = 1.0  # уверенность Whisper 0…1


@dataclass(frozen=True)
class Segment:
    """Фраза Whisper: текст и слова с таймкодами."""

    start: float
    end: float
    text: str
    words: tuple[Word, ...] = ()


@dataclass
class Transcript:
    """Распознанные отрезки исходного видео.

    `ranges` — какие участки видео уже распознаны (в режиме heatmap это не всё
    видео, а окна вокруг пиков). `settings` — параметры распознавания: если они
    поменялись, кэш не используется.
    """

    language: str | None
    settings: dict[str, Any]
    ranges: list[tuple[float, float]]
    segments: list[Segment]

    @property
    def words(self) -> list[Word]:
        return [word for segment in self.segments for word in segment.words]

    def words_between(self, start: float, end: float) -> list[Word]:
        """Слова, которые целиком или частично попадают в [start, end]."""
        return [w for w in self.words if w.end > start and w.start < end]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "language": self.language,
            "settings": self.settings,
            "ranges": [[round(a, 3), round(b, 3)] for a, b in self.ranges],
            "segments": [
                {
                    "start": round(s.start, 3),
                    "end": round(s.end, 3),
                    "text": s.text,
                    # [текст, начало, конец, уверенность] — компактно: слов бывают десятки тысяч
                    "words": [[w.text, round(w.start, 3), round(w.end, 3), round(w.prob, 3)] for w in s.words],
                }
                for s in self.segments
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Transcript":
        return cls(
            language=data.get("language"),
            settings=dict(data.get("settings") or {}),
            ranges=[(float(a), float(b)) for a, b in data.get("ranges") or []],
            segments=[
                Segment(
                    float(s["start"]),
                    float(s["end"]),
                    str(s["text"]),
                    tuple(Word(str(t), float(a), float(b), float(p)) for t, a, b, p in s.get("words") or []),
                )
                for s in data.get("segments") or []
            ],
        )


def save_transcript(work_dir: Path, transcript: Transcript) -> Path:
    path = work_dir / TRANSCRIPT_FILENAME
    write_json_atomic(path, transcript.to_dict(), indent=None)  # кэш: компактно, руками его не правят
    return path


def load_transcript(work_dir: Path) -> Transcript | None:
    """Прочитать кэш распознавания. None — если его нет или он испорчен."""
    try:
        data = json.loads((work_dir / TRANSCRIPT_FILENAME).read_text(encoding="utf-8"))
        return Transcript.from_dict(data)
    except (OSError, ValueError, TypeError, KeyError):
        return None


# --- отрезки времени ------------------------------------------------------------------


def merge_ranges(ranges: list[tuple[float, float]], gap: float = 0.0) -> list[tuple[float, float]]:
    """Объединить пересекающиеся (и отстоящие не больше чем на `gap`) отрезки."""
    result: list[tuple[float, float]] = []
    for start, end in sorted((a, b) for a, b in ranges if b > a):
        if result and start <= result[-1][1] + gap:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def subtract_ranges(
    wanted: list[tuple[float, float]], done: list[tuple[float, float]], min_length: float = 0.0
) -> list[tuple[float, float]]:
    """Части `wanted`, которые не покрыты `done` (короче `min_length` — отбрасываются)."""
    result: list[tuple[float, float]] = []
    covered = merge_ranges(done)
    for start, end in merge_ranges(wanted):
        cursor = start
        for a, b in covered:
            if b <= cursor or a >= end:
                continue
            if a > cursor:
                result.append((cursor, a))
            cursor = max(cursor, b)
        if cursor < end:
            result.append((cursor, end))
    return [(a, b) for a, b in result if b - a >= min_length]


# --- время в тексте -----------------------------------------------------------------


def parse_time(text: str | float | int) -> float:
    """«90», «1:30», «1:02:03.5», «01:02:03,500» → секунды."""
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        if text < 0:
            raise ValueError("время не может быть отрицательным")
        return float(text)
    value = str(text).strip().replace(",", ".")
    parts = value.split(":")
    if not value or len(parts) > 3:
        raise ValueError(f"не понимаю время «{text}»")
    try:
        numbers = [float(part) for part in parts]
    except ValueError:
        raise ValueError(f"не понимаю время «{text}»") from None
    if any(n < 0 for n in numbers) or any(n >= 60 for n in numbers[1:]):
        raise ValueError(f"не понимаю время «{text}»")
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds


def format_time(seconds: float) -> str:
    """Секунды → «00:02:03.400»."""
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
