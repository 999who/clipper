"""Данные, которые этапы передают друг другу и сохраняют на диск.

Здесь только структуры и их (де)сериализация — без обращения к сети и ffmpeg.
"""

import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
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


# --- проект ---------------------------------------------------------------------------

PROJECT_FILENAME = "project.json"  # ← этот файл пользователь правит руками
PROJECT_VERSION = 1

_CLIP_KEYS = {"id", "enabled", "start", "end", "score", "reason", "words"}


class ProjectError(ValueError):
    """project.json испорчен или заполнен неправильно (текст — для пользователя)."""


@dataclass
class Clip:
    id: int
    start: float  # с от начала исходного видео
    end: float
    score: float = 0.0
    reason: str = ""
    enabled: bool = True
    words: list[Word] = field(default_factory=list)  # с запасом за границами клипа

    @property
    def duration(self) -> float:
        return self.end - self.start

    def spoken_words(self) -> list[Word]:
        """Слова внутри границ клипа."""
        return [w for w in self.words if w.end > self.start and w.start < self.end]


@dataclass
class Project:
    source: SourceInfo  # без heatmap: он лежит в source.json
    mode: str
    keywords: list[str]
    language: str | None
    created: str
    clips: list[Clip]

    def to_dict(self) -> dict[str, Any]:
        source = self.source.to_dict()
        source.pop("heatmap", None)
        return {
            "version": PROJECT_VERSION,
            "_help": (
                "Можно править: start/end клипа (ЧЧ:ММ:СС.мс или секунды), enabled (false — пропустить клип), "
                "text слов (исправит субтитры). Время слов — секунды от начала исходного видео."
            ),
            "source": source,
            "analysis": {
                "mode": self.mode,
                "keywords": self.keywords,
                "language": self.language,
                "created": self.created,
            },
            "clips": [
                {
                    "id": clip.id,
                    "enabled": clip.enabled,
                    "start": format_time(clip.start),
                    "end": format_time(clip.end),
                    "score": round(clip.score, 3),
                    "reason": clip.reason,
                    "words": [{"text": w.text, "start": round(w.start, 3), "end": round(w.end, 3)} for w in clip.words],
                }
                for clip in self.clips
            ],
        }


def project_to_json(project: Project) -> str:
    """JSON проекта для ручной правки: разделы — блоками, каждое слово — одной строкой."""
    return _dump(project.to_dict(), 0) + "\n"


def _dump(value: Any, level: int) -> str:
    pad, inner = "  " * level, "  " * (level + 1)
    if isinstance(value, dict) and value:
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {_dump(v, level + 1)}" for k, v in value.items()]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list) and value and all(_is_leaf_dict(v) for v in value):
        items = [inner + json.dumps(v, ensure_ascii=False) for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    if isinstance(value, list) and value and any(isinstance(v, (dict, list)) for v in value):
        return "[\n" + ",\n".join(inner + _dump(v, level + 1) for v in value) + f"\n{pad}]"
    return json.dumps(value, ensure_ascii=False)


def _is_leaf_dict(value: Any) -> bool:
    return isinstance(value, dict) and all(not isinstance(v, (dict, list)) for v in value.values())


def save_project(work_dir: Path, project: Project) -> Path:
    path = work_dir / PROJECT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(project_to_json(project))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return path


def load_project(path: Path) -> Project:
    """Прочитать project.json (в том числе поправленный руками) с проверкой."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ProjectError(f"не удалось прочитать {path}: {exc.strerror}") from None
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProjectError(
            f"строка {exc.lineno}, столбец {exc.colno}: {exc.msg} — "
            "проверьте запятые между элементами и кавычки вокруг текста"
        ) from None
    return parse_project(data)


def parse_project(data: Any) -> Project:
    if not isinstance(data, dict):
        raise ProjectError("ожидался объект { … } с разделами source, analysis, clips")
    if data.get("version") != PROJECT_VERSION:
        raise ProjectError(f"неизвестная версия файла: {data.get('version')!r} (ожидалась {PROJECT_VERSION})")
    try:
        source_data = dict(data["source"])
        source_data["heatmap"] = None
        source = SourceInfo.from_dict(source_data)
    except (KeyError, TypeError, ValueError) as exc:
        raise ProjectError(f"раздел source испорчен: {exc}") from None
    analysis = data.get("analysis") or {}
    raw_clips = data.get("clips")
    if not isinstance(raw_clips, list):
        raise ProjectError("раздел clips должен быть списком [ … ]")
    clips = [_parse_clip(raw, index, source.duration) for index, raw in enumerate(raw_clips, start=1)]
    ids = [clip.id for clip in clips]
    if len(set(ids)) != len(ids):
        raise ProjectError(f"у клипов повторяются id: {sorted(i for i in set(ids) if ids.count(i) > 1)}")
    return Project(
        source=source,
        mode=str(analysis.get("mode") or ""),
        keywords=list(analysis.get("keywords") or []),
        language=analysis.get("language"),
        created=str(analysis.get("created") or ""),
        clips=clips,
    )


def _parse_clip(raw: Any, index: int, duration: float) -> Clip:
    where = f"клип №{index}"
    if not isinstance(raw, dict):
        raise ProjectError(f"{where}: ожидался объект {{ … }}")
    if "id" in raw:
        where = f"клип id={raw['id']}"
    unknown = set(raw) - _CLIP_KEYS
    if unknown:
        raise ProjectError(f"{where}: неизвестные поля {sorted(unknown)}; допустимы {sorted(_CLIP_KEYS)}")
    try:
        clip_id = int(raw["id"])
        start = parse_time(raw["start"])
        end = parse_time(raw["end"])
    except KeyError as exc:
        raise ProjectError(f"{where}: нет поля {exc}") from None
    except (TypeError, ValueError) as exc:
        raise ProjectError(f"{where}: {exc}") from None
    if end <= start:
        raise ProjectError(f"{where}: конец ({format_time(end)}) должен быть позже начала ({format_time(start)})")
    if end > duration + 0.5:
        raise ProjectError(f"{where}: конец {format_time(end)} дальше конца видео ({format_time(duration)})")
    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ProjectError(f"{where}: enabled должно быть true или false")
    words = []
    for number, item in enumerate(raw.get("words") or [], start=1):
        try:
            words.append(Word(str(item["text"]), float(item["start"]), float(item["end"])))
        except (KeyError, TypeError, ValueError):
            raise ProjectError(f"{where}, слово №{number}: нужны text, start и end (числа)") from None
    words.sort(key=lambda w: w.start)
    return Clip(
        id=clip_id,
        start=start,
        end=min(end, duration),
        score=float(raw.get("score") or 0.0),
        reason=str(raw.get("reason") or ""),
        enabled=enabled,
        words=words,
    )
