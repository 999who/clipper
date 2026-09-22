"""Этап 4: что вырезать из клипа — паузы и слова-паразиты.

- **Паузы.** Два способа (`audio.pause_detect`):
  - `volume` — ffmpeg `silencedetect`: тише `audio.silence_db` дольше
    `audio.min_pause`. Отрезки, где Whisper слышал слово, не режутся никогда.
  - `words` — промежутки между распознанными словами. Нужен для стримов: из-за
    игры и музыки настоящей тишины там почти не бывает. Промежутки длиннее
    `audio.max_gap` не трогаются: там обычно смех, крик или игра, которые
    Whisper не записывает, а не пауза.

  От паузы остаётся по `PAUSE_KEEP` с с каждой стороны, чтобы речь не звучала
  рублено.
- **Слова-паразиты** — список `audio.fillers` (можно фразы: «как бы»). «Э-э»,
  «эээ» и «ээ» считаются одним словом.

Результат — `ClipEdit`: Timeline (что остаётся) и что именно вырезано.

Если по громкости пауз не нашлось, `quietest_level` меряет самое тихое место
клипа — по нему интерфейс подсказывает, какой порог silence_db поставить.
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from clipper.core.config import Config
from clipper.core.env import run_command
from clipper.core.errors import ClipperError
from clipper.core.highlights import normalize_word
from clipper.core.models import Clip, Word, merge_ranges, subtract_ranges
from clipper.core.timeline import Timeline

PAUSE_KEEP = 0.12  # с: столько паузы остаётся с каждой стороны
FILLER_PAD = 0.04  # с: запас вокруг слова-паразита (не залезая в соседние слова)
NOISY_FLOOR = -30  # дБ: если тишина не опускается ниже, советуем искать паузы по словам
MAX_PIECES = 60  # больше кусков ffmpeg-граф не усложняем: самые короткие паузы остаются

_SILENCE_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass
class ClipEdit:
    timeline: Timeline
    pauses: list[tuple[float, float]] = field(default_factory=list)  # вырезанные паузы (исходник)
    fillers: list[Word] = field(default_factory=list)  # вырезанные слова-паразиты
    quietest: float | None = None  # дБ: самое тихое место клипа, если по громкости пауз не нашлось

    @property
    def paused_seconds(self) -> float:
        return sum(b - a for a, b in self.pauses)


def plan_edit(clip: Clip, cfg: Config, silences: list[tuple[float, float]] | None = None) -> ClipEdit:
    """Что вырезать из клипа по настройкам `audio.*`.

    `silences` — тишина из silencedetect (для pause_detect: volume); без неё
    паузы ищутся по словам.
    """
    audio = cfg.audio
    words = sorted(clip.words, key=lambda w: w.start)
    cuts: list[tuple[float, float]] = []
    pauses: list[tuple[float, float]] = []
    fillers: list[Word] = []

    if audio.cut_pauses:
        if silences is not None:
            found = silences
        else:
            found = word_gaps(words, clip.start, clip.end)
            if audio.max_gap > 0:
                found = [(a, b) for a, b in found if b - a <= audio.max_gap]
        found = protect_words(found, words)
        pauses = shrink_pauses(found, audio.min_pause, clip.start, clip.end)
        pauses = _limit_pieces(pauses)
        cuts += pauses

    if audio.remove_fillers and audio.fillers:
        spans, fillers = filler_cuts(words, audio.fillers, clip.start, clip.end)
        cuts += spans

    if not cuts:
        return ClipEdit(Timeline.whole(clip.start, clip.end))
    timeline = Timeline.with_cuts(clip.start, clip.end, cuts)
    if timeline.duration < 1.0:  # вырезали почти всё — лучше не трогать клип
        return ClipEdit(Timeline.whole(clip.start, clip.end))
    return ClipEdit(timeline, pauses, fillers)


# --- паузы --------------------------------------------------------------------------


def word_gaps(words: list[Word], start: float, end: float) -> list[tuple[float, float]]:
    """Промежутки без слов внутри клипа, включая начало и конец."""
    inside = [w for w in words if w.end > start and w.start < end]
    gaps, cursor = [], start
    for word in inside:
        if word.start > cursor:
            gaps.append((cursor, word.start))
        cursor = max(cursor, word.end)
    if cursor < end:
        gaps.append((cursor, end))
    return gaps


def protect_words(silences: list[tuple[float, float]], words: list[Word]) -> list[tuple[float, float]]:
    """Убрать из «тишины» всё, где Whisper слышал слово: тихое слово резать нельзя."""
    return subtract_ranges(merge_ranges(silences), [(w.start, w.end) for w in words])


def shrink_pauses(
    silences: list[tuple[float, float]], min_pause: float, start: float, end: float
) -> list[tuple[float, float]]:
    """Паузы не короче min_pause → что вырезать (с каждой стороны остаётся PAUSE_KEEP).

    Пауза в самом начале или конце клипа вырезается целиком до края клипа.
    """
    cuts = []
    for a, b in merge_ranges(silences):
        a, b = max(a, start), min(b, end)
        if b - a < min_pause:
            continue
        cut_start = a if a <= start + 1e-6 else a + PAUSE_KEEP
        cut_end = b if b >= end - 1e-6 else b - PAUSE_KEEP
        if cut_end - cut_start > 0.05:
            cuts.append((round(cut_start, 3), round(cut_end, 3)))
    return cuts


def _limit_pieces(pauses: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(pauses) <= MAX_PIECES:
        return pauses
    longest = sorted(pauses, key=lambda p: p[1] - p[0], reverse=True)[:MAX_PIECES]
    return sorted(longest)


def detect_silences(
    ffmpeg: str, video: Path, start: float, end: float, silence_db: float, min_pause: float
) -> list[tuple[float, float]]:
    """Тишина в [start, end] исходника через ffmpeg silencedetect (время исходника)."""
    args = [
        ffmpeg, "-hide_banner", "-nostdin", "-nostats",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
        "-map", "0:a:0", "-af", f"silencedetect=noise={silence_db:g}dB:d={min_pause:g}",
        "-f", "null", "-",
    ]  # fmt: skip
    try:
        result = run_command(args, timeout=300)
    except OSError as exc:
        raise ClipperError(f"Не удалось найти паузы: {exc}") from None
    if result.returncode != 0:
        lines = [line for line in result.stderr.splitlines() if line.strip()]
        raise ClipperError(f"ffmpeg не смог найти паузы: {lines[-1] if lines else result.returncode}")
    return parse_silences(result.stderr, offset=start, length=end - start)


def quietest_level(ffmpeg: str, video: Path, start: float, end: float, min_pause: float) -> float | None:
    """Пиковая громкость (дБ) самого тихого отрезка длиной min_pause в [start, end].

    silencedetect найдёт паузу, только если порог выше этого числа. None — если
    звук прочитать не удалось или клип короче min_pause.
    """
    import numpy as np

    rate = 8000
    args = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
        "-map", "0:a:0", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-",
    ]  # fmt: skip
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=300)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    samples = np.abs(np.frombuffer(result.stdout, dtype=np.int16).astype(np.float32))
    return window_min_peak(samples / 32768.0, rate, min_pause)


def window_min_peak(samples, rate: int, window: float) -> float | None:
    """Минимум по всем окнам длиной `window` от пика |сэмпла| в окне, в дБ."""
    import numpy as np

    block = rate // 20  # пики по 50 мс, окно — несколько блоков подряд
    count = len(samples) // block
    size = max(1, round(window * 20))
    if count < size:
        return None
    peaks = samples[: count * block].reshape(count, block).max(axis=1)
    windows = np.lib.stride_tricks.sliding_window_view(peaks, size).max(axis=1)
    return round(20 * float(np.log10(max(float(windows.min()), 1e-5))), 1)


def pause_hint(levels: dict[int, float], silence_db: float) -> str:
    """Подсказка, когда по громкости пауз не нашлось: {id клипа: самое тихое место, дБ}."""
    ids = ("клип " if len(levels) == 1 else "клипы ") + ", ".join(str(i) for i in sorted(levels))
    quietest = round(min(levels.values()))
    text = f"Паузы тише {silence_db:g} дБ не найдены ({ids}), самое тихое место: {quietest} дБ"
    if quietest > NOISY_FLOOR:
        return (
            text + ". Звук там почти не затихает (музыка, игра, улица) — лучше искать паузы по словам: "
            f"--pause-detect words. Или поднимите порог: --silence-db {quietest + 5}."
        )
    return text + f". Попробуйте --silence-db {quietest + 5} или --pause-detect words, если это музыка или игра."


def parse_silences(log: str, offset: float = 0.0, length: float | None = None) -> list[tuple[float, float]]:
    """Вывод silencedetect → [(начало, конец)] со сдвигом `offset`.

    Тишина, не закончившаяся до конца куска, длится до `length`.
    """
    silences, current = [], None
    for line in log.splitlines():
        if match := _SILENCE_START.search(line):
            current = max(float(match.group(1)), 0.0)
        elif (match := _SILENCE_END.search(line)) and current is not None:
            silences.append((round(offset + current, 3), round(offset + float(match.group(1)), 3)))
            current = None
    if current is not None and length is not None:
        silences.append((round(offset + current, 3), round(offset + length, 3)))
    return silences


# --- слова-паразиты -----------------------------------------------------------------


def filler_key(text: str) -> str:
    """«Э-э-э» → «э», «Эмм,» → «эм», «Ну» → «ну»: повторы букв и дефисы не важны."""
    word = normalize_word(text).replace("-", "")
    return re.sub(r"(.)\1+", r"\1", word)


def filler_cuts(
    words: list[Word], fillers: list[str], start: float, end: float
) -> tuple[list[tuple[float, float]], list[Word]]:
    """Где в клипе звучат слова-паразиты: отрезки для вырезания и сами слова."""
    patterns = {tuple(filler_key(part) for part in f.split()) for f in fillers}
    patterns = sorted((p for p in patterns if p and all(p)), key=len, reverse=True)  # длинные фразы — первыми
    keys = [filler_key(w.text) for w in words]
    cuts, removed = [], []
    index = 0
    while index < len(words):
        pattern = next((p for p in patterns if tuple(keys[index : index + len(p)]) == p), None)
        if pattern is None or not (start <= words[index].start and words[index + len(pattern) - 1].end <= end):
            index += 1
            continue
        first, last = words[index], words[index + len(pattern) - 1]
        before = words[index - 1].end if index > 0 else start
        after = words[index + len(pattern)].start if index + len(pattern) < len(words) else end
        cuts.append((round(max(first.start - FILLER_PAD, before), 3), round(min(last.end + FILLER_PAD, after), 3)))
        removed.extend(words[index : index + len(pattern)])
        index += len(pattern)
    return cuts, removed
