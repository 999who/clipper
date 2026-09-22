"""Этап 3: выбор моментов — по heatmap YouTube или по ключевым словам.

Здесь только чистая логика над данными (без ffmpeg, сети и GPU):

- `heatmap_candidates` — пики графика «Самые популярные фрагменты»: локальные
  максимумы, окно вокруг пика, объединение близких, топ-N;
- `keyword_candidates` — места транскрипта, где звучат ключевые слова;
- `snap_to_phrases` — границы клипа сдвигаются к началу/концу фраз, чтобы не
  резать слово посередине.
"""

import re
from collections import Counter
from dataclasses import dataclass

from clipper.core.models import HeatPoint, Word

# --- heatmap ---
MIN_PEAK = 0.3  # пик ниже 30 % максимума графика — не пик
SPREAD = 0.6  # окно растёт, пока «интерес» не ниже 60 % от пика
MERGE_GAP = 5.0  # с: окна ближе этого объединяются (если влезают в max_len)
PEAK_POSITION = 0.4  # пик ставится на 40 % клипа: до него — контекст, после — реакция

# --- keywords ---
KEYWORD_LEAD = 0.25  # доля длины клипа до первого ключевого слова
KEYWORD_TAIL = 0.25  # и после последнего

# --- границы фраз ---
SNAP_TOLERANCE = 3.0  # с: насколько можно сдвинуть границу к началу/концу фразы
PAUSE = 0.3  # с: пауза между словами, которая считается границей фразы
START_PAD = 0.15  # с: запас перед первым словом
END_PAD = 0.3  # с: запас после последнего слова
SENTENCE_END = (".", "!", "?", "…")


@dataclass(frozen=True)
class Candidate:
    start: float
    end: float
    score: float
    reason: str

    @property
    def length(self) -> float:
        return self.end - self.start


# --- heatmap ------------------------------------------------------------------------


def heatmap_candidates(
    points: list[HeatPoint], duration: float, count: int, min_len: float, max_len: float
) -> list[Candidate]:
    """Лучшие `count` моментов по heatmap, в порядке времени.

    Первая точка графика не считается пиком: её завышают все, кто просто
    начал смотреть видео. Сглаживать не нужно: YouTube отдаёт уже гладкую кривую.
    """
    if not points or count <= 0:
        return []
    values = [p.value for p in points]
    top = max(values[1:], default=0.0) or 1.0
    peaks = [
        i
        for i in range(1, len(points))
        if values[i] >= values[i - 1] and (i == len(points) - 1 or values[i] >= values[i + 1])
        if values[i] >= MIN_PEAK * top
    ]
    peaks.sort(key=lambda i: (-values[i], i))

    chosen: list[Candidate] = []
    for index in peaks:
        window = _peak_window(points, values, index, duration, min_len, max_len)
        clashes = [c for c in chosen if window.start < c.end + MERGE_GAP and window.end > c.start - MERGE_GAP]
        if clashes:
            if len(clashes) == 1:
                other = clashes[0]
                start, end = min(other.start, window.start), max(other.end, window.end)
                if end - start <= max_len:
                    chosen[chosen.index(other)] = Candidate(start, end, other.score, other.reason)
            continue
        chosen.append(window)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: c.start)


def _peak_window(
    points: list[HeatPoint], values: list[float], index: int, duration: float, min_len: float, max_len: float
) -> Candidate:
    peak = values[index]
    lo = hi = index
    while True:
        left = values[lo - 1] if lo > 0 and values[lo - 1] >= peak * SPREAD else None
        right = values[hi + 1] if hi < len(points) - 1 and values[hi + 1] >= peak * SPREAD else None
        if left is None and right is None:
            break
        new_lo, new_hi = (lo - 1, hi) if right is None or (left is not None and left >= right) else (lo, hi + 1)
        if points[new_hi].end - points[new_lo].start > max_len:
            break
        lo, hi = new_lo, new_hi

    start, end = points[lo].start, points[hi].end
    if not (min_len <= end - start <= max_len):
        # Узкий пик (или одна точка длиннее max_len на длинном видео): клип «средней»
        # длины, пик — на 40 % от начала.
        length = min(max((min_len + max_len) / 2, min_len), max_len)
        center = (points[index].start + points[index].end) / 2
        start = center - PEAK_POSITION * length
        end = start + length
    start, end = _fit(start, end, duration)
    return Candidate(round(start, 3), round(end, 3), round(peak, 3), f"пик heatmap {peak:.2f}")


def _fit(start: float, end: float, duration: float) -> tuple[float, float]:
    """Сдвинуть окно внутрь [0, duration], сохранив длину, если возможно."""
    length = min(end - start, duration)
    if start < 0:
        start, end = 0.0, length
    if end > duration:
        start, end = max(0.0, duration - length), duration
    return start, end


# --- ключевые слова -------------------------------------------------------------------


def normalize_word(text: str) -> str:
    """«Победа!» → «победа», «Ёлки» → «елки», «как-то,» → «как-то»."""
    text = text.lower().replace("ё", "е")
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)


def keyword_tokens(keyword: str) -> list[str]:
    """«да ладно» → ["да", "ладно"]; «побед*» → ["побед*"]."""
    tokens = []
    for raw in keyword.split():
        star = raw.endswith("*")
        token = normalize_word(raw.rstrip("*"))
        if token:
            tokens.append(token + ("*" if star else ""))
    return tokens


def _token_matches(token: str, word: str) -> bool:
    if token.endswith("*"):
        return word.startswith(token[:-1])
    return word == token


def find_keywords(words: list[Word], keywords: list[str]) -> list[tuple[float, float, str]]:
    """Все места, где звучат ключевые слова: (начало, конец, ключевое слово)."""
    normalized = [normalize_word(w.text) for w in words]
    patterns = [(keyword, keyword_tokens(keyword)) for keyword in keywords]
    hits = []
    for index in range(len(words)):
        for keyword, tokens in patterns:
            if not tokens or index + len(tokens) > len(words):
                continue
            if all(_token_matches(token, normalized[index + j]) for j, token in enumerate(tokens)):
                hits.append((words[index].start, words[index + len(tokens) - 1].end, keyword))
    return sorted(hits)


def keyword_candidates(
    words: list[Word], keywords: list[str], duration: float, count: int, min_len: float, max_len: float
) -> list[Candidate]:
    """Лучшие `count` мест с ключевыми словами, в порядке времени.

    Близкие попадания собираются в один клип; клипы ранжируются по числу
    попаданий (и разнообразию слов).
    """
    hits = find_keywords(words, keywords)
    if not hits or count <= 0:
        return []
    target = min(max((min_len + max_len) / 2, min_len), max_len)
    lead, tail = KEYWORD_LEAD * target, KEYWORD_TAIL * target
    groups: list[list[tuple[float, float, str]]] = []
    for hit in hits:
        if groups and hit[1] + tail - (groups[-1][0][0] - lead) <= max_len:
            groups[-1].append(hit)
        else:
            groups.append([hit])

    candidates = []
    for group in groups:
        start = group[0][0] - lead
        end = min(max(group[-1][1] + tail, start + target), start + max_len)
        start, end = _fit(start, end, duration)
        counts = Counter(keyword for _, _, keyword in group)
        score = len(group) + 0.5 * len(counts)
        reason = "ключевые слова: " + ", ".join(
            f"«{keyword}»" + (f" ×{n}" if n > 1 else "") for keyword, n in counts.most_common()
        )
        candidates.append(Candidate(round(start, 3), round(end, 3), score, reason))

    chosen: list[Candidate] = []
    for candidate in sorted(candidates, key=lambda c: (-c.score, c.start)):
        if any(candidate.start < c.end and candidate.end > c.start for c in chosen):
            continue
        chosen.append(candidate)
        if len(chosen) >= count:
            break
    return sorted(chosen, key=lambda c: c.start)


# --- подгонка к фразам ----------------------------------------------------------------


def phrase_starts(words: list[Word]) -> list[float]:
    """Где начинаются фразы: первое слово, после конца предложения или паузы."""
    result = []
    for i, word in enumerate(words):
        if i == 0 or _ends_sentence(words[i - 1].text) or word.start - words[i - 1].end >= PAUSE:
            result.append(word.start)
    return result


def phrase_ends(words: list[Word]) -> list[float]:
    """Где кончаются фразы: конец предложения, пауза после слова или последнее слово."""
    result = []
    for i, word in enumerate(words):
        if i == len(words) - 1 or _ends_sentence(word.text) or words[i + 1].start - word.end >= PAUSE:
            result.append(word.end)
    return result


def _ends_sentence(text: str) -> bool:
    return text.rstrip("\"'»”)]").endswith(SENTENCE_END)


def snap_to_phrases(
    start: float,
    end: float,
    words: list[Word],
    *,
    min_len: float,
    max_len: float,
    duration: float,
    tolerance: float = SNAP_TOLERANCE,
) -> tuple[float, float]:
    """Сдвинуть границы клипа к началу и концу фраз (в пределах `tolerance`).

    Если фразовой границы рядом нет — хотя бы за пределы слова: клип никогда
    не начинается и не кончается посреди слова. Потом добавляется небольшой
    запас по краям и заново соблюдаются min_len/max_len.
    """
    words = sorted(words, key=lambda w: w.start)
    if not words:
        return _clamp(start, end, duration)

    starts = [t for t in phrase_starts(words) if abs(t - start) <= tolerance]
    new_start = min(starts, key=lambda t: (abs(t - start), t)) if starts else _before_word(start, words)

    ends = [t for t in phrase_ends(words) if abs(t - end) <= tolerance and min_len <= t - new_start <= max_len]
    new_end = min(ends, key=lambda t: (abs(t - end), -t)) if ends else _after_word(end, words)

    # Запас по краям — не залезая в соседние слова.
    previous_end = max((w.end for w in words if w.end <= new_start + 1e-6), default=0.0)
    next_start = min((w.start for w in words if w.start >= new_end - 1e-6), default=duration)
    new_start = max(new_start - START_PAD, min(previous_end, new_start), 0.0)
    new_end = min(new_end + END_PAD, max(next_start, new_end), duration)

    # Длина: слишком длинный — режем по последнему влезающему концу фразы, иначе слова.
    if new_end - new_start > max_len:
        limit = new_start + max_len
        low = new_start + min_len
        phrases = [t for t in phrase_ends(words) if low <= t <= limit - END_PAD]
        fitting = phrases or [w.end for w in words if low <= w.end <= limit]
        if fitting:
            cut = max(fitting)
            following = min((w.start for w in words if w.start >= cut - 1e-6), default=duration)
            new_end = min(cut + END_PAD, max(following, cut), limit)
        else:
            new_end = limit
    if new_end - new_start < min_len:
        new_end = _after_word(min(new_start + min_len, duration), words)
        new_end = min(new_end, new_start + max_len, duration)
    return _clamp(new_start, new_end, duration)


def _before_word(t: float, words: list[Word]) -> float:
    """Если t внутри слова — перенести на его начало."""
    for word in words:
        if word.start < t < word.end:
            return word.start
    return t


def _after_word(t: float, words: list[Word]) -> float:
    """Если t внутри слова — перенести на его конец."""
    for word in words:
        if word.start < t < word.end:
            return word.end
    return t


def _clamp(start: float, end: float, duration: float) -> tuple[float, float]:
    start = min(max(start, 0.0), duration)
    end = min(max(end, start), duration)
    return round(start, 3), round(end, 3)
