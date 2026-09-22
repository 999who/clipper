import pytest

from clipper.core import highlights as hl
from clipper.core.models import HeatPoint, Word


def heatmap(values: list[float], duration: float) -> list[HeatPoint]:
    step = duration / len(values)
    return [HeatPoint(i * step, (i + 1) * step, v) for i, v in enumerate(values)]


def speech(start: float, end: float, sentence_every: int = 4) -> list[Word]:
    """Слова по 0,4 с каждые 0,5 с; каждое 4-е слово заканчивает предложение."""
    words, t, n = [], start, 0
    while t + 0.4 <= end:
        n += 1
        words.append(Word("конец." if n % sentence_every == 0 else "слово", round(t, 2), round(t + 0.4, 2)))
        t += 0.5
    return words


# --- heatmap ------------------------------------------------------------------------


def test_heatmap_peaks_skip_intro_merge_neighbours_and_keep_bounds():
    values = [0.1] * 100
    values[0] = 0.9  # «интро»: все начинают смотреть — не момент
    values[30] = 1.0
    values[70], values[72] = 0.8, 0.75  # два близких пика → один клип
    points = heatmap(values, 600.0)  # точки по 6 с
    found = hl.heatmap_candidates(points, 600.0, count=2, min_len=20, max_len=60)

    assert len(found) == 2
    first, second = found
    assert first.start < 30 * 6 + 3 < first.end  # пик внутри клипа
    assert second.start <= 70 * 6 and second.end >= 73 * 6  # оба соседних пика вошли
    for c in found:
        assert 20 <= c.length <= 60
        assert c.reason.startswith("пик heatmap")
    assert first.score > second.score


def test_heatmap_peak_is_placed_at_forty_percent_of_short_window():
    values = [0.1] * 100
    values[50] = 1.0
    (clip,) = hl.heatmap_candidates(heatmap(values, 300.0), 300.0, count=1, min_len=20, max_len=60)
    center = 50 * 3 + 1.5
    assert clip.length == pytest.approx(40.0)
    assert clip.start == pytest.approx(center - 0.4 * 40)


def test_heatmap_on_long_video_where_one_point_is_longer_than_max_len():
    values = [0.2] * 100
    values[40] = 1.0
    duration = 3 * 3600.0  # точка — 108 с
    (clip,) = hl.heatmap_candidates(heatmap(values, duration), duration, count=1, min_len=20, max_len=60)
    assert clip.length == pytest.approx(40.0)
    assert clip.start >= 40 * 108 and clip.end <= 41 * 108


def test_heatmap_respects_count_and_edges():
    values = [0.1] * 100
    for i in (10, 30, 50, 70, 98):
        values[i] = 1.0 - i / 1000
    points = heatmap(values, 500.0)
    found = hl.heatmap_candidates(points, 500.0, count=3, min_len=20, max_len=60)
    assert len(found) == 3
    assert [round(c.score, 2) for c in sorted(found, key=lambda c: -c.score)] == [0.99, 0.97, 0.95]
    last = hl.heatmap_candidates(points, 500.0, count=5, min_len=20, max_len=60)[-1]
    assert last.end <= 500.0  # пик у самого конца не выходит за видео
    assert hl.heatmap_candidates([], 100.0, 3, 20, 60) == []


def test_heatmap_small_bumps_are_not_peaks_but_fill_up_the_count():
    values = [0.1] * 100
    values[20] = 1.0
    values[60] = 0.25  # меньше 30 % максимума — не пик
    found = hl.heatmap_candidates(heatmap(values, 400.0), 400.0, count=3, min_len=20, max_len=60)
    assert [c.reason for c in found if c.reason.startswith("пик")] == ["пик heatmap 1.00"]
    assert len(found) == 3
    assert any(c.start <= 60 * 4 < c.end and c.reason == "heatmap 0.25" for c in found)  # самое «горячее» после пика


def test_heatmap_fills_requested_count_around_peaks():
    """Пиков 3, просят 10: остальные клипы — из самых «горячих» мест, обычно рядом с пиками."""
    values = [0.15] * 100
    for peak in (20, 41, 45):
        for offset, value in ((-2, 0.5), (-1, 0.8), (0, 1.0), (1, 0.8), (2, 0.5)):
            values[peak + offset] = max(values[peak + offset], value if peak != 20 else value * 0.7)
    duration = 7200.0  # 2 часа: точка — 72 с
    found = hl.heatmap_candidates(heatmap(values, duration), duration, count=10, min_len=15, max_len=40)
    assert len(found) == 10
    assert sum(c.reason.startswith("пик") for c in found) == 3
    assert [c.start for c in found] == sorted(c.start for c in found)
    for a, b in zip(found, found[1:], strict=False):
        assert b.start - a.end >= hl.MERGE_GAP  # не пересекаются и не слипаются
    extra = [c for c in found if not c.reason.startswith("пик")]
    assert min(c.score for c in extra) >= 0.5  # добраны из горячих мест, не из «ровного» фона
    assert all(15 <= c.length <= 40 for c in found)


# --- ключевые слова -------------------------------------------------------------------


def test_normalize_and_tokens():
    assert hl.normalize_word("«Победа!»") == "победа"
    assert hl.normalize_word("Ёлки,") == "елки"
    assert hl.normalize_word("как-то,") == "как-то"
    assert hl.keyword_tokens("Побед*") == ["побед*"]
    assert hl.keyword_tokens("да  ладно!") == ["да", "ладно"]


def test_find_keywords_with_prefix_and_phrase():
    words = [Word(t, i, i + 0.5) for i, t in enumerate(["Ну", "победили!", "да", "ладно,", "ПОБЕДА"])]
    hits = hl.find_keywords(words, ["побед*", "да ладно", "победа"])
    assert [(start, keyword) for start, _, keyword in hits] == [
        (1, "побед*"),
        (2, "да ладно"),
        (4, "побед*"),
        (4, "победа"),
    ]


def test_keyword_candidates_group_close_hits_and_rank():
    words = speech(0, 300)
    special = {100.0: "победа", 105.0: "победа", 110.0: "жесть", 250.0: "жесть"}
    words = [Word(special.get(w.start, w.text), w.start, w.end) for w in words]
    found = hl.keyword_candidates(words, ["побед*", "жесть"], 300.0, count=1, min_len=20, max_len=60)
    (best,) = found
    assert best.start <= 100 and best.end >= 110.4  # три попадания — один клип
    assert best.reason == "ключевые слова: «побед*» ×2, «жесть»"
    both = hl.keyword_candidates(words, ["побед*", "жесть"], 300.0, count=5, min_len=20, max_len=60)
    assert len(both) == 2 and both[1].start < 250 < both[1].end
    assert all(20 <= c.length <= 60 for c in both)
    assert hl.keyword_candidates(words, ["нет такого"], 300.0, 5, 20, 60) == []


# --- подгонка к фразам ----------------------------------------------------------------


def test_snap_moves_to_phrase_boundaries_with_padding():
    words = speech(0, 100)  # предложения по 2 с: начала на 0, 2, 4…; концы на 1.9, 3.9…
    start, end = hl.snap_to_phrases(10.7, 40.6, words, min_len=20, max_len=60, duration=100)
    assert start == pytest.approx(10.0 - 0.1)  # начало фразы 10.0, запас не залезает в прошлое слово (9.9)
    assert end == pytest.approx(39.9 + 0.1)  # конец фразы 39.9, запас до следующего слова 40.0


def test_snap_never_cuts_inside_a_word_without_phrase_boundaries():
    words = [Word("слово", i * 0.5, i * 0.5 + 0.45) for i in range(200)]  # без пауз и точек
    start, end = hl.snap_to_phrases(10.2, 40.2, words, min_len=20, max_len=60, duration=100)
    for t in (start, end):
        assert not any(w.start + 1e-6 < t < w.end - 1e-6 for w in words), t


def test_snap_respects_lengths_and_empty_transcript():
    words = speech(0, 200)
    start, end = hl.snap_to_phrases(10, 80, words, min_len=20, max_len=60, duration=200)
    assert end - start <= 60 + 1e-6
    assert end == pytest.approx(67.9 + 0.1)  # последний влезающий конец фразы (67.9) + запас
    start, end = hl.snap_to_phrases(10, 12, words, min_len=20, max_len=60, duration=200)
    assert end - start >= 20
    assert hl.snap_to_phrases(-5, 30, [], min_len=20, max_len=60, duration=25) == (0.0, 25.0)
