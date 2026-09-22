"""Время клипа после вырезок: какие куски исходника остались и где они в готовом ролике.

Вырезание пауз и слов-паразитов превращает клип [start, end] в набор кусков
исходника. `Timeline` переводит время исходного видео во время готового клипа —
по нему сдвигаются субтитры (и позже траектория лица).
"""

from dataclasses import dataclass

from clipper.core.models import Word, merge_ranges, subtract_ranges

MIN_PIECE = 0.1  # с: более короткие куски между вырезками не оставляем
MIN_WORD = 0.02  # с: слово короче этого после вырезок считается вырезанным


@dataclass(frozen=True)
class Timeline:
    start: float  # границы клипа в исходнике
    end: float
    pieces: tuple[tuple[float, float], ...]  # что остаётся, по порядку, без пересечений

    @classmethod
    def whole(cls, start: float, end: float) -> "Timeline":
        return cls(start, end, ((start, end),))

    @classmethod
    def with_cuts(cls, start: float, end: float, cuts: list[tuple[float, float]]) -> "Timeline":
        """Клип [start, end] без отрезков `cuts` (время исходника)."""
        cuts = [(max(a, start), min(b, end)) for a, b in cuts]
        pieces = subtract_ranges([(start, end)], merge_ranges(cuts), min_length=MIN_PIECE)
        return cls(start, end, tuple(pieces))

    @property
    def duration(self) -> float:
        """Длина готового клипа."""
        return sum(b - a for a, b in self.pieces)

    @property
    def removed(self) -> float:
        """Сколько секунд вырезано."""
        return (self.end - self.start) - self.duration

    @property
    def is_whole(self) -> bool:
        return len(self.pieces) == 1 and self.pieces[0] == (self.start, self.end)

    def to_output(self, t: float) -> float:
        """Время исходника → время готового клипа.

        Точка внутри вырезанного отрезка попадает на начало следующего куска.
        """
        passed = 0.0
        for a, b in self.pieces:
            if t < a:
                return passed
            if t <= b:
                return passed + (t - a)
            passed += b - a
        return passed

    def map_words(self, words: list[Word]) -> list[Word]:
        """Слова во времени готового клипа; вырезанные и лежащие за границами — отбрасываются."""
        result = []
        for word in words:
            start, end = self.to_output(word.start), self.to_output(word.end)
            if end - start >= MIN_WORD:
                result.append(Word(word.text, round(start, 3), round(end, 3), word.prob))
        return result

    def relative_pieces(self) -> list[tuple[float, float]]:
        """Куски от начала клипа (для ffmpeg, где -ss уже перешёл к start)."""
        return [(round(a - self.start, 3), round(b - self.start, 3)) for a, b in self.pieces]
