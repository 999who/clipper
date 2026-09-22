"""Этап 5: субтитры в стиле CapCut — файл .ass для libass (фильтр subtitles в ffmpeg).

- На экране 1–3 слова. Новая строка начинается после конца предложения, после
  паузы в речи и когда строка стала бы длиннее `max_chars`.
- На каждое слово — отдельное событие: вся строка белая, текущее слово жёлтое и
  «выпрыгивает» (масштаб через `\\t`).
- КАПС без пунктуации; дефис внутри слова остаётся («КАК-ТО»).
- Оформление — в `styles/<имя>.yaml`. Размеры в стиле заданы для кадра 1080×1920
  и масштабируются под настоящий кадр.
- Слишком длинная строка уменьшается, чтобы влезть по ширине. Ширина текста
  оценивается по средней ширине букв Montserrat Black; для других шрифтов
  оценка грубее, но переносы libass всё равно не дадут вылезти за край.

Время слов — уже время готового клипа (после вырезок этапа 4).
"""

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.core.config import merge_yaml, read_yaml_file
from clipper.core.errors import ConfigError
from clipper.core.models import Word, join_hyphenated

PACKAGE_DIR = Path(__file__).resolve().parent.parent
STYLES_DIR = PACKAGE_DIR / "styles"
FONTS_DIR = PACKAGE_DIR / "fonts"
FONT_SUFFIXES = (".ttf", ".otf", ".ttc")

REFERENCE_WIDTH, REFERENCE_HEIGHT = 1080, 1920  # под этот кадр заданы размеры в стиле
EM_PER_FONT_SIZE = 0.634  # libass: ширина 1 em = 0,634 × размер шрифта (Montserrat, замерено)
MIN_FIT = 0.7  # строку можно уменьшить не больше чем до 70 %
MIN_GAP = 0.25  # с: промежуток между строками короче этого не показываем пустым

_COLOR_KEYS = ("color", "highlight", "outline_color", "shadow_color")
_COLOR = re.compile(r"#?([0-9a-fA-F]{6})")
_NOT_TEXT = re.compile(r"[^\w\-'’%$€₽+#@&]")
_BREAK_MARK = re.compile(r"[.!?…:;—–-]")
_SENTENCE_END = re.compile(r"[.!?…:;]+[\"»”')\]]*$")


@dataclass
class Style:
    """Оформление субтитров. Значения по умолчанию совпадают со styles/capcut.yaml."""

    font: str = "Montserrat Black"  # имя шрифта (семейство); файл ищется в fonts/, потом в системе
    font_file: str | None = None  # свой .ttf/.otf: путь от папки стиля или абсолютный
    size: float = 130  # размер шрифта, px (кадр 1080×1920)
    bold: bool = False  # Montserrat Black уже жирный
    uppercase: bool = True
    strip_punctuation: bool = True  # дефис внутри слова остаётся
    color: str = "#FFFFFF"  # цвет текста
    highlight: str = "#FFE600"  # цвет текущего слова
    outline: float = 5  # толщина обводки, px
    outline_color: str = "#000000"
    shadow: float = 0  # смещение тени, px; 0 — без тени
    shadow_color: str = "#000000"
    position: float = 0.72  # центр строки по высоте кадра: 0 — верх, 1 — низ
    margin: float = 60  # отступ от боковых краёв, px
    max_words: int = 3  # слов на экране
    max_chars: int = 16  # символов в строке (с пробелами); длиннее — новая строка
    pause: float = 0.4  # пауза в речи, после которой начинается новая строка, с
    hold: float = 0.3  # сколько строка держится после последнего слова, с
    pop_from: float = 90  # анимация текущего слова: начальный масштаб, %
    pop_peak: float = 112  # …максимальный, %
    pop_duration: float = 0.15  # …длительность, с; 0 — без анимации


@dataclass(frozen=True)
class Caption:
    """Слово для показа: текст уже в виде для экрана, время — в готовом клипе."""

    text: str
    start: float
    end: float
    sentence_end: bool = False


# --- стиль --------------------------------------------------------------------------


def style_file(name: str) -> Path:
    """Путь к файлу стиля: имя из clipper/styles/ или путь к .yaml."""
    path = Path(name).expanduser()
    if path.suffix.lower() in (".yaml", ".yml") or path.parent != Path("."):
        if not path.is_file():
            raise ConfigError(f"Файл стиля субтитров не найден: {path}", hint="Проверьте subtitles.style / --style.")
        return path
    builtin = STYLES_DIR / f"{name}.yaml"
    if not builtin.is_file():
        known = ", ".join(sorted(p.stem for p in STYLES_DIR.glob("*.yaml"))) or "нет"
        raise ConfigError(
            f"Нет стиля субтитров «{name}».",
            hint=f"Встроенные стили: {known}. Или укажите путь к своему .yaml.",
        )
    return builtin


def load_style(name: str) -> tuple[Style, Path]:
    """Прочитать стиль. Параметры, которых нет в файле, берутся по умолчанию (как в capcut)."""
    path = style_file(name)
    style = Style()
    data = read_yaml_file(path)
    if data is not None and not isinstance(data, dict):
        raise ConfigError(f"{path.name}: ожидался список параметров стиля (font:, size: …)")
    for key in _COLOR_KEYS:
        if data and key in data and data[key] is None:  # `color: #FFFFFF` без кавычек — это комментарий
            raise ConfigError(
                f"{path.name}: {key} — цвет не указан",
                hint='Цвет пишется в кавычках: "#RRGGBB". Без кавычек YAML считает # началом комментария.',
            )
    merge_yaml(style, data, origin=path.name)
    validate_style(style, path.name)
    return style, path


def validate_style(style: Style, origin: str = "стиль") -> None:
    def check(ok: bool, key: str, message: str, hint: str | None = None) -> None:
        if not ok:
            raise ConfigError(f"{origin}: {key} — {message}", hint)

    for key in _COLOR_KEYS:
        value = getattr(style, key)
        check(
            isinstance(value, str) and _COLOR.fullmatch(value.strip()) is not None,
            key,
            f"непонятный цвет {value!r}",
            'Цвет пишется как "#RRGGBB" в кавычках: без кавычек YAML считает # началом комментария.',
        )
    check(bool(style.font.strip()), "font", "не указан шрифт")
    check(style.size > 0, "size", "размер шрифта должен быть больше 0")
    check(style.outline >= 0 and style.shadow >= 0, "outline/shadow", "не могут быть отрицательными")
    check(0 <= style.position <= 1, "position", "от 0 (верх кадра) до 1 (низ)")
    check(0 <= style.margin < REFERENCE_WIDTH / 2, "margin", "отступ должен быть от 0 до 540")
    check(1 <= style.max_words <= 5, "max_words", "допустимо от 1 до 5 слов")
    check(style.max_chars >= 4, "max_chars", "слишком мало символов в строке")
    check(style.pause > 0 and style.hold >= 0, "pause/hold", "должны быть положительными")
    check(style.pop_from > 0 and style.pop_peak > 0, "pop_from/pop_peak", "масштаб в процентах больше 0")
    check(style.pop_duration >= 0, "pop_duration", "не может быть отрицательной")


def font_files(style: Style, style_path: Path | None) -> list[Path]:
    """Файлы шрифтов для fontsdir: встроенные из clipper/fonts/ и font_file стиля."""
    files = sorted(p for p in FONTS_DIR.iterdir() if p.suffix.lower() in FONT_SUFFIXES) if FONTS_DIR.is_dir() else []
    if style.font_file:
        path = Path(style.font_file).expanduser()
        if not path.is_absolute() and style_path is not None:
            path = style_path.parent / path
        if not path.is_file():
            raise ConfigError(
                f"Файл шрифта не найден: {path}",
                hint="font_file в стиле — путь от папки со стилем или абсолютный.",
            )
        files.append(path)
    return files


def prepare_fonts(dest: Path, files: list[Path]) -> Path:
    """Скопировать шрифты в dest (рядом с .ass): ffmpeg получит короткий относительный путь."""
    dest.mkdir(parents=True, exist_ok=True)
    for src in files:
        target = dest / src.name
        if not target.is_file() or target.stat().st_size != src.stat().st_size:
            shutil.copyfile(src, target)
    return dest


# --- текст и строки ---------------------------------------------------------------------


def display_text(text: str, style: Style) -> str:
    """Слово как на экране: «как-то,» → «КАК-ТО». Пустая строка — слово не показывается."""
    if style.strip_punctuation:
        text = _NOT_TEXT.sub("", text).strip("-'’")
    else:
        text = text.replace("{", "(").replace("}", ")").replace("\\", "/").strip()
    return text.upper() if style.uppercase else text


def captions(words: list[Word], style: Style) -> list[Caption]:
    result: list[Caption] = []
    for word in join_hyphenated(sorted(words, key=lambda w: w.start)):  # старые project.json: «Ха» «-ха»
        text = display_text(word.text, style)
        if text:
            result.append(Caption(text, word.start, word.end, bool(_SENTENCE_END.search(word.text.strip()))))
        elif result and _BREAK_MARK.search(word.text):  # «—» или «…» отдельным словом
            result[-1] = Caption(result[-1].text, result[-1].start, result[-1].end, True)
    return result


def group_captions(items: list[Caption], style: Style, max_words: int | None = None) -> list[list[Caption]]:
    """Разбить слова на строки по 1–max_words слов."""
    limit = max_words or style.max_words
    groups: list[list[Caption]] = []
    current: list[Caption] = []
    for item in items:
        if current:
            prev = current[-1]
            length = sum(len(c.text) for c in current) + len(current) + len(item.text)
            if (
                len(current) >= limit
                or length > style.max_chars
                or prev.sentence_end
                or item.start - prev.end >= style.pause
            ):
                groups.append(current)
                current = []
        current.append(item)
    if current:
        groups.append(current)
    return groups


def caption_events(
    groups: list[list[Caption]], style: Style, duration: float
) -> list[tuple[float, float, list[Caption], int]]:
    """(начало, конец, строка, номер текущего слова) — по событию на каждое слово."""
    events = []
    for number, group in enumerate(groups):
        next_start = groups[number + 1][0].start if number + 1 < len(groups) else duration
        end = min(group[-1].end + style.hold, next_start, duration)
        if min(next_start, duration) - end < MIN_GAP:  # не мигать пустым экраном между строками
            end = min(next_start, duration)
        for index, item in enumerate(group):
            start = item.start
            stop = group[index + 1].start if index + 1 < len(group) else end
            start, stop = round(start, 2), round(min(stop, duration), 2)
            if stop - start >= 0.01:
                events.append((start, stop, group, index))
    return events


# --- .ass ------------------------------------------------------------------------------


def text_width(text: str, font_size: float) -> float:
    """Оценка ширины строки в px по ширине букв Montserrat Black."""
    em = 0.0
    for ch in text:
        if ch == " ":
            em += 0.3
        elif ch in "ШЩЖЮЫWMМФ%":
            em += 1.1
        elif ch in "I1-'’.,!ЇІ":
            em += 0.4
        elif ch.isdigit():
            em += 0.65
        else:
            em += 0.78
    return em * font_size * EM_PER_FONT_SIZE


def ass_color(value: str) -> str:
    """#RRGGBB → &HBBGGRR& для тегов \\c."""
    rgb = _COLOR.fullmatch(value.strip()).group(1)  # type: ignore[union-attr]
    return f"&H{rgb[4:6]}{rgb[2:4]}{rgb[0:2]}&".upper()


def _color(value: str):
    import pysubs2

    rgb = _COLOR.fullmatch(value.strip()).group(1)  # type: ignore[union-attr]
    return pysubs2.Color(int(rgb[0:2], 16), int(rgb[2:4], 16), int(rgb[4:6], 16))


def build_ass(
    words: list[Word],
    width: int,
    height: int,
    style: Style,
    duration: float,
    max_words: int | None = None,
) -> Any:
    """Субтитры клипа (pysubs2.SSAFile) для кадра width×height."""
    import pysubs2

    scale = min(width / REFERENCE_WIDTH, height / REFERENCE_HEIGHT)
    size = style.size * scale
    margin = style.margin * scale
    outline = style.outline * scale

    subs = pysubs2.SSAFile()
    subs.info.update(
        {"PlayResX": str(width), "PlayResY": str(height), "ScaledBorderAndShadow": "yes", "WrapStyle": "0"}
    )
    subs.styles["Default"] = pysubs2.SSAStyle(
        fontname=style.font,
        fontsize=round(size, 1),
        primarycolor=_color(style.color),
        secondarycolor=_color(style.color),
        outlinecolor=_color(style.outline_color),
        backcolor=_color(style.shadow_color),
        bold=style.bold,
        outline=round(outline, 1),
        shadow=round(style.shadow * scale, 1),
        alignment=pysubs2.Alignment.MIDDLE_CENTER,
        marginl=round(margin),
        marginr=round(margin),
        marginv=0,
    )

    x, y = round(width / 2), round(height * style.position)
    available = width - 2 * margin - 2 * outline
    for start, end, group, index in caption_events(
        group_captions(captions(words, style), style, max_words), style, duration
    ):
        subs.append(
            pysubs2.SSAEvent(
                start=round(start * 1000),
                end=round(end * 1000),
                text=event_text(group, index, style, size, x, y, available),
            )
        )
    return subs


def event_text(group: list[Caption], index: int, style: Style, size: float, x: int, y: int, available: float) -> str:
    """Текст события: строка целиком, текущее слово подсвечено и «выпрыгивает»."""
    line = " ".join(c.text for c in group)
    peak = max(style.pop_peak, 100) / 100 if style.pop_duration > 0 else 1.0
    needed = text_width(line, size) + text_width(group[index].text, size) * (peak - 1)
    fit = max(MIN_FIT, min(1.0, available / needed)) if needed > 0 else 1.0
    font = f"\\fs{size * fit:.1f}" if fit < 1.0 else ""

    pop = ""
    if style.pop_duration > 0:
        total = round(style.pop_duration * 1000)
        half = total // 2
        pop = (
            f"\\fscx{style.pop_from:g}\\fscy{style.pop_from:g}"
            f"\\t(0,{half},\\fscx{style.pop_peak:g}\\fscy{style.pop_peak:g})"
            f"\\t({half},{total},\\fscx100\\fscy100)"
        )
    parts = []
    for number, item in enumerate(group):
        if number == index:
            parts.append(f"{{\\c{ass_color(style.highlight)}{pop}}}{item.text}{{\\r{font}}}")
        else:
            parts.append(item.text)
    return f"{{\\an5\\pos({x},{y}){font}}}" + " ".join(parts)


def write_ass(path: Path, subs: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(subs.to_string("ass"), encoding="utf-8")
    tmp.replace(path)
    return path
