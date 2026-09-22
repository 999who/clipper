"""Настройки, которые TUI показывает списком: что это, как выглядит, какой флаг CLI.

Значения живут в одном словаре переопределений (как флаги и --set в CLI) и
проверяются тем же `load_config`: ошибка в форме выглядит так же, как ошибка
во флаге. В clipper.yaml TUI ничего не пишет — настройки действуют до выхода,
а внизу экрана видна такая же команда для командной строки.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from clipper.core.config import Config, load_config, parse_value

Kind = Literal["choice", "bool", "int", "float", "text", "list"]


@dataclass(frozen=True)
class Setting:
    key: str  # путь в Config: "select.clips"
    label: str
    kind: Kind
    choices: tuple[tuple[Any, str], ...] = ()  # (значение, подпись)
    flag: str | None = None  # флаг CLI; для bool — «--да/--нет»
    help: str = ""


@dataclass(frozen=True)
class Section:
    title: str


Item = Setting | Section

LANGUAGES = (
    (None, "определять автоматически"),
    ("ru", "русский"),
    ("en", "английский"),
    ("uk", "украинский"),
    ("be", "белорусский"),
    ("kk", "казахский"),
    ("de", "немецкий"),
    ("es", "испанский"),
    ("fr", "французский"),
)

ANALYZE: tuple[Item, ...] = (
    Setting("select.mode", "Как искать моменты", "choice",
            (("heatmap", "по графику «Самые популярные фрагменты»"), ("keywords", "по ключевым словам")),
            flag="--mode-select",
            help="Heatmap есть только у популярных роликов YouTube. Для своих файлов и стримов — ключевые слова."),
    Setting("select.keywords", "Ключевые слова", "list", flag="--keywords",
            help="Через запятую. * — любое окончание: побед*. Фразы можно: да ладно"),
    Setting("select.clips", "Сколько клипов", "int", flag="--clips",
            help="Сколько самых интересных моментов взять (пиков heatmap или мест с ключевыми словами)."),
    Setting("select.min_len", "Длина клипа от, с", "float", flag="--min-len"),
    Setting("select.max_len", "Длина клипа до, с", "float", flag="--max-len"),
    Setting("transcribe.language", "Язык речи", "choice", LANGUAGES, flag="--lang"),
    Setting("transcribe.verbatim", "Записывать «эм», «ээ»", "bool", flag="--verbatim/--no-verbatim",
            help="Нужно, чтобы потом вырезать слова-паразиты. Меняет распознавание — оно пройдёт заново."),
)  # fmt: skip

RENDER: tuple[Item, ...] = (
    Section("Кадр"),
    Setting("reframe.mode", "Режим", "choice",
            (("video", "обычное видео"), ("stream", "стрим: вебка сверху, игра снизу")), flag="--mode"),
    Setting("reframe.aspect", "Формат", "choice",
            (("9:16", "9:16 — на весь экран"), ("1:1", "1:1 — квадрат на фоне"),
             ("original", "как в исходнике, на фоне")), flag="--aspect"),
    Setting("reframe.crop", "Кадрирование", "choice",
            (("face", "следить за лицом"), ("center", "по центру")), flag="--crop",
            help="За лицом — окно 9:16 плавно едет за самым крупным лицом; если лица нет — по центру."),
    Setting("reframe.background", "Фон для 1:1 и original", "choice",
            (("blur", "размытое видео"), ("black", "чёрный")), flag="--background"),
    Setting("reframe.layout", "Пресет стрима", "choice", flag="--layout",
            help="Пресеты описываются в clipper.yaml, раздел layouts."),
    Section("Звук"),
    Setting("audio.cut_pauses", "Вырезать паузы", "bool", flag="--cut-pauses/--no-cut-pauses"),
    Setting("audio.pause_detect", "Как искать паузы", "choice",
            (("volume", "по громкости"), ("words", "между словами (для стримов)")), flag="--pause-detect",
            help="По громкости — где тихо. Между словами — где никто не говорит: для стримов с игрой и музыкой."),
    Setting("audio.silence_db", "Порог тишины, дБ", "float", flag="--silence-db",
            help="Всё тише считается паузой. Если паузы не находятся — поднимите, например до -25."),
    Setting("audio.min_pause", "Паузы не короче, с", "float", flag="--min-pause"),
    Setting("audio.remove_fillers", "Вырезать слова-паразиты", "bool", flag="--remove-fillers/--keep-fillers",
            help="«эм», «ээ», «ну», «как бы»… Список — audio.fillers в clipper.yaml."),
    Section("Субтитры"),
    Setting("subtitles.enabled", "Субтитры", "bool", flag="--subs/--no-subs"),
    Setting("subtitles.style", "Стиль", "choice", flag="--style"),
    Setting("subtitles.max_words", "Слов на экране", "choice",
            ((None, "как в стиле"), (1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5")), flag="--max-words"),
    Section("Файлы"),
    Setting("render.encoder", "Кодировщик", "choice",
            (("auto", "авто (NVENC, если работает)"), ("nvenc", "NVENC — видеокарта"), ("x264", "x264 — процессор")),
            flag="--encoder"),
    Setting("paths.output", "Папка для клипов", "text", flag="--out",
            help="Клипы лягут в эту папку, в подпапку с id видео."),
)  # fmt: skip


def get_value(cfg: Config, key: str) -> Any:
    value: Any = cfg
    for part in key.split("."):
        value = getattr(value, part)
    return value


def choices_for(setting: Setting, cfg: Config) -> tuple[tuple[Any, str], ...]:
    """Варианты выбора; для стиля и пресетов — из файлов и конфига."""
    if setting.key == "reframe.layout":
        return ((None, "не выбран"), *((name, name) for name in sorted(cfg.layouts)))
    if setting.key == "subtitles.style":
        from clipper.core.subtitles import STYLES_DIR

        names = sorted(p.stem for p in STYLES_DIR.glob("*.yaml"))
        current = cfg.subtitles.style
        if current not in names:
            names.append(current)  # свой файл стиля из clipper.yaml
        return tuple((name, name) for name in names)
    return setting.choices


def show_value(setting: Setting, value: Any, cfg: Config | None = None) -> str:
    """Значение для строки списка."""
    if setting.kind == "bool":
        return "да" if value else "нет"
    if setting.kind == "choice":
        for option, label in choices_for(setting, cfg) if cfg is not None else setting.choices:
            if option == value:
                return label
        return "—" if value is None else str(value)
    if setting.kind == "list":
        return ", ".join(value) if value else "—"
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def edit_text(setting: Setting, value: Any) -> str:
    """Значение для поля ввода."""
    if setting.kind == "list":
        return ", ".join(value or [])
    if value is None:
        return ""
    return f"{value:g}" if isinstance(value, float) else str(value)


def parse_input(setting: Setting, text: str) -> Any:
    """Текст из поля ввода → значение для переопределения (проверит load_config)."""
    text = text.strip()
    if setting.kind in ("text", "list"):
        return text or None
    return parse_value(text)


def build_config(config_path: Path | None, overrides: dict[str, Any]) -> Config:
    """Config с переопределениями TUI; ошибка — ConfigError с подсказкой."""
    return load_config(config_path, overrides, overrides_origin="настройки")


def cli_command(command: str, items: tuple[Item, ...], cfg: Config, base: Config, extra: tuple[str, ...] = ()) -> str:
    """Та же операция в командной строке: только то, что отличается от clipper.yaml."""
    parts = ["clipper", command, *extra]
    for item in items:
        if not isinstance(item, Setting):
            continue
        value, before = get_value(cfg, item.key), get_value(base, item.key)
        if value == before:
            continue
        flag = item.flag
        if item.kind == "bool" and flag:
            yes, no = flag.split("/")
            parts.append(yes if value else no)
        elif flag and value is not None:
            parts += [flag, _quote(edit_text(item, value))]
        else:
            parts.append(f"--set {item.key}={_quote(edit_text(item, value))}")
    return " ".join(parts)


def _quote(text: str) -> str:
    if text and all(ch.isalnum() or ch in "-_.:,*/\\" for ch in text):
        return text
    return '"' + text.replace('"', '\\"') + '"' if text else '""'


class SettingsStore:
    """Настройки сессии TUI: clipper.yaml + флаги запуска + то, что выбрано в формах."""

    def __init__(self, config_path: Path | None, overrides: dict[str, Any] | None = None) -> None:
        self.path = config_path
        self.overrides: dict[str, Any] = dict(overrides or {})
        self.base = build_config(config_path, self.overrides)  # для «то же в командной строке»
        self._config = self.base

    def config(self) -> Config:
        return self._config

    def value(self, key: str) -> Any:
        return get_value(self._config, key)

    def try_set(self, key: str, value: Any) -> str | None:
        """Изменить параметр. Ошибка проверки — текст для пользователя, параметр не меняется."""
        from clipper.core.errors import ConfigError

        overrides = {**self.overrides, key: value}
        try:
            config = build_config(self.path, overrides)
        except ConfigError as exc:
            return exc.message + (f"\n{exc.hint}" if exc.hint else "")
        self.overrides, self._config = overrides, config
        return None
