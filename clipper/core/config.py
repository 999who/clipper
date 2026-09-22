"""Конфигурация clipper: один объект `Config` для всего конвейера.

Значения собираются слоями, каждый следующий важнее предыдущего:

1. значения по умолчанию — поля dataclass'ов ниже;
2. файл конфига YAML: `--config PATH`, переменная окружения CLIPPER_CONFIG
   или `./clipper.yaml`;
3. переопределения из интерфейса: флаги CLI и `--set ключ=значение`
   (позже — формы TUI).

Все слои проходят одну и ту же проверку. Ошибка в YAML и ошибка во флаге
выглядят одинаково: где, какой ключ, что ожидалось.
"""

import dataclasses
import difflib
import os
import re
import types
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from clipper.core.errors import ConfigError

CONFIG_FILENAME = "clipper.yaml"
CONFIG_ENV_VAR = "CLIPPER_CONFIG"
EXAMPLE_FILENAME = "clipper.example.yaml"

DEFAULT_FILLERS = ("эм", "эмм", "ээ", "эээ", "мм", "ммм", "хм", "ну", "как бы")


# --- Схема ------------------------------------------------------------------------


@dataclass
class PathsConfig:
    workdir: str = "work"  # рабочие папки видео: work/<id>/
    output: str = "output"  # готовые клипы: output/<id>/


@dataclass
class DownloadConfig:
    max_height: int = 1080  # максимальная высота скачиваемого видео, px
    cookies_from_browser: str | None = None  # браузер, из которого yt-dlp возьмёт cookies


@dataclass
class TranscribeConfig:
    language: str | None = None  # None — автоопределение
    compute_type: Literal["float16", "int8_float16"] = "float16"
    margin: float = 5.0  # запас вокруг окон heatmap при распознавании, с
    model_dir: str | None = None  # None — кэш HuggingFace
    verbatim: bool = False  # подсказать Whisper не выкидывать «эм», «ээ» (для remove_fillers)


@dataclass
class SelectConfig:
    mode: Literal["heatmap", "keywords"] = "heatmap"
    keywords: list[str] = field(default_factory=list)
    clips: int = 5
    min_len: float = 20.0  # с
    max_len: float = 60.0  # с


@dataclass
class AudioConfig:
    cut_pauses: bool = False
    pause_detect: Literal["volume", "words"] = "volume"  # по громкости или по промежуткам между словами
    silence_db: float = -35.0  # порог тишины, дБ
    min_pause: float = 0.6  # вырезаются паузы не короче этого, с
    max_gap: float = 3.0  # words: промежутки без слов длиннее этого не вырезаются, с; 0 — резать все
    remove_fillers: bool = False
    fillers: list[str] = field(default_factory=lambda: list(DEFAULT_FILLERS))


@dataclass
class SubtitlesConfig:
    enabled: bool = True
    style: str = "capcut"  # имя стиля из styles/ или путь к .yaml
    max_words: int | None = None  # слов на экране; None — как в стиле


@dataclass
class ReframeConfig:
    mode: Literal["video", "stream"] = "video"
    aspect: Literal["9:16", "1:1", "original"] = "9:16"
    crop: Literal["face", "center"] = "face"
    background: Literal["blur", "black"] = "blur"  # фон для 1:1 и original
    layout: str | None = None  # пресет из layouts для режима stream


@dataclass
class RenderConfig:
    encoder: Literal["auto", "nvenc", "x264"] = "auto"
    thumbnails: bool = True
    concat: bool = False


@dataclass
class Rect:
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0


@dataclass
class LayoutConfig:
    webcam: Rect = field(default_factory=Rect)
    webcam_zone: float = 0.33  # доля высоты кадра под вебку сверху
    game_crop: Literal["center", "left", "right"] = "center"
    source_size: tuple[int, int] | None = None  # разрешение кадра калибровки


@dataclass
class Config:
    paths: PathsConfig = field(default_factory=PathsConfig)
    download: DownloadConfig = field(default_factory=DownloadConfig)
    transcribe: TranscribeConfig = field(default_factory=TranscribeConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    subtitles: SubtitlesConfig = field(default_factory=SubtitlesConfig)
    reframe: ReframeConfig = field(default_factory=ReframeConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    layouts: dict[str, LayoutConfig] = field(default_factory=dict)


# --- Публичные функции ------------------------------------------------------------


def find_config_file(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Найти файл конфига: явный путь → $CLIPPER_CONFIG → ./clipper.yaml → нет файла."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(f"Файл конфига не найден: {path}", hint="Проверьте путь в --config.")
        return path
    env_value = os.environ.get(CONFIG_ENV_VAR)
    if env_value:
        path = Path(env_value).expanduser()
        if not path.is_file():
            raise ConfigError(
                f"Файл конфига не найден: {path}",
                hint=f"Путь взят из переменной окружения {CONFIG_ENV_VAR} — исправьте или удалите её.",
            )
        return path
    path = Path.cwd() / CONFIG_FILENAME
    return path if path.is_file() else None


def load_config(
    path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    overrides_origin: str = "командная строка",
) -> Config:
    """Собрать Config: значения по умолчанию → файл `path` → `overrides`.

    `overrides` — плоский словарь с ключами через точку: {"select.clips": 3}.
    """
    cfg = Config()
    if path is not None:
        data = _read_config_file(path)
        if data is not None and not isinstance(data, Mapping):
            raise ConfigError(
                f"{path.name}: ожидался набор разделов (select:, render: …), а не {_show(data)}",
                hint="Начните с примера: clipper config --init",
            )
        _apply_layer(cfg, data, origin=path.name)
    if overrides:
        _apply_layer(cfg, _overrides_to_tree(overrides), origin=overrides_origin)
    _normalize(cfg)
    _validate(cfg)
    return cfg


def parse_set_options(items: Iterable[str]) -> dict[str, Any]:
    """Разобрать значения `--set ключ=значение` в плоский словарь переопределений."""
    result: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigError(
                f"--set {item}: нужен формат КЛЮЧ=ЗНАЧЕНИЕ",
                hint="Например: --set select.min_len=15",
            )
        result[key] = parse_value(raw)
    return result


def parse_value(raw: str) -> Any:
    """Превратить строку из командной строки в значение по правилам YAML-загрузчика.

    `15` → 15, `true` → True, `[a, b]` → список, пусто → None,
    всё остальное (включая `9:16`) — строка.
    """
    raw = raw.strip()
    if not raw:
        return None
    try:
        return yaml.load(raw, Loader=_Loader)  # noqa: S506 — _Loader основан на SafeLoader
    except yaml.YAMLError:
        return raw


def config_to_dict(cfg: Config) -> dict[str, Any]:
    """Config → обычный словарь (для вывода, сохранения, форм TUI)."""
    return _plain(dataclasses.asdict(cfg))


def dump_config(cfg: Config) -> str:
    """Config → текст YAML (как его прочитает clipper)."""
    return yaml.dump(
        config_to_dict(cfg),
        Dumper=_Dumper,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def example_config_path() -> Path | None:
    """Путь к clipper.example.yaml в корне репозитория (если clipper запущен из исходников)."""
    path = Path(__file__).resolve().parents[2] / EXAMPLE_FILENAME
    return path if path.is_file() else None


def write_example_config(dest: Path, overwrite: bool = False) -> Path:
    """Создать файл конфига из примера с комментариями."""
    if dest.exists() and not overwrite:
        raise ConfigError(
            f"Файл уже существует: {dest}",
            hint="Отредактируйте его или добавьте --force, чтобы перезаписать примером.",
        )
    example = example_config_path()
    if example is not None:
        text = example.read_text(encoding="utf-8")
    else:
        text = (
            "# Конфиг clipper. Приоритет: значения по умолчанию → этот файл → флаги.\n"
            "# Ниже — все параметры со значениями по умолчанию.\n\n" + dump_config(Config())
        )
    dest.write_text(text, encoding="utf-8")
    return dest


def leaf_paths(cls: type = Config, prefix: str = "") -> list[str]:
    """Все параметры в виде ключей через точку: ["paths.workdir", ...]."""
    result = []
    for name, tp in _field_types(cls).items():
        path = f"{prefix}{name}"
        if dataclasses.is_dataclass(tp):
            result.extend(leaf_paths(tp, path + "."))
        else:
            result.append(path)
    return result


# --- YAML -------------------------------------------------------------------------


class _Loader(yaml.SafeLoader):
    """SafeLoader со скалярами по правилам YAML 1.2 и запретом повторных ключей.

    PyYAML следует YAML 1.1 и читает `9:16` как число 556 (шестидесятеричная запись),
    а `no`/`off` — как false. Здесь числа и булевы значения распознаются только
    в явном виде (`15`, `-35.5`, `true`), всё остальное остаётся строкой.
    """


_REPLACED_TAGS = {"tag:yaml.org,2002:bool", "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}
_Loader.yaml_implicit_resolvers = {
    first: [(tag, rx) for tag, rx in resolvers if tag not in _REPLACED_TAGS]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:int",
    re.compile(r"^[-+]?(?:0|[1-9][0-9_]*)$"),
    list("-+0123456789"),
)
_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?[0-9][0-9_]*\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?\.[0-9][0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def _construct_mapping(loader: _Loader, node: yaml.MappingNode) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=True)
        try:
            duplicate = key in mapping
        except TypeError:
            raise yaml.constructor.ConstructorError(None, None, "недопустимый ключ", key_node.start_mark) from None
        if duplicate:
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"ключ «{key}» указан второй раз (одинаковые разделы не объединяются — оставьте один)",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=True)
    return mapping


_Loader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


class _Dumper(yaml.SafeDumper):
    """Разделы — блоками, короткие списки чисел и слов — в одну строку."""


def _represent_list(dumper: yaml.SafeDumper, data: list[Any]) -> yaml.Node:
    flat = all(isinstance(item, (str, int, float, bool)) or item is None for item in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=flat)


_Dumper.add_representer(list, _represent_list)


def _read_config_file(path: Path) -> Any:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ConfigError(f"Не удалось прочитать файл конфига {path}: {exc.strerror}") from None
    try:
        text = raw.decode("utf-8-sig")  # -sig: Блокнот Windows может добавить BOM
    except UnicodeDecodeError:
        text = raw.decode("cp1251", errors="replace")  # сохранён в «ANSI»
    try:
        return yaml.load(text, Loader=_Loader)  # noqa: S506 — _Loader основан на SafeLoader
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        where = f"строка {mark.line + 1}, столбец {mark.column + 1}: " if mark else ""
        problem = exc.problem or exc.context or "ошибка разметки"
        raise ConfigError(
            f"{path.name}: {where}{problem}",
            hint="Отступы в YAML — только пробелы (не табы); после ключа ставится двоеточие.",
        ) from None
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name}: ошибка разметки YAML — {exc}") from None


# --- Слияние и проверка типов -------------------------------------------------------

_TRUE_WORDS = {"true", "yes", "on", "да", "1"}
_FALSE_WORDS = {"false", "no", "off", "нет", "0"}


def _apply_layer(cfg: Config, data: Any, origin: str) -> None:
    try:
        _merge_into(cfg, data, "")
    except ConfigError as exc:
        raise ConfigError(f"{origin}: {exc.message}", exc.hint) from None


def _overrides_to_tree(overrides: Mapping[str, Any]) -> dict[str, Any]:
    tree: dict[str, Any] = {}
    for dotted, value in overrides.items():
        parts = dotted.split(".")
        if not all(part.strip() for part in parts):
            raise ConfigError(f"неверный ключ «{dotted}»", hint="Формат: раздел.параметр, например select.min_len")
        node = tree
        for part in parts[:-1]:
            child = node.setdefault(part, {})
            if not isinstance(child, dict):
                raise ConfigError(f"ключ «{dotted}» пересекается с другим переопределением")
            node = child
        node[parts[-1]] = value
    return tree


@cache
def _field_types(cls: type) -> dict[str, Any]:
    hints = get_type_hints(cls)
    return {f.name: hints[f.name] for f in dataclasses.fields(cls)}


def _merge_into(obj: Any, data: Any, path: str) -> None:
    """Слить словарь `data` в dataclass `obj` на месте, проверяя ключи и типы."""
    if data is None:
        return  # пустой раздел (`select:` без содержимого) ничего не меняет
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: ожидался раздел с параметрами, получено {_show(data)}")
    types_ = _field_types(type(obj))
    for raw_key, value in data.items():
        key = str(raw_key).replace("-", "_")
        key_path = f"{path}.{raw_key}" if path else str(raw_key)
        if key not in types_:
            raise _unknown_key(str(raw_key), type(obj), path)
        tp = types_[key]
        if dataclasses.is_dataclass(tp):
            _merge_into(getattr(obj, key), value, key_path)
        elif get_origin(tp) is dict:
            _merge_dict(getattr(obj, key), value, get_args(tp)[1], key_path)
        else:
            setattr(obj, key, _convert(value, tp, key_path))


def _merge_dict(target: dict[str, Any], data: Any, value_tp: Any, path: str) -> None:
    if data is None:
        return
    if not isinstance(data, Mapping):
        raise ConfigError(f"{path}: ожидался раздел с параметрами, получено {_show(data)}")
    for raw_key, value in data.items():
        name = str(raw_key)
        item_path = f"{path}.{name}"
        if dataclasses.is_dataclass(value_tp):
            item = target.get(name)
            if item is None:
                item = value_tp()
            _merge_into(item, value, item_path)
            target[name] = item
        else:
            target[name] = _convert(value, value_tp, item_path)


def _convert(value: Any, tp: Any, path: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)

    if tp is Any:
        return value

    if origin is Union or origin is types.UnionType:
        if value is None and type(None) in args:
            return None
        options = [arg for arg in args if arg is not type(None)]
        error: ConfigError | None = None
        for option in options:
            try:
                return _convert(value, option, path)
            except ConfigError as exc:
                error = error or exc
        assert error is not None
        raise error

    if origin is list:
        item_tp = args[0] if args else Any
        if value is None:
            return []
        if isinstance(value, str) and item_tp is str:
            value = [part.strip() for part in value.split(",") if part.strip()]
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: ожидался список, получено {_show(value)}")
        return [_convert(item, item_tp, f"{path}[{i}]") for i, item in enumerate(value)]

    if value is None:
        raise ConfigError(f"{path}: значение не может быть пустым", hint=f"Ожидается {_describe_type(tp)}.")

    if origin is Literal:
        if isinstance(value, str) and value in args:
            return value
        allowed = ", ".join(str(arg) for arg in args)
        raise ConfigError(f"{path}: недопустимое значение {_show(value)}", hint=f"Допустимые значения: {allowed}.")

    if tp is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.strip().lower() in _TRUE_WORDS | _FALSE_WORDS:
            return value.strip().lower() in _TRUE_WORDS
        raise ConfigError(f"{path}: ожидалось true или false, получено {_show(value)}")

    if tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise ConfigError(f"{path}: ожидалось целое число, получено {_show(value)}")

    if tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        raise ConfigError(f"{path}: ожидалось число, получено {_show(value)}")

    if tp is str:
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        raise ConfigError(f"{path}: ожидалась строка, получено {_show(value)}")

    if origin is tuple:
        if not isinstance(value, (list, tuple)) or len(value) != len(args):
            raise ConfigError(f"{path}: ожидался список из {len(args)} значений, получено {_show(value)}")
        return tuple(
            _convert(item, item_tp, f"{path}[{i}]") for i, (item, item_tp) in enumerate(zip(value, args, strict=True))
        )

    if dataclasses.is_dataclass(tp):
        obj = tp()
        _merge_into(obj, value, path)
        return obj

    raise TypeError(f"clipper: неподдерживаемый тип параметра {path}: {tp!r}")


def _unknown_key(key: str, cls: type, path: str) -> ConfigError:
    names = list(_field_types(cls))
    where = f"{path}.{key}" if path else key
    normalized = key.replace("-", "_")
    prefix = f"{path}." if path else ""
    candidates = [prefix + name for name in difflib.get_close_matches(normalized, names, n=1, cutoff=0.6)]
    if not candidates and not path:
        # Ключ без раздела: `clips` вместо `select.clips`.
        leaves = leaf_paths()
        candidates = [leaf for leaf in leaves if leaf.rsplit(".", 1)[-1] == normalized]
        candidates = candidates or difflib.get_close_matches(normalized, leaves, n=1, cutoff=0.6)
    if candidates:
        hint = f"Возможно, вы имели в виду «{candidates[0]}»?"
    elif path:
        hint = f"Параметры раздела {path}: {', '.join(names)}."
    else:
        hint = f"Разделы конфига: {', '.join(names)}."
    return ConfigError(f"неизвестный параметр «{where}»", hint=hint)


def _describe_type(tp: Any) -> str:
    origin = get_origin(tp)
    if origin is Literal:
        return "одно из: " + ", ".join(str(arg) for arg in get_args(tp))
    return {
        bool: "true или false",
        int: "целое число",
        float: "число",
        str: "строка",
    }.get(tp, "значение")


def _show(value: Any) -> str:
    text = f"«{value}»" if isinstance(value, str) else repr(value)
    return text if len(text) <= 60 else text[:57] + "…"


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# --- Нормализация и проверка смысла ----------------------------------------------------


def _normalize(cfg: Config) -> None:
    language = (cfg.transcribe.language or "").strip().lower()
    cfg.transcribe.language = None if language in ("", "auto") else language
    cfg.select.keywords = _unique(word.strip() for word in cfg.select.keywords)
    cfg.audio.fillers = _unique(word.strip().lower() for word in cfg.audio.fillers)
    if cfg.reframe.layout is not None and not cfg.reframe.layout.strip():
        cfg.reframe.layout = None


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def _validate(cfg: Config) -> None:
    def check(ok: bool, path: str, message: str, hint: str | None = None) -> None:
        if not ok:
            raise ConfigError(f"{path}: {message}", hint)

    check(cfg.download.max_height >= 144, "download.max_height", "слишком маленькое значение", "Обычно 720 или 1080.")

    language = cfg.transcribe.language
    check(
        language is None or re.fullmatch(r"[a-z]{2,3}", language) is not None,
        "transcribe.language",
        f"непонятный код языка «{language}»",
        "Используйте двухбуквенный код (ru, en, uk…) или auto.",
    )
    check(cfg.transcribe.margin >= 0, "transcribe.margin", "запас не может быть отрицательным")

    select = cfg.select
    check(select.clips >= 1, "select.clips", "нужен хотя бы 1 клип")
    check(select.min_len > 0, "select.min_len", "длина клипа должна быть больше 0")
    check(
        select.max_len >= select.min_len,
        "select.max_len",
        f"максимальная длина ({select.max_len:g} с) меньше минимальной ({select.min_len:g} с)",
        "Задайте max_len не меньше min_len.",
    )

    audio = cfg.audio
    check(
        audio.silence_db < 0,
        "audio.silence_db",
        "порог тишины задаётся в дБ и должен быть отрицательным",
        "Например, -35.",
    )
    check(audio.min_pause > 0, "audio.min_pause", "минимальная пауза должна быть больше 0")

    max_words = cfg.subtitles.max_words
    check(max_words is None or 1 <= max_words <= 5, "subtitles.max_words", "допустимо от 1 до 5 слов")

    for name, layout in cfg.layouts.items():
        path = f"layouts.{name}"
        cam = layout.webcam
        check(
            cam.width > 0 and cam.height > 0,
            f"{path}.webcam",
            "укажите размер вебки: width и height больше 0",
            "Координаты удобно снять по кадру с сеткой: clipper calibrate ВИДЕО.",
        )
        check(cam.x >= 0 and cam.y >= 0, f"{path}.webcam", "x и y не могут быть отрицательными")
        check(
            0.1 <= layout.webcam_zone <= 0.9,
            f"{path}.webcam_zone",
            "доля высоты под вебку должна быть от 0.1 до 0.9",
        )
        if layout.source_size is not None:
            width, height = layout.source_size
            check(width > 0 and height > 0, f"{path}.source_size", "ширина и высота должны быть больше 0")
            check(
                cam.x + cam.width <= width and cam.y + cam.height <= height,
                f"{path}.webcam",
                f"рамка вебки выходит за кадр {width}×{height}",
            )

    layout_name = cfg.reframe.layout
    if layout_name is not None and layout_name not in cfg.layouts:
        available = ", ".join(cfg.layouts)
        check(
            False,
            "reframe.layout",
            f"пресет «{layout_name}» не найден",
            f"Доступные пресеты: {available}." if available else "Опишите пресет в разделе layouts файла clipper.yaml.",
        )
