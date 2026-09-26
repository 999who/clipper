"""Режим стримера в TUI: пресеты «вебка сверху, игра снизу» без правки clipper.yaml руками.

- Список пресетов из clipper.yaml и «＋ Новый пресет».
- Редактор: видео для примера и момент кадра, координаты вебки, доля высоты под
  вебку, какую часть игры брать. «Кадр с сеткой» и «Превью клипа» запускают
  calibrate и открывают картинку в системном просмотрщике (в терминале её не
  показать). «Сохранить» пишет пресет в clipper.yaml (config.save_layout — только
  его блок, комментарии остаются) и включает его для нарезки.
"""

import re
from pathlib import Path
from typing import Any

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.screen import Screen
from textual.widgets import Footer, Header, Label, OptionList, Static
from textual.widgets.option_list import Option

from clipper.core import pipeline
from clipper.core.config import CONFIG_FILENAME, LayoutConfig, Rect, load_config, save_layout
from clipper.core.errors import ClipperError, ConfigError
from clipper.core.models import parse_time
from clipper.tui.progress import ProgressScreen
from clipper.tui.project import clock
from clipper.tui.system import open_path
from clipper.tui.widgets import CANCEL, ChoiceModal, InputModal, row

NAME = re.compile(r"[\w-]+")
GAME_CROPS = (("center", "середина кадра"), ("left", "левый край"), ("right", "правый край"))
NEW_WEBCAM = Rect(1440, 0, 480, 270)  # правый верхний угол 1080p — частое место вебки


def describe(layout: LayoutConfig) -> str:
    cam = layout.webcam
    return f"вебка {cam.width}×{cam.height} (x {cam.x}, y {cam.y}), сверху {layout.webcam_zone:.0%} клипа"


class PresetsScreen(Screen[Any]):
    """Пресеты режима стримера."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Режим стримера: вебка сверху, игра снизу", classes="screen-title")
        yield Static(
            Text(
                "Пресет — где на кадре стрима вебка. Настройте его один раз, потом при нарезке выберите "
                "«Режим → стрим» и этот пресет.",
                style="dim",
            ),
            id="intro",
        )
        yield OptionList(id="presets")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "режим стримера"
        self._list()
        self.query_one("#presets", OptionList).focus()

    def on_screen_resume(self) -> None:
        self._list()

    def _list(self) -> None:
        self.names = sorted(self.app.store.config().layouts)
        layouts = self.app.store.config().layouts
        options: list[Option | None] = [Option(Text("＋ Новый пресет", style="bold"), id="new")]
        if self.names:
            options.append(None)
        for index, name in enumerate(self.names):
            options.append(
                Option(Text.assemble((name, "bold"), ("   " + describe(layouts[name]), "dim")), id=str(index))
            )
        presets = self.query_one("#presets", OptionList)
        presets.set_options(options)
        presets.highlighted = 0

    @on(OptionList.OptionSelected, "#presets")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        if event.option.id == "new":
            self.app.push_screen(PresetScreen(None))
        else:
            self.app.push_screen(PresetScreen(self.names[int(event.option.id or 0)]))


class PresetScreen(Screen[Any]):
    """Редактор одного пресета. Enter на строке — изменить."""

    BINDINGS = [Binding("escape", "app.pop_screen", "Назад")]

    def __init__(self, name: str | None) -> None:
        super().__init__()
        self.original = name
        self.name_value = name or "my_stream"
        self.source = ""
        self.at: float | None = None
        self.preset = LayoutConfig(webcam=NEW_WEBCAM)

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label("Пресет стрима", classes="screen-title")
        yield OptionList(id="preset")
        yield Static("", id="help")
        yield Footer()

    def on_mount(self) -> None:
        self.sub_title = "режим стримера"
        store = self.app.store
        if self.original and self.original in store.config().layouts:
            self.preset = _copy(store.config().layouts[self.original])
        self.source = self._last_source()
        self._refresh()
        self.query_one("#preset", OptionList).focus()

    def _last_source(self) -> str:
        """Видео последнего проекта — чтобы не вводить путь заново."""
        for item in pipeline.list_projects(self.app.store.config())[:1]:
            try:
                return pipeline.open_project(item.path).source.input
            except ClipperError:
                return ""
        return ""

    # --- строки ---

    def _refresh(self) -> None:
        cam, layout = self.preset.webcam, self.preset
        size = f"{layout.source_size[0]}×{layout.source_size[1]}" if layout.source_size else "узнается по кадру"
        at = clock(self.at) if self.at is not None else "1:00 (или середина видео)"
        options: list[Option | None] = [
            Option(row("Название", self.name_value), id="name"),
            Option(row("Видео стрима для примера", self.source or "— укажите файл или ссылку"), id="source"),
            Option(row("Момент кадра", at), id="at"),
            None,
            Option(Text("Вебка на кадре, пиксели", style="bold underline"), disabled=True),
            Option(row("Левый край (x)", str(cam.x)), id="x"),
            Option(row("Верхний край (y)", str(cam.y)), id="y"),
            Option(row("Ширина", str(cam.width)), id="width"),
            Option(row("Высота", str(cam.height)), id="height"),
            Option(row("Разрешение видео", size), id="size"),
            None,
            Option(Text("Клип 1080×1920", style="bold underline"), disabled=True),
            Option(row("Вебка сверху, доля высоты", f"{layout.webcam_zone:.0%}"), id="zone"),
            Option(row("Игра снизу", dict(GAME_CROPS)[layout.game_crop]), id="game"),
            None,
            Option(Text("▶ Кадр с сеткой — снять координаты вебки", style="bold"), id="grid"),
            Option(Text("▶ Превью клипа 1080×1920", style="bold"), id="preview"),
            Option(Text("✓ Сохранить пресет и нарезать в этом режиме", style="bold"), id="save"),
            Option("← Назад", id="back"),
        ]
        presets = self.query_one("#preset", OptionList)
        highlighted = presets.highlighted
        presets.set_options(options)
        presets.highlighted = highlighted if highlighted is not None else 1

    @on(OptionList.OptionHighlighted, "#preset")
    def _help(self, event: OptionList.OptionHighlighted) -> None:
        texts = {
            "source": "Файл записи стрима или ссылка на YouTube — с него снимается кадр.",
            "at": "Момент, где на кадре хорошо видно вебку.",
            "x": "Координаты — по картинке «Кадр с сеткой»: тонкие линии через 50 px, подписи через 100 px.",
            "size": "Разрешение кадра, по которому сняты координаты. Заполняется само после «Кадр с сеткой».",
            "zone": "Какую часть высоты клипа займёт вебка (остальное — игра). Обычно 30–40 %.",
            "game": "Какую часть кадра игры оставить, если она шире нижней зоны.",
            "grid": "Сохранит кадр с координатной сеткой и откроет его.",
            "preview": "Покажет, как будет выглядеть клип: вебка сверху, игра снизу.",
            "save": "Запишет пресет в clipper.yaml (остальные строки файла не меняются).",
        }
        key = event.option.id or ""
        key = "x" if key in ("y", "width", "height") else key
        self.query_one("#help", Static).update(Text(texts.get(key, "Enter — изменить."), style="italic dim"))

    @on(OptionList.OptionSelected, "#preset")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        key = event.option.id or ""
        cam = self.preset.webcam
        numbers = {"x": cam.x, "y": cam.y, "width": cam.width, "height": cam.height}
        if key == "name":
            self._ask("Название пресета", self.name_value, "Буквы, цифры, _ и -.", self._check_name, self._set_name)
        elif key == "source":
            self._ask("Видео стрима", self.source, "Путь к файлу или ссылка на YouTube.", None, self._set_source)
        elif key == "at":
            current = clock(self.at) if self.at is not None else "1:00"
            self._ask("Момент кадра", current, "Например 1:30 или 90.", self._check_time, self._set_time)
        elif key in numbers:
            label = {"x": "Левый край вебки (x)", "y": "Верхний край вебки (y)", "width": "Ширина вебки",
                     "height": "Высота вебки"}[key]  # fmt: skip
            self._ask(label, str(numbers[key]), "В пикселях кадра стрима.", self._check_int,
                      lambda text: self._set_webcam(key, int(text)))  # fmt: skip
        elif key == "size":
            size = self.preset.source_size
            current = f"{size[0]}x{size[1]}" if size else ""
            self._ask("Разрешение видео", current, "Например 1920x1080. Пусто — узнать по кадру.",
                      self._check_size, self._set_size)  # fmt: skip
        elif key == "zone":
            zone = f"{self.preset.webcam_zone * 100:g}"
            self._ask("Доля высоты под вебку, %", zone, "От 10 до 90.", self._check_zone, self._set_zone)
        elif key == "game":
            self.app.push_screen(ChoiceModal("Игра снизу", GAME_CROPS, self.preset.game_crop), self._set_game)
        elif key == "grid":
            self._grid()
        elif key == "preview":
            self._preview()
        elif key == "save":
            self._save()
        elif key == "back":
            self.app.pop_screen()

    def _ask(self, title: str, value: str, help: str, check: Any, apply: Any) -> None:
        def done(text: Any) -> None:
            if text is not CANCEL:
                apply(text.strip())
                self._refresh()

        self.app.push_screen(InputModal(title, value, help, check), done)

    # --- проверки и изменения ---

    @staticmethod
    def _check_name(text: str) -> str | None:
        return None if NAME.fullmatch(text.strip()) else "Только буквы, цифры, _ и - (без пробелов)."

    @staticmethod
    def _check_int(text: str) -> str | None:
        return None if text.strip().isdigit() else "Целое число пикселей, например 360."

    @staticmethod
    def _check_time(text: str) -> str | None:
        try:
            parse_time(text)
        except ValueError as exc:
            return f"{exc}. Например 1:30"
        return None

    @staticmethod
    def _check_size(text: str) -> str | None:
        if not text.strip() or re.fullmatch(r"\s*\d+\s*[xх×*]\s*\d+\s*", text):
            return None
        return "Например 1920x1080."

    @staticmethod
    def _check_zone(text: str) -> str | None:
        try:
            value = float(text.replace(",", "."))
        except ValueError:
            return "Число процентов, например 33."
        return None if 10 <= value <= 90 else "От 10 до 90 %."

    def _set_name(self, text: str) -> None:
        self.name_value = text

    def _set_source(self, text: str) -> None:
        self.source = text.strip('"')

    def _set_time(self, text: str) -> None:
        self.at = parse_time(text)

    def _set_webcam(self, key: str, value: int) -> None:
        cam = self.preset.webcam
        values = {"x": cam.x, "y": cam.y, "width": cam.width, "height": cam.height, key: value}
        self._update(webcam=Rect(**values))

    def _set_size(self, text: str) -> None:
        numbers = re.findall(r"\d+", text)
        self._update(source_size=(int(numbers[0]), int(numbers[1])) if len(numbers) == 2 else None)

    def _set_zone(self, text: str) -> None:
        self._update(webcam_zone=float(text.replace(",", ".")) / 100)

    def _set_game(self, value: Any) -> None:
        if value is not CANCEL:
            self._update(game_crop=value)
            self._refresh()

    def _update(self, **changes: Any) -> None:
        data = {"webcam": self.preset.webcam, "webcam_zone": self.preset.webcam_zone,
                "game_crop": self.preset.game_crop, "source_size": self.preset.source_size, **changes}  # fmt: skip
        self.preset = LayoutConfig(**data)

    def checked(self) -> LayoutConfig | None:
        """Пресет с той же проверкой, что у clipper.yaml; ошибка — уведомление."""
        cam = self.preset.webcam
        data = {"webcam": {"x": cam.x, "y": cam.y, "width": cam.width, "height": cam.height},
                "webcam_zone": self.preset.webcam_zone, "game_crop": self.preset.game_crop,
                "source_size": list(self.preset.source_size) if self.preset.source_size else None}  # fmt: skip
        try:
            load_config(None, {f"layouts.{self.name_value}": data})
        except ConfigError as exc:
            message = exc.message.split(": ", 1)[-1] if exc.message.startswith("настройки") else exc.message
            self.app.notify(message + (f"\n{exc.hint}" if exc.hint else ""), severity="error", timeout=8)
            return None
        return self.preset

    # --- действия ---

    def _need_source(self) -> bool:
        if not self.source:
            self.app.notify("Сначала укажите видео стрима (строка «Видео стрима для примера»).", severity="warning")
            return False
        return True

    def _grid(self) -> None:
        from clipper.core.calibrate import calibrate

        if not self._need_source():
            return
        cfg, source, at = self.app.store.config(), self.source, self.at
        job = ProgressScreen("Кадр с сеткой", lambda reporter: calibrate(source, cfg, reporter, at=at))
        self.app.push_screen(job, self._grid_done)

    def _grid_done(self, result: Any) -> None:
        if not result:
            return
        size = (result.source.width, result.source.height)
        if self.preset.source_size != size:
            self._update(source_size=size)
            self._refresh()
        open_path(result.frame)
        self.app.notify(f"Открыт кадр с сеткой: {result.frame}", timeout=8)

    def _preview(self) -> None:
        from clipper.core.calibrate import calibrate

        layout = self.checked()
        if layout is None or not self._need_source():
            return
        cfg, source, at, name = self.app.store.config(), self.source, self.at, self.name_value
        job = ProgressScreen(
            "Превью клипа", lambda reporter: calibrate(source, cfg, reporter, at=at, layout=layout, name=name)
        )
        self.app.push_screen(job, self._preview_done)

    def _preview_done(self, result: Any) -> None:
        if not result:
            return
        if self.preset.source_size is None:
            self._update(source_size=(result.source.width, result.source.height))
            self._refresh()
        if result.preview is not None:
            open_path(result.preview)
            self.app.notify(f"Открыто превью. Рамки вебки и игры — на {result.frame.name}.", timeout=8)

    def _save(self) -> None:
        layout = self.checked()
        if layout is None:
            return
        store = self.app.store
        path = store.path or Path.cwd() / CONFIG_FILENAME
        try:
            save_layout(path, self.name_value, layout)
            store.reload(path)
        except (ConfigError, OSError) as exc:
            self.app.notify(f"Не удалось сохранить: {getattr(exc, 'message', exc)}", severity="error", timeout=10)
            return
        store.try_set("reframe.mode", "stream")
        store.try_set("reframe.layout", self.name_value)
        self.app.notify(
            f"Пресет «{self.name_value}» сохранён в {path}. При нарезке уже выбраны «Режим → стрим» и этот пресет.",
            timeout=10,
        )
        self.app.pop_screen()


def _copy(layout: LayoutConfig) -> LayoutConfig:
    cam = layout.webcam
    return LayoutConfig(webcam=Rect(cam.x, cam.y, cam.width, cam.height), webcam_zone=layout.webcam_zone,
                        game_crop=layout.game_crop, source_size=layout.source_size)  # fmt: skip
