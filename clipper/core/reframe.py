"""Этап 6: геометрия вертикального кадра 1080×1920 и граф фильтров ffmpeg для неё.

- **video** (`reframe.mode: video`):
  - `9:16` — из кадра вырезается окно 9:16 во всю высоту и растягивается на
    1080×1920. Окно следует за лицом (траектория — через `sendcmd`, см.
    facetrack.py) или стоит по центру;
  - `1:1` — квадратное окно (тоже за лицом) вписывается в 1080×1920;
  - `original` — весь кадр вписывается в 1080×1920.

  Свободное место в 1:1 и original — размытая копия видео (`blur`) или чёрное (`black`).
- **stream** (`reframe.mode: stream`, пресет из `layouts`): вебка по
  координатам пресета заполняет верхнюю зону (обрезается без искажения
  пропорций), игра обрезается под нижнюю зону, зоны склеиваются `vstack`.

Здесь только чистая геометрия и строки фильтров: без ffmpeg, OpenCV и файлов.
"""

from dataclasses import dataclass

from clipper.core.config import Config, LayoutConfig
from clipper.core.errors import ClipperError

OUT_W, OUT_H = 1080, 1920
BLUR_W, BLUR_H = 270, 480  # фон размывается в уменьшенном виде — в разы быстрее
BLUR_RADIUS = 12
CROP_FILTER = "crop@face"  # имя фильтра, которому sendcmd шлёт координаты окна


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class VideoPlan:
    """Режим video: окно `crop` в исходнике → `fg` в кадре 1080×1920."""

    src_w: int
    src_h: int
    crop_w: int  # размер окна в пикселях исходника
    crop_h: int
    fg_w: int  # размер окна на выходе
    fg_h: int
    background: str  # blur | black (виден, только если окно не заполняет кадр)
    track: bool  # окно следует за лицом

    @property
    def fills_frame(self) -> bool:
        return self.fg_w == OUT_W and self.fg_h == OUT_H

    @property
    def movable(self) -> bool:
        """Есть ли куда двигать окно (иначе следить за лицом бессмысленно)."""
        return self.crop_w < self.src_w or self.crop_h < self.src_h

    def center_box(self) -> Box:
        return Box((self.src_w - self.crop_w) // 2 // 2 * 2, (self.src_h - self.crop_h) // 2 // 2 * 2,
                   self.crop_w, self.crop_h)  # fmt: skip

    def box_at(self, cx: float, cy: float) -> Box:
        """Окно с центром в (cx, cy), прижатое к границам кадра; координаты чётные."""
        x = min(max(round(cx - self.crop_w / 2), 0), self.src_w - self.crop_w)
        y = min(max(round(cy - self.crop_h / 2), 0), self.src_h - self.crop_h)
        return Box(x // 2 * 2, y // 2 * 2, self.crop_w, self.crop_h)


@dataclass(frozen=True)
class StreamPlan:
    """Режим stream: вебка сверху (`top_h`), игра снизу (`bottom_h`)."""

    webcam: Box
    game: Box
    top_h: int
    bottom_h: int


def even(value: float) -> int:
    return max(2, int(round(value / 2)) * 2)


def aspect_ratio(aspect: str, src_w: int, src_h: int) -> float:
    if aspect == "original":
        return src_w / src_h
    num, den = aspect.split(":")
    return int(num) / int(den)


def crop_size(src_w: int, src_h: int, ratio: float) -> tuple[int, int]:
    """Самое большое окно с отношением сторон `ratio` внутри кадра (чётные стороны)."""
    if src_w / src_h > ratio:  # кадр шире окна — окно во всю высоту
        h = src_h // 2 * 2
        w = min(even(h * ratio), src_w // 2 * 2)
    else:
        w = src_w // 2 * 2
        h = min(even(w / ratio), src_h // 2 * 2)
    return w, h


def fit_size(w: int, h: int, box_w: int = OUT_W, box_h: int = OUT_H) -> tuple[int, int]:
    """Вписать w×h в box_w×box_h с сохранением пропорций (чётные стороны)."""
    scale = min(box_w / w, box_h / h)
    return min(even(w * scale), box_w), min(even(h * scale), box_h)


def video_plan(cfg: Config, src_w: int, src_h: int) -> VideoPlan:
    reframe = cfg.reframe
    ratio = aspect_ratio(reframe.aspect, src_w, src_h)
    crop_w, crop_h = crop_size(src_w, src_h, ratio)
    fg_w, fg_h = fit_size(crop_w, crop_h)
    if abs(ratio - OUT_W / OUT_H) < 0.01:  # 9:16 — окно растягивается ровно на весь кадр
        fg_w, fg_h = OUT_W, OUT_H
    plan = VideoPlan(src_w, src_h, crop_w, crop_h, fg_w, fg_h, reframe.background, False)
    track = reframe.crop == "face" and reframe.aspect != "original" and plan.movable
    return VideoPlan(src_w, src_h, crop_w, crop_h, fg_w, fg_h, reframe.background, track)


def layout_for(cfg: Config) -> tuple[str, LayoutConfig]:
    name = cfg.reframe.layout
    if not name:
        raise ClipperError(
            "Для режима stream нужен пресет компоновки: --layout ИМЯ.",
            hint="Пресеты описываются в clipper.yaml в разделе layouts; координаты вебки снимите "
            "по кадру с сеткой: clipper calibrate ВИДЕО.",
        )
    if name not in cfg.layouts:
        known = ", ".join(sorted(cfg.layouts)) or "нет ни одного"
        raise ClipperError(
            f"Нет пресета компоновки «{name}».",
            hint=f"Пресеты в clipper.yaml (раздел layouts): {known}.",
        )
    return name, cfg.layouts[name]


def stream_plan(layout: LayoutConfig, src_w: int, src_h: int) -> StreamPlan:
    """Координаты вебки из пресета (пересчитанные под размер кадра) и окно игры."""
    cam = layout.webcam
    sx = sy = 1.0
    if layout.source_size is not None:
        sx, sy = src_w / layout.source_size[0], src_h / layout.source_size[1]
    x, y = round(cam.x * sx), round(cam.y * sy)
    w, h = round(cam.width * sx), round(cam.height * sy)
    x, y = min(max(x, 0), src_w - 2), min(max(y, 0), src_h - 2)
    w, h = max(2, min(w, src_w - x)), max(2, min(h, src_h - y))
    webcam = Box(x // 2 * 2, y // 2 * 2, max(2, w // 2 * 2), max(2, h // 2 * 2))

    top_h = even(OUT_H * layout.webcam_zone)
    bottom_h = OUT_H - top_h
    gw, gh = crop_size(src_w, src_h, OUT_W / bottom_h)
    gx = {"left": 0, "right": src_w - gw}.get(layout.game_crop, (src_w - gw) // 2)
    game = Box(gx // 2 * 2, (src_h - gh) // 2 // 2 * 2, gw, gh)
    return StreamPlan(webcam, game, top_h, bottom_h)


# --- граф фильтров -----------------------------------------------------------------------


def video_graph(plan: VideoPlan, inp: str, out: str, commands: str | None = None) -> str:
    """Фрагмент filter_complex: `inp` (кадр исходника) → `out` (1080×1920).

    `commands` — файл sendcmd с траекторией окна (относительный путь, ffmpeg
    запускается из его папки). Без него окно стоит по центру.
    """
    box = plan.center_box()
    crop = f"{CROP_FILTER}=w={box.w}:h={box.h}:x={box.x}:y={box.y}"
    if commands:
        crop = f"sendcmd=f={commands}," + crop
    scale = f"scale={plan.fg_w}:{plan.fg_h}:flags=lanczos,setsar=1"
    if plan.fills_frame:
        return f"{inp}{crop},{scale}{out}"
    x, y = (OUT_W - plan.fg_w) // 2, (OUT_H - plan.fg_h) // 2
    if plan.background == "black":
        return f"{inp}{crop},{scale},pad={OUT_W}:{OUT_H}:{x}:{y}:black{out}"
    return (
        f"{inp}split=2[fgsrc][bgsrc];"
        f"[bgsrc]scale={BLUR_W}:{BLUR_H}:force_original_aspect_ratio=increase,crop={BLUR_W}:{BLUR_H},"
        f"boxblur={BLUR_RADIUS}:2,scale={OUT_W}:{OUT_H},setsar=1[bg];"
        f"[fgsrc]{crop},{scale}[fg];"
        f"[bg][fg]overlay={x}:{y}{out}"
    )


def stream_graph(plan: StreamPlan, inp: str, out: str) -> str:
    """Фрагмент filter_complex: вебка сверху, игра снизу → 1080×1920."""
    cam, game = plan.webcam, plan.game
    return (
        f"{inp}split=2[cam][game];"
        f"[cam]crop={cam.w}:{cam.h}:{cam.x}:{cam.y},"
        f"scale={OUT_W}:{plan.top_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={OUT_W}:{plan.top_h},setsar=1[top];"
        f"[game]crop={game.w}:{game.h}:{game.x}:{game.y},scale={OUT_W}:{plan.bottom_h}:flags=lanczos,setsar=1[bottom];"
        f"[top][bottom]vstack=inputs=2{out}"
    )
