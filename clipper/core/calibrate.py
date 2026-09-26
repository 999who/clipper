"""Этап 6: `clipper calibrate` — кадр с координатной сеткой, чтобы снять координаты вебки.

- `calibrate.png` — кадр исходника в полном разрешении: тонкие линии через 50 px,
  линии с подписями через 100 px.
- С пресетом (`--layout ИМЯ`) на кадре дорисовываются рамка вебки и окно игры,
  а `calibrate_<имя>.png` показывает итоговый кадр 1080×1920 — через тот же граф
  фильтров, что и рендер.

Картинки пишутся через cv2.imencode + запись байтов: cv2.imwrite на Windows не
умеет пути с кириллицей.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.core import env
from clipper.core.config import Config, LayoutConfig
from clipper.core.download import prepare_source
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import run_ffmpeg
from clipper.core.models import SourceInfo
from clipper.core.reframe import Box, StreamPlan, layout_for, stream_graph, stream_plan

GRID_STEP = 50
LABEL_STEP = 100


@dataclass(frozen=True)
class CalibrateResult:
    source: SourceInfo
    at: float
    frame: Path  # кадр с сеткой
    preview: Path | None  # итоговый кадр 1080×1920 по пресету
    layout: str | None
    plan: StreamPlan | None


def calibrate(
    source: str,
    cfg: Config,
    reporter: Reporter,
    at: float | None = None,
    *,
    layout: LayoutConfig | None = None,
    name: str | None = None,
) -> CalibrateResult:
    """Кадр с сеткой; с пресетом — ещё рамки и превью клипа.

    Пресет — `layout` (например, ещё не сохранённый из TUI) или reframe.layout из конфига.
    """
    info = prepare_source(source, cfg, reporter)
    if info.width <= 0 or info.height <= 0:
        raise ClipperError("Неизвестен размер кадра видео.", hint="Проверьте файл: clipper download ФАЙЛ --force")
    work_dir = Path(cfg.paths.workdir).resolve() / info.id
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = env.require_ffmpeg().path
    if at is None:
        at = min(60.0, info.duration / 2) if info.duration else 0.0
    if info.duration and at >= info.duration:
        raise ClipperError(
            f"Момент {at:g} с — за концом видео ({info.duration:.0f} с).", hint="Укажите --at поменьше, например 1:30."
        )

    image = read_frame(ffmpeg, Path(info.video), at, info.width, info.height)
    plan = None
    if layout is None and cfg.reframe.layout:
        name, layout = layout_for(cfg)
    if layout is not None:
        name = name or "preview"
        plan = stream_plan(layout, info.width, info.height)
    draw_grid(image)
    if plan is not None:
        draw_box(image, plan.game, (255, 200, 0), "game")
        draw_box(image, plan.webcam, (0, 255, 0), "webcam")
    frame = write_png(work_dir / "calibrate.png", image)

    preview = None
    if plan is not None:
        preview = work_dir / f"calibrate_{name}.png"
        graph = stream_graph(plan, "[0:v:0]", "[v]")
        args = ["-ss", f"{at:.3f}", "-i", info.video, "-filter_complex", graph, "-map", "[v]", "-frames:v", "1",
                "-update", "1", str(preview)]  # fmt: skip
        run_ffmpeg(ffmpeg, args, cancel=reporter.cancel, log_path=work_dir / "clipper.log")
    return CalibrateResult(info, at, frame, preview, name, plan)


def read_frame(ffmpeg: str, video: Path, at: float, width: int, height: int) -> Any:
    """Кадр в момент `at` как массив BGR (размер — как видит зритель, с учётом поворота)."""
    import subprocess

    import numpy as np

    args = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-ss", f"{at:.3f}", "-i", str(video),
            "-frames:v", "1", "-vf", f"scale={width}:{height}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]  # fmt: skip
    try:
        result = subprocess.run(args, stdin=subprocess.DEVNULL, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ClipperError(f"Не удалось получить кадр: {exc}") from None
    size = width * height * 3
    if len(result.stdout) < size:
        error = result.stderr.decode("utf-8", "replace").strip().splitlines()
        raise ClipperError(
            f"Не удалось получить кадр на {at:g} с: {error[-1] if error else 'ffmpeg не вернул кадр'}",
            hint="Попробуйте другой момент: --at 1:30.",
        )
    return np.frombuffer(result.stdout[:size], np.uint8).reshape(height, width, 3).copy()


def draw_grid(image: Any) -> None:
    import cv2

    h, w = image.shape[:2]
    overlay = image.copy()
    for x in range(GRID_STEP, w, GRID_STEP):
        major = x % LABEL_STEP == 0
        cv2.line(overlay, (x, 0), (x, h - 1), (255, 255, 255) if major else (200, 200, 200), 2 if major else 1)
    for y in range(GRID_STEP, h, GRID_STEP):
        major = y % LABEL_STEP == 0
        cv2.line(overlay, (0, y), (w - 1, y), (255, 255, 255) if major else (200, 200, 200), 2 if major else 1)
    cv2.addWeighted(overlay, 0.45, image, 0.55, 0, dst=image)
    scale = max(0.4, min(w, h) / 1800)
    for x in range(LABEL_STEP, w, LABEL_STEP):
        _label(image, str(x), (x + 3, int(22 * scale / 0.6)), scale)
        _label(image, str(x), (x + 3, h - 8), scale)
    for y in range(LABEL_STEP, h, LABEL_STEP):
        _label(image, str(y), (4, y - 5), scale)
        _label(image, str(y), (w - int(60 * scale / 0.6), y - 5), scale)


def draw_box(image: Any, box: Box, color: tuple[int, int, int], text: str) -> None:
    import cv2

    cv2.rectangle(image, (box.x, box.y), (box.x + box.w - 1, box.y + box.h - 1), color, 3)
    scale = max(0.5, min(image.shape[:2]) / 1500)
    label = f"{text}: x={box.x} y={box.y} {box.w}x{box.h}"
    _label(image, label, (box.x + 6, box.y + int(30 * scale / 0.72)), scale, color)


def _label(image: Any, text: str, origin: tuple[int, int], scale: float, color=(255, 255, 255)) -> None:
    import cv2

    thickness = max(1, round(scale * 2))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def write_png(path: Path, image: Any) -> Path:
    import cv2

    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise ClipperError(f"Не удалось сохранить картинку {path.name}.")
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data.tobytes())
    tmp.replace(path)
    return path
