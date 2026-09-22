"""Этап 6: слежение за лицом — куда ставить окно кадрирования.

1. **Анализ.** ffmpeg отдаёт кадры клипа 5 раз в секунду в уменьшенном виде
   (ширина 960). На каждом кадре ищутся лица и считается маленькая миниатюра —
   по её резкому изменению определяется смена сцены.
2. **Детектор** (`reframe.detector`):
   - `yunet` (по умолчанию) — YuNet из OpenCV. Находит и небольшие лица (от ~5 %
     высоты кадра), ~15 мс на кадр;
   - `mediapipe` — BlazeFace short-range из mediapipe Tasks. Модель рассчитана на
     селфи: в горизонтальном кадре лица меньше ~15 % высоты не видит, поэтому
     кадр дополнительно режется на квадратные плитки.
3. **Траектория** (`camera_path`) — чистая функция от найденных лиц:
   - главное лицо — самое крупное; если рядом с прошлым главным есть лицо не
     намного меньше, остаётся прежнее (окно не прыгает между людьми);
   - мелкие движения лица окно не двигают (мёртвая зона);
   - пропало лицо — окно ждёт на месте, потом плавно уходит к центру;
   - смена сцены — окно переходит сразу, без проезда;
   - лицо есть меньше чем в 20 % кадров — кроп по центру.
4. **Для ffmpeg** траектория пересчитывается во время готового клипа (с учётом
   вырезок) и пишется файлом команд `sendcmd` для фильтра `crop@face`.

Результаты анализа кэшируются в work/<id>/tmp/faces_NN.json.
"""

import bisect
import json
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from clipper.core.errors import ClipperError
from clipper.core.events import CancelToken
from clipper.core.models import write_json_atomic
from clipper.core.reframe import CROP_FILTER, VideoPlan
from clipper.core.timeline import Timeline

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
MODELS = {
    "yunet": ASSETS_DIR / "face_detection_yunet_2023mar.onnx",
    "mediapipe": ASSETS_DIR / "blaze_face_short_range.tflite",
}

ANALYSIS_FPS = 5
ANALYSIS_WIDTH = 960
MIN_FACE = 0.03  # лица ниже 3 % высоты кадра не учитываются
MIN_SCORE = 0.6
MIN_COVERAGE = 0.2  # лицо меньше чем в 20 % кадров — кроп по центру
DEADZONE = 0.12  # доля окна: пока лицо смещается меньше, окно стоит
HOLD = 1.0  # с: лицо пропало — окно ждёт, потом уходит к центру
SMOOTH = 5  # отсчётов (1 с) в скользящем среднем; два прохода
STICKY = 0.5  # прежнее лицо остаётся главным, если оно не меньше половины самого крупного
SCENE_CUT = 30.0  # средняя разница миниатюр 32×18 по H, S, V — новая сцена
FACE_HEIGHT = 0.4  # если окно двигается по вертикали — лицо на 40 % высоты окна
CACHE_VERSION = 2


@dataclass(frozen=True)
class Face:
    """Лицо в долях кадра (0…1)."""

    x: float
    y: float
    w: float
    h: float
    score: float = 1.0

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass(frozen=True)
class Sample:
    t: float  # время исходника
    faces: tuple[Face, ...] = ()
    cut: bool = False  # с этого кадра начинается новая сцена


# --- детекторы ------------------------------------------------------------------------


class YuNetDetector:
    def __init__(self) -> None:
        import cv2
        import numpy as np

        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        # Модель — из памяти: OpenCV на Windows не читает пути с кириллицей.
        model = np.frombuffer(_read_model("yunet"), np.uint8)
        self._net = cv2.FaceDetectorYN.create("onnx", model, np.array([], np.uint8), (320, 320), MIN_SCORE, 0.3, 5000)
        self._size: tuple[int, int] | None = None

    def detect(self, bgr: Any) -> list[Face]:
        h, w = bgr.shape[:2]
        if self._size != (w, h):
            self._net.setInputSize((w, h))
            self._size = (w, h)
        _, found = self._net.detect(bgr)
        if found is None:
            return []
        return [Face(float(f[0]) / w, float(f[1]) / h, float(f[2]) / w, float(f[3]) / h, float(f[14])) for f in found]


class MediapipeDetector:
    """BlazeFace short-range: весь кадр + квадратные плитки (модель видит только крупные лица)."""

    def __init__(self) -> None:
        from mediapipe.tasks.python import BaseOptions, vision

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_buffer=_read_model("mediapipe")),
            running_mode=vision.RunningMode.IMAGE,
            min_detection_confidence=MIN_SCORE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _detect(self, rgb: Any) -> list[tuple[float, float, float, float, float]]:
        import mediapipe as mp
        import numpy as np

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(image)
        boxes = []
        for d in result.detections:
            b = d.bounding_box
            boxes.append((b.origin_x, b.origin_y, b.width, b.height, d.categories[0].score if d.categories else 1.0))
        return boxes

    def detect(self, bgr: Any) -> list[Face]:
        rgb = bgr[:, :, ::-1]
        h, w = rgb.shape[:2]
        found = []
        for x0, y0, size in [(0, 0, None), *square_tiles(w, h)]:
            part = rgb if size is None else rgb[y0 : y0 + size, x0 : x0 + size]
            found += [(x + x0, y + y0, bw, bh, s) for x, y, bw, bh, s in self._detect(part)]
        faces = [Face(float(x) / w, float(y) / h, float(bw) / w, float(bh) / h, float(s)) for x, y, bw, bh, s in found]
        return suppress_overlaps(faces)


def square_tiles(w: int, h: int) -> list[tuple[int, int, int]]:
    """Квадратные плитки со стороной в половину короткой стороны, с перекрытием."""
    side = min(w, h) // 2
    tiles = []
    for y0 in _starts(h, side):
        for x0 in _starts(w, side):
            tiles.append((x0, y0, side))
    return tiles


def _starts(length: int, side: int) -> list[int]:
    count = max(1, -(-(length - side) // (side // 2)) + 1)  # шаг — полплитки
    if count == 1:
        return [0]
    return [round(i * (length - side) / (count - 1)) for i in range(count)]


def suppress_overlaps(faces: list[Face], iou: float = 0.3) -> list[Face]:
    kept: list[Face] = []
    for face in sorted(faces, key=lambda f: f.score, reverse=True):
        if all(_iou(face, other) < iou for other in kept):
            kept.append(face)
    return kept


def _iou(a: Face, b: Face) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _read_model(kind: str) -> bytes:
    path = MODELS[kind]
    try:
        return path.read_bytes()
    except OSError:
        raise ClipperError(
            f"Нет модели детектора лиц: {path}",
            hint="Файл входит в репозиторий clipper — обновите проект: git pull.",
        ) from None


def make_detector(kind: str) -> YuNetDetector | MediapipeDetector:
    """Детектор лиц; ошибка загрузки — ClipperError с подсказкой."""
    try:
        return MediapipeDetector() if kind == "mediapipe" else YuNetDetector()
    except ClipperError:
        raise
    except Exception as exc:  # нативные библиотеки падают не только ImportError
        package = "mediapipe" if kind == "mediapipe" else "opencv-contrib-python"
        raise ClipperError(
            f"Детектор лиц ({kind}) не запустился: {exc}",
            hint=f"Проверьте clipper doctor; переустановить: pip install --force-reinstall {package}",
        ) from None


# --- анализ клипа ---------------------------------------------------------------------


def analysis_size(src_w: int, src_h: int) -> tuple[int, int]:
    w = min(ANALYSIS_WIDTH, src_w) // 2 * 2
    return w, max(2, round(src_h * w / src_w / 2) * 2)


def analyze_faces(
    ffmpeg: str,
    video: Path,
    start: float,
    end: float,
    src_w: int,
    src_h: int,
    detector: Any,
    on_progress: Callable[[float], None] | None = None,
    cancel: CancelToken | None = None,
) -> list[Sample]:
    """Лица и смены сцен на отрезке исходника [start, end], ANALYSIS_FPS кадров в секунду."""
    import numpy as np

    w, h = analysis_size(src_w, src_h)
    args = [
        ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(video),
        "-map", "0:v:0", "-vf", f"fps={ANALYSIS_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]  # fmt: skip
    frame_bytes = w * h * 3
    samples: list[Sample] = []
    prev_thumb = None
    try:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise ClipperError(f"Не удалось запустить ffmpeg для поиска лица: {exc}") from None
    try:
        assert process.stdout is not None
        while True:
            if cancel is not None:
                cancel.check()
            data = process.stdout.read(frame_bytes)
            if len(data) < frame_bytes:
                break
            frame = np.frombuffer(data, np.uint8).reshape(h, w, 3)
            thumb = thumbnail(frame)
            cut = prev_thumb is not None and scene_change(prev_thumb, thumb) > SCENE_CUT
            prev_thumb = thumb
            faces = [f for f in detector.detect(frame) if f.h >= MIN_FACE and f.score >= MIN_SCORE]
            t = start + len(samples) / ANALYSIS_FPS
            samples.append(Sample(round(t, 3), tuple(faces), cut))
            if on_progress is not None:
                on_progress(t - start)
        process.wait(timeout=30)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if process.returncode not in (0, None) and not samples:
        error = (
            (process.stderr.read().decode("utf-8", "replace").strip().splitlines() or ["?"])[-1]
            if process.stderr
            else ""
        )
        raise ClipperError(f"ffmpeg не смог прочитать кадры для поиска лица: {error}")
    return samples


def thumbnail(bgr: Any) -> Any:
    """Миниатюра 32×18 в HSV для сравнения кадров."""
    import cv2
    import numpy as np

    small = cv2.resize(bgr, (32, 18), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.int16)


def scene_change(a: Any, b: Any) -> float:
    """Насколько различаются миниатюры: среднее |Δ| по H, S и V (как в PySceneDetect).

    Тон (H) в OpenCV — 0…180 и замкнут в круг: красный 179 и 1 — почти одно и то же.
    """
    import numpy as np

    diff = np.abs(a - b)
    hue = np.minimum(diff[..., 0], 180 - diff[..., 0])
    return float((hue.mean() + diff[..., 1].mean() + diff[..., 2].mean()) / 3)


def load_samples(path: Path, key: dict[str, Any]) -> list[Sample] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if data.get("key") != key:
        return None
    try:
        return [Sample(float(s["t"]), tuple(Face(*f) for f in s["faces"]), bool(s.get("cut"))) for s in data["samples"]]
    except (KeyError, TypeError, ValueError):
        return None


def save_samples(path: Path, key: dict[str, Any], samples: list[Sample]) -> None:
    data = {
        "key": key,
        "samples": [
            {"t": s.t, "cut": s.cut, "faces": [[round(v, 4) for v in asdict(f).values()] for f in s.faces]}
            for s in samples
        ],
    }
    write_json_atomic(path, data, indent=None)


def cache_key(video: Path, start: float, end: float, detector: str) -> dict[str, Any]:
    try:
        stat = video.stat()
        stamp = [stat.st_size, int(stat.st_mtime)]
    except OSError:
        stamp = None
    return {
        "version": CACHE_VERSION, "video": str(video), "stamp": stamp, "start": round(start, 3), "end": round(end, 3),
        "detector": detector, "fps": ANALYSIS_FPS, "width": ANALYSIS_WIDTH,
    }  # fmt: skip


# --- траектория ------------------------------------------------------------------------


def choose_faces(samples: list[Sample]) -> list[Face | None]:
    """Главное лицо на каждом кадре: самое крупное, но прежнее — если оно рядом и не сильно меньше."""
    chosen: list[Face | None] = []
    prev: Face | None = None
    for sample in samples:
        if sample.cut:
            prev = None
        if not sample.faces:
            chosen.append(None)
            continue
        largest = max(sample.faces, key=lambda f: f.area)
        pick = largest
        if prev is not None:
            near = min(sample.faces, key=lambda f: abs(f.cx - prev.cx) + abs(f.cy - prev.cy))
            if abs(near.cx - prev.cx) < max(prev.w, 0.05) * 1.5 and near.area >= STICKY * largest.area:
                pick = near
        chosen.append(pick)
        prev = pick
    return chosen


def camera_path(samples: list[Sample], plan: VideoPlan) -> tuple[list[tuple[float, float]], float]:
    """Центр окна (пиксели исходника) на каждом кадре анализа и доля кадров с лицом."""
    if not samples:
        return [], 0.0
    chosen = choose_faces(samples)
    coverage = sum(face is not None for face in chosen) / len(samples)
    center = (plan.src_w / 2, plan.src_h / 2)
    if coverage < MIN_COVERAGE:
        return [center] * len(samples), coverage

    targets: list[tuple[float, float]] = []
    last: tuple[float, float] | None = None
    last_t = 0.0
    for sample, face in zip(samples, chosen, strict=True):
        if sample.cut:
            last = None
        if face is not None:
            # Если окно двигается и по вертикали — лицо чуть выше центра окна.
            ty = face.cy * plan.src_h + (0.5 - FACE_HEIGHT) * plan.crop_h
            last, last_t = (face.cx * plan.src_w, ty), sample.t
            targets.append(last)
        elif last is not None and sample.t - last_t <= HOLD:
            targets.append(last)
        else:
            targets.append(center)

    cams: list[tuple[float, float]] = []
    cam: list[float] | None = None
    for sample, (tx, ty) in zip(samples, targets, strict=True):
        if cam is None or sample.cut:
            cam = [tx, ty]
        else:
            if abs(tx - cam[0]) > DEADZONE * plan.crop_w:
                cam[0] = tx
            if abs(ty - cam[1]) > DEADZONE * plan.crop_h:
                cam[1] = ty
        cams.append((cam[0], cam[1]))

    path: list[tuple[float, float]] = []
    for first, last_index in _segments(samples):
        xs = _smooth([c[0] for c in cams[first:last_index]])
        ys = _smooth([c[1] for c in cams[first:last_index]])
        path += [_clamp(x, y, plan) for x, y in zip(xs, ys, strict=True)]
    return path, coverage


def _segments(samples: list[Sample]) -> list[tuple[int, int]]:
    starts = [0] + [i for i, s in enumerate(samples) if s.cut and i > 0]
    return list(zip(starts, starts[1:] + [len(samples)], strict=True))


def _smooth(values: list[float], passes: int = 2) -> list[float]:
    half = SMOOTH // 2
    for _ in range(passes):
        padded = [values[0]] * half + values + [values[-1]] * half
        values = [sum(padded[i : i + SMOOTH]) / SMOOTH for i in range(len(values))]
    return values


def _clamp(x: float, y: float, plan: VideoPlan) -> tuple[float, float]:
    half_w, half_h = plan.crop_w / 2, plan.crop_h / 2
    return min(max(x, half_w), plan.src_w - half_w), min(max(y, half_h), plan.src_h - half_h)


def position_at(t: float, samples: list[Sample], path: list[tuple[float, float]]) -> tuple[float, float]:
    """Центр окна в момент t исходника: между кадрами анализа — линейно, через смену сцены — скачком."""
    times = [s.t for s in samples]
    i = bisect.bisect_right(times, t) - 1
    if i < 0:
        return path[0]
    if i >= len(samples) - 1:
        return path[-1]
    if samples[i + 1].cut:
        return path[i]
    k = (t - times[i]) / (times[i + 1] - times[i])
    (x0, y0), (x1, y1) = path[i], path[i + 1]
    return x0 + (x1 - x0) * k, y0 + (y1 - y0) * k


def crop_commands(
    samples: list[Sample], path: list[tuple[float, float]], plan: VideoPlan, timeline: Timeline, fps: float
) -> str:
    """Файл sendcmd: координаты окна на каждом кадре готового клипа (только когда они меняются)."""
    fps = fps if fps and fps > 0 else 30.0
    lines: list[str] = []
    prev = None
    frames = int(timeline.duration * fps) + 1
    for n in range(frames):
        t_out = n / fps
        box = plan.box_at(*position_at(timeline.to_source(t_out), samples, path))
        if (box.x, box.y) != prev:
            at = max(t_out - 0.5 / fps, 0.0)
            lines.append(f"{at:.3f} {CROP_FILTER} x {box.x}, {CROP_FILTER} y {box.y};")
            prev = (box.x, box.y)
    return "\n".join(lines) + "\n"
