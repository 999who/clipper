"""Проверка окружения для `clipper doctor`.

Каждая проверка изолирована: если она упала с исключением, это превращается
в строку отчёта со статусом «ошибка» и не мешает остальным проверкам.
"""

import importlib
import importlib.metadata
import platform
import struct
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from clipper.core import env
from clipper.core.config import find_config_file, load_config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import encoder_works, list_encoders, list_filters

Status = Literal["ok", "warn", "fail", "info"]

# large-v3 в float16 занимает около 4,5 ГБ видеопамяти.
WHISPER_VRAM_MB = 4600

OPENCV_PACKAGES = (
    "opencv-contrib-python",
    "opencv-python",
    "opencv-python-headless",
    "opencv-contrib-python-headless",
)


@dataclass(frozen=True)
class CheckResult:
    group: str
    name: str
    status: Status
    detail: str = ""
    hint: str | None = None


class _Doctor:
    def __init__(self, config_path: str | Path | None) -> None:
        self.config_path = config_path
        self.results: list[CheckResult] = []
        self.ffmpeg: env.Tool | None = None
        self._group = self._name = ""  # текущая проверка — для add()

    def checks(self) -> list[tuple[str, str, Callable[[], None]]]:
        return [
            ("Система", "Python", self.python),
            ("ffmpeg", "ffmpeg", self.ffmpeg_tool),
            ("ffmpeg", "ffprobe", self.ffprobe_tool),
            ("ffmpeg", "Субтитры (libass)", self.libass),
            ("ffmpeg", "Кодирование H.264", self.encoders),
            ("YouTube", "yt-dlp", self.ytdlp),
            ("YouTube", "JS-движок", self.js_runtime),
            ("Распознавание речи", "faster-whisper", self.faster_whisper),
            ("Распознавание речи", "Библиотеки CUDA", self.cuda_libraries),
            ("Распознавание речи", "Видеокарта", self.gpu),
            ("Распознавание речи", "CUDA в ctranslate2", self.ctranslate2_cuda),
            ("Распознавание речи", "Модель large-v3", self.whisper_model),
            ("Кадр и субтитры", "mediapipe", self.mediapipe),
            ("Кадр и субтитры", "OpenCV", self.opencv),
            ("Кадр и субтитры", "Детектор лиц", self.face_detector),
            ("Кадр и субтитры", "pysubs2", self.pysubs2),
            ("Конфиг", "Файл конфига", self.config),
        ]

    def add(self, status: Status, detail: str = "", hint: str | None = None, name: str | None = None) -> None:
        self.results.append(CheckResult(self._group, name or self._name, status, detail, hint))

    def run(self, reporter: Reporter) -> list[CheckResult]:
        checks = self.checks()
        with reporter.stage("doctor", "Проверка окружения", total=len(checks)) as stage:
            for index, (group, name, check) in enumerate(checks):
                stage.update(index, message=name)
                self._group, self._name = group, name
                try:
                    check()
                except Exception as exc:  # проверка не должна ронять весь отчёт
                    self.add("fail", f"проверка упала: {type(exc).__name__}: {exc}")
            stage.update(len(checks))
        return self.results

    # --- Система ---

    def python(self) -> None:
        bits = struct.calcsize("P") * 8
        detail = f"{platform.python_version()}, {bits}-бит — {sys.executable}"
        if sys.prefix != sys.base_prefix:
            detail += " (виртуальное окружение)"
        if bits != 64:
            self.add("fail", detail, "Нужен 64-битный Python: CUDA и ctranslate2 не работают в 32-битном.")
        else:
            self.add("ok", detail)

    # --- ffmpeg ---

    def ffmpeg_tool(self) -> None:
        candidates = env.ffmpeg_candidates()
        chosen = env.choose_ffmpeg(candidates)
        if chosen is None:
            self.add("fail", "не найден в PATH", env.ffmpeg_install_hint())
            return
        self.ffmpeg = env.Tool("ffmpeg", chosen.path, env.tool_version(chosen.path), chosen.libass)
        self.add("ok", f"{self.ffmpeg.version} — {self.ffmpeg.path}")
        earlier = [c.path for c in candidates[: candidates.index(chosen)] if c.libass is False]
        if earlier:
            self.add(
                "info",
                "в PATH раньше стоит сборка без libass, она пропущена: " + "; ".join(earlier),
                "Уберите её папку из PATH (часто она в системном PATH — он идёт раньше пользовательского). "
                "Порядок показывает команда: where.exe ffmpeg",
                name="ffmpeg без libass",
            )

    def ffprobe_tool(self) -> None:
        tool = env.find_ffprobe(self.ffmpeg)
        if tool is None:
            self.add("fail", "не найден в PATH (входит в комплект ffmpeg)", env.ffmpeg_install_hint())
        else:
            self.add("ok", f"{tool.version} — {tool.path}")

    def libass(self) -> None:
        if self.ffmpeg is None:
            self.add("info", "пропущено — нет ffmpeg")
            return
        has_libass = self.ffmpeg.libass
        if has_libass is None:  # -buildconf не сработал — смотрим список фильтров
            has_libass = {"ass", "subtitles"} <= list_filters(self.ffmpeg.path)
        if has_libass:
            self.add("ok", "фильтр subtitles есть — субтитры можно вшивать")
        else:
            self.add(
                "fail",
                "ни одна сборка ffmpeg в PATH не собрана с libass — субтитры не получится вшить",
                "Поставьте полную сборку ffmpeg (например, winget install --id Gyan.FFmpeg -e).",
            )

    def encoders(self) -> None:
        if self.ffmpeg is None:
            self.add("info", "пропущено — нет ffmpeg")
            return
        available = list_encoders(self.ffmpeg.path)
        x264 = "libx264" in available
        nvenc_ok, nvenc_error = False, ""
        if "h264_nvenc" in available:
            nvenc_ok, nvenc_error = encoder_works(self.ffmpeg.path, "h264_nvenc")

        if nvenc_ok:
            self.add("ok", "NVENC работает — рендер на видеокарте", name="NVENC (видеокарта)")
        elif "h264_nvenc" in available:
            self.add(
                "warn",
                f"есть в сборке, но не запускается: {nvenc_error}",
                "Обновите драйвер NVIDIA. Без NVENC рендер пойдёт через libx264 — медленнее, но работает.",
                name="NVENC (видеокарта)",
            )
        else:
            self.add("info", "в этой сборке ffmpeg нет NVENC — будет libx264", name="NVENC (видеокарта)")

        if x264:
            self.add("ok", "libx264 есть", name="libx264 (процессор)")
        elif nvenc_ok:
            self.add("warn", "нет libx264 — рендер возможен только через NVENC", name="libx264 (процессор)")
        else:
            self.add(
                "fail",
                "нет ни libx264, ни рабочего NVENC — кодировать H.264 нечем",
                "Поставьте полную сборку ffmpeg (например, winget install --id Gyan.FFmpeg -e).",
                name="libx264 (процессор)",
            )

    # --- YouTube ---

    def ytdlp(self) -> None:
        version = env.ytdlp_version()
        if version is None:
            self.add("fail", "не установлен", f"Установите: {env.YTDLP_UPDATE_COMMAND}")
            return
        age = env.ytdlp_age_days(version)
        if age is not None and age > env.YTDLP_MAX_AGE_DAYS:
            self.add(
                "warn",
                f"{version} — версии {age} дн.; YouTube часто меняется",
                f"Обновите: {env.YTDLP_UPDATE_COMMAND}",
            )
        else:
            self.add("ok", version)

    def js_runtime(self) -> None:
        runtime = env.find_js_runtime()
        if runtime is None:
            self.add(
                "warn",
                "не найден — без него YouTube может не отдать видео",
                f"Установите Deno вместе с yt-dlp: {env.YTDLP_UPDATE_COMMAND}",
            )
        else:
            self.add("ok", f"{runtime.name} {runtime.version} — {runtime.path}")

    # --- Распознавание речи ---

    def faster_whisper(self) -> None:
        env.setup_cuda_dlls()  # до импорта ctranslate2
        try:
            faster_whisper = importlib.import_module("faster_whisper")
            ctranslate2 = importlib.import_module("ctranslate2")
        except ImportError as exc:
            self.add("fail", f"не импортируется: {exc}", "Переустановите зависимости: pip install -r requirements.txt")
            return
        self.add("ok", f"{faster_whisper.__version__} (ctranslate2 {ctranslate2.__version__})")

    def cuda_libraries(self) -> None:
        dirs = env.setup_cuda_dlls()
        loaded, missing = env.check_cuda_libraries()
        source = "из пакетов nvidia-*" if dirs else "из системы"
        if not missing:
            self.add("ok", f"cuBLAS и cuDNN загружаются ({source})")
        else:
            self.add(
                "fail",
                "не найдены: " + ", ".join(missing),
                f"Установите: {env.CUDA_PIP_COMMAND}",
            )

    def gpu(self) -> None:
        gpus = env.nvidia_gpus()
        if not gpus:
            self.add(
                "fail",
                "видеокарта NVIDIA не найдена (nvidia-smi недоступен)",
                "Установите или обновите драйвер NVIDIA.",
            )
            return
        gpu = max(gpus, key=lambda g: g.memory_total_mb)
        detail = (
            f"{gpu.name} — {gpu.memory_total_mb / 1024:.0f} ГБ, свободно "
            f"{gpu.memory_free_mb / 1024:.1f} ГБ, драйвер {gpu.driver}"
        )
        if gpu.memory_free_mb < WHISPER_VRAM_MB:
            self.add(
                "warn",
                detail,
                "Для large-v3 нужно ~4,5 ГБ свободной видеопамяти. Закройте игры и тяжёлые программы "
                "на время распознавания или задайте transcribe.compute_type: int8_float16.",
            )
        else:
            self.add("ok", detail)

    def ctranslate2_cuda(self) -> None:
        env.setup_cuda_dlls()
        try:
            ctranslate2 = importlib.import_module("ctranslate2")
        except ImportError:
            self.add("info", "пропущено — ctranslate2 не установлен")
            return
        count = ctranslate2.get_cuda_device_count()
        if count == 0:
            self.add(
                "fail",
                "ctranslate2 не видит видеокарту",
                "Нужны видеокарта NVIDIA и драйвер с поддержкой CUDA 12. Обновите драйвер.",
            )
            return
        compute_types = ctranslate2.get_supported_compute_types("cuda")
        if "float16" in compute_types:
            self.add("ok", f"устройств CUDA: {count}, float16 поддерживается")
        else:
            self.add("warn", f"устройств CUDA: {count}, но float16 не поддерживается: {sorted(compute_types)}")

    def whisper_model(self) -> None:
        from clipper.core.transcribe import MODEL_NAME, model_is_downloaded

        model_dir = None
        try:
            path = find_config_file(self.config_path)
            model_dir = load_config(path).transcribe.model_dir if path else None
        except ClipperError:
            pass  # ошибку конфига покажет отдельная проверка
        if model_is_downloaded(model_dir):
            self.add("ok", f"{MODEL_NAME} скачана")
        else:
            self.add("info", f"{MODEL_NAME} скачается при первом распознавании (~3 ГБ, один раз)")

    # --- Кадр и субтитры ---

    def mediapipe(self) -> None:
        try:
            mediapipe = importlib.import_module("mediapipe")
            importlib.import_module("mediapipe.tasks.python.vision")
        except Exception as exc:  # у mediapipe нативная библиотека: ошибки бывают не только ImportError
            self.add("fail", f"не импортируется: {exc}", "Переустановите: pip install --force-reinstall mediapipe")
            return
        self.add("ok", str(mediapipe.__version__))

    def opencv(self) -> None:
        installed = [name for name in OPENCV_PACKAGES if _dist_version(name)]
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            self.add("fail", f"не импортируется: {exc}", "Установите: pip install opencv-contrib-python")
            return
        detail = f"{cv2.__version__} ({', '.join(installed) or 'пакет не определён'})"
        if len(installed) > 1:
            self.add(
                "warn",
                detail + " — несколько пакетов OpenCV конфликтуют",
                "Оставьте только opencv-contrib-python: pip uninstall -y "
                + " ".join(name for name in installed if name != "opencv-contrib-python")
                + " && pip install --force-reinstall opencv-contrib-python",
            )
        else:
            self.add("ok", detail)

    def face_detector(self) -> None:
        from clipper.core.errors import ClipperError
        from clipper.core.facetrack import make_detector

        try:
            make_detector("yunet")
        except ClipperError as exc:
            self.add("fail", exc.message, exc.hint or "Без него --crop face работает как --crop center.")
            return
        self.add("ok", "YuNet (OpenCV) загружается")

    def pysubs2(self) -> None:
        version = _dist_version("pysubs2")
        if version is None:
            self.add("fail", "не установлен", "Установите: pip install pysubs2")
        else:
            self.add("ok", version)

    # --- Конфиг ---

    def config(self) -> None:
        try:
            path = find_config_file(self.config_path)
            if path is None:
                self.add(
                    "info",
                    "clipper.yaml не найден — используются значения по умолчанию",
                    "Создать файл с комментариями: clipper config --init",
                )
                return
            load_config(path)
        except ClipperError as exc:
            self.add("fail", exc.message, exc.hint)
            return
        self.add("ok", str(path.resolve()))


def _dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_doctor(reporter: Reporter | None = None, config_path: str | Path | None = None) -> list[CheckResult]:
    """Проверить всё, что нужно clipper: программы, пакеты, GPU, конфиг."""
    return _Doctor(config_path).run(reporter or Reporter())
