"""Внешнее окружение: ffmpeg, yt-dlp, JS-движок для YouTube, библиотеки CUDA.

Функции `require_*` вызываются в начале команд, которым нужна соответствующая
программа. Если программы нет, они выбрасывают `DependencyError` с подсказкой,
как её установить. Полная проверка для `clipper doctor` собрана в `doctor.py`.
"""

import ctypes
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from clipper.core.errors import DependencyError

YTDLP_UPDATE_COMMAND = 'pip install -U "yt-dlp[default,deno]"'
YTDLP_MAX_AGE_DAYS = 60
CUDA_PIP_COMMAND = 'pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"'

# Что загружает ctranslate2 4.5+ (движок faster-whisper) при работе на GPU.
CUDA_LIBRARIES = {
    "win32": ("cublas64_12.dll", "cublasLt64_12.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    "linux": ("libcublas.so.12", "libcublasLt.so.12", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"),
}


@dataclass(frozen=True)
class Tool:
    name: str
    path: str
    version: str


@dataclass(frozen=True)
class GpuInfo:
    name: str
    memory_total_mb: int
    memory_used_mb: int
    driver: str

    @property
    def memory_free_mb(self) -> int:
        return self.memory_total_mb - self.memory_used_mb


def run_command(args: Sequence[str], timeout: float = 30) -> subprocess.CompletedProcess[str]:
    """Запустить короткую служебную команду и вернуть её вывод.

    Вывод декодируется как UTF-8 с заменой нечитаемых байтов. Если команду не
    удалось запустить или она не уложилась в `timeout`, выбрасывается OSError.
    """
    try:
        return subprocess.run(
            list(args),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise OSError(f"{Path(args[0]).name} не ответил за {timeout:g} с") from exc


# --- ffmpeg ---------------------------------------------------------------------


def ffmpeg_install_hint() -> str:
    if sys.platform == "win32":
        return (
            "Установите ffmpeg и откройте новый терминал:\n"
            "  winget install --id Gyan.FFmpeg -e\n"
            "Или скачайте сборку с https://www.gyan.dev/ffmpeg/builds/ и добавьте её папку bin в PATH."
        )
    if sys.platform == "darwin":
        return "Установите ffmpeg: brew install ffmpeg"
    return "Установите ffmpeg: sudo apt install ffmpeg (или пакетным менеджером вашего дистрибутива)"


def find_tool(name: str) -> Tool | None:
    """Найти ffmpeg/ffprobe в PATH и узнать версию."""
    path = shutil.which(name)
    if path is None:
        return None
    version = "?"
    try:
        result = run_command([path, "-hide_banner", "-version"], timeout=15)
        match = re.search(r"version\s+(\S+)", result.stdout)
        if match:
            version = match.group(1)
    except OSError:
        pass
    return Tool(name, path, version)


def require_ffmpeg() -> Tool:
    tool = find_tool("ffmpeg")
    if tool is None:
        raise DependencyError("ffmpeg не найден в PATH.", hint=ffmpeg_install_hint())
    return tool


def require_ffprobe() -> Tool:
    tool = find_tool("ffprobe")
    if tool is None:
        raise DependencyError("ffprobe не найден в PATH (он входит в комплект ffmpeg).", hint=ffmpeg_install_hint())
    return tool


# --- yt-dlp и JS-движок -----------------------------------------------------------


def ytdlp_version() -> str | None:
    try:
        from yt_dlp.version import __version__
    except ImportError:
        return None
    return str(__version__)


def require_ytdlp() -> str:
    version = ytdlp_version()
    if version is None:
        raise DependencyError("Python-пакет yt-dlp не установлен.", hint=f"Установите его: {YTDLP_UPDATE_COMMAND}")
    return version


def ytdlp_age_days(version: str, today: date | None = None) -> int | None:
    """Возраст версии yt-dlp в днях (версия — это дата выпуска: 2026.08.19)."""
    match = re.match(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", version)
    if not match:
        return None
    try:
        released = date(*(int(part) for part in match.groups()))
    except ValueError:
        return None
    return ((today or date.today()) - released).days


def find_js_runtime() -> Tool | None:
    """JS-движок для yt-dlp (нужен для YouTube).

    Ищем так же, как yt-dlp: сначала папка скриптов текущего Python (туда ставит
    Deno pip-пакет `deno`), потом PATH. Node.js подходит как запасной вариант
    с версии 22.
    """
    exe_suffix = sysconfig.get_config_var("EXE") or ""
    candidates: list[tuple[str, str]] = []
    scripts = sysconfig.get_path("scripts")
    if scripts:
        candidates.append(("deno", os.path.join(scripts, "deno" + exe_suffix)))
    for name in ("deno", "node"):
        found = shutil.which(name)
        if found:
            candidates.append((name, found))

    for name, path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            result = run_command([path, "--version"], timeout=15)
        except OSError:
            continue
        pattern = r"^deno (\S+)" if name == "deno" else r"^v(\S+)"
        match = re.search(pattern, result.stdout.strip(), re.M)
        if not match:
            continue
        version = match.group(1)
        if name == "node" and _major(version) < 22:
            continue
        return Tool(name, path, version)
    return None


def _major(version: str) -> int:
    match = re.match(r"(\d+)", version)
    return int(match.group(1)) if match else 0


# --- CUDA -------------------------------------------------------------------------

_registered_dirs: list[Path] = []
_dll_handles: list[object] = []  # держим ссылки, чтобы каталоги не выпали из поиска DLL


def nvidia_library_dirs(package_paths: Sequence[str] | None = None) -> list[Path]:
    """Папки с библиотеками из pip-пакетов nvidia-* (cuBLAS, cuDNN, ...).

    `package_paths` — где лежит пакет `nvidia`; по умолчанию ищется в текущем Python.
    """
    if package_paths is None:
        try:
            spec = importlib.util.find_spec("nvidia")
        except (ImportError, ValueError):
            return []
        if spec is None or not spec.submodule_search_locations:
            return []
        package_paths = list(spec.submodule_search_locations)
    subdir = "bin" if sys.platform == "win32" else "lib"
    dirs: list[Path] = []
    for base in package_paths:
        for candidate in sorted(Path(base).glob(f"*/{subdir}")):
            if candidate.is_dir():
                dirs.append(candidate)
    return dirs


def setup_cuda_dlls() -> list[Path]:
    """Сделать DLL из пакетов nvidia-* видимыми для ctranslate2 (только Windows).

    Python на Windows сам не ищет DLL в site-packages/nvidia/*/bin: отсюда
    классическая ошибка «cublas64_12.dll not found». Каталоги добавляются и через
    os.add_dll_directory, и в PATH — ctranslate2 и cuDNN загружают библиотеки
    разными способами. Вызывать до `import faster_whisper`. Повторный вызов безопасен.
    """
    if sys.platform != "win32":
        return []
    new_dirs = [d for d in nvidia_library_dirs() if d not in _registered_dirs]
    for directory in new_dirs:
        try:
            _dll_handles.append(os.add_dll_directory(str(directory)))
        except OSError:
            continue
        _registered_dirs.append(directory)
    added = [str(d) for d in new_dirs if d in _registered_dirs]
    if added:
        os.environ["PATH"] = os.pathsep.join([*added, os.environ.get("PATH", "")])
    return list(_registered_dirs)


def check_cuda_libraries() -> tuple[list[str], list[str]]:
    """Попробовать загрузить cuBLAS и cuDNN. Вернуть (загрузились, не нашлись)."""
    names = CUDA_LIBRARIES.get(sys.platform, CUDA_LIBRARIES["linux"])
    loaded: list[str] = []
    missing: list[str] = []
    for name in names:
        if _try_load_library(name):
            loaded.append(name)
        else:
            missing.append(name)
    return loaded, missing


def _try_load_library(name: str) -> bool:
    attempts: list[dict[str, int]] = [{}]
    if sys.platform == "win32":
        # winmode=0 — стандартный порядок поиска Windows (с PATH), как у ctranslate2;
        # без winmode — каталоги из os.add_dll_directory.
        attempts = [{"winmode": 0}, {}]
    for kwargs in attempts:
        try:
            ctypes.CDLL(name, **kwargs)
            return True
        except OSError:
            continue
    return False


def nvidia_gpus() -> list[GpuInfo]:
    """Список видеокарт NVIDIA через nvidia-smi (ставится вместе с драйвером)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    try:
        result = run_command(
            [exe, "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
            timeout=20,
        )
    except OSError:
        return []
    if result.returncode != 0:
        return []
    gpus = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        name, total, used, driver = parts
        try:
            gpus.append(GpuInfo(name, int(float(total)), int(float(used)), driver))
        except ValueError:
            continue
    return gpus
