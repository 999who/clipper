"""Этап 1: получить исходное видео.

- Ссылка → yt-dlp скачивает видео (H.264 до `download.max_height` + AAC) и его
  метаданные в work/<id>/. Из метаданных берётся heatmap — график «Самые
  популярные фрагменты» YouTube.
- Локальный файл → параметры читаются через ffprobe, файл не копируется,
  heatmap у него нет.

Итог — `SourceInfo`, он же сохраняется в work/<id>/source.json. Повторный запуск
с той же ссылкой берёт готовое из рабочей папки (флаг --force — скачать заново).
"""

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from clipper.core import env
from clipper.core.config import Config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter, Stage
from clipper.core.ffmpeg import probe
from clipper.core.models import INFO_FILENAME, SourceInfo, load_source, parse_heatmap, save_source, write_json_atomic

log = logging.getLogger(__name__)

SOURCE_BASENAME = "source"  # work/<id>/source.mp4
_VIDEO_EXTENSIONS = (".mp4", ".mkv", ".webm", ".mov", ".m4v")
_YOUTUBE_HOST = re.compile(r"^https?://(?:[\w-]+\.)*(?:youtube\.com|youtu\.be|youtube-nocookie\.com)/", re.I)
_YOUTUBE_ID = re.compile(r"(?:[?&]v=|youtu\.be/|/shorts/|/live/|/embed/|/v/)([A-Za-z0-9_-]{11})(?![A-Za-z0-9_-])")
_BARE_YOUTUBE = re.compile(r"^(?:www\.|m\.)?(?:youtube\.com|youtu\.be)/", re.I)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def prepare_source(source: str, cfg: Config, reporter: Reporter, *, force: bool = False) -> SourceInfo:
    """Ссылка или путь к файлу → SourceInfo (с видео в рабочей папке или на месте)."""
    source = source.strip().strip('"').strip("'")
    if _BARE_YOUTUBE.match(source) and not Path(source).exists():
        source = "https://" + source  # «youtube.com/watch?v=…» без https://
    if is_url(source):
        return download_url(source, cfg, reporter, force=force)
    return use_local_file(source, cfg, reporter)


def is_url(source: str) -> bool:
    return re.match(r"^https?://", source, re.I) is not None


def youtube_id(url: str) -> str | None:
    """ID ролика из ссылки YouTube (watch?v=, youtu.be/, shorts/, live/, embed/)."""
    if not _YOUTUBE_HOST.match(url):
        return None
    match = _YOUTUBE_ID.search(url)
    return match.group(1) if match else None


# --- Ссылка -------------------------------------------------------------------------


def download_url(url: str, cfg: Config, reporter: Reporter, *, force: bool = False) -> SourceInfo:
    env.require_ytdlp()
    ffmpeg = env.require_ffmpeg()
    ffprobe = env.require_ffprobe()
    root = Path(cfg.paths.workdir).resolve()

    # Для YouTube ID виден прямо в ссылке: готовое видео находится без обращения к сети.
    known_id = youtube_id(url)
    if known_id and not force:
        cached = _cached_source(root / known_id)
        if cached is not None:
            reporter.info(f"Видео уже скачано — беру из {root / known_id}")
            return cached

    import yt_dlp  # импорт здесь: yt-dlp нужен только для ссылок

    options = ytdlp_options(cfg, root, ffmpeg.path, reporter, force=force)
    try:
        with yt_dlp.YoutubeDL(options) as ydl:  # type: ignore[arg-type]
            with reporter.stage("metadata", "Данные о видео") as stage:
                info = ydl.extract_info(url, download=False)
                _check_info(info, url)
                stage.result = str(info.get("title") or "")
            work_dir = Path(ydl.prepare_filename(info)).parent
            if not force and (cached := _cached_source(work_dir)) is not None:
                reporter.info(f"Видео уже скачано — беру из {work_dir}")
                return cached

            with reporter.stage("download", "Загрузка видео", total=_expected_size(info), unit="bytes") as stage:
                progress = DownloadProgress(stage, info)
                ydl.add_progress_hook(progress.on_progress)
                ydl.add_postprocessor_hook(progress.on_postprocess)
                info = ydl.process_ie_result(info, download=True)
                stage.result = f"{progress.done / 1_000_000:.1f} МБ" if progress.done else ""
            write_json_atomic(work_dir / INFO_FILENAME, ydl.sanitize_info(info))
    except yt_dlp.utils.YoutubeDLError as exc:
        raise download_error(str(exc)) from None

    video = _downloaded_file(info, work_dir)
    with reporter.stage("probe", "Чтение параметров видео"):
        media = probe(str(video), ffprobe.path)
    heatmap = parse_heatmap(info.get("heatmap"))
    extractor = str(info.get("extractor_key") or info.get("extractor") or "").lower()
    source = SourceInfo(
        id=work_dir.name,
        kind="youtube" if extractor.startswith("youtube") else "url",
        input=url,
        video=str(video),
        title=str(info.get("title") or work_dir.name),
        duration=media.duration,
        width=media.width,
        height=media.height,
        fps=media.fps,
        has_audio=media.has_audio,
        heatmap=heatmap,
    )
    save_source(work_dir, source)
    return source


def ytdlp_options(cfg: Config, root: Path, ffmpeg_path: str, reporter: Reporter, *, force: bool) -> dict[str, Any]:
    """Параметры YoutubeDL: формат, куда сохранять, наш ffmpeg, JS-движок, cookies."""
    height = cfg.download.max_height
    options: dict[str, Any] = {
        # H.264 + AAC быстро декодируются и читаются OpenCV; если их нет — лучшее до max_height.
        "format": (
            f"bv*[vcodec^=avc1][height<={height}]+ba[ext=m4a]"
            f"/bv*[vcodec^=avc1][height<={height}]+ba"
            f"/bv*[height<={height}]+ba"
            f"/b[height<={height}]/b"
        ),
        "merge_output_format": "mp4",
        # «%» в пути к папке yt-dlp принял бы за шаблон — экранируем.
        "outtmpl": {"default": os.path.join(str(root).replace("%", "%%"), "%(id)s", f"{SOURCE_BASENAME}.%(ext)s")},
        "noplaylist": True,
        "overwrites": force,
        "continuedl": True,
        "ffmpeg_location": ffmpeg_path,  # тот ffmpeg, что выбрал clipper (с libass)
        "quiet": True,
        "no_warnings": False,
        "noprogress": True,
        "logger": _YtdlpLogger(reporter),
    }
    runtime = env.find_js_runtime()
    if runtime is not None:
        options["js_runtimes"] = {runtime.name: {"path": runtime.path}}
    else:
        reporter.warning(
            "JS-движок для YouTube не найден — часть форматов может быть недоступна. "
            f"Установите: {env.YTDLP_UPDATE_COMMAND}"
        )
    if cfg.download.cookies_from_browser:
        options["cookiesfrombrowser"] = (cfg.download.cookies_from_browser,)
    return options


class DownloadProgress:
    """Сводит прогресс yt-dlp по нескольким файлам (видео и звук) в один этап."""

    def __init__(self, stage: Stage, info: Mapping[str, Any]) -> None:
        self._stage = stage
        self._files: dict[str, tuple[float, float | None]] = {}  # файл → (скачано, всего)
        self._expected = _expected_size(info)

    @property
    def done(self) -> float:
        return sum(done for done, _ in self._files.values())

    @property
    def total(self) -> float | None:
        known = sum(total or done for done, total in self._files.values())
        if self._expected:
            return max(self._expected, known)
        return known or None

    def on_progress(self, data: Mapping[str, Any]) -> None:
        filename = str(data.get("filename") or data.get("tmpfilename") or "?")
        total = data.get("total_bytes") or data.get("total_bytes_estimate")
        if data.get("status") == "finished":
            size = float(total or data.get("downloaded_bytes") or 0)
            self._files[filename] = (size, size)
        elif data.get("status") == "downloading":
            self._files[filename] = (float(data.get("downloaded_bytes") or 0), float(total) if total else None)
        else:
            return
        self._stage.update(self.done, self.total, message=_progress_message(data))

    def on_postprocess(self, data: Mapping[str, Any]) -> None:
        if data.get("status") == "started" and data.get("postprocessor") == "Merger":
            self._stage.update(self.done, self.total, message="склейка видео и звука")


def download_error(message: str) -> ClipperError:
    """Ошибка yt-dlp → понятная ошибка с подсказкой."""
    text = _ANSI.sub("", message).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    lower = text.lower()
    cookies = (
        "Укажите браузер, где вы вошли в YouTube: --set download.cookies_from_browser=firefox "
        "(из Chrome на Windows cookies не читаются)."
    )
    if "sign in to confirm" in lower or "cookies" in lower:
        hint = "YouTube просит войти в аккаунт. " + cookies
    elif "private video" in lower:
        hint = "Видео приватное. Если у вашего аккаунта есть к нему доступ — " + cookies[0].lower() + cookies[1:]
    elif "video unavailable" in lower or "this video is not available" in lower:
        hint = "Видео недоступно: проверьте ссылку и откройте её в браузере."
    elif "http error 404" in lower or "http error 410" in lower:
        hint = "По ссылке ничего нет: проверьте, что она открывается в браузере."
    elif "requested format is not available" in lower:
        hint = f"Обновите yt-dlp и проверьте JS-движок (clipper doctor): {env.YTDLP_UPDATE_COMMAND}"
    elif any(word in lower for word in ("unable to download", "timed out", "connection", "getaddrinfo", "urlopen")):
        hint = f"Проверьте подключение к интернету. Если сеть в порядке — обновите yt-dlp: {env.YTDLP_UPDATE_COMMAND}"
    else:
        hint = f"YouTube часто меняется — обновите yt-dlp: {env.YTDLP_UPDATE_COMMAND}"
    return ClipperError(f"Не удалось скачать видео: {text}", hint=hint)


def _check_info(info: Mapping[str, Any] | None, url: str) -> None:
    if not info:
        raise ClipperError(f"yt-dlp не вернул данных о видео: {url}")
    if info.get("_type") in ("playlist", "multi_video"):
        raise ClipperError(
            "Это ссылка на плейлист или канал, а нужна ссылка на одно видео.",
            hint="Откройте нужное видео и скопируйте его ссылку (watch?v=…).",
        )
    if info.get("is_live"):
        raise ClipperError(
            "Идёт прямой эфир — его нельзя скачать целиком.",
            hint="Дождитесь окончания трансляции: запись появится по той же ссылке.",
        )


def _expected_size(info: Mapping[str, Any]) -> float | None:
    formats = info.get("requested_formats") or [info]
    sizes = [f.get("filesize") or f.get("filesize_approx") for f in formats]
    return float(sum(sizes)) if sizes and all(sizes) else None


def _progress_message(data: Mapping[str, Any]) -> str:
    fmt = data.get("info_dict") or {}
    if fmt.get("vcodec") == "none":
        part = "звук"
    elif fmt.get("acodec") == "none":
        part = "видеодорожка"
    else:
        part = "видео"
    speed = data.get("speed")
    return f"{part}, {speed / 1_000_000:.1f} МБ/с" if speed else part


def _downloaded_file(info: Mapping[str, Any], work_dir: Path) -> Path:
    for item in info.get("requested_downloads") or []:
        path = item.get("filepath")
        if path and Path(path).is_file():
            return Path(path).resolve()
    found = _find_video(work_dir)
    if found is None:
        raise ClipperError(f"yt-dlp закончил работу, но видеофайла в {work_dir} нет.", hint="Запустите с --force.")
    return found


def _find_video(work_dir: Path) -> Path | None:
    for ext in _VIDEO_EXTENSIONS:
        path = work_dir / f"{SOURCE_BASENAME}{ext}"
        if path.is_file():
            return path.resolve()
    return None


def _cached_source(work_dir: Path) -> SourceInfo | None:
    source = load_source(work_dir)
    if source is None or source.kind == "file" or not Path(source.video).is_file():
        return None
    return source


class _YtdlpLogger:
    """Сообщения yt-dlp: предупреждения — пользователю, остальное — в отладочный лог."""

    def __init__(self, reporter: Reporter) -> None:
        self._reporter = reporter
        self._seen: set[str] = set()

    def debug(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def info(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)

    def warning(self, message: str) -> None:
        text = re.sub(r"^WARNING:\s*", "", _ANSI.sub("", message)).strip()
        if text and text not in self._seen:
            self._seen.add(text)
            self._reporter.warning(f"yt-dlp: {text}")

    def error(self, message: str) -> None:
        log.debug("yt-dlp: %s", message)  # ошибка придёт исключением DownloadError


# --- Локальный файл ------------------------------------------------------------------


def use_local_file(path_text: str, cfg: Config, reporter: Reporter) -> SourceInfo:
    path = Path(path_text).expanduser()
    if not path.is_file():
        hint = "Проверьте путь. Если в нём есть пробелы, возьмите его в кавычки."
        if path.is_dir():
            hint = "Это папка, а нужен видеофайл."
        raise ClipperError(f"Файл не найден: {path}", hint=hint)
    path = path.resolve()
    ffprobe = env.require_ffprobe()
    with reporter.stage("probe", "Чтение параметров видео") as stage:
        media = probe(str(path), ffprobe.path)
        work_dir = work_dir_for_file(Path(cfg.paths.workdir).resolve(), path)
        source = SourceInfo(
            id=work_dir.name,
            kind="file",
            input=path_text,
            video=str(path),
            title=path.stem,
            duration=media.duration,
            width=media.width,
            height=media.height,
            fps=media.fps,
            has_audio=media.has_audio,
            heatmap=None,
        )
        save_source(work_dir, source)
        stage.result = f"{media.width}×{media.height}"
    if not media.has_audio:
        reporter.warning("В файле нет звуковой дорожки — распознавать будет нечего.")
    return source


def local_id(path: Path) -> str:
    """Имя рабочей папки для локального файла: имя файла без расширения и спецсимволов."""
    name = re.sub(r"[^\w.-]+", "_", path.stem).strip("._")[:60]
    return name or "video"


def work_dir_for_file(root: Path, path: Path) -> Path:
    """work/<имя файла>; если такое имя уже занято другим файлом — work/<имя>-2, -3, …"""
    base = local_id(path)
    for index in range(1, 1000):
        work_dir = root / (base if index == 1 else f"{base}-{index}")
        existing = load_source(work_dir)
        if existing is None and not (work_dir.exists() and any(work_dir.iterdir())):
            return work_dir
        if existing is not None and _same_file(existing.video, path):
            return work_dir
    raise ClipperError(f"Слишком много рабочих папок с именем {base} в {root}")


def _same_file(a: str, b: Path) -> bool:
    return os.path.normcase(os.path.realpath(a)) == os.path.normcase(os.path.realpath(b))
