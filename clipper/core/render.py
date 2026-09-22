"""Рендер клипов через ffmpeg.

Сейчас (этап 3) клип нарезается как есть: исходный кадр, H.264 + AAC. Вырезание
пауз, субтитры и вертикальный кадр добавятся следующими этапами, поверх этого же
модуля.

Ошибка в одном клипе не останавливает остальные: она попадает в результат,
а в конце интерфейс показывает сводку. Готовый файл сначала пишется во
временный и потом переименовывается — прерванный рендер не оставит битых mp4.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from clipper.core import env
from clipper.core.config import Config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import encoder_works, list_encoders, run_ffmpeg
from clipper.core.models import Clip, Project

LOG_FILENAME = "clipper.log"

# Параметры кодирования: качество примерно одинаковое у обоих кодировщиков.
VIDEO_ARGS = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"],
    "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
}
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]


@dataclass(frozen=True)
class RenderResult:
    clip_id: int
    path: Path | None  # None — клип не получился
    error: str | None = None
    duration: float = 0.0


def clip_filename(clip: Clip) -> str:
    return f"clip_{clip.id:02d}.mp4"


def output_dir(cfg: Config, project: Project) -> Path:
    return Path(cfg.paths.output).resolve() / project.source.id


def choose_encoder(cfg: Config, ffmpeg: str, reporter: Reporter) -> str:
    """nvenc или x264: auto берёт NVENC, если он реально кодирует на этой машине."""
    wanted = cfg.render.encoder
    available = list_encoders(ffmpeg)
    if wanted in ("auto", "nvenc") and "h264_nvenc" in available:
        works, reason = encoder_works(ffmpeg, "h264_nvenc")
        if works:
            return "nvenc"
        if wanted == "nvenc":
            reporter.warning(f"NVENC не запускается ({reason}) — кодирую через libx264.")
    elif wanted == "nvenc":
        reporter.warning("В этой сборке ffmpeg нет NVENC — кодирую через libx264.")
    if "libx264" not in available:
        raise ClipperError(
            "В ffmpeg нет libx264, а NVENC недоступен — кодировать H.264 нечем.",
            hint="Поставьте полную сборку ffmpeg: winget install --id Gyan.FFmpeg -e",
        )
    return "x264"


def render_project(
    project: Project,
    work_dir: Path,
    cfg: Config,
    reporter: Reporter,
    only: set[int] | None = None,
) -> list[RenderResult]:
    """Отрендерить включённые клипы проекта (или только `only`) в output/<id>/."""
    video = Path(project.source.video)
    if not video.is_file():
        raise ClipperError(
            f"Исходное видео не найдено: {video}",
            hint="Файл перемещён или удалён. Запустите clipper analyze заново.",
        )
    clips = [c for c in project.clips if c.enabled and (only is None or c.id in only)]
    if only:
        missing = sorted(only - {c.id for c in project.clips})
        if missing:
            raise ClipperError(
                f"В проекте нет клипов с id {missing}.",
                hint="Номера клипов — поле id в project.json.",
            )
    if not clips:
        raise ClipperError("Нечего рендерить: все клипы выключены (enabled: false).")

    ffmpeg = env.require_ffmpeg()
    encoder = choose_encoder(cfg, ffmpeg.path, reporter)
    out = output_dir(cfg, project)
    out.mkdir(parents=True, exist_ok=True)
    reporter.info(f"Кодировщик: {'NVENC (видеокарта)' if encoder == 'nvenc' else 'libx264 (процессор)'}")

    results = []
    for number, clip in enumerate(clips, start=1):
        title = f"Клип {number}/{len(clips)} (id {clip.id})"
        target = out / clip_filename(clip)
        try:
            with reporter.stage(f"clip_{clip.id}", title, total=clip.duration, unit="seconds") as stage:
                render_clip(clip, video, target, ffmpeg.path, encoder, reporter, stage.update, work_dir)
                stage.result = target.name
            results.append(RenderResult(clip.id, target, duration=clip.duration))
        except ClipperError as exc:
            results.append(RenderResult(clip.id, None, error=exc.message))
    return results


def render_clip(
    clip: Clip,
    video: Path,
    target: Path,
    ffmpeg: str,
    encoder: str,
    reporter: Reporter,
    on_progress,
    work_dir: Path,
) -> None:
    """Вырезать клип [start, end] из исходника и закодировать в H.264 + AAC."""
    tmp = target.with_name(target.stem + ".part.mp4")
    args = [
        # -ss перед -i: быстрый переход к нужному месту; при перекодировании он точный до кадра.
        "-ss", f"{clip.start:.3f}", "-i", str(video), "-t", f"{clip.duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0?",
        *VIDEO_ARGS[encoder], "-pix_fmt", "yuv420p",
        *AUDIO_ARGS,
        "-movflags", "+faststart",
        str(tmp),
    ]  # fmt: skip
    try:
        run_ffmpeg(ffmpeg, args, on_progress=on_progress, cancel=reporter.cancel, log_path=work_dir / LOG_FILENAME)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target)
