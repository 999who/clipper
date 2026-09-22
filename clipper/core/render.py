"""Рендер клипов через ffmpeg.

Клип вырезается из исходника в H.264 + AAC. Если включены audio.cut_pauses или
audio.remove_fillers, из него вырезаются паузы и слова-паразиты (куски склеиваются
через trim/concat). Субтитры и вертикальный кадр добавятся следующими этапами.

Ошибка в одном клипе не останавливает остальные: она попадает в результат,
а в конце интерфейс показывает сводку. Готовый файл сначала пишется во
временный и потом переименовывается — прерванный рендер не оставит битых mp4.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from clipper.core import env
from clipper.core.audio import ClipEdit, detect_silences, pause_hint, plan_edit, quietest_level
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
FADE = 0.01  # с: фейд звука на стыках вырезок


@dataclass(frozen=True)
class RenderResult:
    clip_id: int
    path: Path | None  # None — клип не получился
    error: str | None = None
    duration: float = 0.0  # длина готового клипа
    removed: float = 0.0  # сколько секунд вырезано (паузы и паразиты)
    fillers: int = 0  # сколько слов-паразитов вырезано


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
    levels: dict[int, float] = {}  # клипы, где по громкости пауз не нашлось
    for number, clip in enumerate(clips, start=1):
        title = f"Клип {number}/{len(clips)} (id {clip.id})"
        target = out / clip_filename(clip)
        try:
            edit = clip_edit(clip, video, ffmpeg.path, cfg, project.source.has_audio)
            length = edit.timeline.duration
            if edit.quietest is not None:
                levels[clip.id] = edit.quietest
            with reporter.stage(f"clip_{clip.id}", title, total=length, unit="seconds") as stage:
                render_clip(clip, edit, video, target, ffmpeg.path, encoder, reporter, stage.update, work_dir,
                            has_audio=project.source.has_audio)  # fmt: skip
                stage.result = target.name + _edit_note(edit)
            results.append(
                RenderResult(clip.id, target, duration=length, removed=edit.timeline.removed, fillers=len(edit.fillers))
            )
            if not edit.timeline.is_whole and length < cfg.select.min_len:
                reporter.warning(
                    f"Клип {clip.id} после вырезок — {length:.0f} с, короче min_len ({cfg.select.min_len:g} с). "
                    f"Расширьте его в project.json или ослабьте вырезание (--min-pause, --set audio.max_gap=…)."
                )
        except ClipperError as exc:
            results.append(RenderResult(clip.id, None, error=exc.message))
    if levels:
        reporter.warning(pause_hint(levels, cfg.audio.silence_db))
    return results


def clip_edit(clip: Clip, video: Path, ffmpeg: str, cfg: Config, has_audio: bool) -> ClipEdit:
    """План вырезок клипа по audio.*: паузы (по громкости или по словам) и паразиты."""
    audio = cfg.audio
    silences = None
    if audio.cut_pauses and audio.pause_detect == "volume" and has_audio:
        silences = detect_silences(ffmpeg, video, clip.start, clip.end, audio.silence_db, audio.min_pause)
    edit = plan_edit(clip, cfg, silences)
    if silences is not None and not edit.pauses:
        edit.quietest = quietest_level(ffmpeg, video, clip.start, clip.end, audio.min_pause)
    return edit


def _edit_note(edit: ClipEdit) -> str:
    parts = []
    if edit.pauses:
        parts.append(f"паузы −{edit.paused_seconds:.1f} с")
    if edit.fillers:
        parts.append(f"паразиты: {len(edit.fillers)}")
    return "".join(f", {part}" for part in parts)


def render_clip(
    clip: Clip,
    edit: ClipEdit,
    video: Path,
    target: Path,
    ffmpeg: str,
    encoder: str,
    reporter: Reporter,
    on_progress,
    work_dir: Path,
    *,
    has_audio: bool = True,
) -> None:
    """Вырезать клип из исходника (с вырезками по edit) и закодировать в H.264 + AAC."""
    tmp = target.with_name(target.stem + ".part.mp4")
    # -ss перед -i: быстрый переход к нужному месту; при перекодировании он точный до кадра.
    args = ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(video)]
    if edit.timeline.is_whole:
        args += ["-map", "0:v:0", "-map", "0:a:0?"]
    else:
        graph = cut_filter(edit.timeline.relative_pieces(), has_audio)
        args += ["-filter_complex", graph, "-map", "[v]"] + (["-map", "[a]"] if has_audio else [])
    args += [*VIDEO_ARGS[encoder], "-pix_fmt", "yuv420p", *AUDIO_ARGS, "-movflags", "+faststart", str(tmp)]
    try:
        run_ffmpeg(ffmpeg, args, on_progress=on_progress, cancel=reporter.cancel, log_path=work_dir / LOG_FILENAME)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, target)


def cut_filter(pieces: list[tuple[float, float]], audio: bool) -> str:
    """filter_complex: оставить куски [a, b] (время от начала клипа) и склеить их.

    На стыках звука — короткие фейды, иначе слышны щелчки.
    """
    count = len(pieces)
    parts: list[str] = []
    video_in = [f"[vi{i}]" for i in range(count)] if count > 1 else ["[0:v:0]"]
    audio_in = [f"[ai{i}]" for i in range(count)] if count > 1 else ["[0:a:0]"]
    if count > 1:
        parts.append("[0:v:0]split=" + str(count) + "".join(video_in))
        if audio:
            parts.append("[0:a:0]asplit=" + str(count) + "".join(audio_in))
    outputs = []
    for i, (start, end) in enumerate(pieces):
        parts.append(f"{video_in[i]}trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{i}]")
        outputs.append(f"[v{i}]")
        if audio:
            length = end - start
            fade = round(min(FADE, length / 4), 3)
            parts.append(
                f"{audio_in[i]}atrim=start={start:.3f}:end={end:.3f},asetpts=PTS-STARTPTS,"
                f"afade=t=in:d={fade},afade=t=out:st={max(length - fade, 0):.3f}:d={fade}[a{i}]"
            )
            outputs.append(f"[a{i}]")
    parts.append("".join(outputs) + f"concat=n={count}:v=1:a={1 if audio else 0}[v]" + ("[a]" if audio else ""))
    return ";".join(parts)
