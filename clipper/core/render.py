"""Рендер клипов через ffmpeg.

Один запуск ffmpeg на клип, граф фильтров:

    вход → [вырезки: trim/concat] → кадр 1080×1920 (reframe) → субтитры → H.264 + AAC

- Вырезки пауз и паразитов (этап 4) — куски склеиваются через trim/concat.
- Кадр (этап 6): окно 9:16/1:1 за лицом или по центру, фон для 1:1 и original,
  режим stream — вебка сверху, игра снизу. Траектория окна — файл sendcmd.
- Субтитры (этап 5) накладываются на готовый кадр 1080×1920.

.ass, файл sendcmd и шрифты лежат в work/<id>/tmp/; ffmpeg запускается из этой
папки и получает относительные пути — на Windows так не нужно экранировать «C:».

Ошибка в одном клипе не останавливает остальные: она попадает в результат,
а в конце интерфейс показывает сводку. Готовый файл сначала пишется во
временный и потом переименовывается — прерванный рендер не оставит битых mp4.
"""

import contextlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipper.core import env
from clipper.core.audio import ClipEdit, detect_silences, pause_hint, plan_edit, quietest_level
from clipper.core.config import Config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import encoder_works, list_encoders, list_filters, run_ffmpeg
from clipper.core.models import Clip, Project
from clipper.core.reframe import (
    OUT_H,
    OUT_W,
    StreamPlan,
    VideoPlan,
    layout_for,
    stream_graph,
    stream_plan,
    video_graph,
    video_plan,
)
from clipper.core.subtitles import Style, build_ass, font_files, load_style, prepare_fonts, write_ass

LOG_FILENAME = "clipper.log"
TMP_DIRNAME = "tmp"

# Параметры кодирования: качество примерно одинаковое у обоих кодировщиков.
VIDEO_ARGS = {
    "nvenc": ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr", "-cq", "21", "-b:v", "0"],
    "x264": ["-c:v", "libx264", "-preset", "medium", "-crf", "20"],
}
AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
FADE = 0.01  # с: фейд звука на стыках вырезок
REPLACE_ATTEMPTS = 5  # Windows: файл может ненадолго держать антивирус или проводник
REPLACE_DELAY = 0.4  # с между попытками


@dataclass(frozen=True)
class RenderResult:
    clip_id: int
    path: Path | None  # None — клип не получился
    error: str | None = None
    duration: float = 0.0  # длина готового клипа
    removed: float = 0.0  # сколько секунд вырезано (паузы и паразиты)
    fillers: int = 0  # сколько слов-паразитов вырезано
    subtitles: bool = False  # наложены ли субтитры
    frame: str = ""  # как поставлено окно: «лицо в 85 % кадров», «по центру»


@dataclass(frozen=True)
class SubtitleSetup:
    style: Style
    folder: Path  # здесь .ass и fonts/; ffmpeg запускается отсюда
    max_words: int | None


@dataclass
class FrameSetup:
    """Как строится кадр 1080×1920: план video или stream и детектор лиц (создаётся при первой нужде)."""

    video: VideoPlan | None
    stream: StreamPlan | None
    detector_kind: str
    folder: Path
    detector: Any = None
    detector_error: str | None = None
    notes: dict[int, str] = field(default_factory=dict)


def clip_filename(clip: Clip) -> str:
    return f"clip_{clip.id:02d}.mp4"


def output_dir(cfg: Config, project: Project) -> Path:
    """Папка клипов: output из project.json (если выбрана) или paths.output, плюс id видео."""
    return Path(project.output or cfg.paths.output).expanduser().resolve() / project.source.id


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
    video = Path(project.source.video).resolve()
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
    frame = frame_setup(cfg, project, work_dir)
    encoder = choose_encoder(cfg, ffmpeg.path, reporter)
    out = output_dir(cfg, project)
    out.mkdir(parents=True, exist_ok=True)
    reporter.info(f"Кодировщик: {'NVENC (видеокарта)' if encoder == 'nvenc' else 'libx264 (процессор)'}")
    reporter.info(f"Кадр: {frame_description(cfg)}")
    subs = subtitle_setup(cfg, ffmpeg.path, project, work_dir, reporter)

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
            commands = clip_commands(clip, edit, project, frame, video, ffmpeg.path, reporter, title)
            ass = clip_subtitles(clip, edit, subs, length)
            graph = frame_graph(frame, commands, ass)
            with reporter.stage(f"clip_{clip.id}", title, total=length, unit="seconds") as stage:
                saved = render_clip(clip, edit, video, target, ffmpeg.path, encoder, reporter, stage.update, work_dir,
                                    has_audio=project.source.has_audio, graph=graph, cwd=frame.folder)  # fmt: skip
                stage.result = saved.name + _edit_note(edit) + (", без субтитров: нет слов" if subs and not ass else "")
            if saved != target:
                reporter.warning(
                    f"{target.name} открыт в другой программе (плеер?) и не заменён — новый клип сохранён как "
                    f"{saved.name}. Закройте файл, чтобы в следующий раз он перезаписался."
                )
            results.append(
                RenderResult(clip.id, saved, duration=length, removed=edit.timeline.removed, fillers=len(edit.fillers),
                             subtitles=ass is not None, frame=frame.notes.get(clip.id, ""))
            )  # fmt: skip
            if not edit.timeline.is_whole and length < cfg.select.min_len:
                reporter.warning(
                    f"Клип {clip.id} после вырезок — {length:.0f} с, короче min_len ({cfg.select.min_len:g} с). "
                    f"Расширьте его в project.json или ослабьте вырезание (--min-pause, --set audio.max_gap=…)."
                )
        except ClipperError as exc:
            results.append(RenderResult(clip.id, None, error=exc.message))
        except OSError as exc:  # диск, права, занятый файл — это ошибка одного клипа, не всего рендера
            results.append(RenderResult(clip.id, None, error=f"{exc.strerror or exc}: {exc.filename or target}"))
    if levels:
        reporter.warning(pause_hint(levels, cfg.audio.silence_db))
    if frame.detector_error:
        reporter.warning(f"Слежение за лицом не работает — кроп по центру. {frame.detector_error}")
    return results


# --- кадр ------------------------------------------------------------------------------


def frame_description(cfg: Config) -> str:
    reframe = cfg.reframe
    if reframe.mode == "stream":
        return f"1080×1920, стрим — пресет {reframe.layout}"
    parts = [f"1080×1920, {reframe.aspect}"]
    if reframe.aspect != "original":
        parts.append("за лицом" if reframe.crop == "face" else "по центру")
    if reframe.aspect != "9:16":
        parts.append("фон размытый" if reframe.background == "blur" else "фон чёрный")
    return ", ".join(parts)


def frame_setup(cfg: Config, project: Project, work_dir: Path) -> FrameSetup:
    width, height = project.source.width, project.source.height
    if width <= 0 or height <= 0:
        raise ClipperError(
            "Неизвестен размер кадра исходника — не из чего строить вертикальный кадр.",
            hint="Скачайте или откройте видео заново: clipper download ССЫЛКА_ИЛИ_ФАЙЛ --force",
        )
    folder = Path(work_dir).resolve() / TMP_DIRNAME
    folder.mkdir(parents=True, exist_ok=True)
    if cfg.reframe.mode == "stream":
        _, layout = layout_for(cfg)
        return FrameSetup(None, stream_plan(layout, width, height), cfg.reframe.detector, folder)
    return FrameSetup(video_plan(cfg, width, height), None, cfg.reframe.detector, folder)


def clip_commands(
    clip: Clip,
    edit: ClipEdit,
    project: Project,
    frame: FrameSetup,
    video: Path,
    ffmpeg: str,
    reporter: Reporter,
    title: str,
) -> Path | None:
    """Найти лицо в клипе и записать траекторию окна (файл sendcmd). None — окно по центру."""
    from clipper.core import facetrack

    plan = frame.video
    if plan is None or not plan.track or frame.detector_error:
        return None
    cache = frame.folder / f"faces_{clip.id:02d}.json"
    key = facetrack.cache_key(video, clip.start, clip.end, frame.detector_kind)
    samples = facetrack.load_samples(cache, key)
    if samples is None:
        if frame.detector is None:
            try:
                frame.detector = facetrack.make_detector(frame.detector_kind)
            except ClipperError as exc:
                frame.detector_error = exc.message + (f" {exc.hint}" if exc.hint else "")
                return None
        with reporter.stage(f"face_{clip.id}", f"{title}: поиск лица", total=clip.duration, unit="seconds") as stage:
            samples = facetrack.analyze_faces(
                ffmpeg, video, clip.start, clip.end, plan.src_w, plan.src_h, frame.detector,
                on_progress=stage.update, cancel=reporter.cancel,
            )  # fmt: skip
            stage.result = f"кадров: {len(samples)}"
        facetrack.save_samples(cache, key, samples)
    path, coverage = facetrack.camera_path(samples, plan)
    if coverage < facetrack.MIN_COVERAGE:
        frame.notes[clip.id] = "лицо не найдено — по центру"
        return None
    frame.notes[clip.id] = f"лицо в {coverage:.0%} кадров"
    commands = frame.folder / f"clip_{clip.id:02d}.cmd"
    text = facetrack.crop_commands(samples, path, plan, edit.timeline, project.source.fps)
    commands.write_text(text, encoding="utf-8")
    return commands


def frame_graph(frame: FrameSetup, commands: Path | None, ass: Path | None) -> Callable[[str], str]:
    """Фрагмент графа: вход (метка) → кадр 1080×1920 → субтитры → [v]."""

    def build(inp: str) -> str:
        framed = "[framed]" if ass else "[v]"
        if frame.stream is not None:
            graph = stream_graph(frame.stream, inp, framed)
        else:
            assert frame.video is not None
            graph = video_graph(frame.video, inp, framed, commands.name if commands else None)
        if ass:
            graph += f";[framed]subtitles=filename={ass.name}:fontsdir=fonts[v]"
        return graph

    return build


def subtitle_setup(
    cfg: Config, ffmpeg: str, project: Project, work_dir: Path, reporter: Reporter
) -> SubtitleSetup | None:
    """Стиль и шрифты для субтитров; None — субтитры выключены или их не наложить."""
    if not cfg.subtitles.enabled:
        return None
    style, style_path = load_style(cfg.subtitles.style)  # ошибка в стиле останавливает рендер
    if "subtitles" not in list_filters(ffmpeg):
        reporter.warning(
            "В этой сборке ffmpeg нет libass — клипы будут без субтитров. "
            "Поставьте сборку с libass: winget install --id Gyan.FFmpeg -e (или --no-subs, чтобы не видеть это)."
        )
        return None
    folder = Path(work_dir).resolve() / TMP_DIRNAME
    prepare_fonts(folder / "fonts", font_files(style, style_path))
    return SubtitleSetup(style, folder, cfg.subtitles.max_words)


def clip_subtitles(clip: Clip, edit: ClipEdit, subs: SubtitleSetup | None, length: float) -> Path | None:
    """Записать .ass клипа (время — после вырезок). None — субтитров не будет."""
    if subs is None:
        return None
    words = edit.timeline.map_words(clip.words)
    if not words:
        return None
    ass = build_ass(words, OUT_W, OUT_H, subs.style, length, subs.max_words)
    return write_ass(subs.folder / f"clip_{clip.id:02d}.ass", ass)


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
    graph: Callable[[str], str],
    cwd: Path,
) -> Path:
    """Вырезать клип из исходника (с вырезками по edit), построить кадр и закодировать в H.264 + AAC.

    `graph(вход)` — фрагмент filter_complex от метки входа до [v] (кадр и субтитры).
    ffmpeg запускается из `cwd`: файлы .ass/.cmd в графе — относительные.
    Возвращает путь готового файла (см. publish).
    """
    tmp = target.with_name(target.stem + ".part.mp4")
    # -ss перед -i: быстрый переход к нужному месту; при перекодировании он точный до кадра.
    args = ["-ss", f"{clip.start:.3f}", "-t", f"{clip.duration:.3f}", "-i", str(video)]
    if not edit.timeline.is_whole:
        full = cut_filter(edit.timeline.relative_pieces(), has_audio, post=graph("[vcat]"))
        args += ["-filter_complex", full, "-map", "[v]"] + (["-map", "[a]"] if has_audio else [])
    else:
        args += ["-filter_complex", graph("[0:v:0]"), "-map", "[v]", "-map", "0:a:0?"]
    args += [*VIDEO_ARGS[encoder], "-pix_fmt", "yuv420p", *AUDIO_ARGS, "-movflags", "+faststart", str(tmp)]
    try:
        run_ffmpeg(
            ffmpeg,
            args,
            on_progress=on_progress,
            cancel=reporter.cancel,
            log_path=work_dir / LOG_FILENAME,
            cwd=cwd,
        )
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return publish(tmp, target)


def publish(tmp: Path, target: Path) -> Path:
    """Переименовать готовый .part.mp4 в target.

    На Windows открытый в плеере файл заменить нельзя. Тогда после нескольких
    попыток клип сохраняется рядом как clip_NN.new.mp4 — готовый рендер не пропадает.
    """
    fallback = target.with_name(target.stem + ".new" + target.suffix)
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, target)
        except PermissionError:
            if attempt + 1 < REPLACE_ATTEMPTS:
                time.sleep(REPLACE_DELAY)
            continue
        with contextlib.suppress(OSError):
            fallback.unlink(missing_ok=True)  # запасной файл от прошлого раза больше не нужен
        return target
    try:
        os.replace(tmp, fallback)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise ClipperError(
            f"Не удалось сохранить {target.name}: файл занят другой программой ({exc.strerror}).",
            hint="Закройте плеер или проводник с этим файлом и запустите рендер ещё раз.",
        ) from None
    return fallback


def cut_filter(pieces: list[tuple[float, float]], audio: bool, post: str | None = None) -> str:
    """filter_complex: оставить куски [a, b] (время от начала клипа) и склеить их.

    На стыках звука — короткие фейды, иначе слышны щелчки. `post` — фрагмент
    графа от метки [vcat] (склеенное видео) до [v].
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
    video_out = "[vcat]" if post else "[v]"
    parts.append("".join(outputs) + f"concat=n={count}:v=1:a={1 if audio else 0}{video_out}" + ("[a]" if audio else ""))
    if post:
        parts.append(post)
    return ";".join(parts)
