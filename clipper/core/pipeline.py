"""Оркестрация этапов: analyze → project.json → render.

Интерфейс (CLI сейчас, TUI позже) вызывает только эти функции. Они ничего не
печатают: ход работы — события через Reporter, итог — возвращаемые данные.
"""

import shutil
from datetime import datetime
from pathlib import Path

from clipper.core.config import Config
from clipper.core.download import prepare_source
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter
from clipper.core.highlights import Candidate, heatmap_candidates, keyword_candidates, snap_to_phrases
from clipper.core.models import (
    PROJECT_FILENAME,
    Clip,
    Project,
    ProjectError,
    SourceInfo,
    Transcript,
    format_time,
    load_project,
    save_project,
)
from clipper.core.render import RenderResult, output_dir, render_project
from clipper.core.transcribe import EngineFactory, transcribe_source

PREVIOUS_PROJECT = "project.prev.json"


def work_dir_for(cfg: Config, source_id: str) -> Path:
    return Path(cfg.paths.workdir).resolve() / source_id


# --- analyze ------------------------------------------------------------------------


def analyze(
    source_input: str,
    cfg: Config,
    reporter: Reporter,
    *,
    force: bool = False,
    engine_factory: EngineFactory | None = None,
) -> tuple[Project, Path]:
    """Видео → моменты → распознавание → project.json. Возвращает проект и путь к файлу."""
    source = prepare_source(source_input, cfg, reporter)
    work_dir = work_dir_for(cfg, source.id)
    select = cfg.select

    if select.mode == "heatmap":
        if not source.heatmap:
            raise ClipperError(
                "У этого видео нет heatmap («Самые популярные фрагменты») — выбрать моменты по нему нельзя.",
                hint=f'Выберите моменты по ключевым словам: clipper analyze "{source_input}" '
                '--mode-select keywords --keywords "слово1,слово2"',
            )
        candidates = heatmap_candidates(source.heatmap, source.duration, select.clips, select.min_len, select.max_len)
        if not candidates:
            raise ClipperError(
                "На графике heatmap нет выраженных пиков.",
                hint='Попробуйте режим ключевых слов: --mode-select keywords --keywords "…"',
            )
        margin = cfg.transcribe.margin
        ranges = [(c.start - margin, c.end + margin) for c in candidates]
        transcript = transcribe_source(
            source, work_dir, cfg, reporter, ranges, force=force, engine_factory=engine_factory
        )
    else:
        if not select.keywords:
            raise ClipperError(
                "Для режима keywords нужны ключевые слова.",
                hint='Например: --keywords "победа,жесть,да ладно". * — любое окончание: побед*',
            )
        transcript = transcribe_source(
            source, work_dir, cfg, reporter, None, force=force, engine_factory=engine_factory
        )
        candidates = keyword_candidates(
            transcript.words, select.keywords, source.duration, select.clips, select.min_len, select.max_len
        )
        if not candidates:
            raise ClipperError(
                "Ни одно из ключевых слов не прозвучало: " + ", ".join(f"«{k}»" for k in select.keywords),
                hint="Проверьте написание. Для разных окончаний используйте *, например «побед*». "
                f"Весь распознанный текст — в {work_dir / 'transcript.srt'}",
            )

    with reporter.stage("select", "Подгонка границ к фразам", total=len(candidates)) as stage:
        clips = build_clips(candidates, transcript, source, cfg, reporter)
        stage.update(len(candidates))
        stage.result = f"клипов: {len(clips)}"
    if len(clips) < select.clips:
        reporter.info(f"Нашлось клипов: {len(clips)} из запрошенных {select.clips}.")

    project = Project(
        source=_without_heatmap(source),
        mode=select.mode,
        keywords=list(select.keywords),
        language=transcript.language,
        created=datetime.now().isoformat(timespec="seconds"),
        clips=clips,
    )
    path = work_dir / PROJECT_FILENAME
    if path.exists():
        shutil.copy2(path, work_dir / PREVIOUS_PROJECT)  # ручные правки не пропадут
        reporter.info(f"Прежний {PROJECT_FILENAME} сохранён как {PREVIOUS_PROJECT}.")
    save_project(work_dir, project)
    return project, path


def build_clips(
    candidates: list[Candidate], transcript: Transcript, source: SourceInfo, cfg: Config, reporter: Reporter
) -> list[Clip]:
    """Кандидаты → клипы: границы по фразам, без пересечений, слова с запасом, id по порядку."""
    select, margin = cfg.select, cfg.transcribe.margin
    words = transcript.words
    snapped = []
    for candidate in candidates:
        start, end = snap_to_phrases(
            candidate.start,
            candidate.end,
            words,
            min_len=select.min_len,
            max_len=select.max_len,
            duration=source.duration,
        )
        snapped.append((start, end, candidate))
    snapped.sort(key=lambda item: item[0])

    kept: list[tuple[float, float, Candidate]] = []
    for item in snapped:
        if kept and item[0] < kept[-1][1]:  # после подгонки клипы пересеклись — оставляем лучший
            loser = item if item[2].score <= kept[-1][2].score else kept[-1]
            reporter.info(f"Клип {format_time(loser[0])} пересёкся с соседним и пропущен.")
            if loser is kept[-1]:
                kept[-1] = item
            continue
        kept.append(item)

    return [
        Clip(
            id=number,
            start=start,
            end=end,
            score=candidate.score,
            reason=candidate.reason,
            words=transcript.words_between(start - margin, end + margin),
        )
        for number, (start, end, candidate) in enumerate(kept, start=1)
    ]


def _without_heatmap(source: SourceInfo) -> SourceInfo:
    data = source.to_dict()
    data["heatmap"] = None
    return SourceInfo.from_dict(data)


# --- проекты ------------------------------------------------------------------------


def find_project(cfg: Config, project: str | None = None) -> Path:
    """Путь к project.json: явный файл, папка, id видео — или самый свежий проект в work/."""
    root = Path(cfg.paths.workdir).resolve()
    if project is None:
        found = sorted(root.glob(f"*/{PROJECT_FILENAME}"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise ClipperError(
                f"В {root} нет ни одного проекта.",
                hint='Сначала выберите моменты: clipper analyze "ССЫЛКА_ИЛИ_ФАЙЛ"',
            )
        return found[0]
    candidates = [Path(project), Path(project) / PROJECT_FILENAME, root / project / PROJECT_FILENAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise ClipperError(
        f"Проект не найден: {project}",
        hint="Укажите путь к project.json, папку work/<id> или просто id видео.",
    )


def open_project(path: Path) -> Project:
    try:
        return load_project(path)
    except ProjectError as exc:
        raise ClipperError(
            f"Ошибка в {path}: {exc}",
            hint=f"Исправьте файл или запустите clipper analyze заново (прежняя версия — {PREVIOUS_PROJECT}).",
        ) from None


# --- render -------------------------------------------------------------------------


def render(
    cfg: Config, reporter: Reporter, project: str | None = None, only: set[int] | None = None
) -> tuple[Project, Path, list[RenderResult]]:
    """project.json → output/<id>/clip_NN.mp4. Возвращает проект, папку вывода и результаты."""
    path = find_project(cfg, project)
    loaded = open_project(path)
    reporter.info(f"Проект: {path}")
    results = render_project(loaded, path.parent, cfg, reporter, only)
    return loaded, output_dir(cfg, loaded), results


def run(
    source_input: str,
    cfg: Config,
    reporter: Reporter,
    *,
    force: bool = False,
    engine_factory: EngineFactory | None = None,
) -> tuple[Project, Path, list[RenderResult]]:
    """analyze + render одной командой."""
    project, path = analyze(source_input, cfg, reporter, force=force, engine_factory=engine_factory)
    results = render_project(project, path.parent, cfg, reporter)
    return project, output_dir(cfg, project), results
