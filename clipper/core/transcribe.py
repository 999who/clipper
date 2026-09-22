"""Этап 2: распознавание речи — faster-whisper large-v3 на видеокарте.

1. Звук один раз извлекается из видео в work/<id>/audio16k.wav (16 кГц, моно).
2. Whisper распознаёт нужные отрезки: всё видео или окна вокруг пиков heatmap.
   Таймкоды каждого слова (`word_timestamps=True`) переводятся во время
   исходного видео.
3. Результат копится в work/<id>/transcript.json. Уже распознанные отрезки
   повторно не распознаются, а новые просьбы дораспознают только недостающее.
   Для проверки рядом пишется transcript.srt.

Модель фиксирована: large-v3, `device="cuda"`. Тип вычислений —
`transcribe.compute_type` (float16 или экономный int8_float16).
"""

import logging
import os
import wave
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Protocol

from clipper.core import env
from clipper.core.config import Config
from clipper.core.errors import ClipperError, DependencyError
from clipper.core.events import Reporter
from clipper.core.ffmpeg import run_ffmpeg
from clipper.core.models import (
    join_hyphenated,
    join_hyphenated_text,
    SRT_FILENAME,
    Segment,
    SourceInfo,
    Transcript,
    Word,
    load_transcript,
    merge_ranges,
    save_transcript,
    subtract_ranges,
)
from clipper.core.text import plural

log = logging.getLogger(__name__)

MODEL_NAME = "large-v3"
DEVICE = "cuda"
SAMPLE_RATE = 16_000
AUDIO_FILENAME = "audio16k.wav"
LOG_FILENAME = "clipper.log"
WHISPER_VRAM_MB = 4600  # large-v3 в float16

# Параметры Whisper. Меняются — меняется и SETTINGS_VERSION, чтобы кэш пересчитался.
WHISPER_OPTIONS: dict[str, Any] = {
    "beam_size": 5,
    "word_timestamps": True,
    "vad_filter": True,  # Silero VAD: пропускает тишину, меньше галлюцинаций
    "vad_parameters": {"min_silence_duration_ms": 500},
    "condition_on_previous_text": False,  # меньше зацикливаний на длинных записях
    "hallucination_silence_threshold": 2.0,
}
SETTINGS_VERSION = 1
MIN_RANGE = 1.0  # с: более короткие недостающие кусочки не распознаём

# Типичные «галлюцинации» Whisper на музыке и тишине. Эти подписи из обучающих
# субтитров в настоящей речи не встречаются — такие сегменты выбрасываются всегда.
HALLUCINATION_MARKERS = (
    "dimatorzok",
    "субтитры сделал",
    "субтитры создавал",
    "субтитры подготовил",
    "редактор субтитров",
    "корректор субтитров",
    "семкин",
    "amara.org",
)
# Эти фразы бывают и настоящими, поэтому выбрасываются, только если Whisper в них не уверен.
SUSPICIOUS_PHRASES = {
    "продолжение следует",
    "спасибо за просмотр",
    "спасибо за внимание",
    "субтитры",
    "thank you for watching",
    "thanks for watching",
    "thank you",
    "you",
}


class Engine(Protocol):
    """То, что нужно от модели: метод transcribe как у faster_whisper.WhisperModel."""

    def transcribe(self, audio: Any, **kwargs: Any) -> tuple[Iterable[Any], Any]: ...


EngineFactory = Callable[[Config, Reporter], Engine]


def transcribe_source(
    source: SourceInfo,
    work_dir: Path,
    cfg: Config,
    reporter: Reporter,
    ranges: list[tuple[float, float]] | None = None,
    *,
    force: bool = False,
    engine_factory: EngineFactory | None = None,
) -> Transcript:
    """Распознать `ranges` исходного видео (по умолчанию — всё видео).

    Возвращает транскрипт со всеми распознанными к этому моменту отрезками.
    """
    if not source.has_audio:
        raise ClipperError("В видео нет звуковой дорожки — распознавать нечего.")
    wanted = merge_ranges([(max(0.0, a), min(source.duration, b)) for a, b in (ranges or [(0.0, source.duration)])])
    settings = transcript_settings(cfg)
    transcript = None if force else load_transcript(work_dir)
    if transcript is not None and transcript.settings != settings:
        reporter.info("Параметры распознавания изменились — распознаю заново.")
        transcript = None
    if transcript is None:
        transcript = Transcript(language=None, settings=settings, ranges=[], segments=[])

    todo = subtract_ranges(wanted, transcript.ranges, min_length=MIN_RANGE)
    if not todo:
        reporter.info("Эти отрезки уже распознаны — беру из кэша.")
        write_srt(transcript, work_dir / SRT_FILENAME)
        return transcript

    audio_path = extract_audio(source, work_dir, reporter)
    engine = (engine_factory or load_whisper)(cfg, reporter)
    language = cfg.transcribe.language or transcript.language
    total = sum(b - a for a, b in todo)
    done = 0.0
    with reporter.stage("transcribe", "Распознавание речи", total=total, unit="seconds") as stage:
        for start, end in todo:
            audio = read_audio(audio_path, start, end)
            try:
                segments, info = engine.transcribe(audio, language=language, **whisper_options(cfg, language))
                if language is None:
                    language = info.language
                    reporter.info(f"Язык речи: {language} (уверенность {info.language_probability:.0%})")
                new_segments: list[Segment] = []
                for raw in segments:
                    if is_hallucination(raw):
                        log.debug("галлюцинация отброшена: %r", raw.text)
                    elif (segment := convert_segment(raw, start, end)) is not None:
                        new_segments.append(segment)
                    stage.update(done + min(float(raw.end), end - start), message=_preview(raw.text))
            except (RuntimeError, OSError, ValueError) as exc:
                raise whisper_error(exc) from None
            done += end - start
            transcript = add_range(transcript, (start, end), new_segments)
            transcript.language = language
            save_transcript(work_dir, transcript)  # сохраняем после каждого отрезка: сбой не теряет сделанное
        stage.update(total)
        stage.result = plural(len(transcript.words), "слово", "слова", "слов")
    write_srt(transcript, work_dir / SRT_FILENAME)
    return transcript


def transcript_settings(cfg: Config) -> dict[str, Any]:
    """То, от чего зависит результат распознавания (compute_type на текст почти не влияет)."""
    settings: dict[str, Any] = {"model": MODEL_NAME, "language": cfg.transcribe.language, "version": SETTINGS_VERSION}
    if cfg.transcribe.verbatim:
        settings["verbatim"] = True
    return settings


# Подсказка-«затравка» для Whisper: пример речи с паразитами. Без неё Whisper обычно
# сам выбрасывает «эм» и «ээ» из текста — и вырезать их потом нечем.
VERBATIM_PROMPTS = {
    "ru": "Эм, ну, как бы, ээ... Мм, короче, вот. Ну, эм, в общем, так.",
    "en": "Umm, let me think like, hmm... Okay, here's what I'm, like, thinking.",
}


def whisper_options(cfg: Config, language: str | None) -> dict[str, Any]:
    options = dict(WHISPER_OPTIONS)
    if cfg.transcribe.verbatim:
        options["initial_prompt"] = VERBATIM_PROMPTS.get(language or "ru", VERBATIM_PROMPTS["ru"])
    return options


def add_range(transcript: Transcript, done: tuple[float, float], segments: list[Segment]) -> Transcript:
    """Добавить распознанный отрезок: старые сегменты внутри него заменяются новыми."""
    start, end = done
    kept = [s for s in transcript.segments if s.end <= start or s.start >= end]
    return Transcript(
        language=transcript.language,
        settings=transcript.settings,
        ranges=merge_ranges([*transcript.ranges, done]),
        segments=sorted([*kept, *segments], key=lambda s: s.start),
    )


def convert_segment(raw: Any, offset: float, range_end: float) -> Segment | None:
    """Сегмент faster-whisper (время от начала отрезка) → Segment (время исходного видео)."""
    limit = range_end
    words: list[Word] = []
    for raw_word in raw.words or []:
        text = str(raw_word.word).strip()
        if not text:
            continue
        start = min(offset + float(raw_word.start), limit)
        end = min(max(offset + float(raw_word.end), start), limit)
        if words and start < words[-1].start:  # слова строго по порядку
            start = words[-1].start
            end = max(end, start)
        words.append(Word(text, round(start, 3), round(end, 3), round(float(raw_word.probability), 3)))
    words = join_hyphenated(words)
    text = join_hyphenated_text(" ".join(str(raw.text).split()))
    if not text or not words:
        return None
    return Segment(
        start=round(min(offset + float(raw.start), words[0].start), 3),
        end=round(min(max(offset + float(raw.end), words[-1].end), limit), 3),
        text=text,
        words=tuple(words),
    )


def is_hallucination(raw: Any) -> bool:
    text = _normalize(str(raw.text))
    if not text:
        return True
    if any(marker in text for marker in HALLUCINATION_MARKERS):
        return True
    unsure = float(getattr(raw, "no_speech_prob", 0.0)) > 0.3 or float(getattr(raw, "avg_logprob", 0.0)) < -0.7
    return text in SUSPICIOUS_PHRASES and unsure


def _normalize(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in " .-" else " " for ch in text.lower().replace("ё", "е"))
    return " ".join(cleaned.replace("...", " ").split()).strip(" .-")


def _preview(text: str, width: int = 50) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


# --- звук ---------------------------------------------------------------------------


def extract_audio(source: SourceInfo, work_dir: Path, reporter: Reporter) -> Path:
    """Извлечь звук в work/<id>/audio16k.wav (один раз)."""
    path = work_dir / AUDIO_FILENAME
    if path.is_file() and abs(audio_duration(path) - source.duration) < 2.0:
        return path
    ffmpeg = env.require_ffmpeg()
    tmp = path.with_name(path.stem + ".part.wav")
    work_dir.mkdir(parents=True, exist_ok=True)
    args = ["-i", source.video, "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE)]
    args += ["-c:a", "pcm_s16le", "-f", "wav", str(tmp)]
    with reporter.stage("audio", "Извлечение звука", total=source.duration, unit="seconds") as stage:
        try:
            run_ffmpeg(
                ffmpeg.path,
                args,
                on_progress=stage.update,
                cancel=reporter.cancel,
                log_path=work_dir / LOG_FILENAME,
            )
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        os.replace(tmp, path)
    return path


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / float(wav.getframerate())
    except (OSError, EOFError, wave.Error):
        return -1.0


def read_audio(path: Path, start: float, end: float) -> Any:
    """Кусок WAV [start, end] как numpy float32 в диапазоне -1…1 (так ждёт faster-whisper)."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ClipperError(f"Неожиданный формат {path.name}: удалите его, он создастся заново.")
        first = min(int(start * SAMPLE_RATE), wav.getnframes())
        wav.setpos(first)
        data = wav.readframes(max(int((end - start) * SAMPLE_RATE), 0))
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


# --- модель -------------------------------------------------------------------------


def load_whisper(cfg: Config, reporter: Reporter) -> Engine:
    """Загрузить large-v3 на видеокарту (при первом запуске — скачать, ~3 ГБ)."""
    env.setup_cuda_dlls()  # до импорта ctranslate2
    _quiet_huggingface()
    try:
        import ctranslate2
        from faster_whisper import WhisperModel
        from faster_whisper.utils import download_model
    except ImportError as exc:
        raise DependencyError(
            f"faster-whisper не импортируется: {exc}",
            hint="Переустановите зависимости: pip install -r requirements.txt",
        ) from None

    if ctranslate2.get_cuda_device_count() == 0:
        raise DependencyError(
            "Видеокарта NVIDIA не найдена — распознавание работает только на GPU.",
            hint="Проверьте драйвер NVIDIA; подробности покажет clipper doctor.",
        )
    _check_free_vram(cfg, reporter)
    _check_language(cfg.transcribe.language)

    cache_dir = cfg.transcribe.model_dir
    try:
        model_path = download_model(MODEL_NAME, cache_dir=cache_dir, local_files_only=True)
    except Exception:
        with reporter.stage("model_download", f"Загрузка модели {MODEL_NAME} (~3 ГБ, только в первый раз)"):
            try:
                model_path = download_model(MODEL_NAME, cache_dir=cache_dir)
            except Exception as exc:
                raise ClipperError(
                    f"Не удалось скачать модель {MODEL_NAME}: {exc}",
                    hint="Проверьте интернет. Модель скачивается с huggingface.co один раз; "
                    "папку для неё можно задать через transcribe.model_dir.",
                ) from None

    with reporter.stage("model_load", "Загрузка модели в видеопамять") as stage:
        try:
            model = WhisperModel(model_path, device=DEVICE, compute_type=cfg.transcribe.compute_type)
        except (RuntimeError, OSError, ValueError) as exc:
            raise whisper_error(exc) from None
        stage.result = f"{MODEL_NAME}, {cfg.transcribe.compute_type}"
    return model


def model_is_downloaded(model_dir: str | None = None) -> bool:
    """Скачана ли уже модель large-v3 (без обращения к сети)."""
    _quiet_huggingface()
    try:
        from faster_whisper.utils import download_model

        download_model(MODEL_NAME, cache_dir=model_dir, local_files_only=True)
    except Exception:
        return False
    return True


def _quiet_huggingface() -> None:
    """Без служебных сообщений HuggingFace Hub поверх нашего прогресса.

    Например, «You are sending unauthenticated requests to the HF Hub…» — токен
    для скачивания открытой модели не нужен. У логгера huggingface_hub свой
    обработчик, поэтому такие сообщения выводились дважды и ломали строку прогресса.
    """
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # прогресс рисуем сами
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def whisper_error(exc: BaseException) -> ClipperError:
    """Ошибка CUDA/ctranslate2 → понятная ошибка с подсказкой."""
    text = str(exc).strip() or type(exc).__name__
    lower = text.lower()
    if "out of memory" in lower:
        return ClipperError(
            "Не хватило видеопамяти для распознавания.",
            hint="Закройте игры и программы, занимающие видеокарту, или задайте "
            "--set transcribe.compute_type=int8_float16 (та же модель, ~3 ГБ вместо ~4,5).",
        )
    if any(name in lower for name in ("cublas", "cudnn", ".dll", "cannot load", "library")):
        return DependencyError(
            f"Не загружаются библиотеки CUDA: {text}",
            hint=f"Установите их: {env.CUDA_PIP_COMMAND} — и проверьте clipper doctor.",
        )
    if "cuda" in lower:
        return DependencyError(f"Ошибка CUDA: {text}", hint="Обновите драйвер NVIDIA и проверьте clipper doctor.")
    return ClipperError(f"Ошибка распознавания: {text}")


def _check_free_vram(cfg: Config, reporter: Reporter) -> None:
    gpus = env.nvidia_gpus()
    if not gpus:
        return
    free = max(gpu.memory_free_mb for gpu in gpus)
    need = WHISPER_VRAM_MB if cfg.transcribe.compute_type == "float16" else 3200
    if free < need:
        reporter.warning(
            f"Свободно {free / 1024:.1f} ГБ видеопамяти, а модели нужно ~{need / 1024:.1f} ГБ. "
            "Если распознавание упадёт — закройте игры или задайте transcribe.compute_type: int8_float16."
        )


def _check_language(language: str | None) -> None:
    if language is None:
        return
    try:
        from faster_whisper.tokenizer import _LANGUAGE_CODES
    except ImportError:
        return
    if language not in _LANGUAGE_CODES:
        raise ClipperError(
            f"Whisper не знает язык «{language}».",
            hint="Используйте двухбуквенный код (ru, en, uk, de…) или auto.",
        )


# --- SRT ----------------------------------------------------------------------------


def write_srt(transcript: Transcript, path: Path) -> Path:
    """Сегменты транскрипта как субтитры .srt — чтобы проверить распознавание в плеере."""
    blocks = []
    for index, segment in enumerate(transcript.segments, start=1):
        blocks.append(f"{index}\n{_srt_time(segment.start)} --> {_srt_time(segment.end)}\n{segment.text}\n")
    tmp = path.with_name(path.name + ".tmp")
    # BOM — чтобы кириллицу понимали все плееры Windows; CRLF — обычный формат SRT на любой ОС.
    tmp.write_text("\n".join(blocks), encoding="utf-8-sig", newline="\r\n")
    os.replace(tmp, path)
    return path


def _srt_time(seconds: float) -> str:
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
