"""Командная строка clipper.

Здесь только разбор команд и флагов, сборка Config и вызов ядра. Обработка
живёт в clipper.core и ничего не знает о терминале.
"""

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.markup import escape

from clipper import __version__
from clipper.console import (
    ConsoleSink,
    console,
    print_config,
    print_doctor,
    print_error,
    print_source,
    setup_logging,
    setup_stdio,
)
from clipper.core.config import (
    CONFIG_FILENAME,
    Config,
    dump_config,
    find_config_file,
    load_config,
    parse_set_options,
    write_example_config,
)
from clipper.core.doctor import run_doctor
from clipper.core.download import prepare_source
from clipper.core.errors import Cancelled, ClipperError
from clipper.core.events import CancelToken, Reporter

app = typer.Typer(
    name="clipper",
    help="Нарезка коротких вертикальных клипов с субтитрами из длинных видео и стримов.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)

PANEL_COMMON = "Общие"

ConfigOption = Annotated[
    Optional[Path],
    typer.Option(
        "--config",
        "-c",
        help=f"Файл конфига. По умолчанию ./{CONFIG_FILENAME}.",
        show_default=False,
        rich_help_panel=PANEL_COMMON,
    ),
]
SetOption = Annotated[
    Optional[list[str]],
    typer.Option(
        "--set",
        metavar="КЛЮЧ=ЗНАЧЕНИЕ",
        help="Задать любой параметр конфига, например --set select.min_len=15. Можно повторять.",
        show_default=False,
        rich_help_panel=PANEL_COMMON,
    ),
]
VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Подробный вывод и трейсбеки ошибок.", rich_help_panel=PANEL_COMMON),
]


def _show_version(value: bool) -> None:
    if value:
        console.print(f"clipper {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_show_version, is_eager=True, help="Показать версию и выйти."),
    ] = False,
) -> None:
    """Нарезка коротких вертикальных клипов с субтитрами из длинных видео и стримов."""


@contextmanager
def _command(verbose: bool) -> Iterator[CancelToken]:
    """Общая обёртка команд: логи, отмена по Ctrl+C, понятные ошибки без трейсбека."""
    setup_logging(verbose)
    token = CancelToken()
    try:
        yield token
    except typer.Exit:
        raise
    except ClipperError as exc:
        print_error(exc)
        if verbose:
            console.print_exception()
        raise typer.Exit(1) from None
    except (KeyboardInterrupt, Cancelled):
        token.cancel()
        console.print("[yellow]Отменено.[/]")
        raise typer.Exit(130) from None
    except Exception as exc:
        if verbose:
            console.print_exception()
        else:
            console.print(
                f"[bold red]Непредвиденная ошибка:[/] {escape(type(exc).__name__)}: {escape(str(exc))}\n"
                "[dim]Запустите команду с -v, чтобы увидеть подробности.[/]"
            )
        raise typer.Exit(1) from None


def build_config(
    config_file: Path | None,
    set_items: list[str] | None,
    flags: dict[str, Any] | None = None,
) -> tuple[Config, Path | None]:
    """Config из трёх слоёв. `flags` — {"select.clips": значение флага или None}."""
    path = find_config_file(config_file)
    overrides = {key: value for key, value in (flags or {}).items() if value is not None}
    overrides.update(parse_set_options(set_items or []))
    return load_config(path, overrides), path


@app.command()
def doctor(config: ConfigOption = None, verbose: VerboseOption = False) -> None:
    """Проверить окружение: ffmpeg, yt-dlp, видеокарту и CUDA, пакеты, конфиг."""
    with _command(verbose) as token:
        with ConsoleSink() as sink:
            results = run_doctor(Reporter(sink, token), config_path=config)
        print_doctor(results)
    if any(result.status == "fail" for result in results):
        raise typer.Exit(1)


@app.command("config")
def config_command(
    config: ConfigOption = None,
    set_items: SetOption = None,
    init: Annotated[
        bool,
        typer.Option("--init", help=f"Создать {CONFIG_FILENAME} из примера с комментариями."),
    ] = False,
    force: Annotated[bool, typer.Option("--force", help="Вместе с --init: перезаписать существующий файл.")] = False,
    verbose: VerboseOption = False,
) -> None:
    """Показать итоговые параметры: значения по умолчанию → clipper.yaml → флаги."""
    with _command(verbose):
        if init:
            dest = write_example_config(config or Path.cwd() / CONFIG_FILENAME, overwrite=force)
            console.print(
                f"Создан файл [bold]{escape(str(dest))}[/]. Раскомментируйте нужные параметры и поменяйте значения."
            )
            return
        cfg, path = build_config(config, set_items)
        print_config(dump_config(cfg), path, parse_set_options(set_items or []))


@app.command()
def download(
    source: Annotated[str, typer.Argument(metavar="ССЫЛКА_ИЛИ_ФАЙЛ", help="Ссылка на видео YouTube или путь к файлу.")],
    max_height: Annotated[
        Optional[int],
        typer.Option("--max-height", help="Максимальная высота видео, px (по умолчанию 1080).", show_default=False),
    ] = None,
    force: Annotated[bool, typer.Option("--force", help="Скачать заново, даже если видео уже есть в work/.")] = False,
    config: ConfigOption = None,
    set_items: SetOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Скачать видео и heatmap в work/<id>/ (для файла — прочитать параметры)."""
    with _command(verbose) as token:
        cfg, _ = build_config(config, set_items, {"download.max_height": max_height})
        with ConsoleSink() as sink:
            source_info = prepare_source(source, cfg, Reporter(sink, token), force=force)
        print_source(source_info, Path(cfg.paths.workdir).resolve() / source_info.id)


def main() -> None:
    setup_stdio()
    # Без аргументов — справка с кодом выхода 0 (click в этом случае возвращает 2).
    # Позже здесь будет запуск интерактивного интерфейса.
    app(sys.argv[1:] or ["--help"], prog_name="clipper")
