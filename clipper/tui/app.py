"""Приложение Textual: стек экранов поверх того же ядра, что и CLI."""

from textual.app import App
from textual.binding import Binding

from clipper.tui.progress import ProgressScreen
from clipper.tui.screens import HomeScreen
from clipper.tui.settings import SettingsStore


class ClipperApp(App[None]):
    CSS_PATH = "clipper.tcss"
    TITLE = "clipper"
    ENABLE_COMMAND_PALETTE = False
    BINDINGS = [Binding("ctrl+q", "quit", "Выход", show=False)]

    def __init__(self, store: SettingsStore) -> None:
        super().__init__()
        self.store = store

    def on_mount(self) -> None:
        self.push_screen(HomeScreen())

    async def action_quit(self) -> None:
        for screen in self.screen_stack:  # остановить ffmpeg и распознавание, если они идут
            if isinstance(screen, ProgressScreen):
                screen.cancel_now()
        self.exit()
