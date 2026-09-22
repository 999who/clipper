"""Ошибки, которые можно показать пользователю.

Каждая ошибка несёт сообщение («что случилось») и подсказку («что сделать»).
Интерфейс показывает их без трейсбека.
"""


class ClipperError(Exception):
    """Понятная пользователю ошибка: сообщение + необязательная подсказка."""

    def __init__(self, message: str, hint: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint

    def __str__(self) -> str:
        if self.hint:
            return f"{self.message}\n{self.hint}"
        return self.message


class ConfigError(ClipperError):
    """Ошибка в файле конфига, флагах или --set."""


class DependencyError(ClipperError):
    """Не хватает внешней программы или пакета (ffmpeg, yt-dlp, CUDA...)."""


class Cancelled(BaseException):
    """Пользователь отменил операцию.

    Наследуется от BaseException, как KeyboardInterrupt: обработчики
    `except Exception` (например, «ошибка одного клипа не мешает остальным»)
    не должны случайно проглотить отмену.
    """
