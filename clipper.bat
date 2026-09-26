@echo off
rem Запуск clipper без активации окружения: двойной щелчок — интерфейс,
rem из терминала — .\clipper doctor и другие команды.
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".venv\Scripts\clipper.exe" (
    echo Не найдено окружение .venv — сначала установите clipper, см. README.md.
    pause
    exit /b 1
)
".venv\Scripts\clipper.exe" %*
if errorlevel 1 pause
