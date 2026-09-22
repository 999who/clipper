# clipper

Нарезает самые интересные моменты длинного видео или стрима в вертикальные
клипы 1080×1920 с субтитрами. Работает локально на вашем компьютере.

## Установка (Windows)

Что нужно заранее:
- **Python 3.10+, 64-бит** (проверено на 3.14);
- **видеокарта NVIDIA** со свежим драйвером;
- **Git**.

1. Установите ffmpeg и откройте **новый** терминал, чтобы обновился PATH:
   ```
   winget install --id Gyan.FFmpeg -e
   ```
2. Скачайте проект и создайте виртуальное окружение:
   ```
   git clone https://github.com/999who/clipper.git
   cd clipper
   git checkout claude/jolly-allen-lfr00v
   py -3.14 -m venv .venv
   .venv\Scripts\activate
   python -m pip install -U pip
   ```
   Если PowerShell не даёт выполнить `activate`, один раз разрешите локальные
   скрипты: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
3. Установите зависимости:
   ```
   pip install -r requirements.txt
   ```
   Скачается около 1,6 ГБ. Модель распознавания речи (~3 ГБ) скачается один
   раз при первом запуске распознавания.
4. Проверьте, что всё готово к работе:
   ```
   clipper doctor
   ```
   Для каждой проблемы команда подскажет, как её исправить.

## Запуск

Каждый раз перед работой откройте терминал в папке проекта и активируйте
окружение:
```
cd clipper
.venv\Scripts\activate
```

Затем запустите:
```
clipper
```

Откроется интерфейс в терминале. Всё выбирается стрелками и Enter, Esc —
назад, q — выход. Лучше всего он выглядит в Windows Terminal.

## Обновление

```
git pull
pip install -r requirements.txt
```

Если перестало скачиваться видео с YouTube, обновите yt-dlp:
```
pip install -U "yt-dlp[default,deno]"
```
