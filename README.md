# clipper

Нарезает самые интересные фрагменты длинного видео или стрима в короткие
вертикальные клипы 1080×1920 с субтитрами в стиле CapCut.

- Источник — ссылка на YouTube или локальный файл.
- Моменты выбираются по графику «Самые популярные фрагменты» YouTube (heatmap)
  или по ключевым словам.
- Всё работает локально, без LLM и облачных сервисов. Сеть нужна только для
  скачивания видео и для однократной загрузки модели распознавания речи.

Устройство проекта описано в [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Статус

| Этап | Что умеет | Готово |
|---|---|---|
| 0 | каркас, конфиг, `clipper doctor`, `clipper config` | ✅ |
| 1 | загрузка видео и heatmap, `clipper download` | ✅ |
| 2 | распознавание речи (Whisper large-v3 на GPU) | |
| 3 | выбор моментов, `project.json`, нарезка | |
| 4 | вырезание пауз и слов-паразитов | |
| 5 | субтитры в стиле CapCut | |
| 6 | вертикальный кадр: слежение за лицом, режим стрима, калибровка вебки | |
| 7 | склейка, обложки, папка вывода | |

## Установка (Windows)

Что нужно заранее:
- **Python 3.10+, 64-бит** (проверено на 3.14);
- **видеокарта NVIDIA** со свежим драйвером;
- **ffmpeg** в PATH.

1. Поставьте ffmpeg и откройте **новый** терминал, чтобы обновился PATH:
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
3. Установите clipper со всеми зависимостями:
   ```
   pip install -r requirements.txt
   ```
   Скачается около 1,6 ГБ, на диске это займёт ~2,7 ГБ — в основном
   библиотеки CUDA. Модель распознавания (~3 ГБ) скачается один раз при
   первом распознавании.
4. Проверьте окружение:
   ```
   clipper doctor
   ```
   Команда проверяет:
   - ffmpeg (есть ли libass для субтитров и работает ли NVENC);
   - yt-dlp и JS-движок для YouTube;
   - видеокарту, CUDA и библиотеки cuBLAS/cuDNN;
   - mediapipe, OpenCV и файл конфига.

   Для каждой проблемы выводится подсказка, как её исправить.

Каждый раз перед работой активируйте окружение: `.venv\Scripts\activate`.

### Обновление yt-dlp

YouTube регулярно меняет сайт, поэтому yt-dlp нужно обновлять. Если загрузка
перестала работать, первым делом выполните:
```
pip install -U "yt-dlp[default,deno]"
```
`clipper doctor` предупреждает, если версии yt-dlp больше 60 дней.

## Конфиг

Параметры собираются в таком порядке, каждый следующий важнее:
1. значения по умолчанию;
2. файл `clipper.yaml` в текущей папке (или файл из `--config`);
3. флаги командной строки.

```
clipper config --init        создать clipper.yaml из примера с комментариями
clipper config               показать итоговые параметры и откуда они взяты
clipper config --set select.clips=3 --set reframe.aspect=1:1
```

- `--set раздел.параметр=значение` задаёт любой параметр на один запуск, без
  правки файла. Работает с каждой командой.
- Все параметры с пояснениями — в [clipper.example.yaml](clipper.example.yaml).
- Ваш `clipper.yaml` не хранится в git, поэтому обновления проекта не
  конфликтуют с вашими правками.

## Загрузка видео (этап 1)

```
clipper download "https://www.youtube.com/watch?v=..."
clipper download "D:\Видео\стрим.mp4"
```

- Видео с YouTube скачивается в `work/<id>/source.mp4` (H.264 до 1080p + AAC),
  рядом кладутся `info.json` (все метаданные yt-dlp) и `source.json`
  (параметры видео и heatmap в удобном виде).
- Локальный файл не копируется: clipper читает его параметры через ffprobe,
  а `source.json` пишет в `work/<имя файла>/`.
- В конце печатаются путь, длительность, разрешение и мини-график heatmap
  («Самые популярные фрагменты»). Heatmap есть только у достаточно
  популярных роликов YouTube.
- Повторный запуск с той же ссылкой не качает видео заново; `--force` —
  скачать ещё раз. `--max-height 720` — ограничить качество.
- Если YouTube просит войти («подтвердите, что вы не бот»), укажите браузер,
  в котором вы вошли в YouTube:
  `--set download.cookies_from_browser=firefox`. Из Chrome на Windows
  cookies прочитать нельзя.

## Проверка этапа 0

```
clipper
clipper doctor
clipper config --init
clipper config --set select.clips=3 --set reframe.aspect=1:1
clipper config --set select.clipz=3
```

- `clipper` без аргументов показывает справку.
- `clipper doctor` показывает таблицу проверок. На вашем ПК всё должно быть
  «ОК», кроме пункта «Файл конфига» до `--init`.
- `clipper config --init` создаёт `clipper.yaml`.
- `clipper config --set …` показывает параметры с `clips: 3` и `aspect: '1:1'`.
- `clipper config --set select.clipz=3` должен выдать понятную ошибку с
  подсказкой «Возможно, вы имели в виду «select.clips»?».

## Частые проблемы

- **`cublas64_12.dll` / `cudnn_ops64_9.dll` не найдены.** Переустановите
  библиотеки CUDA:
  ```
  pip install --force-reinstall nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
  ```
  clipper сам подключает их DLL из `site-packages\nvidia\…\bin`.
- **Не хватает видеопамяти при распознавании.** Модели large-v3 нужно ~4,5 ГБ.
  Закройте игры на время распознавания или задайте
  `transcribe.compute_type: int8_float16` — это та же модель, но ей нужно ~3 ГБ.
- **Предупреждение о нескольких пакетах OpenCV.** mediapipe требует
  `opencv-contrib-python`, а `opencv-python` с ним конфликтует. Удалите лишний
  пакет командой, которую подскажет `clipper doctor`.

## Разработка

```
pytest
```

Тесты проверяют логику без видеокарты, сети и видео.
