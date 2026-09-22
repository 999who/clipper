import functools
import http.server
import shutil
import subprocess
import threading
from pathlib import Path

import pytest

from clipper.core import download
from clipper.core.config import Config, load_config
from clipper.core.errors import ClipperError
from clipper.core.events import Reporter, StageProgress
from clipper.core.ffmpeg import parse_probe
from clipper.core.models import HeatPoint, SourceInfo, load_source, save_source

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="нет ffmpeg/ffprobe"
)


def make_video(path: Path, seconds: float = 2, size: str = "320x240", audio: bool = True) -> Path:
    """Короткое тестовое видео, сгенерированное ffmpeg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={size}:rate=25:duration={seconds}",
    ]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    args += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(args, check=True)
    return path


# --- ссылки ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ&t=42",
        "https://youtu.be/dQw4w9WgXcQ?si=abc",
        "https://m.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/live/dQw4w9WgXcQ?feature=share",
        "https://www.youtube-nocookie.com/embed/dQw4w9WgXcQ",
    ],
)
def test_youtube_id(url):
    assert download.youtube_id(url) == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "url",
    ["https://vimeo.com/123", "https://www.youtube.com/@channel", "https://notyoutube.com/watch?v=dQw4w9WgXcQ"],
)
def test_youtube_id_absent(url):
    assert download.youtube_id(url) is None


def test_bare_youtube_link_gets_https(monkeypatch):
    seen = []
    monkeypatch.setattr(download, "download_url", lambda url, cfg, reporter, force: seen.append(url))
    download.prepare_source("youtube.com/watch?v=dQw4w9WgXcQ", Config(), Reporter())
    download.prepare_source('  "https://youtu.be/dQw4w9WgXcQ"  ', Config(), Reporter())
    assert seen == ["https://youtube.com/watch?v=dQw4w9WgXcQ", "https://youtu.be/dQw4w9WgXcQ"]


# --- ffprobe ------------------------------------------------------------------------


def test_parse_probe_rotation_and_cover_art():
    data = {
        "format": {"duration": "12.5"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "mjpeg",
                "width": 600,
                "height": 600,
                "disposition": {"attached_pic": 1},
            },
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
                "side_data_list": [{"side_data_type": "Display Matrix", "rotation": -90}],
            },
            {"codec_type": "audio", "codec_name": "aac"},
        ],
    }
    media = parse_probe(data)
    assert (media.width, media.height) == (1080, 1920)  # снято на телефон вертикально
    assert media.fps == pytest.approx(29.97)
    assert media.duration == 12.5
    assert media.video_codec == "h264"
    assert media.has_audio


def test_parse_probe_fallbacks_and_errors():
    data = {
        "format": {},
        "streams": [
            {
                "codec_type": "video",
                "width": 640,
                "height": 360,
                "duration": "3.0",
                "avg_frame_rate": "0/0",
                "r_frame_rate": "25/1",
            }
        ],
    }
    media = parse_probe(data)
    assert (media.duration, media.fps, media.has_audio) == (3.0, 25.0, False)
    with pytest.raises(ClipperError, match="нет видеодорожки"):
        parse_probe({"format": {"duration": "1"}, "streams": [{"codec_type": "audio"}]})


# --- локальный файл -------------------------------------------------------------------


@needs_ffmpeg
def test_local_file(tmp_path):
    video = make_video(tmp_path / "видео" / "мой стрим 01.mp4")
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    source = download.prepare_source(str(video), cfg, Reporter())
    assert source.kind == "file"
    assert source.id == "мой_стрим_01"
    assert Path(source.video) == video.resolve()  # файл не копируется
    assert (source.width, source.height, source.fps, source.has_audio) == (320, 240, 25.0, True)
    assert source.duration == pytest.approx(2.0, abs=0.1)
    assert source.heatmap is None
    assert load_source(tmp_path / "work" / "мой_стрим_01") == source


@needs_ffmpeg
def test_local_files_with_same_name_get_separate_folders(tmp_path):
    first = make_video(tmp_path / "a" / "clip.mp4")
    second = make_video(tmp_path / "b" / "clip.mp4")
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    assert download.prepare_source(str(first), cfg, Reporter()).id == "clip"
    assert download.prepare_source(str(second), cfg, Reporter()).id == "clip-2"
    assert download.prepare_source(str(first), cfg, Reporter()).id == "clip"  # тот же файл — та же папка


@needs_ffmpeg
def test_file_without_audio_warns(tmp_path):
    events = []
    video = make_video(tmp_path / "silent.mp4", audio=False)
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    source = download.prepare_source(str(video), cfg, Reporter(events.append))
    assert not source.has_audio
    assert any("нет звуковой дорожки" in getattr(e, "text", "") for e in events)


def test_missing_file_and_folder(tmp_path):
    with pytest.raises(ClipperError, match="Файл не найден"):
        download.prepare_source(str(tmp_path / "нет.mp4"), Config(), Reporter())
    with pytest.raises(ClipperError) as err:
        download.prepare_source(str(tmp_path), Config(), Reporter())
    assert "папка" in err.value.hint


def test_local_id():
    assert download.local_id(Path("C:/video/Мой стрим (часть 2).mp4")) == "Мой_стрим_часть_2"
    assert download.local_id(Path("???.mp4")) == "video"


# --- кэш и yt-dlp -------------------------------------------------------------------


@needs_ffmpeg
def test_cached_youtube_video_is_reused_without_network(tmp_path, monkeypatch):
    work = tmp_path / "work" / "dQw4w9WgXcQ"
    video = make_video(work / "source.mp4")
    cached = SourceInfo(
        id="dQw4w9WgXcQ",
        kind="youtube",
        input="https://youtu.be/dQw4w9WgXcQ",
        video=str(video),
        title="Кэш",
        duration=2.0,
        width=320,
        height=240,
        fps=25.0,
        has_audio=True,
        heatmap=[HeatPoint(0, 1, 1.0)],
    )
    save_source(work, cached)

    yt_dlp = pytest.importorskip("yt_dlp")

    def no_network(*args, **kwargs):
        raise AssertionError("кэшированное видео не должно скачиваться заново")

    monkeypatch.setattr(yt_dlp, "YoutubeDL", no_network)
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    assert download.prepare_source("https://www.youtube.com/watch?v=dQw4w9WgXcQ", cfg, Reporter()) == cached


class FakeStage:
    def __init__(self):
        self.updates = []

    def update(self, done, total=None, message=""):
        self.updates.append((done, total, message))


def test_download_progress_sums_video_and_audio():
    stage = FakeStage()
    info = {"requested_formats": [{"filesize": 800}, {"filesize_approx": 200}]}
    progress = download.DownloadProgress(stage, info)
    video = {"vcodec": "avc1", "acodec": "none"}
    audio = {"vcodec": "none", "acodec": "mp4a"}
    progress.on_progress(
        {
            "status": "downloading",
            "filename": "v",
            "downloaded_bytes": 400,
            "total_bytes": 800,
            "speed": 2_000_000,
            "info_dict": video,
        }
    )
    assert stage.updates[-1] == (400, 1000, "видеодорожка, 2.0 МБ/с")
    progress.on_progress({"status": "finished", "filename": "v", "total_bytes": 800, "info_dict": video})
    progress.on_progress(
        {
            "status": "downloading",
            "filename": "a",
            "downloaded_bytes": 50,
            "total_bytes_estimate": 210,
            "info_dict": audio,
        }
    )
    assert stage.updates[-1] == (850, 1010, "звук")
    progress.on_postprocess({"status": "started", "postprocessor": "Merger"})
    assert stage.updates[-1][2] == "склейка видео и звука"


def test_download_progress_without_known_sizes():
    stage = FakeStage()
    progress = download.DownloadProgress(stage, {})
    progress.on_progress({"status": "downloading", "filename": "f", "downloaded_bytes": 10, "info_dict": {}})
    assert stage.updates[-1][:2] == (10, 10)


@pytest.mark.parametrize(
    ("message", "hint_part"),
    [
        ("ERROR: [youtube] abc: Sign in to confirm you’re not a bot.", "cookies_from_browser"),
        ("ERROR: [youtube] abc: Private video. Sign in if you've been granted access", "cookies_from_browser"),
        ("ERROR: [youtube] abc: Video unavailable", "проверьте ссылку"),
        ("ERROR: [generic] x: Unable to download webpage: HTTP Error 404: Not Found", "открывается в браузере"),
        ("ERROR: [youtube] abc: Requested format is not available", "JS-движок"),
        ("ERROR: \x1b[0;31mUnable to download webpage: <urlopen error timed out>\x1b[0m", "интернету"),
        ("ERROR: что-то новое", "обновите yt-dlp"),
    ],
)
def test_download_error_hints(message, hint_part):
    error = download.download_error(message)
    assert not error.message.startswith("Не удалось скачать видео: ERROR")
    assert "\x1b" not in error.message
    assert hint_part in error.hint


def test_ytdlp_options(tmp_path, monkeypatch):
    monkeypatch.setattr(download.env, "find_js_runtime", lambda: None)
    events = []
    cfg = load_config(None, {"download.max_height": 720, "download.cookies_from_browser": "firefox"})
    options = download.ytdlp_options(cfg, tmp_path, "/opt/ffmpeg", Reporter(events.append), force=False)
    assert "[height<=720]" in options["format"] and "avc1" in options["format"]
    assert options["ffmpeg_location"] == "/opt/ffmpeg"
    assert options["outtmpl"]["default"].endswith("source.%(ext)s")
    odd = download.ytdlp_options(cfg, tmp_path / "100%", "/opt/ffmpeg", Reporter(), force=False)
    assert "100%%" in odd["outtmpl"]["default"]
    assert options["cookiesfrombrowser"] == ("firefox",)
    assert options["noplaylist"] is True
    assert any("JS-движок" in e.text for e in events)


@pytest.fixture
def http_server(tmp_path, monkeypatch):
    """Локальный HTTP-сервер: проверяем весь путь через yt-dlp без YouTube."""
    for name in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(name, "127.0.0.1,localhost")
    root = tmp_path / "site"
    root.mkdir()
    handler = functools.partial(QuietHandler, directory=str(root))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield root, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@needs_ffmpeg
def test_download_by_url_through_ytdlp(http_server, tmp_path):
    pytest.importorskip("yt_dlp")
    site, base = http_server
    make_video(site / "clip.mp4", seconds=3)
    events = []
    cfg = load_config(None, {"paths.workdir": str(tmp_path / "work")})
    source = download.prepare_source(f"{base}/clip.mp4", cfg, Reporter(events.append))

    work = tmp_path / "work" / "clip"
    assert source.kind == "url"
    assert Path(source.video) == (work / "source.mp4").resolve()
    assert source.duration == pytest.approx(3.0, abs=0.1)
    assert source.heatmap is None
    assert (work / "info.json").is_file()
    assert load_source(work) == source
    assert any(isinstance(e, StageProgress) and e.stage == "download" for e in events)

    # Второй запуск берёт готовое, а 404 даёт понятную ошибку.
    assert download.prepare_source(f"{base}/clip.mp4", cfg, Reporter()) == source
    with pytest.raises(ClipperError) as err:
        download.prepare_source(f"{base}/missing.mp4", cfg, Reporter())
    assert "HTTP Error 404" in err.value.message
