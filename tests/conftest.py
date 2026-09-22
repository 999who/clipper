import pytest


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    """Каждый тест — в пустой папке и без внешнего CLIPPER_CONFIG."""
    monkeypatch.delenv("CLIPPER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def fast_x264(monkeypatch):
    """Тестам не нужно качество: кадр 1080×1920 с preset medium кодируется в разы дольше."""
    from clipper.core import render

    monkeypatch.setitem(render.VIDEO_ARGS, "x264", ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "28"])
