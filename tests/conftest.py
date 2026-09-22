import pytest


@pytest.fixture(autouse=True)
def isolated_config(monkeypatch, tmp_path):
    """Каждый тест — в пустой папке и без внешнего CLIPPER_CONFIG."""
    monkeypatch.delenv("CLIPPER_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path
