import json

import pytest

from clipper.core.models import (
    HeatPoint,
    SourceInfo,
    heatmap_peak,
    heatmap_profile,
    load_source,
    parse_heatmap,
    save_source,
    write_json_atomic,
)


def ytdlp_heatmap(values: list[float], duration: float = 200.0) -> list[dict]:
    """Heatmap в том виде, как его отдаёт yt-dlp: равные отрезки на всё видео."""
    step = duration / len(values)
    return [{"start_time": i * step, "end_time": (i + 1) * step, "value": value} for i, value in enumerate(values)]


def test_parse_heatmap_from_ytdlp_format():
    values = [i / 99 for i in range(100)]
    points = parse_heatmap(ytdlp_heatmap(values))
    assert len(points) == 100
    assert points[0] == HeatPoint(0.0, 2.0, 0.0)
    assert points[-1].end == pytest.approx(200.0)
    assert points[-1].value == pytest.approx(1.0)


def test_parse_heatmap_skips_broken_points_sorts_and_clamps():
    raw = [
        {"start_time": 10, "end_time": 20, "value": 1.7},  # value > 1 → 1
        {"start_time": 0, "end_time": 10, "value": -0.2},  # value < 0 → 0
        {"start_time": 20, "end_time": 20, "value": 0.5},  # нулевая длина
        {"start_time": 30, "end_time": 40, "value": "0.5"},  # не число
        {"start_time": 40, "end_time": 50, "value": float("nan")},
        {"start_time": 50, "end_time": 60, "value": True},  # bool — не число
        {"start_time": 60, "end_time": 70},  # нет value
        "мусор",
    ]
    assert parse_heatmap(raw) == [HeatPoint(0, 10, 0.0), HeatPoint(10, 20, 1.0)]


@pytest.mark.parametrize("raw", [None, [], {}, "heatmap", [{"start_time": 5, "end_time": 1, "value": 1}]])
def test_parse_heatmap_without_usable_points_is_none(raw):
    assert parse_heatmap(raw) is None


def test_heatmap_profile_keeps_points_one_to_one():
    values = [0.1 * (i % 10) for i in range(100)]
    points = parse_heatmap(ytdlp_heatmap(values))
    assert heatmap_profile(points, 100) == pytest.approx(values)


def test_heatmap_profile_compresses_by_maximum():
    values = [0.0] * 100
    values[37] = 0.9  # узкий пик не должен пропасть при сжатии в 10 раз
    profile = heatmap_profile(parse_heatmap(ytdlp_heatmap(values)), 10)
    assert profile == pytest.approx([0, 0, 0, 0.9, 0, 0, 0, 0, 0, 0])


def test_heatmap_profile_stretches_and_handles_edges():
    points = [HeatPoint(0, 5, 0.2), HeatPoint(5, 10, 0.8)]
    assert heatmap_profile(points, 4) == pytest.approx([0.2, 0.2, 0.8, 0.8])
    assert heatmap_profile(points, 4, duration=20) == pytest.approx([0.2, 0.8, 0.0, 0.0])
    assert heatmap_profile([], 3) == [0.0, 0.0, 0.0]
    assert heatmap_profile(points, 0) == []


def test_heatmap_peak_prefers_earlier_on_tie():
    points = [HeatPoint(0, 1, 0.5), HeatPoint(1, 2, 1.0), HeatPoint(2, 3, 1.0)]
    assert heatmap_peak(points) == HeatPoint(1, 2, 1.0)


def make_source(**changes) -> SourceInfo:
    values = dict(
        id="abc",
        kind="youtube",
        input="https://youtu.be/abc",
        video="/tmp/abc/source.mp4",
        title="Видео",
        duration=200.0,
        width=1920,
        height=1080,
        fps=30.0,
        has_audio=True,
        heatmap=[HeatPoint(0, 2, 0.3)],
    )
    values.update(changes)
    return SourceInfo(**values)


def test_source_round_trip(tmp_path):
    source = make_source()
    path = save_source(tmp_path / "abc", source)
    assert json.loads(path.read_text(encoding="utf-8"))["heatmap"] == [{"start": 0, "end": 2, "value": 0.3}]
    assert load_source(tmp_path / "abc") == source
    assert load_source(tmp_path / "missing") is None


def test_load_source_ignores_broken_file(tmp_path):
    (tmp_path / "source.json").write_text("{битый", encoding="utf-8")
    assert load_source(tmp_path) is None
    (tmp_path / "source.json").write_text('{"id": "x"}', encoding="utf-8")
    assert load_source(tmp_path) is None


def test_write_json_atomic_leaves_no_temp_files(tmp_path):
    target = tmp_path / "sub" / "data.json"
    write_json_atomic(target, {"текст": "да"})
    write_json_atomic(target, {"текст": "ещё"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"текст": "ещё"}
    assert [p.name for p in target.parent.iterdir()] == ["data.json"]
