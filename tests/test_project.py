import json

import pytest

from clipper.core.models import (
    Clip,
    Project,
    ProjectError,
    SourceInfo,
    Word,
    load_project,
    project_to_json,
    save_project,
)


def make_project() -> Project:
    source = SourceInfo(
        "abc", "youtube", "https://youtu.be/abc", "C:/work/abc/source.mp4", "Видео", 600.0, 1920, 1080, 30.0, True
    )
    words = [Word("Короче,", 123.52, 123.9), Word("смотри", 123.95, 124.3)]
    clips = [
        Clip(1, 123.4, 161.85, 0.93, "пик heatmap 0.93", True, words),
        Clip(2, 300.0, 330.0, 0.5, "пик heatmap 0.50", True, []),
    ]
    return Project(source, "heatmap", [], "ru", "2026-09-22T18:00:00", clips)


def test_project_round_trip_and_readable_format(tmp_path):
    project = make_project()
    path = save_project(tmp_path, project)
    text = path.read_text(encoding="utf-8")
    assert '{"text": "Короче,", "start": 123.52, "end": 123.9}' in text  # слово — одна строка
    assert '"start": "00:02:03.400"' in text  # границы клипа — ЧЧ:ММ:СС.мс
    assert '"_help"' in text
    loaded = load_project(path)
    assert loaded.clips[0].start == pytest.approx(123.4)
    assert loaded.clips[0].words == [Word("Короче,", 123.52, 123.9), Word("смотри", 123.95, 124.3)]
    assert loaded.source.heatmap is None
    assert project_to_json(loaded) == text


def edited(tmp_path, change) -> object:
    path = save_project(tmp_path, make_project())
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_hand_edits_are_accepted(tmp_path):
    def change(data):
        clip = data["clips"][0]
        clip["start"] = "2:00.5"  # короткая запись
        clip["end"] = 170  # секунды числом
        clip["words"][0]["text"] = "Короче"
        data["clips"][1]["enabled"] = False

    project = load_project(edited(tmp_path, change))
    assert (project.clips[0].start, project.clips[0].end) == (120.5, 170.0)
    assert project.clips[0].words[0].text == "Короче"
    assert project.clips[1].enabled is False


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d["clips"][0].update(end="00:01:00"), "должен быть позже начала"),
        (lambda d: d["clips"][0].update(strat="1:00"), r"неизвестные поля \['strat'\]"),
        (lambda d: d["clips"][0].update(start="полторы минуты"), "клип id=1"),
        (lambda d: d["clips"][0].update(enabled="нет"), "true или false"),
        (lambda d: d["clips"][1].update(id=1), "повторяются id"),
        (lambda d: d["clips"][0].update(end="11:00"), "дальше конца видео"),
        (lambda d: d["clips"][0]["words"][0].pop("start"), "слово №1"),
        (lambda d: d.update(version=2), "версия"),
    ],
)
def test_hand_edit_errors_are_understandable(tmp_path, change, message):
    with pytest.raises(ProjectError, match=message):
        load_project(edited(tmp_path, change))


def test_broken_json_reports_line(tmp_path):
    path = save_project(tmp_path, make_project())
    text = path.read_text(encoding="utf-8").replace('"enabled": true,', '"enabled": true', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ProjectError, match=r"строка \d+, столбец \d+"):
        load_project(path)
