import re
from pathlib import Path

import pytest

from clipper.core.config import (
    DEFAULT_FILLERS,
    Config,
    config_to_dict,
    dump_config,
    example_config_path,
    find_config_file,
    leaf_paths,
    load_config,
    parse_set_options,
    write_example_config,
)
from clipper.core.errors import ConfigError


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults():
    cfg = load_config()
    assert cfg.select.clips == 5
    assert cfg.reframe.aspect == "9:16"
    assert cfg.reframe.background == "blur"
    assert cfg.transcribe.language is None
    assert cfg.audio.fillers == list(DEFAULT_FILLERS)


def test_file_then_overrides(tmp_path):
    path = write(tmp_path / "c.yaml", "select:\n  clips: 7\n  min_len: 15\n")
    cfg = load_config(path, {"select.clips": 3})
    assert cfg.select.clips == 3  # флаг важнее файла
    assert cfg.select.min_len == 15.0  # из файла, приведено к float
    assert cfg.select.max_len == 60.0  # значение по умолчанию


def test_unknown_key_suggests_close_name(tmp_path):
    path = write(tmp_path / "c.yaml", "select:\n  min_lenght: 15\n")
    with pytest.raises(ConfigError) as err:
        load_config(path)
    assert err.value.message.startswith("c.yaml: ")
    assert "select.min_lenght" in err.value.message
    assert "select.min_len" in err.value.hint


def test_key_without_section_suggests_full_path():
    with pytest.raises(ConfigError) as err:
        load_config(None, {"clips": 3})
    assert "select.clips" in err.value.hint


def test_type_error_names_the_key(tmp_path):
    path = write(tmp_path / "c.yaml", "select:\n  clips: много\n")
    with pytest.raises(ConfigError, match="select.clips: ожидалось целое число"):
        load_config(path)


def test_literal_lists_allowed_values():
    with pytest.raises(ConfigError) as err:
        load_config(None, {"reframe.aspect": "4:3"})
    assert "9:16, 1:1, original" in err.value.hint


def test_yaml_1_1_quirks_are_disabled(tmp_path):
    # PyYAML по умолчанию читает 9:16 как 556, а no — как false.
    path = write(
        tmp_path / "c.yaml",
        "reframe:\n  aspect: 9:16\ntranscribe:\n  language: no\nsubtitles:\n  enabled: yes\n",
    )
    cfg = load_config(path)
    assert cfg.reframe.aspect == "9:16"
    assert cfg.transcribe.language == "no"
    assert cfg.subtitles.enabled is True


def test_parse_set_options():
    parsed = parse_set_options(
        [
            "select.clips=3",
            "reframe.aspect=1:1",
            "select.keywords=побед*, да ладно",
            "reframe.layout=",
            "select.min_len=12.5",
            "paths.workdir=D:\\clips\\work",
        ]
    )
    assert parsed == {
        "select.clips": 3,
        "reframe.aspect": "1:1",
        "select.keywords": "побед*, да ладно",
        "reframe.layout": None,
        "select.min_len": 12.5,
        "paths.workdir": "D:\\clips\\work",
    }
    cfg = load_config(None, parsed)
    assert cfg.select.keywords == ["побед*", "да ладно"]


def test_set_requires_key_and_value():
    with pytest.raises(ConfigError, match="КЛЮЧ=ЗНАЧЕНИЕ"):
        parse_set_options(["select.clips"])


def test_language_normalization():
    assert load_config(None, {"transcribe.language": "auto"}).transcribe.language is None
    assert load_config(None, {"transcribe.language": "RU"}).transcribe.language == "ru"
    with pytest.raises(ConfigError, match="transcribe.language"):
        load_config(None, {"transcribe.language": "russian"})


def test_min_len_greater_than_max_len():
    with pytest.raises(ConfigError, match="select.max_len"):
        load_config(None, {"select.min_len": 90})


def test_layouts_from_file_and_overrides(tmp_path):
    path = write(
        tmp_path / "c.yaml",
        """
layouts:
  my_setup:
    source_size: [1920, 1080]
    webcam: { x: 1560, y: 810, width: 360, height: 270 }
    webcam_zone: 0.33
    game_crop: center
""",
    )
    cfg = load_config(path, {"layouts.my_setup.webcam.x": 1500, "reframe.layout": "my_setup"})
    layout = cfg.layouts["my_setup"]
    assert (layout.webcam.x, layout.webcam.width) == (1500, 360)
    assert layout.source_size == (1920, 1080)
    assert cfg.reframe.layout == "my_setup"


def test_layout_needs_webcam_size(tmp_path):
    path = write(tmp_path / "c.yaml", "layouts:\n  cam:\n    webcam_zone: 0.3\n")
    with pytest.raises(ConfigError, match="layouts.cam.webcam"):
        load_config(path)


def test_webcam_outside_the_frame():
    with pytest.raises(ConfigError, match="выходит за кадр 1920×1080"):
        load_config(
            None,
            {
                "layouts.cam.webcam": {"x": 1800, "y": 0, "width": 360, "height": 270},
                "layouts.cam.source_size": [1920, 1080],
            },
        )


def test_reframe_layout_must_exist():
    with pytest.raises(ConfigError, match="пресет «nope» не найден"):
        load_config(None, {"reframe.layout": "nope"})


def test_duplicate_section_is_an_error(tmp_path):
    path = write(tmp_path / "c.yaml", "select:\n  clips: 3\nselect:\n  min_len: 10\n")
    with pytest.raises(ConfigError, match="строка 3"):
        load_config(path)


def test_empty_section_and_empty_file(tmp_path):
    assert load_config(write(tmp_path / "a.yaml", "select:\n")).select.clips == 5
    assert load_config(write(tmp_path / "b.yaml", "")).select.clips == 5


def test_dashes_in_keys(tmp_path):
    cfg = load_config(write(tmp_path / "c.yaml", "select:\n  min-len: 12\n"))
    assert cfg.select.min_len == 12.0


def test_syntax_error_has_position(tmp_path):
    path = write(tmp_path / "c.yaml", "select:\n  clips: [1, 2\n")
    with pytest.raises(ConfigError, match="строка"):
        load_config(path)


def test_top_level_must_be_mapping(tmp_path):
    with pytest.raises(ConfigError, match="набор разделов"):
        load_config(write(tmp_path / "c.yaml", "- a\n- b\n"))


def test_bom_and_cp1251_files(tmp_path):
    bom = tmp_path / "bom.yaml"
    bom.write_bytes("\ufeffaudio:\n  fillers: [ну]\n".encode())
    assert load_config(bom).audio.fillers == ["ну"]
    ansi = tmp_path / "ansi.yaml"
    ansi.write_bytes("audio:\n  fillers: [ну, короче]\n".encode("cp1251"))
    assert load_config(ansi).audio.fillers == ["ну", "короче"]


def test_find_config_file(tmp_path, monkeypatch):
    assert find_config_file() is None
    local = write(tmp_path / "clipper.yaml", "")
    assert find_config_file().resolve() == local.resolve()
    other = write(tmp_path / "other.yaml", "")
    monkeypatch.setenv("CLIPPER_CONFIG", str(other))
    assert find_config_file() == other
    assert find_config_file(local) == local  # явный путь важнее переменной
    with pytest.raises(ConfigError, match="не найден"):
        find_config_file(tmp_path / "missing.yaml")
    monkeypatch.setenv("CLIPPER_CONFIG", str(tmp_path / "missing.yaml"))
    with pytest.raises(ConfigError) as err:
        find_config_file()
    assert "CLIPPER_CONFIG" in err.value.hint


def test_dump_round_trip(tmp_path):
    cfg = load_config(None, {"reframe.aspect": "1:1", "select.keywords": ["a", "да ладно"]})
    reloaded = load_config(write(tmp_path / "dump.yaml", dump_config(cfg)))
    assert config_to_dict(reloaded) == config_to_dict(cfg)


def test_example_documents_every_parameter_with_its_default(tmp_path):
    example = example_config_path()
    assert example is not None
    text = example.read_text(encoding="utf-8")
    for path in leaf_paths():
        assert f"{path.rsplit('.', 1)[-1]}:" in text, f"{path} не описан в {example.name}"

    # Если раскомментировать все параметры, получатся ровно значения по умолчанию.
    uncommented = re.sub(r"(?m)^# (\s*[a-z_]+:)", r"\1", text)
    actual = config_to_dict(load_config(write(tmp_path / "all.yaml", uncommented)))
    expected = config_to_dict(Config())
    assert "my_setup" in actual.pop("layouts")
    assert expected.pop("layouts") == {}
    assert actual == expected


def test_example_is_valid_as_is():
    cfg = load_config(example_config_path())
    assert cfg.layouts["my_setup"].webcam.width == 360
    assert cfg.select == Config().select  # всё, кроме layouts, закомментировано


def test_write_example_config(tmp_path):
    dest = write_example_config(tmp_path / "clipper.yaml")
    assert "layouts:" in dest.read_text(encoding="utf-8")
    with pytest.raises(ConfigError, match="уже существует"):
        write_example_config(dest)
    write_example_config(dest, overwrite=True)


def test_removed_parameters_explain_themselves(tmp_path):
    path = tmp_path / "clipper.yaml"
    path.write_text("render:\n  concat: true\n", encoding="utf-8")
    with pytest.raises(ConfigError) as info:
        load_config(path)
    assert "render.concat" in info.value.message and "больше не поддерживается" in info.value.message
    assert "удалите эту строку" in (info.value.hint or "")


def test_save_layout_keeps_comments_and_replaces_or_appends(tmp_path):
    from clipper.core.config import LayoutConfig, Rect, save_layout

    path = tmp_path / "clipper.yaml"
    path.write_text(
        "# мой конфиг\nselect:\n  clips: 7   # мне нужно 7\n\n"
        "layouts:\n  old:\n    webcam: { x: 1, y: 2, width: 30, height: 40 }\n    game_crop: left\n\n"
        "# хвост\nrender:\n  encoder: x264\n",
        encoding="utf-8",
    )
    cam = LayoutConfig(webcam=Rect(100, 200, 300, 240), webcam_zone=0.4, game_crop="right", source_size=(1920, 1080))
    save_layout(path, "cam", cam)
    save_layout(path, "old", LayoutConfig(webcam=Rect(10, 20, 30, 40)))
    text = path.read_text(encoding="utf-8")
    assert "# мой конфиг" in text and "# мне нужно 7" in text and "# хвост" in text
    assert text.count("old:") == 1 and "x: 1," not in text
    cfg = load_config(path)
    assert cfg.select.clips == 7 and cfg.render.encoder == "x264"
    assert cfg.layouts["cam"] == cam
    assert cfg.layouts["old"].webcam == Rect(10, 20, 30, 40)
    assert (tmp_path / "clipper.yaml.bak").is_file()


def test_save_layout_creates_file_and_section(tmp_path):
    from clipper.core.config import LayoutConfig, Rect, save_layout

    fresh = tmp_path / "new.yaml"
    save_layout(fresh, "стрим", LayoutConfig(webcam=Rect(0, 0, 100, 100)))  # файла нет — из примера
    assert "стрим" in load_config(fresh).layouts

    plain = tmp_path / "plain.yaml"
    plain.write_text("select:\n  clips: 3", encoding="utf-8")
    save_layout(plain, "a", LayoutConfig(webcam=Rect(5, 5, 50, 50)))
    assert load_config(plain).layouts["a"].webcam.x == 5

    bad = LayoutConfig(webcam=Rect(1900, 0, 100, 100), source_size=(1920, 1080))  # вебка за кадром
    before = plain.read_text(encoding="utf-8")
    with pytest.raises(ConfigError):
        save_layout(plain, "a", bad)
    assert plain.read_text(encoding="utf-8") == before
