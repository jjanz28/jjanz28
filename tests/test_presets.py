import json
from pathlib import Path

import pytest

import presets


def test_load_presets_returns_empty_for_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    assert presets.load_presets(missing) == {}


def test_get_preset_returns_named_values(tmp_path: Path) -> None:
    path = tmp_path / "presets.json"
    path.write_text(
        json.dumps({"fantasy-vivid": {"steps": 40, "guidance_scale": 9.0}}),
        encoding="utf-8",
    )
    loaded = presets.get_preset("fantasy-vivid", path)
    assert loaded["steps"] == 40
    assert loaded["guidance_scale"] == 9.0


def test_get_preset_raises_for_unknown_name(tmp_path: Path) -> None:
    path = tmp_path / "presets.json"
    path.write_text(json.dumps({"known": {"steps": 30}}), encoding="utf-8")
    with pytest.raises(KeyError, match="Unknown preset"):
        presets.get_preset("unknown", path)


def test_save_preset_validates_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "presets.json"
    presets.save_preset(
        name="my-style",
        values={"scheduler": "euler", "steps": 25, "guidance_scale": 7.0},
        presets_file=path,
    )
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert "my-style" in stored
    assert stored["my-style"]["scheduler"] == "euler"
    assert stored["my-style"]["steps"] == 25


def test_save_preset_rejects_invalid_scheduler(tmp_path: Path) -> None:
    path = tmp_path / "presets.json"
    with pytest.raises(ValueError, match="Preset scheduler must be one of"):
        presets.save_preset(
            name="bad-style",
            values={"scheduler": "invalid", "steps": 20},
            presets_file=path,
        )
