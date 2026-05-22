
import json
from pathlib import Path

import pytest

import generate


def test_parse_args_defaults() -> None:
    args = generate.parse_args(["--prompt", "test prompt"])
    assert args.prompt == "test prompt"
    assert args.model == "runwayml/stable-diffusion-v1-5"
    assert args.scheduler == "default"
    assert args.width == 512
    assert args.height == 512
    assert args.steps == 30
    assert args.guidance_scale == 7.5
    assert args.num_images == 1
    assert args.low_memory is False
    assert args.save_metadata is True
    assert args.metadata_file is None


def test_parse_args_custom_values() -> None:
    args = generate.parse_args(
        [
            "--prompt",
            "test prompt",
            "--model",
            "custom/model",
            "--scheduler",
            "euler",
            "--width",
            "768",
            "--height",
            "512",
            "--steps",
            "40",
            "--guidance-scale",
            "8.0",
            "--num-images",
            "2",
            "--low-memory",
        ]
    )
    assert args.model == "custom/model"
    assert args.scheduler == "euler"
    assert args.width == 768
    assert args.height == 512
    assert args.steps == 40
    assert args.guidance_scale == 8.0
    assert args.num_images == 2
    assert args.low_memory is True


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--width", "0"),
        ("--height", "-1"),
        ("--steps", "0"),
        ("--num-images", "0"),
        ("--guidance-scale", "0"),
    ],
)
def test_parse_args_rejects_non_positive_values(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(["--prompt", "test prompt", flag, value])


def test_parse_args_rejects_invalid_scheduler() -> None:
    with pytest.raises(SystemExit):
        generate.parse_args(["--prompt", "test prompt", "--scheduler", "bad_scheduler"])


def test_parse_args_applies_preset_defaults(tmp_path: Path) -> None:
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(
        json.dumps(
            {
                "fantasy-vivid": {
                    "scheduler": "dpmpp_2m",
                    "steps": 41,
                    "guidance_scale": 9.5,
                    "negative_prompt": "muted colors",
                }
            }
        ),
        encoding="utf-8",
    )

    args = generate.parse_args(
        [
            "--prompt",
            "test prompt",
            "--preset",
            "fantasy-vivid",
            "--presets-file",
            str(presets_file),
        ]
    )

    assert args.scheduler == "dpmpp_2m"
    assert args.steps == 41
    assert args.guidance_scale == 9.5
    assert args.negative_prompt == "muted colors"


def test_parse_args_cli_values_override_preset(tmp_path: Path) -> None:
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(
        json.dumps({"fantasy-vivid": {"steps": 40, "guidance_scale": 9.0}}),
        encoding="utf-8",
    )

    args = generate.parse_args(
        [
            "--prompt",
            "test prompt",
            "--preset",
            "fantasy-vivid",
            "--presets-file",
            str(presets_file),
            "--steps",
            "25",
            "--guidance-scale",
            "7.0",
        ]
    )

    assert args.steps == 25
    assert args.guidance_scale == 7.0


def test_parse_args_lists_presets_and_exits(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(
        json.dumps({"b-preset": {"steps": 30}, "a-preset": {"steps": 20}}),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exit_info:
        generate.parse_args(["--list-presets", "--presets-file", str(presets_file)])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert output == ["a-preset", "b-preset"]


def test_parse_args_rejects_unknown_preset(tmp_path: Path) -> None:
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(json.dumps({"known": {"steps": 20}}), encoding="utf-8")
    with pytest.raises(SystemExit):
        generate.parse_args(
            [
                "--prompt",
                "test prompt",
                "--preset",
                "unknown",
                "--presets-file",
                str(presets_file),
            ]
        )


def test_parse_args_from_metadata_defaults(tmp_path: Path) -> None:
    metadata_file = tmp_path / "run.json"
    metadata_file.write_text(
        json.dumps(
            {
                "parameters": {
                    "prompt": "from metadata prompt",
                    "steps": 44,
                    "guidance_scale": 8.2,
                    "scheduler": "euler_a",
                    "seed": 999,
                    "width": 768,
                    "height": 512,
                }
            }
        ),
        encoding="utf-8",
    )

    args = generate.parse_args(["--from-metadata", str(metadata_file)])

    assert args.prompt == "from metadata prompt"
    assert args.steps == 44
    assert args.guidance_scale == 8.2
    assert args.scheduler == "euler_a"
    assert args.seed == 999
    assert args.width == 768
    assert args.height == 512


def test_parse_args_cli_values_override_metadata(tmp_path: Path) -> None:
    metadata_file = tmp_path / "run.json"
    metadata_file.write_text(
        json.dumps({"parameters": {"prompt": "from metadata prompt", "steps": 44}}),
        encoding="utf-8",
    )

    args = generate.parse_args(
        [
            "--from-metadata",
            str(metadata_file),
            "--prompt",
            "from cli prompt",
            "--steps",
            "22",
        ]
    )

    assert args.prompt == "from cli prompt"
    assert args.steps == 22
