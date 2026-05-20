
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
