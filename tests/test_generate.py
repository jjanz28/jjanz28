import argparse
import sys
import types
from pathlib import Path

import pytest

import generate


class DummyImage:
    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image-bytes")


class DummyPipeline:
    def __init__(self, images: list[DummyImage]) -> None:
        self.images = images
        self.calls: list[dict] = []
        self.device: str | None = None
        self.scheduler = types.SimpleNamespace(config={"kind": "default"}, name="default")

    def to(self, device: str) -> "DummyPipeline":
        self.device = device
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(images=self.images)


def install_fake_runtime(monkeypatch: pytest.MonkeyPatch, pipeline: DummyPipeline) -> list[tuple]:
    from_pretrained_calls: list[tuple] = []

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int) -> "FakeGenerator":
            self.seed = seed
            return self

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        Generator = FakeGenerator

        class cuda:
            @staticmethod
            def is_available() -> bool:
                return False

    class FakeStableDiffusionPipeline:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            from_pretrained_calls.append((model, kwargs))
            return pipeline

    class FakeDDIMScheduler:
        @staticmethod
        def from_config(config: dict):
            return types.SimpleNamespace(config=config, name="ddim")

    class FakeEulerScheduler:
        @staticmethod
        def from_config(config: dict):
            return types.SimpleNamespace(config=config, name="euler")

    class FakeEulerAncestralScheduler:
        @staticmethod
        def from_config(config: dict):
            return types.SimpleNamespace(config=config, name="euler_a")

    class FakeDpmppScheduler:
        @staticmethod
        def from_config(config: dict):
            return types.SimpleNamespace(config=config, name="dpmpp_2m")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            StableDiffusionPipeline=FakeStableDiffusionPipeline,
            DDIMScheduler=FakeDDIMScheduler,
            EulerDiscreteScheduler=FakeEulerScheduler,
            EulerAncestralDiscreteScheduler=FakeEulerAncestralScheduler,
            DPMSolverMultistepScheduler=FakeDpmppScheduler,
        ),
    )
    return from_pretrained_calls


def build_args(tmp_path: Path, **overrides):
    args = {
        "prompt": "test prompt",
        "negative_prompt": None,
        "negative_preset": "none",
        "model": "fake/model",
        "model_revision": None,
        "scheduler": "default",
        "output": str(tmp_path / "out.png"),
        "width": 512,
        "height": 512,
        "steps": 2,
        "guidance_scale": 7.5,
        "seed": 1,
        "num_images": 1,
        "cpu": True,
    }
    args.update(overrides)
    return argparse.Namespace(**args)


def test_generate_rejects_invalid_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(generate, "parse_args", lambda: build_args(tmp_path, width=513))

    with pytest.raises(ValueError, match="multiples of 8"):
        generate.main()


def test_generate_rejects_invalid_num_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(generate, "parse_args", lambda: build_args(tmp_path, num_images=0))

    with pytest.raises(ValueError, match="--num-images must be >= 1"):
        generate.main()


def test_generate_smoke_single_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "single.png"
    pipeline = DummyPipeline(images=[DummyImage()])
    from_pretrained_calls = install_fake_runtime(monkeypatch, pipeline)

    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: build_args(
            tmp_path,
            output=str(output),
            negative_prompt="extra fingers",
            negative_preset="photo",
            scheduler="euler",
            model_revision="v1.0.0",
            guidance_scale=6.0,
            seed=42,
        ),
    )

    generate.main()

    assert output.exists()
    assert pipeline.device == "cpu"
    assert pipeline.scheduler.name == "euler"
    assert from_pretrained_calls == [
        ("fake/model", {"torch_dtype": "float32", "revision": "v1.0.0"})
    ]
    assert pipeline.calls[0]["num_images_per_prompt"] == 1
    assert "extra fingers, blurry, low quality" in pipeline.calls[0]["negative_prompt"]


def test_generate_smoke_multiple_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "batch.png"
    pipeline = DummyPipeline(images=[DummyImage(), DummyImage(), DummyImage()])
    install_fake_runtime(monkeypatch, pipeline)

    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: build_args(tmp_path, output=str(output), num_images=3, scheduler="ddim", seed=7),
    )

    generate.main()

    assert (tmp_path / "batch_01.png").exists()
    assert (tmp_path / "batch_02.png").exists()
    assert (tmp_path / "batch_03.png").exists()
    assert pipeline.scheduler.name == "ddim"


def test_negative_prompt_preset_only() -> None:
    prompt = generate.build_negative_prompt(None, "anime")
    assert "bad hands" in prompt


def test_negative_prompt_user_and_preset() -> None:
    prompt = generate.build_negative_prompt("extra fingers", "illustration")
    assert prompt.startswith("extra fingers, ")
