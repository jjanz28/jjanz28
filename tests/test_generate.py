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
        def from_pretrained(model: str, torch_dtype=None):
            from_pretrained_calls.append((model, torch_dtype))
            return pipeline

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(StableDiffusionPipeline=FakeStableDiffusionPipeline),
    )
    return from_pretrained_calls


def test_generate_rejects_invalid_resolution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="test",
            negative_prompt=None,
            model="dummy",
            output=str(tmp_path / "out.png"),
            width=513,
            height=512,
            steps=1,
            guidance_scale=7.5,
            seed=1,
            num_images=1,
            cpu=True,
        ),
    )

    with pytest.raises(ValueError, match="multiples of 8"):
        generate.main()


def test_generate_rejects_invalid_num_images(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="test",
            negative_prompt=None,
            model="dummy",
            output=str(tmp_path / "out.png"),
            width=512,
            height=512,
            steps=1,
            guidance_scale=7.5,
            seed=1,
            num_images=0,
            cpu=True,
        ),
    )

    with pytest.raises(ValueError, match="--num-images must be >= 1"):
        generate.main()


def test_generate_smoke_single_image(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "single.png"
    pipeline = DummyPipeline(images=[DummyImage()])
    from_pretrained_calls = install_fake_runtime(monkeypatch, pipeline)

    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="test prompt",
            negative_prompt="bad",
            model="fake/model",
            output=str(output),
            width=512,
            height=512,
            steps=2,
            guidance_scale=6.0,
            seed=42,
            num_images=1,
            cpu=True,
        ),
    )

    generate.main()

    assert output.exists()
    assert pipeline.device == "cpu"
    assert from_pretrained_calls == [("fake/model", "float32")]
    assert pipeline.calls[0]["num_images_per_prompt"] == 1


def test_generate_smoke_multiple_images(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "batch.png"
    pipeline = DummyPipeline(images=[DummyImage(), DummyImage(), DummyImage()])
    install_fake_runtime(monkeypatch, pipeline)

    monkeypatch.setattr(
        generate,
        "parse_args",
        lambda: argparse.Namespace(
            prompt="test prompt",
            negative_prompt=None,
            model="fake/model",
            output=str(output),
            width=512,
            height=512,
            steps=2,
            guidance_scale=7.5,
            seed=7,
            num_images=3,
            cpu=True,
        ),
    )

    generate.main()

    assert (tmp_path / "batch_01.png").exists()
    assert (tmp_path / "batch_02.png").exists()
    assert (tmp_path / "batch_03.png").exists()
