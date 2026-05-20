import argparse
import sys
import types
from pathlib import Path

import pytest

import verify_pipeline


class DummyImage:
    def save(self, path: str) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"image-bytes")


class DummyPipeline:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.device: str | None = None
        self.progress_disabled: bool | None = None

    def to(self, device: str) -> "DummyPipeline":
        self.device = device
        return self

    def set_progress_bar_config(self, disable: bool) -> None:
        self.progress_disabled = disable

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return types.SimpleNamespace(images=[DummyImage()])


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

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(StableDiffusionPipeline=FakeStableDiffusionPipeline),
    )
    return from_pretrained_calls


def test_verify_rejects_invalid_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        verify_pipeline,
        "parse_args",
        lambda: argparse.Namespace(
            model="fake/model",
            prompt="test",
            output=str(tmp_path / "verify.png"),
            steps=1,
            width=129,
            height=128,
            seed=1,
            cpu=True,
        ),
    )

    with pytest.raises(ValueError, match="multiples of 8"):
        verify_pipeline.main()


def test_verify_rejects_invalid_steps(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        verify_pipeline,
        "parse_args",
        lambda: argparse.Namespace(
            model="fake/model",
            prompt="test",
            output=str(tmp_path / "verify.png"),
            steps=0,
            width=128,
            height=128,
            seed=1,
            cpu=True,
        ),
    )

    with pytest.raises(ValueError, match="--steps must be >= 1"):
        verify_pipeline.main()


def test_verify_pipeline_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "verify.png"
    pipeline = DummyPipeline()
    from_pretrained_calls = install_fake_runtime(monkeypatch, pipeline)

    monkeypatch.setattr(
        verify_pipeline,
        "parse_args",
        lambda: argparse.Namespace(
            model="fake/model",
            prompt="test prompt",
            output=str(output),
            steps=2,
            width=128,
            height=128,
            seed=123,
            cpu=True,
        ),
    )

    verify_pipeline.main()

    assert output.exists()
    assert pipeline.device == "cpu"
    assert pipeline.progress_disabled is True
    assert pipeline.calls[0]["num_inference_steps"] == 2
    assert from_pretrained_calls == [
        (
            "fake/model",
            {
                "torch_dtype": "float32",
                "safety_checker": None,
                "requires_safety_checker": False,
            },
        )
    ]
