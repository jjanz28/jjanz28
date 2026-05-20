import sys
import types

import model_loader


def test_select_device_and_dtype_prefers_cpu_when_forced() -> None:
    fake_torch = types.SimpleNamespace(
        float16="float16",
        float32="float32",
        cuda=types.SimpleNamespace(is_available=lambda: True),
    )
    device, dtype = model_loader.select_device_and_dtype(fake_torch, force_cpu=True)
    assert device == "cpu"
    assert dtype == "float32"


def test_load_runtime_applies_scheduler_revision_and_safety_flags(monkeypatch) -> None:
    calls = []

    class FakeGenerator:
        def __init__(self, device: str) -> None:
            self.device = device
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    class FakeTorch:
        float16 = "float16"
        float32 = "float32"
        Generator = FakeGenerator
        cuda = types.SimpleNamespace(is_available=lambda: False)

    class FakePipeline:
        def __init__(self) -> None:
            self.scheduler = types.SimpleNamespace(config={"kind": "default"}, name="default")
            self.device = None

        def to(self, device: str):
            self.device = device
            return self

    class FakeStableDiffusionPipeline:
        @staticmethod
        def from_pretrained(model: str, **kwargs):
            calls.append((model, kwargs))
            return FakePipeline()

    class FakeEulerScheduler:
        @staticmethod
        def from_config(config: dict):
            return types.SimpleNamespace(config=config, name="euler")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        types.SimpleNamespace(
            StableDiffusionPipeline=FakeStableDiffusionPipeline,
            DDIMScheduler=object,
            EulerDiscreteScheduler=FakeEulerScheduler,
            EulerAncestralDiscreteScheduler=object,
            DPMSolverMultistepScheduler=object,
        ),
    )

    runtime = model_loader.load_stable_diffusion_runtime(
        model="fake/model",
        force_cpu=True,
        scheduler_name="euler",
        model_revision="v2",
        disable_safety_checker=True,
    )

    assert runtime.device == "cpu"
    assert runtime.dtype == "float32"
    assert runtime.pipe.device == "cpu"
    assert runtime.pipe.scheduler.name == "euler"
    assert calls == [
        (
            "fake/model",
            {
                "torch_dtype": "float32",
                "revision": "v2",
                "safety_checker": None,
                "requires_safety_checker": False,
            },
        )
    ]
