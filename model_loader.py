from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LoadedModelRuntime:
    torch: Any
    pipe: Any
    device: str
    dtype: Any


def select_device_and_dtype(torch_module: Any, force_cpu: bool) -> tuple[str, Any]:
    use_cuda = torch_module.cuda.is_available() and not force_cpu
    if use_cuda:
        return "cuda", torch_module.float16
    return "cpu", torch_module.float32


def configure_scheduler(pipe: Any, scheduler_name: str, scheduler_map: dict[str, Any]) -> None:
    if scheduler_name == "default":
        return
    scheduler_cls = scheduler_map[scheduler_name]
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)


def create_generator(torch_module: Any, device: str, seed: int) -> Any:
    return torch_module.Generator(device=device).manual_seed(seed)


def load_stable_diffusion_runtime(
    *,
    model: str,
    force_cpu: bool,
    scheduler_name: str = "default",
    model_revision: str | None = None,
    disable_safety_checker: bool = False,
) -> LoadedModelRuntime:
    import torch
    from diffusers import StableDiffusionPipeline

    device, dtype = select_device_and_dtype(torch, force_cpu=force_cpu)
    pipe_kwargs: dict[str, Any] = {"torch_dtype": dtype}
    if model_revision:
        pipe_kwargs["revision"] = model_revision
    if disable_safety_checker:
        pipe_kwargs["safety_checker"] = None
        pipe_kwargs["requires_safety_checker"] = False

    pipe = StableDiffusionPipeline.from_pretrained(model, **pipe_kwargs)
    pipe = pipe.to(device)

    if scheduler_name != "default":
        from diffusers import (
            DDIMScheduler,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
        )

        scheduler_map = {
            "ddim": DDIMScheduler,
            "euler": EulerDiscreteScheduler,
            "euler_a": EulerAncestralDiscreteScheduler,
            "dpmpp_2m": DPMSolverMultistepScheduler,
        }
        configure_scheduler(pipe, scheduler_name, scheduler_map)
    return LoadedModelRuntime(torch=torch, pipe=pipe, device=device, dtype=dtype)
