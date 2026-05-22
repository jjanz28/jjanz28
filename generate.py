import argparse
import time
from pathlib import Path
from typing import Sequence

from model_loader import create_generator, load_stable_diffusion_runtime

NEGATIVE_PROMPT_PRESETS = {
    "none": None,
    "photo": "blurry, low quality, artifacts, overexposed, underexposed, watermark, text",
    "illustration": "blurry, muddy colors, bad anatomy, extra limbs, watermark, text",
    "anime": "low quality, blurry, bad hands, malformed face, watermark, text",
}

SCHEDULER_CHOICES = ("default", "ddim", "euler", "euler_a", "dpmpp_2m")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image with Stable Diffusion.")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate.")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--negative-preset",
        choices=tuple(NEGATIVE_PROMPT_PRESETS.keys()),
        default="none",
        help="Optional negative prompt preset.",
    )
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model ID.",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model revision/tag/commit to pin for reproducibility.",
    )
    parser.add_argument(
        "--scheduler",
        choices=SCHEDULER_CHOICES,
        default="default",
        help="Scheduler to use for inference.",
    )
    parser.add_argument("--output", default="output.png", help="Output image path.")
    parser.add_argument("--width", type=positive_int, default=512, help="Output width.")
    parser.add_argument("--height", type=positive_int, default=512, help="Output height.")
    parser.add_argument(
        "--steps",
        type=positive_int,
        default=30,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=positive_float,
        default=7.5,
        help="Classifier-free guidance scale.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--num-images",
        type=positive_int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (slow).",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable memory-saving pipeline options when available.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_negative_prompt(user_negative_prompt: str | None, preset_name: str) -> str | None:
    preset_prompt = NEGATIVE_PROMPT_PRESETS[preset_name]
    if user_negative_prompt and preset_prompt:
        return f"{user_negative_prompt}, {preset_prompt}"
    if user_negative_prompt:
        return user_negative_prompt
    return preset_prompt



def format_device_diagnostics(torch, device: str, dtype) -> str:
    diagnostics = [
        f"device={device}",
        f"dtype={dtype}",
        f"cuda_available={torch.cuda.is_available()}",
    ]
    if device == "cuda":
        diagnostics.append(f"gpu={torch.cuda.get_device_name(0)}")
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        diagnostics.append(f"vram_gb={total_memory_gb:.1f}")
    return "Diagnostics: " + ", ".join(diagnostics)


def enable_low_memory_mode(pipe) -> list[str]:
    applied = []
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
        applied.append("attention_slicing")
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
        applied.append("vae_slicing")
    return applied


def main() -> None:

    args = parse_args()
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1.")

    runtime = load_stable_diffusion_runtime(
        model=args.model,
        model_revision=args.model_revision,
        force_cpu=args.cpu,
        scheduler_name=args.scheduler,
        disable_safety_checker=True,
    )
    torch = runtime.torch
    pipe = runtime.pipe
    device = runtime.device
    dtype = runtime.dtype
    print(f"Loading model '{args.model}' on {device}...")
    negative_prompt = build_negative_prompt(args.negative_prompt, args.negative_preset)
    print(format_device_diagnostics(torch, device, dtype))

    if args.low_memory:
        memory_features = enable_low_memory_mode(pipe)
        if memory_features:
            print(f"Low-memory mode enabled: {', '.join(memory_features)}")
        else:
            print("Low-memory mode requested, but no memory-saving features were available.")

    revision_label = args.model_revision or "default"
    print(
        f"Run config: seed={args.seed}, scheduler={args.scheduler}, "
        f"model={args.model}@{revision_label}"
    )

    generator = create_generator(torch, device=device, seed=args.seed)
    print("Generating images...")
    start_time = time.perf_counter()
    result = pipe(
        prompt=args.prompt,
        negative_prompt=negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
    )
    elapsed_seconds = time.perf_counter() - start_time
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_images == 1:
        result.images[0].save(output_path)
        print(f"Saved image to: {output_path.resolve()}")
        print(
            f"Runtime summary: elapsed={elapsed_seconds:.2f}s, "
            f"images={len(result.images)}, resolution={args.width}x{args.height}, "
            f"scheduler={args.scheduler}, low_memory={args.low_memory}"
        )
        return

    stem = output_path.stem
    suffix = output_path.suffix or ".png"
    for idx, image in enumerate(result.images, start=1):
        numbered_path = output_path.with_name(f"{stem}_{idx:02d}{suffix}")
        image.save(numbered_path)
        print(f"Saved image to: {numbered_path.resolve()}")

    print(
        f"Runtime summary: elapsed={elapsed_seconds:.2f}s, "
        f"images={len(result.images)}, resolution={args.width}x{args.height}, "
        f"scheduler={args.scheduler}, low_memory={args.low_memory}"
    )


if __name__ == "__main__":
    main()
