import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from model_loader import create_generator, load_stable_diffusion_runtime
from presets import DEFAULT_PRESETS_PATH, get_preset, list_presets

NEGATIVE_PROMPT_PRESETS = {
    "none": None,
    "photo": "blurry, low quality, artifacts, overexposed, underexposed, watermark, text",
    "illustration": "blurry, muddy colors, bad anatomy, extra limbs, watermark, text",
    "anime": "low quality, blurry, bad hands, malformed face, watermark, text",
}

SCHEDULER_CHOICES = ("default", "ddim", "euler", "euler_a", "dpmpp_2m")
METADATA_PARAMETER_FIELDS = (
    "prompt",
    "negative_prompt",
    "negative_preset",
    "model",
    "model_revision",
    "scheduler",
    "width",
    "height",
    "steps",
    "guidance_scale",
    "seed",
    "num_images",
    "cpu",
    "low_memory",
)


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


def load_metadata_defaults(metadata_file: str) -> dict:
    metadata_path = Path(metadata_file).expanduser()
    if not metadata_path.exists():
        raise ValueError(f"Metadata file not found: {metadata_path}")

    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Metadata file must be a JSON object.")

    raw_parameters = data.get("parameters", data)
    if not isinstance(raw_parameters, dict):
        raise ValueError("Metadata file must contain a 'parameters' object.")

    defaults: dict = {}
    for key in METADATA_PARAMETER_FIELDS:
        if key in raw_parameters:
            defaults[key] = raw_parameters[key]
    return defaults


def resolve_metadata_output_path(
    output_path: Path, num_images: int, metadata_file: str | None
) -> Path:
    if metadata_file:
        return Path(metadata_file).expanduser()
    if num_images == 1:
        return output_path.with_suffix(".json")
    return output_path.with_name(f"{output_path.stem}_run.json")


def write_generation_metadata(
    *,
    metadata_path: Path,
    args: argparse.Namespace,
    negative_prompt: str | None,
    output_paths: list[Path],
    elapsed_seconds: float,
) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed_seconds, 4),
        "parameters": {
            "prompt": args.prompt,
            "negative_prompt": negative_prompt,
            "negative_preset": args.negative_preset,
            "model": args.model,
            "model_revision": args.model_revision,
            "scheduler": args.scheduler,
            "width": args.width,
            "height": args.height,
            "steps": args.steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "num_images": args.num_images,
            "cpu": args.cpu,
            "low_memory": args.low_memory,
        },
        "outputs": [str(path.resolve()) for path in output_paths],
    }
    metadata_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an image with Stable Diffusion.")
    parser.add_argument("--prompt", default=None, help="Text prompt to generate.")
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
        "--preset",
        default=None,
        help="Optional preset name loaded from the presets file.",
    )
    parser.add_argument(
        "--presets-file",
        default=str(DEFAULT_PRESETS_PATH),
        help="Path to JSON preset definitions.",
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List available preset names and exit.",
    )
    parser.add_argument(
        "--from-metadata",
        default=None,
        help="Optional metadata JSON file to reuse generation parameters.",
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
    parser.add_argument(
        "--save-metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write metadata JSON for reproducibility (default: enabled).",
    )
    parser.add_argument(
        "--metadata-file",
        default=None,
        help="Optional output path for metadata JSON.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    preset_probe = argparse.ArgumentParser(add_help=False)
    preset_probe.add_argument("--preset", default=None)
    preset_probe.add_argument("--presets-file", default=str(DEFAULT_PRESETS_PATH))
    preset_probe.add_argument("--list-presets", action="store_true")
    preset_probe.add_argument("--from-metadata", default=None)
    probe_args, _ = preset_probe.parse_known_args(argv)

    try:
        preset_names = list_presets(probe_args.presets_file)
        preset_defaults = get_preset(probe_args.preset, probe_args.presets_file)
        metadata_defaults = (
            load_metadata_defaults(probe_args.from_metadata)
            if probe_args.from_metadata
            else {}
        )
    except (ValueError, KeyError) as err:
        parser.error(str(err))

    if probe_args.list_presets:
        if preset_names:
            print("\n".join(preset_names))
        else:
            print("No presets configured.")
        raise SystemExit(0)

    parser.set_defaults(**preset_defaults)
    parser.set_defaults(**metadata_defaults)
    return parser.parse_args(argv)


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
    if not args.prompt:
        raise ValueError("Prompt is required (provide --prompt or --from-metadata).")
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
    saved_paths: list[Path] = []

    if args.num_images == 1:
        result.images[0].save(output_path)
        print(f"Saved image to: {output_path.resolve()}")
        saved_paths = [output_path]
        print(
            f"Runtime summary: elapsed={elapsed_seconds:.2f}s, "
            f"images={len(result.images)}, resolution={args.width}x{args.height}, "
            f"scheduler={args.scheduler}, low_memory={args.low_memory}"
        )
    else:
        stem = output_path.stem
        suffix = output_path.suffix or ".png"
        for idx, image in enumerate(result.images, start=1):
            numbered_path = output_path.with_name(f"{stem}_{idx:02d}{suffix}")
            image.save(numbered_path)
            saved_paths.append(numbered_path)
            print(f"Saved image to: {numbered_path.resolve()}")

        print(
            f"Runtime summary: elapsed={elapsed_seconds:.2f}s, "
            f"images={len(result.images)}, resolution={args.width}x{args.height}, "
            f"scheduler={args.scheduler}, low_memory={args.low_memory}"
        )

    if args.save_metadata:
        metadata_path = resolve_metadata_output_path(
            output_path=output_path,
            num_images=args.num_images,
            metadata_file=args.metadata_file,
        )
        write_generation_metadata(
            metadata_path=metadata_path,
            args=args,
            negative_prompt=negative_prompt,
            output_paths=saved_paths,
            elapsed_seconds=elapsed_seconds,
        )
        print(f"Saved metadata to: {metadata_path.resolve()}")


if __name__ == "__main__":
    main()
