import argparse
from pathlib import Path

from model_loader import create_generator, load_stable_diffusion_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test that a Stable Diffusion pipeline can load and generate."
    )
    parser.add_argument(
        "--model",
        default="hf-internal-testing/tiny-stable-diffusion-pipe",
        help="Hugging Face model ID to test.",
    )
    parser.add_argument(
        "--prompt",
        default="a small blue cube on a table",
        help="Prompt used for the verification generation.",
    )
    parser.add_argument(
        "--output",
        default="outputs/pipeline_verify.png",
        help="Path to save generated verification image.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2,
        help="Number of inference steps for smoke test.",
    )
    parser.add_argument("--width", type=int, default=128, help="Output width.")
    parser.add_argument("--height", type=int, default=128, help="Output height.")
    parser.add_argument("--seed", type=int, default=123, help="Random seed.")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")

    runtime = load_stable_diffusion_runtime(
        model=args.model,
        force_cpu=args.cpu,
        disable_safety_checker=True,
    )
    torch = runtime.torch
    pipe = runtime.pipe
    device = runtime.device
    print(f"Loading verification model '{args.model}' on {device}...")
    pipe.set_progress_bar_config(disable=True)
    generator = create_generator(torch, device=device, seed=args.seed)
    result = pipe(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        generator=generator,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(output_path)
    print(f"VERIFIED: pipeline generated image at {output_path.resolve()}")


if __name__ == "__main__":
    main()
