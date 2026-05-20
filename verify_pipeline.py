import argparse
from pathlib import Path


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
    import torch
    from diffusers import StableDiffusionPipeline
    args = parse_args()
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")
    if args.steps < 1:
        raise ValueError("--steps must be >= 1.")

    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32

    print(f"Loading verification model '{args.model}' on {device}...")
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=device).manual_seed(args.seed)
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
