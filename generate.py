import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with Stable Diffusion.")
    parser.add_argument("--prompt", required=True, help="Text prompt to generate.")
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="Optional negative prompt.",
    )
    parser.add_argument(
        "--model",
        default="runwayml/stable-diffusion-v1-5",
        help="Hugging Face model ID.",
    )
    parser.add_argument("--output", default="output.png", help="Output image path.")
    parser.add_argument("--width", type=int, default=512, help="Output width.")
    parser.add_argument("--height", type=int, default=512, help="Output height.")
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="Number of inference steps.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
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
        type=int,
        default=1,
        help="Number of images to generate.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference (slow).",
    )
    return parser.parse_args()


def main() -> None:
    import torch
    from diffusers import StableDiffusionPipeline
    args = parse_args()
    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError("Width and height must be multiples of 8.")
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1.")
    use_cuda = torch.cuda.is_available() and not args.cpu
    device = "cuda" if use_cuda else "cpu"
    dtype = torch.float16 if use_cuda else torch.float32
    print(f"Loading model '{args.model}' on {device}...")

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    result = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        width=args.width,
        height=args.height,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        num_images_per_prompt=args.num_images,
        generator=generator,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.num_images == 1:
        result.images[0].save(output_path)
        print(f"Saved image to: {output_path.resolve()}")
        return

    stem = output_path.stem
    suffix = output_path.suffix or ".png"
    for idx, image in enumerate(result.images, start=1):
        numbered_path = output_path.with_name(f"{stem}_{idx:02d}{suffix}")
        image.save(numbered_path)
        print(f"Saved image to: {numbered_path.resolve()}")


if __name__ == "__main__":
    main()
