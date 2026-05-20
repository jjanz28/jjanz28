# Stable Diffusion Starter
Simple Python starter project for text-to-image generation with Hugging Face Diffusers.

## Requirements
- Python 3.10+
- `pip`
- (Recommended) NVIDIA GPU with CUDA for fast generation

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Generate an image
```bash
python generate.py \
  --prompt "a cinematic photo of a robot walking through neon rain at night" \
  --output outputs/robot.png
```

## Useful options
- `--model`: Hugging Face model id (default: `runwayml/stable-diffusion-v1-5`)
- `--negative-prompt`: discourage unwanted elements
- `--steps`: inference steps (higher is slower, often better quality)
- `--guidance-scale`: prompt adherence strength
- `--width` / `--height`: output size (must be multiples of 8)
- `--seed`: reproducibility
- `--num-images`: images per prompt
- `--cpu`: force CPU (much slower)

## Example: multiple images
```bash
python generate.py \
  --prompt "isometric cyberpunk city block" \
  --num-images 4 \
  --output outputs/city.png
```
This saves:
- `outputs/city_01.png`
- `outputs/city_02.png`
- `outputs/city_03.png`
- `outputs/city_04.png`

## Notes
- Some models may require Hugging Face authentication and license acceptance.
- CPU generation is supported but can be very slow.
