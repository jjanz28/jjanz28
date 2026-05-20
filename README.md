# Stable Diffusion Project
Text-to-image generation starter project using Python, PyTorch, and Hugging Face Diffusers.

## Prerequisites
- Python 3.10+
- `pip`
- Optional but recommended: NVIDIA GPU with CUDA support

## Project files
- `generate.py`: CLI script for image generation
- `requirements.txt`: Python dependencies
- `outputs/`: generated images (created automatically)

## Setup
From the project directory:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Quick start
```bash
python generate.py \
  --prompt "a cinematic photo of a robot walking through neon rain at night" \
  --output outputs/robot.png
```

## CLI options
- `--prompt` (required): text prompt for generation
- `--negative-prompt`: optional text describing what to avoid
- `--model`: Hugging Face model ID (default: `runwayml/stable-diffusion-v1-5`)
- `--output`: output file path (default: `output.png`)
- `--width` / `--height`: output resolution (must be multiples of 8)
- `--steps`: inference steps (default: `30`)
- `--guidance-scale`: prompt adherence strength (default: `7.5`)
- `--seed`: random seed for reproducibility (default: `42`)
- `--num-images`: number of images to generate (default: `1`)
- `--cpu`: force CPU inference (much slower)

## Examples
Single image:
```bash
python generate.py \
  --prompt "an astronaut riding a horse, ultra detailed" \
  --output outputs/astronaut.png
```

Multiple images:
```bash
python generate.py \
  --prompt "isometric cyberpunk city block" \
  --num-images 4 \
  --output outputs/city.png
```
This writes:
- `outputs/city_01.png`
- `outputs/city_02.png`
- `outputs/city_03.png`
- `outputs/city_04.png`

## Troubleshooting
- If model downloads fail or rate-limit, authenticate with Hugging Face (`huggingface-cli login`).
- If generation is slow, confirm GPU is available and avoid `--cpu`.
- If you get size errors, use `--width` and `--height` values divisible by 8.

## Notes
- Some models require license acceptance on Hugging Face before download.
- The first run can be slow because model files are downloaded and cached.
