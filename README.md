# Stable Diffusion Project
Starter project for text-to-image generation using Python, PyTorch, and Hugging Face Diffusers.

## What this project includes
- `generate.py`: command-line image generation script
- `verify_pipeline.py`: smoke-test script for pipeline load + generation
- `requirements.txt`: Python package dependencies
- `outputs/`: generated image files (created automatically)

## Prerequisites
- Python 3.10 or newer
- `pip`
- Optional but recommended: CUDA-capable NVIDIA GPU

## Development environment setup
From the project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
Basic generation:

```bash
python generate.py \
  --prompt "a cinematic photo of a robot walking through neon rain at night" \
  --output outputs/robot.png
```

Generate multiple images:

```bash
python generate.py \
  --prompt "isometric cyberpunk city block" \
  --num-images 4 \
  --output outputs/city.png
```

Expected output files:
- `outputs/city_01.png`
- `outputs/city_02.png`
- `outputs/city_03.png`
- `outputs/city_04.png`

## Main options
- `--prompt` (required): generation prompt
- `--negative-prompt`: optional exclusions
- `--model`: model ID (default `runwayml/stable-diffusion-v1-5`)
- `--output`: output image path (default `output.png`)
- `--width` and `--height`: image size (must be divisible by 8)
- `--steps`: inference steps (default `30`)
- `--guidance-scale`: guidance strength (default `7.5`)
- `--seed`: deterministic seed (default `42`)
- `--num-images`: number of images per run (default `1`)
- `--cpu`: force CPU inference

## Development workflow
- Activate environment:
  - `source .venv/bin/activate`
- Run generation script while iterating:
  - `python generate.py --prompt "test prompt" --output outputs/test.png`
- Run smoke-test verification:
  - `python verify_pipeline.py --output outputs/pipeline_verify_smoke.png`
- Deactivate when done:
  - `deactivate`

## Smoke test results
Latest smoke-test run status: **PASS**

Command:
- `/home/jay/stable-diffusion/.venv/bin/python /home/jay/stable-diffusion/verify_pipeline.py --output /home/jay/stable-diffusion/outputs/pipeline_verify_smoke.png`

Result artifact:
- `outputs/pipeline_verify_smoke.png` (generated successfully)

Notes:
- Dependency import smoke test passed (`torch`, `diffusers`, `transformers`, `accelerate`).
- Non-blocking warnings were observed about `torchvision` fallback in `transformers`, but pipeline generation completed successfully.

## Troubleshooting
- GitHub/Hugging Face rate limiting or model access issues:
  - Authenticate with Hugging Face CLI and accept model license terms if required.
- Very slow inference:
  - Confirm GPU availability and avoid `--cpu`.
- Resolution errors:
  - Use width and height values that are multiples of 8.
