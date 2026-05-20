# Stable Diffusion CLI Project
Local text-to-image generation project built with Python, PyTorch, and Hugging Face Diffusers.

## Project structure
```text
stable-diffusion/
├── generate.py                # Main CLI for image generation
├── model_loader.py            # Shared model/runtime loading helpers
├── verify_pipeline.py         # Smoke test for pipeline load + generation
├── gui_app.py                 # Desktop launcher UI that wraps the CLI
├── requirements.txt           # Runtime dependencies
├── requirements-dev.txt       # Lint/test dependencies
├── pyproject.toml             # Tooling config (pytest + ruff)
├── tests/                     # Unit tests
├── outputs/                   # Generated images
└── .github/workflows/ci.yml   # CI checks
```

## Setup instructions
### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional dev tools:
```bash
pip install -r requirements-dev.txt
```

## CLI usage examples
### Basic generation
```bash
python generate.py \
  --prompt "a cinematic photo of a robot in neon rain" \
  --output outputs/robot.png
```

### Generate multiple images
```bash
python generate.py \
  --prompt "isometric cyberpunk city block" \
  --num-images 4 \
  --output outputs/city.png
```

This creates:
- `outputs/city_01.png`
- `outputs/city_02.png`
- `outputs/city_03.png`
- `outputs/city_04.png`

### Use scheduler, preset, and reproducible seed
```bash
python generate.py \
  --prompt "portrait photo of a traveler in golden hour light" \
  --negative-preset photo \
  --scheduler euler_a \
  --seed 1234 \
  --model-revision main \
  --output outputs/portrait.png
```

### Low-memory mode
```bash
python generate.py \
  --prompt "detailed skyline at dusk" \
  --scheduler dpmpp_2m \
  --low-memory \
  --output outputs/city_low_mem.png
```

## Key CLI options
- `--prompt` (required): text prompt
- `--negative-prompt`: custom negative prompt
- `--negative-preset`: `none`, `photo`, `illustration`, `anime`
- `--model`: model ID (default: `runwayml/stable-diffusion-v1-5`)
- `--model-revision`: model tag/branch/commit
- `--scheduler`: `default`, `ddim`, `euler`, `euler_a`, `dpmpp_2m`
- `--output`: output file path
- `--width`, `--height`: image size (must be multiples of 8)
- `--steps`: inference steps
- `--guidance-scale`: classifier-free guidance strength
- `--seed`: reproducible random seed
- `--num-images`: images per prompt
- `--cpu`: force CPU inference
- `--low-memory`: enable memory-saving options when available

## Verification and testing
Run a quick smoke check:
```bash
python verify_pipeline.py --output outputs/pipeline_verify_smoke.png
```

Run lint + tests:
```bash
ruff check .
pytest
```
