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

Quality-control example:

```bash
python generate.py \
  --prompt "portrait photo of a traveler in golden hour light" \
  --negative-preset photo \
  --scheduler euler_a \
  --seed 1234 \
  --model-revision main \
  --output outputs/portrait.png
```

Low-memory example:

```bash
python generate.py \
  --prompt "detailed city skyline at dusk" \
  --scheduler dpmpp_2m \
  --low-memory \
  --output outputs/city_low_mem.png
```

## Main options
- `--prompt` (required): generation prompt
- `--negative-prompt`: optional exclusions
- `--negative-preset`: preset exclusions (`none`, `photo`, `illustration`, `anime`)
- `--model`: model ID (default `runwayml/stable-diffusion-v1-5`)
- `--model-revision`: optional model tag/branch/commit pin for reproducibility
- `--scheduler`: sampler scheduler (`default`, `ddim`, `euler`, `euler_a`, `dpmpp_2m`)
- `--output`: output image path (default `output.png`)
- `--width` and `--height`: image size (must be divisible by 8)
- `--steps`: inference steps (default `30`)
- `--guidance-scale`: guidance strength (default `7.5`)
- `--seed`: deterministic seed (default `42`)
- `--num-images`: number of images per run (default `1`)
- `--cpu`: force CPU inference
- `--low-memory`: enable memory-saving pipeline options when available

## Reproducibility guidance
- Pin model identity with `--model` and `--model-revision`.
- Set a fixed `--seed` to reproduce the same prompt run.
- Keep `--scheduler`, `--steps`, resolution, and guidance scale constant across runs.
- Save generated outputs with deterministic filenames per run configuration.

## Runtime diagnostics and summary
Each generation run now prints:
- Device diagnostics (device, dtype, CUDA availability, and GPU memory hints on CUDA runs)
- Effective run configuration (seed, scheduler, and model revision)
- Progress marker before generation starts
- Runtime summary with elapsed time, image count, resolution, scheduler, and low-memory status

## Development workflow
- Activate environment:
  - `source .venv/bin/activate`
- Run generation script while iterating:
  - `python generate.py --prompt "test prompt" --output outputs/test.png`
- Run smoke-test verification:
  - `python verify_pipeline.py --output outputs/pipeline_verify_smoke.png`
- Run lint + tests:
  - `ruff check .`
  - `pytest`
- Deactivate when done:
  - `deactivate`

## Testing framework
- Test runner: `pytest`
- Linter: `ruff`
- Test location: `tests/`
- CI workflow: `.github/workflows/ci.yml`

Install dev tooling:

```bash
pip install -r requirements-dev.txt
```

## Smoke test results
Latest smoke-test run status: **PASS**

Command:
- `/home/jay/stable-diffusion/.venv/bin/python /home/jay/stable-diffusion/verify_pipeline.py --output /home/jay/stable-diffusion/outputs/pipeline_verify_smoke.png`

Result artifact:
- `outputs/pipeline_verify_smoke.png` (generated successfully)

Notes:
- Dependency import smoke test passed (`torch`, `diffusers`, `transformers`, `accelerate`).
- Non-blocking warnings were observed about `torchvision` fallback in `transformers`, but pipeline generation completed successfully.

## Next phase development roadmap
### Milestone 1: reliability and test coverage
- Add automated smoke tests for both `generate.py` and `verify_pipeline.py`.
- Add argument validation tests (resolution rules, step bounds, output path behavior).
- Add CI checks for linting and test execution on push/PR.

### Milestone 2: generation quality and controls
- Add sampler/scheduler selection flags and default presets.
- Add optional negative-prompt presets for common artifacts.
- Add reproducibility docs for seed strategy and model/version pinning.

### Milestone 3: performance and usability
- Add device diagnostics output (`cuda` availability, dtype, memory hints).
- Add optional low-memory mode and better CPU fallback messaging.
- Add progress and runtime summary output for generation runs.

### Milestone 4: packaging and deployment readiness
- Add a `Makefile`/task runner for setup, test, and smoke-test commands.
- Add containerized runtime option for reproducible environments.
- Document release checklist (dependency lock refresh, smoke test, changelog).

### Exit criteria for next phase
- CI is green on `main`.
- Smoke test artifacts can be generated consistently from a clean environment.
- README reflects production-ready setup, usage, and troubleshooting paths.

## Troubleshooting
- GitHub/Hugging Face rate limiting or model access issues:
  - Authenticate with Hugging Face CLI and accept model license terms if required.
- Very slow inference:
  - Confirm GPU availability and avoid `--cpu`.
- Resolution errors:
  - Use width and height values that are multiples of 8.
