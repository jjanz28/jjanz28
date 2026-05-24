# Stable Diffusion CLI Project
Local text-to-image generation project built with Python, PyTorch, and Hugging Face Diffusers.

## Docs Index
- `docs/monitoring.md` — AMD CPU monitoring setup, persistence, and reboot verification

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
### Use a built-in preset
```bash
python generate.py \
  --prompt "ancient floating citadel above glowing forests" \
  --preset fantasy-vivid \
  --output outputs/fantasy_castle.png
```

## Key CLI options
- `--prompt`: text prompt (required unless `--from-metadata` provides one)
- `--preset`: named preset from `config/presets.json`
- `--presets-file`: optional custom preset file path
- `--list-presets`: print available preset names and exit
- `--from-metadata`: load generation defaults from a metadata JSON file
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
- `--save-metadata` / `--no-save-metadata`: enable/disable metadata sidecar output
- `--metadata-file`: explicit metadata output path
## Preset system
- Preset definitions live in `config/presets.json`.
- Built-in presets:
  - `fantasy-vivid`
  - `portrait-photo`
  - `anime-clean`
  - `realistic`
- CLI behavior:
  - Preset values become defaults.
  - Explicit CLI flags override preset values for the same fields.
- GUI behavior:
  - Choose a preset from the **Preset** dropdown.
  - Click **Apply** to load preset values.
  - Click **Save** to store your current settings as a new preset.

## Metadata and reproducibility
- Metadata is saved by default for each generation run.
- Default metadata paths:
  - single image output (`outputs/image.png`) → `outputs/image.json`
  - multi-image output (`outputs/batch.png`) → `outputs/batch_run.json`
- Disable metadata output with `--no-save-metadata`.

Re-run from metadata:
```bash
python generate.py \
  --from-metadata outputs/portrait.json \
  --output outputs/portrait_rerun.png
```

Reuse metadata but override selected values:
```bash
python generate.py \
  --from-metadata outputs/portrait.json \
  --steps 50 \
  --guidance-scale 9.0 \
  --output outputs/portrait_variant.png
```

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
 
# Sudo Convenience Aliases
This machine is configured with Bash aliases that automatically prefix common admin commands with `sudo`.

## Alias file
Aliases are defined in:

- `~/.bash_aliases`

They are loaded from:

- `~/.bashrc`

## Configured aliases
- `apt='sudo apt'`
- `apt-get='sudo apt-get'`
- `apt-cache='sudo apt-cache'`
- `dpkg='sudo dpkg'`
- `systemctl='sudo systemctl'`
- `service='sudo service'`
- `journalctl='sudo journalctl'`
- `ufw='sudo ufw'`
- `mount='sudo mount'`
- `umount='sudo umount'`
- `snap='sudo snap'`

## Activate in current shell
Run:

```bash
source ~/.bashrc
```

## Verify aliases
Run:

```bash
alias apt
alias systemctl
```

## Safely add or remove aliases
### Add an alias
1. Edit `~/.bash_aliases`.
2. Add one alias per line, for example:

```bash
alias ll='ls -alF'
```

3. Reload shell config:

```bash
source ~/.bashrc
```

4. Verify:

```bash
alias ll
```

### Remove an alias
- Permanent removal:
  1. Delete the alias line from `~/.bash_aliases`.
  2. Run `source ~/.bashrc`.
- Temporary removal (current shell only):

```bash
unalias ll
```

### Safety tips
- Avoid aliasing critical commands to risky behavior (for example `rm` with destructive flags).
- After changes, always test with `alias <name>` before relying on the alias.
- If an alias interferes with a command, run the real command with a leading backslash, e.g. `\ls`.

# AMD CPU Monitoring
Monitoring setup and persistence details were moved to:

- `docs/monitoring.md`
