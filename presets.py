from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_PRESETS_PATH = Path(__file__).resolve().parent / "config" / "presets.json"
SCHEDULER_CHOICES = {"default", "ddim", "euler", "euler_a", "dpmpp_2m"}
PRESET_KEYS = {
    "model",
    "model_revision",
    "scheduler",
    "negative_prompt",
    "width",
    "height",
    "steps",
    "guidance_scale",
    "seed",
    "num_images",
    "cpu",
    "low_memory",
}
POSITIVE_INT_KEYS = {"width", "height", "steps", "num_images"}
BOOLEAN_KEYS = {"cpu", "low_memory"}


def _resolve_presets_path(presets_file: str | Path | None = None) -> Path:
    if presets_file is None:
        return DEFAULT_PRESETS_PATH
    return Path(presets_file).expanduser()


def _validate_preset_name(name: str) -> str:
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("Preset name cannot be empty.")
    return cleaned


def _validate_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ValueError(f"Preset field '{name}' must be an integer.")
    if value < 1:
        raise ValueError(f"Preset field '{name}' must be >= 1.")
    return value


def _validate_positive_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Preset field '{name}' must be a number.")
    result = float(value)
    if result <= 0:
        raise ValueError(f"Preset field '{name}' must be > 0.")
    return result


def _normalize_preset_values(values: dict[str, Any]) -> dict[str, Any]:
    unknown_keys = sorted(set(values) - PRESET_KEYS)
    if unknown_keys:
        joined = ", ".join(unknown_keys)
        raise ValueError(f"Unsupported preset field(s): {joined}")

    normalized = dict(values)
    if "scheduler" in normalized:
        scheduler = normalized["scheduler"]
        if not isinstance(scheduler, str):
            raise ValueError("Preset field 'scheduler' must be a string.")
        if scheduler not in SCHEDULER_CHOICES:
            allowed = ", ".join(sorted(SCHEDULER_CHOICES))
            raise ValueError(f"Preset scheduler must be one of: {allowed}")

    if "model" in normalized and not isinstance(normalized["model"], str):
        raise ValueError("Preset field 'model' must be a string.")
    if "model_revision" in normalized and normalized["model_revision"] is not None:
        if not isinstance(normalized["model_revision"], str):
            raise ValueError("Preset field 'model_revision' must be a string or null.")
    if "negative_prompt" in normalized and normalized["negative_prompt"] is not None:
        if not isinstance(normalized["negative_prompt"], str):
            raise ValueError("Preset field 'negative_prompt' must be a string or null.")

    for key in POSITIVE_INT_KEYS:
        if key in normalized:
            normalized[key] = _validate_positive_int(key, normalized[key])

    if "seed" in normalized and not isinstance(normalized["seed"], int):
        raise ValueError("Preset field 'seed' must be an integer.")
    if "guidance_scale" in normalized:
        normalized["guidance_scale"] = _validate_positive_float(
            "guidance_scale", normalized["guidance_scale"]
        )

    for key in BOOLEAN_KEYS:
        if key in normalized and not isinstance(normalized[key], bool):
            raise ValueError(f"Preset field '{key}' must be true/false.")

    return normalized


def load_presets(presets_file: str | Path | None = None) -> dict[str, dict[str, Any]]:
    path = _resolve_presets_path(presets_file)
    if not path.exists():
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Preset file must contain a JSON object of named presets.")

    presets: dict[str, dict[str, Any]] = {}
    for raw_name, raw_values in data.items():
        if not isinstance(raw_name, str):
            raise ValueError("Preset names must be strings.")
        if not isinstance(raw_values, dict):
            raise ValueError(f"Preset '{raw_name}' must be a JSON object.")
        name = _validate_preset_name(raw_name)
        presets[name] = _normalize_preset_values(raw_values)
    return presets


def list_presets(presets_file: str | Path | None = None) -> list[str]:
    return sorted(load_presets(presets_file).keys())


def get_preset(name: str | None, presets_file: str | Path | None = None) -> dict[str, Any]:
    if not name:
        return {}
    cleaned = _validate_preset_name(name)
    presets = load_presets(presets_file)
    if cleaned not in presets:
        raise KeyError(f"Unknown preset: {cleaned}")
    return dict(presets[cleaned])


def save_preset(
    name: str, values: dict[str, Any], presets_file: str | Path | None = None
) -> dict[str, Any]:
    path = _resolve_presets_path(presets_file)
    cleaned_name = _validate_preset_name(name)
    normalized_values = _normalize_preset_values(values)
    presets = load_presets(path)
    presets[cleaned_name] = normalized_values
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(presets, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dict(normalized_values)
