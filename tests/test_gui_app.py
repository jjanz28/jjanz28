from gui_app import DEFAULT_MODEL, DEFAULT_MODEL_LABEL, resolve_model_choice


def test_resolve_model_choice_returns_default_when_empty() -> None:
    assert resolve_model_choice("   ") == DEFAULT_MODEL


def test_resolve_model_choice_maps_dropdown_label() -> None:
    assert resolve_model_choice(DEFAULT_MODEL_LABEL) == DEFAULT_MODEL


def test_resolve_model_choice_allows_custom_model_id() -> None:
    custom_model = "stabilityai/stable-diffusion-xl-base-1.0"
    assert resolve_model_choice(custom_model) == custom_model
