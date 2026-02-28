"""Dual interface helpers for a shared analysis configuration.

This module implements a "One Engine, Two Cockpits" pattern:
- Guided mode asks for a small set of biological inputs.
- Expert mode exposes the full configuration for direct editing.
- Both modes operate on the same universal configuration object.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

SCHEMA_VERSION = 1

SESSION_KEY_UI_MODE = "ui_mode"
SESSION_KEY_UNIVERSAL_CONFIG = "universal_session_config"
SESSION_KEY_GUIDED_INPUTS = "guided_inputs"
SESSION_KEY_PROTOCOL_REGISTRY = "guided_protocol_registry"

NOVICE_CHAT_SYSTEM_PROMPT = (
    "You are an empathetic biophysics tutor. The user is a wet-lab biologist. "
    "Explain metrics with biological analogies and avoid raw math unless asked."
)

EXPERT_CHAT_SYSTEM_PROMPT = (
    "You are a post-doctoral computational biophysicist. Provide rigorous "
    "mathematical explanations, API-level scripting support, and methods prose."
)


class UIMode(str, Enum):
    """Top-level user workspace mode."""

    GUIDED = "guided"
    EXPERT = "expert"


class BiologyPreset(str, Enum):
    """Guided-mode biological context presets."""

    MEMBRANE_RECEPTOR = "membrane_receptor"
    CHROMATIN_DNA_BINDING = "chromatin_dna_binding"
    CYTOSOLIC_PROTEIN = "cytosolic_protein"


class TrafficLightStatus(str, Enum):
    """Data quality gate status returned by a quality checker."""

    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class GuidedInputs:
    """Small input surface shown in Guided mode."""

    biology_preset: BiologyPreset
    traffic_light: TrafficLightStatus = TrafficLightStatus.GREEN
    pixel_size_um: float = 0.1
    frame_interval_s: float = 0.1
    custom_protocol: Optional[str] = None


@dataclass
class UniversalSessionConfig:
    """Canonical configuration consumed by analysis backends."""

    schema_version: int = SCHEMA_VERSION
    protocol_name: str = "custom"
    dimensionality: str = "2d"

    # Core diffusion/HMM controls
    msd_fit_fraction: float = 0.25
    msd_max_lag_fraction: float = 0.5
    hmm_state_count: int = 3
    ihmm_enabled: bool = False
    ihmm_static_state_prior: float = 0.35
    hmm_dirichlet_prior: float = 1.0
    bocpd_hazard_lambda: float = 100.0

    # Tracking/quality controls
    search_radius_px: float = 10.0
    max_jump_um: float = 0.5
    min_track_length: int = 5
    savin_doyle_noise_correction: bool = False
    localization_error_subtraction: bool = True

    # Rheology controls
    rheology_freq_min_hz: float = 0.1
    rheology_freq_max_hz: float = 100.0

    # Runtime controls
    batch_enabled: bool = True
    pixel_size_um: float = 0.1
    frame_interval_s: float = 0.1

    # Future-proof extension point
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate bounds for core parameters."""
        self.dimensionality = self.dimensionality.lower().strip()
        if self.dimensionality not in {"2d", "3d"}:
            raise ValueError("dimensionality must be '2d' or '3d'")

        if not 0.0 < self.msd_fit_fraction <= 1.0:
            raise ValueError("msd_fit_fraction must be in (0, 1]")
        if not 0.0 < self.msd_max_lag_fraction <= 1.0:
            raise ValueError("msd_max_lag_fraction must be in (0, 1]")
        if self.hmm_state_count < 1:
            raise ValueError("hmm_state_count must be >= 1")
        if self.search_radius_px <= 0.0:
            raise ValueError("search_radius_px must be positive")
        if self.max_jump_um <= 0.0:
            raise ValueError("max_jump_um must be positive")
        if self.min_track_length < 2:
            raise ValueError("min_track_length must be >= 2")
        if self.pixel_size_um <= 0.0:
            raise ValueError("pixel_size_um must be positive")
        if self.frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s must be positive")
        if self.rheology_freq_min_hz <= 0.0 or self.rheology_freq_max_hz <= 0.0:
            raise ValueError("rheology frequency bounds must be positive")
        if self.rheology_freq_min_hz >= self.rheology_freq_max_hz:
            raise ValueError("rheology_freq_min_hz must be < rheology_freq_max_hz")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "UniversalSessionConfig":
        """Build config from mapping, preserving unknown keys in extra_params."""
        known_fields = {f.name for f in fields(cls)}
        known_payload: Dict[str, Any] = {}
        unknown_payload: Dict[str, Any] = {}

        for key, value in dict(payload).items():
            if key in known_fields:
                known_payload[key] = value
            else:
                unknown_payload[key] = value

        cfg = cls(**known_payload)
        cfg.extra_params.update(unknown_payload)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config while keeping unknown extension params."""
        payload: Dict[str, Any] = {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name != "extra_params"
        }
        payload.update(self.extra_params)
        return payload


_PRESET_TO_TEMPLATE = {
    BiologyPreset.MEMBRANE_RECEPTOR: "membrane_receptor.json",
    BiologyPreset.CHROMATIN_DNA_BINDING: "chromatin_dna_binding.json",
    BiologyPreset.CYTOSOLIC_PROTEIN: "cytosolic_protein.json",
}

_FALLBACK_SESSION_STATE: Dict[str, Any] = {}


def _protocol_dir(protocol_dir: Optional[Path] = None) -> Path:
    base = protocol_dir or (Path(__file__).resolve().parent / "protocols")
    return base.resolve()


def _resolve_session_state(
    session_state: Optional[MutableMapping[str, Any]] = None,
) -> MutableMapping[str, Any]:
    if session_state is not None:
        return session_state

    try:
        import streamlit as st

        return st.session_state
    except Exception:
        return _FALLBACK_SESSION_STATE


def _coerce_preset(value: BiologyPreset | str) -> BiologyPreset:
    if isinstance(value, BiologyPreset):
        return value
    return BiologyPreset(str(value))


def _coerce_traffic(value: TrafficLightStatus | str) -> TrafficLightStatus:
    if isinstance(value, TrafficLightStatus):
        return value
    return TrafficLightStatus(str(value))


def _coerce_mode(value: UIMode | str) -> UIMode:
    if isinstance(value, UIMode):
        return value
    return UIMode(str(value))


def list_guided_presets() -> list[str]:
    """Return supported guided preset ids."""
    return [preset.value for preset in BiologyPreset]


def load_guided_template(
    preset: BiologyPreset | str,
    protocol_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load a guided preset template from JSON."""
    preset_enum = _coerce_preset(preset)
    template_name = _PRESET_TO_TEMPLATE[preset_enum]
    template_path = _protocol_dir(protocol_dir) / template_name

    if not template_path.exists():
        raise FileNotFoundError(f"Guided preset template missing: {template_path}")

    with template_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(f"Template {template_path} must contain a JSON object")

    return payload


def translate_guided_inputs(
    guided_inputs: GuidedInputs,
    protocol_dir: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> UniversalSessionConfig:
    """Translate small guided inputs into full universal configuration."""
    preset = _coerce_preset(guided_inputs.biology_preset)
    traffic = _coerce_traffic(guided_inputs.traffic_light)

    if guided_inputs.custom_protocol:
        custom_path = Path(guided_inputs.custom_protocol)
        if not custom_path.is_absolute():
            custom_path = _protocol_dir(protocol_dir) / custom_path
        with custom_path.open("r", encoding="utf-8") as handle:
            custom_payload = json.load(handle)
        if isinstance(custom_payload, dict) and "template" in custom_payload:
            payload = dict(custom_payload["template"])
        elif isinstance(custom_payload, dict):
            payload = dict(custom_payload)
        else:
            raise ValueError("Custom protocol must be a JSON object")
    else:
        payload = load_guided_template(preset, protocol_dir=protocol_dir)

    payload["pixel_size_um"] = float(guided_inputs.pixel_size_um)
    payload["frame_interval_s"] = float(guided_inputs.frame_interval_s)

    if traffic == TrafficLightStatus.YELLOW:
        payload["savin_doyle_noise_correction"] = True
        payload["quality_warning"] = "motion_blur_detected"
    elif traffic == TrafficLightStatus.RED:
        payload["savin_doyle_noise_correction"] = True
        payload["quality_gate_failed"] = True
        payload["requires_expert_review"] = True

    if overrides:
        payload.update(dict(overrides))

    config = UniversalSessionConfig.from_mapping(payload)
    config.extra_params.setdefault("guided_preset", preset.value)
    config.extra_params.setdefault("traffic_light", traffic.value)
    return config


def init_dual_mode_state(
    session_state: Optional[MutableMapping[str, Any]] = None,
    default_mode: UIMode = UIMode.GUIDED,
) -> None:
    """Ensure dual-mode session keys exist."""
    state = _resolve_session_state(session_state)

    if SESSION_KEY_UI_MODE not in state:
        state[SESSION_KEY_UI_MODE] = default_mode.value

    if SESSION_KEY_UNIVERSAL_CONFIG not in state:
        state[SESSION_KEY_UNIVERSAL_CONFIG] = UniversalSessionConfig().to_dict()

    if SESSION_KEY_GUIDED_INPUTS not in state:
        state[SESSION_KEY_GUIDED_INPUTS] = {}

    if SESSION_KEY_PROTOCOL_REGISTRY not in state:
        state[SESSION_KEY_PROTOCOL_REGISTRY] = {}


def get_ui_mode(session_state: Optional[MutableMapping[str, Any]] = None) -> UIMode:
    """Read current UI mode from session state."""
    state = _resolve_session_state(session_state)
    value = state.get(SESSION_KEY_UI_MODE, UIMode.GUIDED.value)
    return _coerce_mode(value)


def set_ui_mode(
    mode: UIMode | str,
    session_state: Optional[MutableMapping[str, Any]] = None,
) -> UIMode:
    """Set UI mode and return normalized enum value."""
    state = _resolve_session_state(session_state)
    normalized = _coerce_mode(mode)
    state[SESSION_KEY_UI_MODE] = normalized.value
    return normalized


def get_universal_config(
    session_state: Optional[MutableMapping[str, Any]] = None,
) -> UniversalSessionConfig:
    """Read universal config from session state."""
    state = _resolve_session_state(session_state)
    payload = state.get(SESSION_KEY_UNIVERSAL_CONFIG, {})
    if isinstance(payload, UniversalSessionConfig):
        cfg = payload
    elif isinstance(payload, Mapping):
        cfg = UniversalSessionConfig.from_mapping(payload)
    else:
        raise TypeError("Stored universal config must be a mapping or UniversalSessionConfig")

    return cfg


def set_universal_config(
    config: UniversalSessionConfig | Mapping[str, Any],
    session_state: Optional[MutableMapping[str, Any]] = None,
) -> UniversalSessionConfig:
    """Write universal config to session state."""
    state = _resolve_session_state(session_state)
    if isinstance(config, UniversalSessionConfig):
        normalized = config
        normalized.validate()
    else:
        normalized = UniversalSessionConfig.from_mapping(config)

    state[SESSION_KEY_UNIVERSAL_CONFIG] = normalized.to_dict()
    return normalized


def apply_guided_inputs_to_state(
    guided_inputs: GuidedInputs,
    session_state: Optional[MutableMapping[str, Any]] = None,
    protocol_dir: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> UniversalSessionConfig:
    """Translate guided inputs and store result in shared session config."""
    state = _resolve_session_state(session_state)
    init_dual_mode_state(state)
    preset = _coerce_preset(guided_inputs.biology_preset)
    traffic = _coerce_traffic(guided_inputs.traffic_light)

    config = translate_guided_inputs(
        guided_inputs,
        protocol_dir=protocol_dir,
        overrides=overrides,
    )

    state[SESSION_KEY_GUIDED_INPUTS] = {
        "biology_preset": preset.value,
        "traffic_light": traffic.value,
        "pixel_size_um": guided_inputs.pixel_size_um,
        "frame_interval_s": guided_inputs.frame_interval_s,
        "custom_protocol": guided_inputs.custom_protocol,
    }
    state[SESSION_KEY_UNIVERSAL_CONFIG] = config.to_dict()
    return config


def chatbot_prompt_for_mode(mode: UIMode | str) -> str:
    """Switch chatbot persona by active UI mode."""
    normalized = _coerce_mode(mode)
    if normalized == UIMode.EXPERT:
        return EXPERT_CHAT_SYSTEM_PROMPT
    return NOVICE_CHAT_SYSTEM_PROMPT


def _slugify(name: str) -> str:
    cleaned = [ch.lower() if ch.isalnum() else "_" for ch in name.strip()]
    slug = "".join(cleaned)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "custom_protocol"


def deploy_as_guided_protocol(
    config: UniversalSessionConfig | Mapping[str, Any],
    name: str,
    description: str = "",
    output_dir: Optional[Path] = None,
) -> Path:
    """Save an expert-tuned config as a guided protocol template."""
    normalized = config if isinstance(config, UniversalSessionConfig) else UniversalSessionConfig.from_mapping(config)

    destination_dir = Path(output_dir) if output_dir is not None else (_protocol_dir() / "custom")
    destination_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "description": description,
        "source": "expert_workspace",
        "template": normalized.to_dict(),
    }

    file_path = destination_dir / f"{_slugify(name)}.json"
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    return file_path


def load_custom_guided_protocol(path: str | Path) -> UniversalSessionConfig:
    """Load a custom guided protocol file exported from Expert mode."""
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "template" in payload:
        template = payload["template"]
    else:
        template = payload

    if not isinstance(template, dict):
        raise ValueError("Custom guided protocol must contain a JSON object template")

    return UniversalSessionConfig.from_mapping(template)


def eject_to_expert_workspace(
    session_state: Optional[MutableMapping[str, Any]] = None,
) -> UniversalSessionConfig:
    """Set mode to expert and return the current shared config."""
    state = _resolve_session_state(session_state)
    init_dual_mode_state(state)
    set_ui_mode(UIMode.EXPERT, state)
    return get_universal_config(state)


__all__ = [
    "BiologyPreset",
    "EXPERT_CHAT_SYSTEM_PROMPT",
    "GuidedInputs",
    "NOVICE_CHAT_SYSTEM_PROMPT",
    "SCHEMA_VERSION",
    "SESSION_KEY_GUIDED_INPUTS",
    "SESSION_KEY_PROTOCOL_REGISTRY",
    "SESSION_KEY_UI_MODE",
    "SESSION_KEY_UNIVERSAL_CONFIG",
    "TrafficLightStatus",
    "UIMode",
    "UniversalSessionConfig",
    "apply_guided_inputs_to_state",
    "chatbot_prompt_for_mode",
    "deploy_as_guided_protocol",
    "eject_to_expert_workspace",
    "get_ui_mode",
    "get_universal_config",
    "init_dual_mode_state",
    "list_guided_presets",
    "load_custom_guided_protocol",
    "load_guided_template",
    "set_ui_mode",
    "set_universal_config",
    "translate_guided_inputs",
]
