from __future__ import annotations

from pathlib import Path

from spt2025b.ui.dual_mode import (
    BiologyPreset,
    GuidedInputs,
    TrafficLightStatus,
    UIMode,
    apply_guided_inputs_to_state,
    chatbot_prompt_for_mode,
    deploy_as_guided_protocol,
    get_ui_mode,
    get_universal_config,
    init_dual_mode_state,
    load_custom_guided_protocol,
    set_ui_mode,
    set_universal_config,
    translate_guided_inputs,
)


def test_translate_membrane_preset_defaults() -> None:
    cfg = translate_guided_inputs(
        GuidedInputs(
            biology_preset=BiologyPreset.MEMBRANE_RECEPTOR,
            traffic_light=TrafficLightStatus.GREEN,
            pixel_size_um=0.107,
            frame_interval_s=0.05,
        )
    )

    assert cfg.protocol_name == "Membrane Receptors"
    assert cfg.dimensionality == "2d"
    assert cfg.msd_fit_fraction == 0.25
    assert cfg.hmm_state_count == 3
    assert cfg.ihmm_enabled is False
    assert cfg.pixel_size_um == 0.107
    assert cfg.frame_interval_s == 0.05


def test_translate_yellow_quality_enables_noise_correction() -> None:
    cfg = translate_guided_inputs(
        GuidedInputs(
            biology_preset=BiologyPreset.CYTOSOLIC_PROTEIN,
            traffic_light=TrafficLightStatus.YELLOW,
        )
    )

    assert cfg.savin_doyle_noise_correction is True
    assert cfg.extra_params["quality_warning"] == "motion_blur_detected"


def test_translate_red_quality_sets_expert_review_flags() -> None:
    cfg = translate_guided_inputs(
        GuidedInputs(
            biology_preset=BiologyPreset.CHROMATIN_DNA_BINDING,
            traffic_light=TrafficLightStatus.RED,
        )
    )

    assert cfg.savin_doyle_noise_correction is True
    assert cfg.extra_params["quality_gate_failed"] is True
    assert cfg.extra_params["requires_expert_review"] is True


def test_mode_toggle_preserves_universal_config() -> None:
    state = {}
    init_dual_mode_state(state)

    cfg = translate_guided_inputs(
        GuidedInputs(biology_preset=BiologyPreset.MEMBRANE_RECEPTOR)
    )
    set_universal_config(cfg, state)
    set_ui_mode(UIMode.EXPERT, state)

    roundtrip = get_universal_config(state)
    assert get_ui_mode(state) == UIMode.EXPERT
    assert roundtrip.to_dict() == cfg.to_dict()


def test_apply_guided_inputs_to_state_records_inputs_and_config() -> None:
    state = {}
    guided = GuidedInputs(
        biology_preset=BiologyPreset.CYTOSOLIC_PROTEIN,
        traffic_light=TrafficLightStatus.GREEN,
        pixel_size_um=0.2,
        frame_interval_s=0.02,
    )

    cfg = apply_guided_inputs_to_state(guided, state)
    assert state["guided_inputs"]["biology_preset"] == BiologyPreset.CYTOSOLIC_PROTEIN.value
    assert state["guided_inputs"]["traffic_light"] == TrafficLightStatus.GREEN.value
    assert get_universal_config(state).to_dict() == cfg.to_dict()


def test_deploy_and_load_custom_guided_protocol_roundtrip(tmp_path: Path) -> None:
    cfg = translate_guided_inputs(
        GuidedInputs(biology_preset=BiologyPreset.CHROMATIN_DNA_BINDING)
    )

    output_path = deploy_as_guided_protocol(
        cfg,
        name="Core Facility - Noisy Fluorophore",
        description="Expert tuned defaults for difficult low SNR videos.",
        output_dir=tmp_path,
    )

    loaded = load_custom_guided_protocol(output_path)
    assert output_path.exists()
    assert loaded.to_dict() == cfg.to_dict()


def test_chatbot_prompt_switches_by_mode() -> None:
    assert "wet-lab biologist" in chatbot_prompt_for_mode(UIMode.GUIDED)
    assert "post-doctoral computational biophysicist" in chatbot_prompt_for_mode(
        UIMode.EXPERT
    )
