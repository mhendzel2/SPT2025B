import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from plotly.subplots import make_subplots

from enhanced_report_generator import EnhancedSPTReportGenerator
from report_plot_utils import (
    assert_report_health,
    find_report_health_issues,
    nonempty_array,
    safe_hover_data,
)
from visualization import plot_clustering_analysis


def _make_tracks(n_tracks: int = 12, n_frames: int = 20, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for track_id in range(n_tracks):
        x = np.cumsum(rng.normal(0, 0.03, size=n_frames))
        y = np.cumsum(rng.normal(0, 0.03, size=n_frames))
        for frame in range(n_frames):
            rows.append(
                {
                    "track_id": track_id,
                    "frame": frame,
                    "x": float(x[frame]),
                    "y": float(y[frame]),
                }
            )
    return pd.DataFrame(rows)


def test_safe_hover_data_filters_missing_columns():
    df = pd.DataFrame({"a": [1], "b": [2]})
    assert safe_hover_data(df, ["a", "missing", "b"]) == ["a", "b"]


def test_nonempty_array_handles_numpy_and_python_lists():
    assert nonempty_array([1, 2])
    assert nonempty_array(np.array([1, 2]))
    assert not nonempty_array([])
    assert not nonempty_array(np.array([]))
    assert not nonempty_array(None)


def test_spatial_organization_plot_handles_missing_n_tracks():
    cluster_tracks = pd.DataFrame(
        {
            "frame": [0, 0, 0],
            "centroid_x": [0.1, 0.5, 0.9],
            "centroid_y": [0.2, 0.4, 0.8],
            "cluster_track_id": [0, 1, 2],
            "n_points": [4, 6, 5],
            "radius": [0.05, 0.08, 0.06],
        }
    )

    fig = plot_clustering_analysis({"success": True, "cluster_tracks": cluster_tracks})
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
    hover_templates = " ".join((trace.hovertemplate or "") for trace in fig.data)
    assert "n_tracks" in hover_templates


def test_energy_landscape_plot_accepts_numpy_array_input():
    generator = EnhancedSPTReportGenerator()
    energy_map = np.arange(16, dtype=float).reshape(4, 4)
    result = {"success": True, "energy_map": energy_map}

    fig = generator._plot_energy_landscape(result)
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 2
    annotation_text = " ".join((ann.text or "") for ann in (fig.layout.annotations or []))
    assert "Visualization failed:" not in annotation_text


def test_energy_landscape_plot_handles_empty_arrays_gracefully():
    generator = EnhancedSPTReportGenerator()
    fig = generator._plot_energy_landscape({"success": True, "energy_map": np.array([])})

    assert isinstance(fig, go.Figure)
    annotation_text = " ".join((ann.text or "") for ann in (fig.layout.annotations or []))
    assert "empty" in annotation_text.lower() or "missing" in annotation_text.lower()
    assert "Visualization failed:" not in annotation_text


def test_plotly_hline_with_pie_subplots_regression():
    fig = make_subplots(
        rows=2,
        cols=1,
        specs=[[{"type": "pie"}], [{"type": "xy"}]],
    )
    fig.add_trace(go.Pie(labels=["A", "B"], values=[1, 2]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines"), row=2, col=1)

    # Regression target: this should not raise with pie traces present.
    fig.add_hline(y=0, row=2, col=1, exclude_empty_subplots=False)


def test_loop_extrusion_plot_no_pd_unboundlocal_error():
    generator = EnhancedSPTReportGenerator()
    track_results = pd.DataFrame(
        {
            "track_id": [1, 2],
            "is_confined": [True, False],
            "confinement_radius": [0.08, 0.12],
            "plateau_msd": [0.01, 0.02],
            "has_periodicity": [True, False],
            "period": [6.0, 0.0],
            "return_tendency": [0.3, -0.1],
        }
    )

    result = {
        "success": True,
        "loop_detected": True,
        "periodic_tracks": [1],
        "full_results": {
            "n_tracks_analyzed": 2,
            "n_confined_tracks": 1,
            "n_periodic_tracks": 1,
            "confinement_fraction": 0.5,
            "periodicity_fraction": 0.5,
            "mean_loop_size": 0.12,
            "track_results": track_results,
        },
    }

    figs = generator._plot_loop_extrusion(result)
    assert isinstance(figs, list)
    assert figs
    assert isinstance(figs[0], go.Figure)
    annotation_text = " ".join((ann.text or "") for ann in (figs[0].layout.annotations or []))
    assert "Error creating loop extrusion visualization" not in annotation_text


def test_report_health_guardrail_detects_failure_markers():
    bad_html = "<html><body><div>Plotting failed: boom</div></body></html>"
    issues = find_report_health_issues(bad_html)
    assert issues
    with pytest.raises(AssertionError):
        assert_report_health(bad_html)


def test_generated_report_html_passes_health_guardrail():
    generator = EnhancedSPTReportGenerator()
    tracks_df = _make_tracks()
    analyses = [
        key
        for key in ["spatial_organization", "energy_landscape", "biased_inference", "loop_extrusion"]
        if key in generator.available_analyses
    ]
    if not analyses:
        pytest.skip("No target analyses are registered in this environment.")

    report = generator.generate_batch_report(tracks_df, analyses, "health-guardrail")
    generator.report_results = report["analysis_results"]
    generator.report_figures = report["figures"]
    html = generator._export_html_report(
        config={"include_raw": True},
        current_units={"pixel_size": 0.1, "frame_interval": 0.1},
    ).decode("utf-8")

    assert_report_health(html, source="generated report")
