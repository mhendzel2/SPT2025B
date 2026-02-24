"""Generate synthetic SPT datasets, validate analyses, and export sample reports."""

from __future__ import annotations

import argparse
import json
import math
import re
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enhanced_report_generator import EnhancedSPTReportGenerator
from report_plot_utils import assert_report_health


def _track_df(track_id: int, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    n = len(x)
    return pd.DataFrame(
        {
            "track_id": track_id,
            "frame": np.arange(n, dtype=int),
            "x": x.astype(float),
            "y": y.astype(float),
        }
    )


def _add_common_columns(df: pd.DataFrame, rng: np.random.Generator, n_cells: int = 8) -> pd.DataFrame:
    out = df.copy()
    out["cell_id"] = out["track_id"].astype(int) % n_cells
    out["z"] = 0.05 * np.sin(0.08 * out["frame"].values) + rng.normal(0.0, 0.01, size=len(out))
    base = 300 + 40 * np.sin(0.05 * out["frame"].values)
    bursts = 80 * (np.sin(0.15 * out["frame"].values + out["track_id"].values * 0.2) > 0.9)
    out["intensity"] = np.clip(base + bursts + rng.normal(0.0, 10.0, size=len(out)), 50, None)
    return out


def _with_intensity_channels(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Add channelized intensity columns expected by intensity analysis."""
    out = df.copy()
    if "intensity" not in out.columns:
        out["intensity"] = 250.0 + rng.normal(0.0, 15.0, size=len(out))

    mean_signal = np.clip(out["intensity"].astype(float) + rng.normal(0.0, 5.0, size=len(out)), 5.0, None)
    std_signal = np.clip(rng.normal(6.0, 1.5, size=len(out)), 0.5, None)
    out["mean_intensity_ch1"] = mean_signal
    out["median_intensity_ch1"] = np.clip(mean_signal + rng.normal(0.0, 2.0, size=len(out)), 5.0, None)
    out["min_intensity_ch1"] = np.clip(mean_signal - np.abs(rng.normal(10.0, 3.0, size=len(out))), 1.0, None)
    out["max_intensity_ch1"] = mean_signal + np.abs(rng.normal(12.0, 4.0, size=len(out)))
    out["std_intensity_ch1"] = std_signal
    return out


def _simulate_brownian(n_tracks: int, n_frames: int, rng: np.random.Generator, start_id: int = 0) -> pd.DataFrame:
    tracks = []
    for i in range(n_tracks):
        x = np.cumsum(rng.normal(0.0, 0.03, size=n_frames)) + rng.uniform(0.0, 2.0)
        y = np.cumsum(rng.normal(0.0, 0.03, size=n_frames)) + rng.uniform(0.0, 2.0)
        tracks.append(_track_df(start_id + i, x, y))
    return pd.concat(tracks, ignore_index=True)


def _simulate_directed(n_tracks: int, n_frames: int, rng: np.random.Generator, start_id: int = 1000) -> pd.DataFrame:
    tracks = []
    for i in range(n_tracks):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        drift = np.array([0.02 * np.cos(angle), 0.02 * np.sin(angle)])
        noise = rng.normal(0.0, 0.015, size=(n_frames, 2))
        steps = noise + drift
        pos = np.cumsum(steps, axis=0)
        pos += rng.uniform(0.0, 2.0, size=(1, 2))
        tracks.append(_track_df(start_id + i, pos[:, 0], pos[:, 1]))
    return pd.concat(tracks, ignore_index=True)


def _simulate_confined_periodic(
    n_tracks: int, n_frames: int, rng: np.random.Generator, start_id: int = 2000
) -> pd.DataFrame:
    tracks = []
    for i in range(n_tracks):
        center = rng.uniform(0.5, 1.5, size=2)
        radius = rng.uniform(0.08, 0.2)
        period = rng.integers(8, 20)
        phase = rng.uniform(0.0, 2.0 * math.pi)
        t = np.arange(n_frames)
        x = center[0] + radius * np.cos(2.0 * math.pi * t / period + phase) + rng.normal(0.0, 0.01, size=n_frames)
        y = center[1] + radius * np.sin(2.0 * math.pi * t / period + phase) + rng.normal(0.0, 0.01, size=n_frames)
        tracks.append(_track_df(start_id + i, x, y))
    return pd.concat(tracks, ignore_index=True)


def _simulate_clustered(n_tracks: int, n_frames: int, rng: np.random.Generator, start_id: int = 3000) -> pd.DataFrame:
    tracks = []
    centers = np.array([[0.4, 0.4], [1.2, 0.5], [0.8, 1.3], [1.5, 1.1]])
    for i in range(n_tracks):
        center = centers[i % len(centers)] + rng.normal(0.0, 0.05, size=2)
        steps = rng.normal(0.0, 0.02, size=(n_frames, 2))
        pos = np.cumsum(steps, axis=0)
        pos += center
        tracks.append(_track_df(start_id + i, pos[:, 0], pos[:, 1]))
    return pd.concat(tracks, ignore_index=True)


def _simulate_regime_switch(
    n_tracks: int, n_frames: int, rng: np.random.Generator, start_id: int = 4000
) -> pd.DataFrame:
    tracks = []
    switch = n_frames // 2
    for i in range(n_tracks):
        angle = rng.uniform(0.0, 2.0 * math.pi)
        drift = np.array([0.03 * np.cos(angle), 0.03 * np.sin(angle)])
        first = np.cumsum(rng.normal(0.0, 0.015, size=(switch, 2)) + drift, axis=0)
        center = first[-1]
        second_steps = rng.normal(0.0, 0.01, size=(n_frames - switch, 2))
        second = center + np.cumsum(second_steps - 0.3 * (second_steps), axis=0)
        pos = np.vstack([first, second])
        pos += rng.uniform(0.0, 1.0, size=(1, 2))
        tracks.append(_track_df(start_id + i, pos[:, 0], pos[:, 1]))
    return pd.concat(tracks, ignore_index=True)


def build_synthetic_datasets(seed: int = 7) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    brownian = _simulate_brownian(n_tracks=30, n_frames=160, rng=rng, start_id=0)
    directed = _simulate_directed(n_tracks=24, n_frames=140, rng=rng, start_id=1000)
    confined = _simulate_confined_periodic(n_tracks=24, n_frames=180, rng=rng, start_id=2000)
    clustered = _simulate_clustered(n_tracks=36, n_frames=120, rng=rng, start_id=3000)
    regime = _simulate_regime_switch(n_tracks=20, n_frames=160, rng=rng, start_id=4000)
    dense_long = _simulate_brownian(n_tracks=60, n_frames=260, rng=rng, start_id=5000)
    bayesian = _simulate_brownian(n_tracks=12, n_frames=90, rng=rng, start_id=9000)
    md_quick = _simulate_brownian(n_tracks=8, n_frames=40, rng=rng, start_id=9500)
    two_point_base = _simulate_brownian(n_tracks=10, n_frames=60, rng=rng, start_id=9800)

    mixed = pd.concat([brownian, directed, confined], ignore_index=True)

    datasets = {
        "brownian": _add_common_columns(brownian, rng),
        "directed": _add_common_columns(directed, rng),
        "confined_periodic": _add_common_columns(confined, rng),
        "clustered": _add_common_columns(clustered, rng),
        "regime_switch": _add_common_columns(regime, rng),
        "dense_long": _add_common_columns(dense_long, rng),
        "bayesian": _add_common_columns(bayesian, rng),
        "md_quick": _add_common_columns(md_quick, rng),
        "two_point": _add_common_columns(two_point_base, rng),
        "mixed": _add_common_columns(mixed, rng),
    }

    intensity = _with_intensity_channels(datasets["mixed"].copy(), rng)
    datasets["intensity"] = intensity

    # Spread track coordinates to satisfy two-point distance bins (0.5-10 μm)
    # when pixel_size is ~0.03 μm/pixel.
    two_point = datasets["two_point"].copy()
    two_point["x"] = two_point["x"] * 120.0 + 300.0
    two_point["y"] = two_point["y"] * 120.0 + 300.0
    datasets["two_point"] = two_point

    return datasets


ANALYSIS_DATASET_HINT = {
    "active_transport": "directed",
    "confinement_analysis": "confined_periodic",
    "loop_extrusion": "confined_periodic",
    "changepoint_detection": "regime_switch",
    "spatial_organization": "clustered",
    "multi_particle_interactions": "clustered",
    "territory_mapping": "clustered",
    "local_diffusion_map": "clustered",
    "spatial_microrheology": "dense_long",
    "two_point_microrheology": "two_point",
    "microrheology": "dense_long",
    "creep_compliance": "dense_long",
    "relaxation_modulus": "dense_long",
    "intensity_analysis": "intensity",
    "bayesian_posterior": "bayesian",
    "md_comparison": "md_quick",
}


class StepTimeoutError(RuntimeError):
    """Raised when an analysis/visualization step exceeds timeout."""


def _run_with_timeout(fn, timeout_sec: int):
    """Run a callable with a SIGALRM timeout (Unix environments)."""
    if timeout_sec <= 0:
        return fn(), None

    def _handler(signum, frame):
        raise StepTimeoutError(f"Timed out after {timeout_sec}s")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(timeout_sec)
    try:
        return fn(), None
    except Exception as e:  # noqa: BLE001
        return None, e
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _json_default(value):
    try:
        import numpy as _np
        import pandas as _pd

        if isinstance(value, (_np.integer,)):
            return int(value)
        if isinstance(value, (_np.floating,)):
            return float(value)
        if isinstance(value, (_np.ndarray,)):
            return value.tolist()
        if isinstance(value, (_pd.DataFrame,)):
            return value.to_dict(orient="records")
        if isinstance(value, (_pd.Series,)):
            return value.tolist()
    except Exception:
        pass
    return str(value)


def _save_figure(fig_obj, output_base: Path) -> List[str]:
    output_files: List[str] = []
    if fig_obj is None:
        return output_files

    fig_list = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    for idx, fig in enumerate(fig_list, start=1):
        if fig is None:
            continue
        suffix = f"_{idx}" if len(fig_list) > 1 else ""
        out_path = output_base.with_name(output_base.name + suffix)
        try:
            if isinstance(fig, go.Figure):
                html_path = out_path.with_suffix(".html")
                pio.write_html(fig, file=str(html_path), include_plotlyjs="cdn", full_html=True)
                output_files.append(str(html_path))
            else:
                png_path = out_path.with_suffix(".png")
                fig.savefig(str(png_path), dpi=150, bbox_inches="tight")
                output_files.append(str(png_path))
        except Exception:
            continue

    return output_files


def run_validation(output_dir: Path, seed: int, analysis_timeout_sec: int = 30) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = output_dir / "datasets"
    figure_dir = output_dir / "figures"
    dataset_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)

    datasets = build_synthetic_datasets(seed=seed)
    for name, df in datasets.items():
        df.to_csv(dataset_dir / f"{name}.csv", index=False)

    generator = EnhancedSPTReportGenerator()
    units = {
        "pixel_size": 0.03,
        "frame_interval": 0.1,
        "distance": "μm",
        "time": "s",
        # Performance settings for heavier analyses.
        "md_bootstrap": 20,
        "md_analyze_compartments": False,
        "bayes_tracks": 6,
        "bayes_walkers": 12,
        "bayes_steps": 500,
        "bayes_warmup": 250,
        "bayes_backend": "auto",
        "bayes_use_gpu": True,
    }

    validation_rows: List[Dict[str, Any]] = []
    for analysis_key, info in generator.available_analyses.items():
        dataset_name = ANALYSIS_DATASET_HINT.get(analysis_key, "mixed")
        tracks_df = datasets[dataset_name]
        generator.track_df = tracks_df
        generator.tracks = tracks_df
        print(f"[analysis] {analysis_key} (dataset={dataset_name})")

        t0 = time.perf_counter()
        result = None
        analysis_success = False
        analysis_error = None
        try:
            result, analysis_exc = _run_with_timeout(
                lambda: info["function"](tracks_df.copy(), units),
                timeout_sec=analysis_timeout_sec,
            )
            if analysis_exc is not None:
                raise analysis_exc
            if isinstance(result, dict):
                success_flag = result.get("success")
                error_value = result.get("error")
                has_error = bool(error_value)
                if success_flag is None:
                    analysis_success = not has_error
                else:
                    analysis_success = bool(success_flag) and not has_error
                analysis_error = error_value if has_error else None
            else:
                analysis_success = result is not None
        except Exception as e:
            analysis_error = str(e)
        runtime_sec = time.perf_counter() - t0

        figure_files: List[str] = []
        visualization_success = False
        visualization_error = None
        if result is not None:
            try:
                fig, viz_exc = _run_with_timeout(
                    lambda: info["visualization"](result),
                    timeout_sec=analysis_timeout_sec,
                )
                if viz_exc is not None:
                    raise viz_exc
                figure_files = _save_figure(fig, figure_dir / analysis_key)
                visualization_success = len(figure_files) > 0 or fig is not None
            except Exception as e:
                visualization_error = str(e)

        result_json_path = output_dir / "results" / f"{analysis_key}.json"
        result_json_path.parent.mkdir(exist_ok=True)
        with result_json_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, default=_json_default)

        validation_rows.append(
            {
                "analysis_key": analysis_key,
                "analysis_name": info.get("name", analysis_key),
                "dataset": dataset_name,
                "analysis_success": analysis_success,
                "visualization_success": visualization_success,
                "runtime_sec": round(runtime_sec, 3),
                "analysis_error": analysis_error,
                "visualization_error": visualization_error,
                "figure_count": len(figure_files),
                "result_json": str(result_json_path),
                "figure_files": figure_files,
            }
        )

    validation_df = pd.DataFrame(validation_rows).sort_values(["analysis_success", "visualization_success"], ascending=False)
    validation_df.to_csv(output_dir / "analysis_validation_summary.csv", index=False)

    # Build a report from analyses that passed end-to-end.
    selected_for_report = [
        row["analysis_key"]
        for row in validation_rows
        if row["analysis_success"] and row["visualization_success"]
    ]
    if not selected_for_report:
        selected_for_report = [key for key in ("basic_statistics", "diffusion_analysis", "spatial_organization") if key in generator.available_analyses]

    report = generator.generate_batch_report(
        tracks_df=datasets["mixed"],
        selected_analyses=selected_for_report,
        condition_name="Synthetic Validation",
    )
    generator.report_results = report["analysis_results"]
    generator.report_figures = report["figures"]

    export_config = {"include_raw": True}
    html_bytes = generator._export_html_report(config=export_config, current_units=units)
    pdf_bytes = generator._export_pdf_report(current_units=units, config=export_config)

    html_path = output_dir / "synthetic_validation_report.html"
    pdf_path = output_dir / "synthetic_validation_report.pdf"
    html_path.write_bytes(html_bytes)
    if pdf_bytes:
        pdf_path.write_bytes(pdf_bytes)

    html_text = html_bytes.decode("utf-8", errors="replace")
    assert_report_health(html_text, source=str(html_path))

    sections = generator._collect_report_sections(config=export_config)
    pdf_text = pdf_bytes.decode("latin-1", errors="ignore") if pdf_bytes else ""

    def _normalize_for_match(text: str) -> str:
        # Collapse differences from PDF escaping/formatting so title matching is stable.
        cleaned = text.replace("\\(", "(").replace("\\)", ")")
        greek = {
            # ReportLab may emit Greek letters as single Latin glyphs in PDF text streams.
            "α": "a",
            "β": "b",
            "μ": "u",
            "Δ": "d",
            "δ": "d",
        }
        for symbol, replacement in greek.items():
            cleaned = cleaned.replace(symbol, replacement)
        cleaned = re.sub(r"[^a-zA-Z0-9]+", " ", cleaned)
        return " ".join(cleaned.lower().split())

    def _tokens_in_order(haystack: str, tokens: List[str]) -> bool:
        pos = 0
        for token in tokens:
            idx = haystack.find(token, pos)
            if idx < 0:
                return False
            pos = idx + len(token)
        return True

    pdf_text_norm = _normalize_for_match(pdf_text)
    missing_titles_pdf = []
    for section in sections:
        title_norm = _normalize_for_match(section["title"])
        if not title_norm:
            continue
        tokens = [tok for tok in title_norm.split() if tok]
        if title_norm not in pdf_text_norm and not _tokens_in_order(pdf_text_norm, tokens):
            missing_titles_pdf.append(section["title"])

    summary = {
        "generated_at": datetime.now().isoformat(),
        "output_dir": str(output_dir),
        "datasets": {name: len(df) for name, df in datasets.items()},
        "analyses_total": int(len(validation_rows)),
        "analysis_success_count": int(sum(1 for row in validation_rows if row["analysis_success"])),
        "visualization_success_count": int(sum(1 for row in validation_rows if row["visualization_success"])),
        "selected_for_report": selected_for_report,
        "html_report": str(html_path),
        "pdf_report": str(pdf_path) if pdf_bytes else None,
        "pdf_missing_section_titles": missing_titles_pdf,
    }
    with (output_dir / "validation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    # Quick human-readable report.
    md_lines = [
        "# Synthetic Analysis Validation",
        "",
        f"- Generated at: `{summary['generated_at']}`",
        f"- Analyses tested: `{summary['analyses_total']}`",
        f"- Analysis successes: `{summary['analysis_success_count']}`",
        f"- Visualization successes: `{summary['visualization_success_count']}`",
        f"- HTML report: `{summary['html_report']}`",
        f"- PDF report: `{summary['pdf_report']}`",
        "",
        "## Report parity check",
        "",
        f"- Missing PDF section titles vs HTML sections: `{len(missing_titles_pdf)}`",
    ]
    if missing_titles_pdf:
        for title in missing_titles_pdf:
            md_lines.append(f"- `{title}`")
    md_lines.extend(["", "## Dataset files", ""])
    for name in sorted(datasets):
        md_lines.append(f"- `{name}`: `{dataset_dir / (name + '.csv')}`")
    (output_dir / "README.md").write_text("\n".join(md_lines), encoding="utf-8")

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic validation assets for SPT analyses.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write outputs. Defaults to test_reports/synthetic_validation_<timestamp>",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility.")
    parser.add_argument(
        "--analysis-timeout-sec",
        type=int,
        default=30,
        help="Timeout per analysis or visualization call (seconds). Use 0 to disable.",
    )
    args = parser.parse_args()

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path("test_reports") / f"synthetic_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    summary = run_validation(
        output_dir=out_dir,
        seed=args.seed,
        analysis_timeout_sec=args.analysis_timeout_sec,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
