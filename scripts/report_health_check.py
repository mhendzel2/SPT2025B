"""Generate a report and fail if known visualization error markers are present."""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from enhanced_report_generator import EnhancedSPTReportGenerator
from report_plot_utils import assert_report_health_file


def make_tracks(n_tracks: int = 12, n_frames: int = 20, seed: int = 7) -> pd.DataFrame:
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


def run_health_check(output_html: str | None = None) -> None:
    generator = EnhancedSPTReportGenerator()
    tracks_df = make_tracks()

    selected_analyses = [
        key
        for key in ("spatial_organization", "energy_landscape", "biased_inference", "loop_extrusion")
        if key in generator.available_analyses
    ]
    if not selected_analyses:
        raise RuntimeError("No report analyses were selected for health checking.")

    report = generator.generate_batch_report(tracks_df, selected_analyses, "ci-health-check")
    generator.report_results = report["analysis_results"]
    generator.report_figures = report["figures"]

    html_bytes = generator._export_html_report(
        config={"include_raw": True},
        current_units={"pixel_size": 0.1, "frame_interval": 0.1},
    )
    html_text = html_bytes.decode("utf-8")

    # Validate the written HTML output, matching CI behavior.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_report = Path(tmp_dir) / "report_health_check.html"
        tmp_report.write_text(html_text, encoding="utf-8")
        assert_report_health_file(tmp_report)

    if output_html:
        with open(output_html, "w", encoding="utf-8") as handle:
            handle.write(html_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run report health guardrail check.")
    parser.add_argument(
        "--output-html",
        default=None,
        help="Optional output path for the generated report HTML.",
    )
    args = parser.parse_args()

    run_health_check(output_html=args.output_html)
    print("Report health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
