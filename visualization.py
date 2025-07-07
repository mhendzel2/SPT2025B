"""
================================================================================
  SPT‑VIS  ──  Single‑Particle Tracking Visualisation Toolkit
  ---------------------------------------------------------------------
  * Optimised, duplicate‑free rewrite – 03 Jul 2025
  * Zero hard Streamlit dependency – library code is now UI‑agnostic.
  * Unified error handling via EmptyFigure() helper.
  * Deterministic colour selection and random‑free sampling.
  * Ternary‑free vector maths (no silent precedence bugs).
  * PEP‑8 names and ≤ 79‑char lines where practical.
  ---------------------------------------------------------------------
  External requirements:
      pandas        >= 1.4
      numpy         >= 1.22
      plotly        >= 5.20
      matplotlib    >= 3.5     (only if you request a Matplotlib fig)
  Optional UI:
      streamlit     – not imported here; call st.pyplot/plotly_chart from
                      your app layer if desired.
================================================================================
"""

from __future__ import annotations

import io
import base64
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from matplotlib.figure import Figure as MplFigure
import matplotlib.pyplot as plt

# ------------------------------------------------------------------ #
#   General helpers
# ------------------------------------------------------------------ #

QualColours = tuple(px.colors.qualitative.Plotly)
DEFAULT_CM = "viridis"


def _assert_cols(df: pd.DataFrame, required: List[str], func: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{func} – missing required columns: {', '.join(missing)}"
        )


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font_size=14
    )
    fig.update_xaxes(visible=False).update_yaxes(visible=False)
    fig.update_layout(template="plotly_white")
    return fig


def _qual_colour(i: int) -> str:
    """Deterministic qualitative colour cycling."""
    return QualColours[i % len(QualColours)]


# ------------------------------------------------------------------ #
#   Track‑level visualisations
# ------------------------------------------------------------------ #

def plot_tracks(
    tracks: pd.DataFrame,
    *,
    frame_range: Optional[Tuple[int, int]] = None,
    color_by: str = "track_id",
    colormap: str = DEFAULT_CM,
    alpha: float = 0.7,
) -> go.Figure:
    """
    2‑D trajectory plot (Plotly).
    Required columns – track_id, frame, x, y
    """
    if tracks.empty:
        return _empty_fig("No track data available")

    _assert_cols(tracks, ["track_id", "frame", "x", "y"], "plot_tracks")

    if frame_range:
        tracks = tracks.loc[
            tracks["frame"].between(frame_range[0], frame_range[1])
        ]

    track_ids = tracks["track_id"].unique()
    fig = go.Figure()

    if color_by == "track_id" or color_by not in tracks.columns:
        for i, (tid, tdf) in enumerate(tracks.groupby("track_id")):
            if tid not in track_ids:
                continue
            tdf = tdf.sort_values("frame")
            colour = _qual_colour(i)
            fig.add_trace(
                go.Scatter(
                    x=tdf["x"],
                    y=tdf["y"],
                    mode="lines+markers",
                    name=f"Track {tid}",
                    line=dict(color=colour, width=1),
                    marker=dict(color=colour, size=4),
                    showlegend=len(track_ids) <= 25,
                    opacity=alpha,
                )
            )
            # Start marker
            fig.add_trace(
                go.Scatter(
                    x=[tdf["x"].iloc[0]],
                    y=[tdf["y"].iloc[0]],
                    mode="markers",
                    marker=dict(
                        symbol="circle-open",
                        size=10,
                        color=colour,
                        line=dict(width=2),
                    ),
                    showlegend=False,
                )
            )
    else:  # continuous colour mapping
        if not pd.api.types.is_numeric_dtype(tracks[color_by]):
            raise TypeError(f"{color_by} must be numeric for continuous colouring")

        for tid in track_ids:
            tdf = tracks.query("track_id == @tid").sort_values("frame")
            fig.add_trace(
                go.Scatter(
                    x=tdf["x"],
                    y=tdf["y"],
                    mode="lines+markers",
                    name=f"Track {tid}",
                    marker=dict(
                        color=tdf[color_by],
                        colorscale=colormap,
                        showscale=True,
                        colorbar=dict(title=color_by),
                        size=4,
                    ),
                    line=dict(
                        color=px.colors.sample_colorscale(
                            colormap, [tdf[color_by].mean()]
                        )[0],
                        width=1,
                    ),
                    showlegend=len(track_ids) <= 25,
                    opacity=alpha,
                )
            )

    fig.update_layout(
        title=f"Particle Tracks ({len(track_ids)} tracks)",
        xaxis_title="X (px)",
        yaxis_title="Y (px)",
        yaxis_scaleanchor="x",
        template="plotly_white",
    )
    return fig


def plot_tracks_3d(
    tracks: pd.DataFrame,
    *,
    max_tracks: int = 20,
    pixel_size: float = 0.1,
    frame_interval: float = 1.0,
    use_real_z: bool = False,
    color_by: str = "track_id",
    colormap: str = DEFAULT_CM,
) -> go.Figure:
    """
    Interactive 3‑D trajectory plot.  
    If no z column is present or use_real_z=False, time acts as z.
    """
    if tracks.empty:
        return _empty_fig("No track data available (3‑D)")

    _assert_cols(tracks, ["track_id", "frame", "x", "y"], "plot_tracks_3d")
    has_z = "z" in tracks.columns and use_real_z

    # choose deterministic first N tracks (longest)
    track_lengths = (
        tracks.groupby("track_id")["frame"].count().sort_values(ascending=False)
    )
    track_ids = track_lengths.head(max_tracks).index.to_numpy()

    fig = go.Figure()

    for i, tid in enumerate(track_ids):
        tdf = tracks.query("track_id == @tid").sort_values("frame")
        if len(tdf) < 2:  # skip degenerate
            continue

        x = tdf["x"].to_numpy() * pixel_size
        y = tdf["y"].to_numpy() * pixel_size
        z = (
            tdf["z"].to_numpy() * pixel_size
            if has_z
            else tdf["frame"].to_numpy() * frame_interval
        )
        z_label = "Z (µm)" if has_z else "Time (s)"

        if color_by == "track_id":
            colour = _qual_colour(i)
            line_colour = colour
            marker_kwargs = dict(color=colour)
        else:
            if color_by not in tdf.columns:
                raise ValueError(f"{color_by} column not in data")
            vals = tdf[color_by]
            line_colour = None  # let Plotly colour per‑point
            marker_kwargs = dict(
                color=vals,
                colorscale=colormap,
                showscale=True,
                colorbar=dict(title=color_by) if i == 0 else None,
            )

        fig.add_trace(
            go.Scatter3d(
                x=x,
                y=y,
                z=z,
                mode="lines+markers",
                name=f"Track {tid}",
                line=dict(color=line_colour, width=3),
                marker=dict(size=4, **marker_kwargs),
                showlegend=len(track_ids) <= 15,
            )
        )
        # start / end markers
        fig.add_trace(
            go.Scatter3d(
                x=[x[0]], y=[y[0]], z=[z[0]],
                mode="markers",
                marker=dict(symbol="diamond", size=8, color="green"),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=[x[-1]], y=[y[-1]], z=[z[-1]],
                mode="markers",
                marker=dict(symbol="square", size=8, color="red"),
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"3‑D Trajectories ({len(track_ids)} tracks)",
        scene=dict(
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            zaxis_title=z_label,
            aspectmode="cube",
        ),
        template="plotly_white",
        width=900,
        height=700,
    )
    return fig


def plot_tracks_time_series(
    tracks: pd.DataFrame,
    y_vars: List[str],
    *,
    time_var: str = "frame",
    time_label: str = "Frame",
    y_labels: Optional[Dict[str, str]] = None,
) -> go.Figure:
    """One or many track properties vs time."""
    if tracks.empty:
        return _empty_fig("No track data available")

    required = ["track_id", time_var] + y_vars
    _assert_cols(tracks, required, "plot_tracks_time_series")

    n_rows = len(y_vars)
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[y_labels.get(v, v) if y_labels else v for v in y_vars],
    )

    colours = [ _qual_colour(i) for i in range(tracks["track_id"].nunique()) ]

    for row, yv in enumerate(y_vars, start=1):
        for i, (tid, tdf) in enumerate(tracks.groupby("track_id")):
            tdf = tdf.sort_values(time_var)
            fig.add_trace(
                go.Scatter(
                    x=tdf[time_var],
                    y=tdf[yv],
                    mode="lines",
                    name=f"Track {tid}",
                    legendgroup=f"Track {tid}",
                    showlegend=row == 1 and tracks["track_id"].nunique() <= 10,
                    line=dict(color=colours[i]),
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text=y_labels.get(yv, yv) if y_labels else yv, row=row, col=1)

    fig.update_xaxes(title_text=time_label, row=n_rows, col=1)
    fig.update_layout(
        title="Track variables over time",
        height=300 * n_rows,
        template="plotly_white",
    )
    return fig


# ------------------------------------------------------------------ #
#   Track‑statistics histogram / box utilities
# ------------------------------------------------------------------ #

def _generic_histogram(
    df: pd.DataFrame,
    column: str,
    *,
    title: str,
    xlabel: str,
    colour: str,
    nbins: int = 30,
    log_x: bool = False,
) -> go.Figure:
    if df.empty:
        return _empty_fig("No statistics available")

    fig = px.histogram(
        df,
        x=column,
        nbins=nbins,
        title=title,
        labels={column: xlabel, "count": "N"},
        log_x=log_x,
        color_discrete_sequence=[colour],
    )
    fig.update_layout(template="plotly_white")
    return fig


# ------------------------------------------------------------------ #
#   Track‑statistics panel
# ------------------------------------------------------------------ #

def plot_track_statistics(
    stats: pd.DataFrame, *, mode: str = "histogram"
) -> Dict[str, go.Figure]:
    """
    Returns individual figures keyed by statistic name.
    Recognised columns in *stats*: track_length, duration, net_displacement,
    mean_speed, straightness
    """
    if stats.empty:
        return {"empty": _empty_fig("No track statistics available")}

    figs: Dict[str, go.Figure] = {}
    h = _generic_histogram

    if "track_length" in stats:
        figs["track_length"] = h(
            stats,
            "track_length",
            title="Track length distribution",
            xlabel="Length (frames)",
            colour=QualColours[0],
        )

    if "duration" in stats:
        figs["duration"] = h(
            stats,
            "duration",
            title="Track duration distribution",
            xlabel="Duration (s)",
            colour=QualColours[1],
        )

    if "net_displacement" in stats:
        figs["net_displacement"] = h(
            stats,
            "net_displacement",
            title="Net displacement distribution",
            xlabel="Displacement (px)",
            colour=QualColours[2],
        )

    if "mean_speed" in stats:
        figs["mean_speed"] = h(
            stats,
            "mean_speed",
            title="Mean speed distribution",
            xlabel="Speed (px/s)",
            colour=QualColours[3],
        )

    if "straightness" in stats:
        figs["straightness"] = h(
            stats,
            "straightness",
            title="Straightness distribution",
            xlabel="Straightness (0–1)",
            colour=QualColours[4],
        )
    return figs


# ------------------------------------------------------------------ #
#   MSD
# ------------------------------------------------------------------ #

def plot_msd_curves(
    msd: pd.DataFrame,
    *,
    show_tracks: bool = True,
    show_average: bool = True,
    log_scale: bool = False,
) -> go.Figure:
    if msd.empty:
        return _empty_fig("No MSD data")

    _assert_cols(msd, ["track_id", "lag_time", "msd"], "plot_msd_curves")

    fig = go.Figure()
    colours = [ _qual_colour(i) for i in range(msd["track_id"].nunique()) ]

    if show_tracks:
        for i, (tid, tdf) in enumerate(msd.groupby("track_id")):
            fig.add_trace(
                go.Scatter(
                    x=tdf["lag_time"],
                    y=tdf["msd"],
                    mode="lines",
                    line=dict(color=colours[i], width=1),
                    opacity=0.6,
                    name=f"Track {tid}",
                    showlegend=msd["track_id"].nunique() <= 25,
                )
            )

    if show_average:
        g = msd.groupby("lag_time")["msd"]
        mean, sem = g.mean(), g.std() / np.sqrt(g.count())
        fig.add_trace(
            go.Scatter(
                x=mean.index,
                y=mean.values,
                mode="markers+lines",
                line=dict(color="black", width=3),
                marker=dict(size=6),
                name="Average",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([mean.index, mean.index[::-1]]),
                y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                fill="toself",
                fillcolor="rgba(0,0,0,0.15)",
                line=dict(width=0),
                showlegend=False,
            )
        )

    if log_scale:
        fig.update_xaxes(type="log").update_yaxes(type="log")
        suffix = " (Log‑log)"
    else:
        suffix = ""

    fig.update_layout(
        title=f"MSD vs lag time{suffix}",
        xaxis_title="Lag time",
        yaxis_title="MSD (px²)",
        template="plotly_white",
    )
    return fig


# ------------------------------------------------------------------ #
#   Diffusion coefficient distribution
# ------------------------------------------------------------------ #

def plot_diffusion_coefficients(tracks: Union[pd.DataFrame, Dict[str, Any]]) -> go.Figure:
    """
    Accepts a DataFrame with column 'diffusion_coefficient' or a dict with diffusion results.
    """
    # Handle case where tracks is a dictionary (diffusion analysis results)
    if isinstance(tracks, dict):
        if "diffusion_coefficients" in tracks:
            # Extract diffusion coefficients from analysis results
            coeffs = tracks["diffusion_coefficients"]
            if isinstance(coeffs, (list, np.ndarray)):
                data = pd.Series(coeffs).dropna()
            elif isinstance(coeffs, pd.Series):
                data = coeffs.dropna()
            else:
                return _empty_fig("Invalid diffusion coefficient format in results")
        else:
            return _empty_fig("No diffusion coefficients in analysis results")
    else:
        # Handle DataFrame input
        if tracks.empty or "diffusion_coefficient" not in tracks:
            return _empty_fig("No diffusion coefficients")
        data = tracks["diffusion_coefficient"].dropna()
    
    data = data[data > 0]
    if data.empty:
        return _empty_fig("All diffusion coefficients are 0 / NaN")

    fig = _generic_histogram(
        pd.DataFrame({"D": data}),
        "D",
        title="Diffusion coefficient distribution",
        xlabel="D (px²/s)",
        colour=QualColours[0],
        log_x=True,
    )
    fig.add_vline(
        x=data.mean(),
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {data.mean():.2e}",
    )
    return fig


# ------------------------------------------------------------------ #
#   Active‑transport analysis (merged version – no duplicate)
# ------------------------------------------------------------------ #

def plot_active_transport(
    transport: Dict[str, Any],
) -> Dict[str, go.Figure]:
    """Return dict with 'speeds', 'directions' etc."""
    figs: Dict[str, go.Figure] = {}

    if not transport.get("success") or transport.get("directed_segments") is None:
        return {"empty": _empty_fig("No active‑transport results")}

    seg = transport["directed_segments"]
    if seg.empty:
        return {"empty": _empty_fig("No directed segments")}

    # speed histogram
    figs["speeds"] = _generic_histogram(
        seg,
        "speed",
        title="Directed‑segment speeds",
        xlabel="Speed (µm/s)",
        colour=QualColours[0],
    )

    # polar histogram of angles – custom
    if "angle" in seg.columns:
        theta = np.degrees(seg["angle"])
        bins = np.linspace(-180, 180, 19)  # 18 × 20°
        counts, edges = np.histogram(theta, bins=bins)
        centres = (edges[:-1] + edges[1:]) / 2

        polar = go.Figure()
        polar.add_trace(
            go.Scatterpolar(
                r=counts,
                theta=centres,
                mode="lines",
                fill="toself",
                line_color=QualColours[1],
            )
        )
        polar.update_layout(
            title="Directed‑motion directionality",
            polar=dict(radialaxis=dict(showticklabels=False)),
            template="plotly_white",
        )
        figs["directions"] = polar

    return figs


# Alias for backwards compatibility
def plot_motion_analysis(
    transport: Dict[str, Any],
) -> Dict[str, go.Figure]:
    """
    Alias for plot_active_transport for backwards compatibility.
    Return dict with 'speeds', 'directions' etc.
    """
    return plot_active_transport(transport)


# ------------------------------------------------------------------ #
#   Comparative utilities (bar / box / hist / scatter / line)
def comparative_histogram(
    data: Dict[str, pd.DataFrame],
    column: str,
    *,
    bins: int = 20,
    log_x: bool = False,
    title: Optional[str] = None,
) -> go.Figure:
    if not data:
        return _empty_fig("No data sets provided")

    fig = go.Figure()
    for i, (name, df) in enumerate(data.items()):
        if column not in df or df.empty:
            continue
        fig.add_trace(
            go.Histogram(
                x=df[column],
                name=name,
                nbinsx=bins,
                opacity=0.7,
                marker_color=_qual_colour(i),
            )
        )
    fig.update_layout(
        barmode="overlay",
        title=title or f"{column} histogram",
        xaxis_title=column,
        yaxis_title="Count",
        template="plotly_white",
    )
    if log_x:
        fig.update_xaxes(type="log")
    return fig


# ------------------------------------------------------------------ #
#   Export helpers
# ------------------------------------------------------------------ #

def fig_to_base64(fig: Union[go.Figure, MplFigure], fmt: str = "png") -> str:
    """
    Encode a figure as base64 PNG/JPG/SVG/PDF string.
    """
    valid = {"png", "jpg", "svg", "pdf"}
    if fmt not in valid:
        raise ValueError(f"Format must be one of {valid}")

    if isinstance(fig, go.Figure):
        img_bytes = fig.to_image(format="png" if fmt == "jpg" else fmt, scale=2)
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_bytes = buf.read()
    return base64.b64encode(img_bytes).decode("utf-8")


def download_bytes(
    fig: Union[go.Figure, MplFigure], *, fmt: str = "png"
) -> bytes:
    """Return raw bytes so a UI layer can expose them to users."""
    if isinstance(fig, go.Figure):
        return fig.to_image(format=fmt, scale=2)
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
    buf.seek(0)
    return buf.read()
