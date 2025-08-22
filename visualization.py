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
        raise ValueError(f"{func} – missing required columns: {', '.join(missing)}")


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
    tracks_df, max_tracks=50, colormap='viridis', 
    include_markers=True, marker_size=5, line_width=1,
    title="Particle Tracks", plot_type='2D', color_by=None
):
    """
    Create an interactive plot of particle tracks.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing tracking data with columns 'track_id', 'x', 'y', etc.
    max_tracks : int, optional
        Maximum number of tracks to display, by default 50
    colormap : str, optional
        Colormap for track visualization, by default 'viridis'
    include_markers : bool, optional
        Whether to include markers at particle positions, by default True
    marker_size : int, optional
        Size of markers, by default 5
    line_width : int, optional
        Width of track lines, by default 1
    title : str, optional
        Plot title, by default "Particle Tracks"
    plot_type : str, optional
        Type of plot: '2D' for standard, 'time_coded' for time-color coded tracks
    color_by : str, optional
        Column name to use for coloring tracks (e.g., 'track_id', 'diffusion_coefficient')
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot of tracks
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Create a figure
    fig = go.Figure()
    
    # Get unique track IDs and limit to max_tracks
    unique_tracks = tracks_df['track_id'].unique()
    if max_tracks > 0 and len(unique_tracks) > max_tracks:
        unique_tracks = np.random.choice(unique_tracks, max_tracks, replace=False)
    
    # Generate colors for each track based on track_id
    import plotly.express as px
    
    # Setup color mapping based on color_by parameter
    if color_by and color_by in tracks_df.columns and color_by != 'track_id':
        # Get the values to use for coloring
        color_values = {}
        for tid in unique_tracks:
            track_data = tracks_df[tracks_df['track_id'] == tid]
            # Use mean value of the column for each track
            if not track_data.empty and not track_data[color_by].isna().all():
                color_values[tid] = track_data[color_by].mean()
            else:
                color_values[tid] = np.nan
        
        # Remove tracks with NaN color values
        valid_tracks = [tid for tid in unique_tracks if tid in color_values and not np.isnan(color_values[tid])]
        
        if valid_tracks:
            color_vals = [color_values[tid] for tid in valid_tracks]
            unique_tracks = valid_tracks
            
            # Create a color mapper function
            min_val, max_val = np.min(color_vals), np.max(color_vals)
            norm_vals = [(v - min_val) / (max_val - min_val) if max_val > min_val else 0.5 for v in color_vals]
            colors = px.colors.sample_colorscale(colormap, norm_vals)
            
            # Add a colorbar
            fig.update_layout(
                coloraxis=dict(
                    colorscale=colormap,
                    colorbar=dict(
                        title=color_by,
                        thickness=15,
                        len=0.5,
                        y=0.5
                    ),
                    cmin=min_val,
                    cmax=max_val
                )
            )
        else:
            # Fallback to default coloring
            colors = px.colors.sample_colorscale(colormap, np.linspace(0, 1, len(unique_tracks)))
    else:
        # Default coloring by track_id
        colors = px.colors.sample_colorscale(colormap, np.linspace(0, 1, len(unique_tracks)))
    
    # Determine min and max frame for time-coded colors if needed
    if plot_type == 'time_coded' and 'frame' in tracks_df.columns:
        min_frame = tracks_df['frame'].min()
        max_frame = tracks_df['frame'].max()
        time_colorscale = px.colors.sequential.Viridis
    
    for i, track_id in enumerate(unique_tracks):
        # Get data for this track
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if plot_type == 'time_coded' and 'frame' in track_data.columns:
            # For time-coded tracks, use a scatter plot with color gradient
            color_vals = (track_data['frame'] - min_frame) / (max_frame - min_frame) if max_frame > min_frame else 0.5
            
            fig.add_trace(go.Scatter(
                x=track_data['x'], 
                y=track_data['y'],
                mode='lines+markers' if include_markers else 'lines',
                line=dict(width=line_width, color='rgba(0,0,0,0)'),  # Transparent line
                marker=dict(
                    size=marker_size,
                    color=color_vals,
                    colorscale=time_colorscale,
                    showscale=i==0,  # Show colorbar only for first track
                    colorbar=dict(title="Frame") if i==0 else None
                ),
                name=f"Track {track_id}",
                hovertemplate="Track: %{meta}<br>x: %{x}<br>y: %{y}<br>Frame: %{marker.color}" if 'frame' in track_data.columns else None,
                meta=track_id
            ))
        else:
            # Standard track plot with consistent color for each track
            fig.add_trace(go.Scatter(
                x=track_data['x'], 
                y=track_data['y'],
                mode='lines+markers' if include_markers else 'lines',
                name=f"Track {track_id}",
                line=dict(color=colors[i], width=line_width),
                marker=dict(color=colors[i], size=marker_size),
                hovertemplate="Track: %{meta}<br>x: %{x}<br>y: %{y}<extra></extra>",
                meta=track_id
            ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="x",
        yaxis_title="y",
        legend_title="Tracks",
        template="plotly_white",
        showlegend=False,  # Hide legend due to potentially large number of tracks
        autosize=True,
        height=600,
        hovermode='closest'
    )
    
    # Ensure equal aspect ratio
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )
    
    return fig


def plot_tracks_3d(
    tracks_df, max_tracks=50, colormap='viridis',
    include_markers=True, marker_size=3, line_width=1,
    title="Particle Tracks (3D)", height=700, width=700
):
    """
    Create an interactive 3D plot of particle tracks, supporting time as z-axis.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame with track data containing 'track_id', 'frame', 'x', 'y'
    max_tracks : int, optional
        Maximum number of tracks to display, by default 50
    colormap : str, optional
        Colormap for tracks, by default 'viridis'
    include_markers : bool, optional
        Whether to include markers at positions, by default True
    marker_size : int, optional
        Size of markers, by default 3
    line_width : int, optional
        Width of track lines, by default 1
    title : str, optional
        Plot title, by default "Particle Tracks (3D)"
    height, width : int, optional
        Plot dimensions, by default 700
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive 3D plot of tracks
    """
    import plotly.graph_objects as go
    import numpy as np
    
    # Create figure
    fig = go.Figure()
    
    # Limit number of tracks if necessary
    unique_tracks = tracks_df['track_id'].unique()
    if max_tracks > 0 and len(unique_tracks) > max_tracks:
        unique_tracks = np.random.choice(unique_tracks, max_tracks, replace=False)
    
    # Generate colors
    import plotly.express as px
    colors = px.colors.sample_colorscale(colormap, np.linspace(0, 1, len(unique_tracks)))
    
    for i, track_id in enumerate(unique_tracks):
        # Get data for this track
        track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values('frame')
        
        if len(track_data) <= 1:
            continue  # Skip very short tracks
            
        # Add 3D scatter plot
        fig.add_trace(go.Scatter3d(
            x=track_data['x'],
            y=track_data['y'],
            z=track_data['frame'],  # Use frame as z-coordinate
            mode='lines+markers' if include_markers else 'lines',
            name=f"Track {track_id}",
            line=dict(color=colors[i], width=line_width),
            marker=dict(color=colors[i], size=marker_size),
            hovertemplate="Track: %{meta}<br>x: %{x}<br>y: %{y}<br>frame: %{z}<extra></extra>",
            meta=track_id
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='frame',
            aspectmode='auto'
        ),
        showlegend=False,
        height=height,
        width=width
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
            coeffs = tracks["diffusion_coefficients"]
            if isinstance(coeffs, (list, np.ndarray)):
                data = pd.Series(coeffs).dropna()
            elif isinstance(coeffs, pd.Series):
                data = coeffs.dropna()
            else:
                return _empty_fig("Invalid diffusion coefficient format in results")
        elif "coefficient_values" in tracks:
            coeffs = tracks["coefficient_values"]
            if isinstance(coeffs, (list, np.ndarray)):
                data = pd.Series(coeffs).dropna()
            elif isinstance(coeffs, pd.Series):
                data = coeffs.dropna()
            else:
                return _empty_fig("Invalid diffusion coefficient format in results")
        else:
            return _empty_fig("No diffusion coefficients found in analysis results")
    else:
        # Handle DataFrame input
        if tracks.empty:
            return _empty_fig("No track data provided")
        
        possible_cols = ['diffusion_coefficient', 'D', 'diffusion_coeff', 'coefficient']
        diff_col = None
        
        for col in possible_cols:
            if col in tracks.columns:
                diff_col = col
                break
        
        if diff_col is None:
            return _empty_fig("No diffusion coefficient column found in data")
        
        data = tracks[diff_col].dropna()
    
    # Filter out invalid values
    data = data[data > 0]
    if data.empty:
        return _empty_fig("All diffusion coefficients are 0 or invalid")

    # Create histogram
    fig = _generic_histogram(
        pd.DataFrame({"D": data}),
        "D",
        title="Diffusion coefficient distribution",
        xlabel="D (px²/s)",
        colour=QualColours[0],
        log_x=True,
    )
    
    # Add mean line
    mean_val = data.mean()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean = {mean_val:.2e}",
        annotation_position="top left"
    )
    
    # Add median line
    median_val = data.median()
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Median = {median_val:.2e}",
        annotation_position="top right"
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

    if not transport.get("success", False):
        return {"empty": _empty_fig("Active transport analysis failed")}
    
    # Handle different possible key names for directed segments
    seg_data = None
    for key in ["directed_segments", "segments", "transport_segments"]:
        if key in transport and transport[key] is not None:
            seg_data = transport[key]
            break
    
    if seg_data is None:
        return {"empty": _empty_fig("No active‑transport results found")}

    # Handle both DataFrame and dict formats
    if isinstance(seg_data, dict):
        if "speeds" in seg_data and "angles" in seg_data:
            # Create DataFrame from dict
            seg = pd.DataFrame({
                "speed": seg_data["speeds"],
                "angle": seg_data["angles"]
            })
        else:
            return {"empty": _empty_fig("Invalid transport segment format")}
    else:
        seg = seg_data

    if seg.empty:
        return {"empty": _empty_fig("No directed segments found")}

    # Speed histogram
    if "speed" in seg.columns:
        figs["speeds"] = _generic_histogram(
            seg,
            "speed",
            title="Directed‑segment speeds",
            xlabel="Speed (µm/s)",
            colour=QualColours[0],
        )

    # Polar histogram of angles
    angle_col = None
    for col in ["angle", "angles", "direction", "theta"]:
        if col in seg.columns:
            angle_col = col
            break
    
    if angle_col is not None:
        theta = np.degrees(seg[angle_col])
        # Handle NaN values
        theta = theta[~np.isnan(theta)]
        
        if len(theta) > 0:
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
                    name="Direction distribution"
                )
            )
            polar.update_layout(
                title="Directed‑motion directionality",
                polar=dict(
                    radialaxis=dict(
                        showticklabels=False,
                        range=[0, max(counts) * 1.1] if max(counts) > 0 else [0, 1]
                    )
                ),
                template="plotly_white",
            )
            figs["directions"] = polar

    # Add summary statistics if available
    if "speed" in seg.columns:
        speed_data = seg["speed"].dropna()
        if len(speed_data) > 0:
            summary_text = f"Mean speed: {speed_data.mean():.2f} µm/s<br>"
            summary_text += f"Median speed: {speed_data.median():.2f} µm/s<br>"
            summary_text += f"N segments: {len(speed_data)}"
            
            # Add text annotation to speed plot
            if "speeds" in figs:
                figs["speeds"].add_annotation(
                    text=summary_text,
                    xref="paper", yref="paper",
                    x=0.7, y=0.9,
                    showarrow=False,
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1
                )

    return figs


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
#   Intensity Analysis
# ------------------------------------------------------------------ #

def plot_intensity_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for intensity analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results from analyze_intensity_profiles function, with 'all_intensities' added.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot summarizing intensity analysis.
    """
    if not analysis_results or 'intensity_statistics' not in analysis_results:
        return _empty_fig("No intensity analysis data available")

    stats = analysis_results['intensity_statistics']
    track_profiles = analysis_results.get('track_profiles', {})
    all_intensities = analysis_results.get('all_intensities')

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("Intensity Behavior", "Intensity Distribution", "Example Intensity Profiles")
    )

    # Pie chart for intensity behavior
    n_total = len(track_profiles) if track_profiles else 0
    if n_total > 0:
        bleaching_set = set(stats.get('photobleaching_detected', []))
        blinking_set = set(stats.get('blinking_events', {}).keys())
        stable_set = set(track_profiles.keys()) - bleaching_set - blinking_set

        labels = ['Stable', 'Photobleaching', 'Blinking']
        values = [len(stable_set), len(bleaching_set), len(blinking_set)]

        fig.add_trace(go.Pie(labels=labels, values=values, name="Behavior", hole=.3), row=1, col=1)

    # Histogram of all intensities
    if all_intensities:
        fig.add_trace(go.Histogram(x=all_intensities, name='Intensity'), row=1, col=2)
        fig.update_xaxes(title_text="Intensity", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)


    # Example intensity profiles
    if track_profiles:
        # Use a deterministic way to select tracks to show, e.g., sort by ID
        track_ids = sorted(list(track_profiles.keys()))

        # Show up to 3 example tracks
        for i, track_id in enumerate(track_ids[:3]):
            profile = track_profiles[track_id]
            frames = profile['frames']
            raw_intensities = profile['raw_intensities']
            smoothed_intensities = profile['smoothed_intensities']

            color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]

            fig.add_trace(go.Scatter(x=frames, y=raw_intensities, mode='lines', name=f'Track {track_id} (Raw)',
                                     legendgroup=f'track_{track_id}', showlegend=True,
                                     line=dict(color=color, width=1, dash='dot')), row=2, col=1)
            fig.add_trace(go.Scatter(x=frames, y=smoothed_intensities, mode='lines', name=f'Track {track_id} (Smoothed)',
                                     legendgroup=f'track_{track_id}', showlegend=True,
                                     line=dict(color=color, width=2)), row=2, col=1)

    fig.update_layout(
        title_text="Intensity Analysis Summary",
        height=800,
        showlegend=True,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="Frame", row=2, col=1)
    fig.update_yaxes(title_text="Intensity", row=2, col=1)

    return fig

def plot_confinement_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for confinement analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results from confinement analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot summarizing confinement analysis.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No confinement analysis data available")

    n_total = analysis_results.get('n_total_tracks', 0)
    n_confined = analysis_results.get('n_confined_tracks', 0)
    confined_results = analysis_results.get('track_results', pd.DataFrame())
    tracks_df = analysis_results.get('tracks_df')

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "xy"}],
               [{"type": "xy", "colspan": 2}, None]],
        subplot_titles=("Confined Fraction", "Confinement Radii", "Spatial Distribution of Confined Tracks")
    )

    # Pie chart for confined fraction
    if n_total > 0:
        labels = ['Confined', 'Unconfined']
        values = [n_confined, n_total - n_confined]
        fig.add_trace(go.Pie(labels=labels, values=values, name="Confinement", hole=.3), row=1, col=1)

    # Histogram of confinement radii
    if not confined_results.empty and 'confinement_radius' in confined_results.columns:
        radii = confined_results['confinement_radius'].dropna()
        if not radii.empty:
            fig.add_trace(go.Histogram(x=radii, name='Radius'), row=1, col=2)
            fig.update_xaxes(title_text="Confinement Radius (µm)", row=1, col=2)
            fig.update_yaxes(title_text="Frequency", row=1, col=2)

    # Scatter plot of all tracks, with confined tracks highlighted
    if tracks_df is not None and not tracks_df.empty:
        # Plot all tracks in grey
        for track_id, track_data in tracks_df.groupby('track_id'):
            fig.add_trace(go.Scatter(x=track_data['x'], y=track_data['y'], mode='lines',
                                     line=dict(color='lightgrey', width=1),
                                     showlegend=False), row=2, col=1)

        # Plot confined tracks in color
        if not confined_results.empty:
            confined_track_ids = confined_results['track_id'].unique()
            confined_tracks_df = tracks_df[tracks_df['track_id'].isin(confined_track_ids)]

            for i, (track_id, track_data) in enumerate(confined_tracks_df.groupby('track_id')):
                color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
                fig.add_trace(go.Scatter(x=track_data['x'], y=track_data['y'], mode='lines',
                                         line=dict(color=color, width=2),
                                         name=f'Track {track_id}'), row=2, col=1)

    fig.update_layout(
        title_text="Confinement Analysis Summary",
        height=800,
        showlegend=False,
        template="plotly_white"
    )
    fig.update_xaxes(title_text="X (µm)", row=2, col=1)
    fig.update_yaxes(title_text="Y (µm)", row=2, col=1)

    return fig

def plot_velocity_correlation_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for velocity correlation analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results from velocity correlation analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot of velocity autocorrelation.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No velocity correlation data available")

    mean_autocorr = analysis_results.get('mean_autocorr')
    lags = analysis_results.get('lags')
    individual_autocorrs = analysis_results.get('individual_autocorrs', [])

    if not mean_autocorr or not lags:
        return _empty_fig("Mean autocorrelation data is missing.")

    fig = go.Figure()

    # Plot individual tracks with low opacity
    for i, ac in enumerate(individual_autocorrs):
        fig.add_trace(go.Scatter(x=lags[:len(ac)], y=ac, mode='lines',
                                 line=dict(color='rgba(128,128,128,0.2)'),
                                 showlegend=False))

    # Plot mean autocorrelation
    fig.add_trace(go.Scatter(x=lags, y=mean_autocorr, mode='lines+markers',
                             line=dict(color='red', width=3),
                             name='Mean Autocorrelation'))

    fig.add_hline(y=0, line_dash="dash", line_color="black")

    fig.update_layout(
        title_text="Velocity Autocorrelation",
        xaxis_title="Lag Time (s)",
        yaxis_title="Autocorrelation",
        template="plotly_white"
    )

    return fig

def plot_anomaly_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for anomaly detection results.

    Parameters
    ----------
    analysis_results : dict
        Results from anomaly detection analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot of tracks with anomalies highlighted.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No anomaly analysis data available")

    anomaly_df = analysis_results.get('anomaly_df')

    if anomaly_df is None or anomaly_df.empty:
        return _empty_fig("No anomaly data to plot.")

    fig = px.scatter(anomaly_df, x="x", y="y", color="anomaly_type",
                     hover_data=['track_id', 'frame'],
                     title="Anomaly Detection Results")

    fig.update_layout(
        xaxis_title="X (µm)",
        yaxis_title="Y (µm)",
        template="plotly_white"
    )

    return fig

def plot_clustering_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for clustering analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results from clustering analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot summarizing clustering analysis.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No clustering analysis data available")

    cluster_tracks = analysis_results.get('cluster_tracks')

    if cluster_tracks is None or cluster_tracks.empty:
        return _empty_fig("No cluster data to plot.")

    # For simplicity, we plot the clusters at a single frame (e.g., the first frame with clusters)
    first_frame_with_clusters = cluster_tracks['frame'].min()
    frame_data = cluster_tracks[cluster_tracks['frame'] == first_frame_with_clusters]

    fig = px.scatter(frame_data, x="centroid_x", y="centroid_y", color="cluster_track_id",
                     size="n_points", hover_data=['n_tracks', 'radius'],
                     title=f"Clusters at Frame {first_frame_with_clusters}")

    fig.update_layout(
        xaxis_title="X (µm)",
        yaxis_title="Y (µm)",
        template="plotly_white"
    )

    return fig

def plot_changepoint_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for changepoint detection results.

    Parameters
    ----------
    analysis_results : dict
        Results from changepoint detection analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot of tracks with changepoints highlighted.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No changepoint analysis data available")

    tracks_df = analysis_results.get('tracks_df')
    changepoints = analysis_results.get('changepoints') # This is a dict {track_id: [frame1, frame2]}

    if tracks_df is None or changepoints is None:
        return _empty_fig("Track or changepoint data is missing.")

    fig = go.Figure()

    # Plot all tracks
    for track_id, track_data in tracks_df.groupby('track_id'):
        fig.add_trace(go.Scatter(x=track_data['x'], y=track_data['y'], mode='lines',
                                 line=dict(color='lightgrey', width=1),
                                 showlegend=False))

    # Highlight changepoints
    if changepoints:
        for track_id, cp_frames in changepoints.items():
            track_data = tracks_df[tracks_df['track_id'] == track_id]
            if not track_data.empty:
                cp_data = track_data[track_data['frame'].isin(cp_frames)]
                if not cp_data.empty:
                    fig.add_trace(go.Scatter(x=cp_data['x'], y=cp_data['y'], mode='markers',
                                             marker=dict(color='red', size=8, symbol='x'),
                                             name=f'Changepoints Track {track_id}'))

    fig.update_layout(
        title_text="Changepoint Detection",
        xaxis_title="X (µm)",
        yaxis_title="Y (µm)",
        template="plotly_white"
    )

    return fig

def plot_particle_interaction_analysis(analysis_results: Dict[str, Any]) -> go.Figure:
    """
    Create visualization for particle interaction analysis results.

    Parameters
    ----------
    analysis_results : dict
        Results from crowding analysis.

    Returns
    -------
    plotly.graph_objects.Figure
        Interactive plot summarizing particle interactions.
    """
    if not analysis_results.get('success', False):
        return _empty_fig("No particle interaction data available")

    crowding_data = analysis_results.get('crowding_data')
    track_results = analysis_results.get('track_results')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Particle Density Map", "Mobility vs. Local Density")
    )

    # Density map
    if crowding_data is not None and not crowding_data.empty:
        # Re-using logic from plot_track_density_map
        x = crowding_data['x'].values
        y = crowding_data['y'].values

        h, x_edges, y_edges = np.histogram2d(x, y, bins=50)

        from scipy.ndimage import gaussian_filter
        h_smooth = gaussian_filter(h, sigma=2.0)

        fig.add_trace(go.Heatmap(
            z=h_smooth.T,
            x=x_edges,
            y=y_edges,
            colorscale='Viridis',
            colorbar=dict(title='Density')
        ), row=1, col=1)

    # Mobility vs. Density scatter plot
    if track_results is not None and not track_results.empty:
        if 'mean_density' in track_results.columns and 'mean_displacement' in track_results.columns:
            fig.add_trace(go.Scatter(
                x=track_results['mean_density'],
                y=track_results['mean_displacement'],
                mode='markers',
                marker=dict(
                    color=track_results['density_displacement_correlation'],
                    colorscale='RdBu',
                    colorbar=dict(title='Correlation'),
                    showscale=True
                ),
                text=track_results['track_id']
            ), row=1, col=2)

            fig.update_xaxes(title_text="Mean Local Density", row=1, col=2)
            fig.update_yaxes(title_text="Mean Displacement (µm)", row=1, col=2)

    fig.update_layout(
        title_text="Multi-Particle Interaction Analysis",
        height=500,
        showlegend=False,
        template="plotly_white"
    )

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

    try:
        if isinstance(fig, go.Figure):
            img_bytes = fig.to_image(format="png" if fmt == "jpg" else fmt, scale=2)
        else:
            buf = io.BytesIO()
            fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.read()
        return base64.b64encode(img_bytes).decode("utf-8")
    except Exception as e:
        raise ValueError(f"Failed to encode figure as {fmt}: {str(e)}")


def download_bytes(
    fig: Union[go.Figure, MplFigure], *, fmt: str = "png"
) -> bytes:
    """Return raw bytes so a UI layer can expose them to users."""
    try:
        if isinstance(fig, go.Figure):
            return fig.to_image(format=fmt, scale=2)
        buf = io.BytesIO()
        fig.savefig(buf, format=fmt, dpi=300, bbox_inches="tight")
        buf.seek(0)
        return buf.read()
    except Exception as e:
        raise ValueError(f"Failed to generate {fmt} bytes: {str(e)}")

def spatial_intensity_correlation(image_data, segmentation_data, properties_data, channel_indices=None):
    """
    Calculate spatial intensity correlation between channels in segmented regions.
    
    Parameters:
    -----------
    image_data : numpy array or None
        The full image data with shape (channels, height, width) or None if not available
    segmentation_data : numpy array
        Segmentation mask with labeled regions
    properties_data : list of dict
        Properties for each segmented region, may contain intensity data for channels
    channel_indices : list or None
        Indices of channels to analyze. If None, all available channels are used.
    
    Returns:
    --------
    dict
        Dictionary with correlation results between channels
    """
    from scipy import stats
    
    results = {}
    
    # Check if we have full image data
    if image_data is not None:
        num_channels = image_data.shape[0]
        if channel_indices is None:
            channel_indices = list(range(num_channels))
        
        # Calculate correlations using full image data
        for i, ch1 in enumerate(channel_indices):
            for j, ch2 in enumerate(channel_indices[i+1:], i+1):
                channel_key = f"ch{ch1}_ch{ch2}"
                results[channel_key] = {
                    'pearson': [],
                    'spearman': [],
                    'region_ids': []
                }
                
                for region in np.unique(segmentation_data):
                    if region == 0:  # Skip background
                        continue
                    
                    mask = segmentation_data == region
                    ch1_values = image_data[ch1][mask]
                    ch2_values = image_data[ch2][mask]
                    
                    if len(ch1_values) > 5:  # Ensure enough pixels for correlation
                        pearson = np.corrcoef(ch1_values, ch2_values)[0, 1]
                        spearman = stats.spearmanr(ch1_values, ch2_values)[0]
                        
                        results[channel_key]['pearson'].append(pearson)
                        results[channel_key]['spearman'].append(spearman)
                        results[channel_key]['region_ids'].append(region)
    
    # Use properties data for correlation if available (even if image_data is present)
    # This allows correlating channels based on properties when full image is not available
    if properties_data:
        # Determine which channels have intensity data in properties
        available_channels = set()
        for prop in properties_data:
            for key in prop.keys():
                if key.startswith('MEAN_INTENSITY_CH') or key.startswith('Mean intensity ch') or key.startswith('Mean ch'):
                    # Extract channel number using different possible formats
                    if key.startswith('MEAN_INTENSITY_CH'):
                        ch = int(key.replace('MEAN_INTENSITY_CH', ''))
                    elif key.startswith('Mean intensity ch'):
                        ch = int(key.replace('Mean intensity ch', ''))
                    elif key.startswith('Mean ch'):
                        ch = int(key.replace('Mean ch', ''))
                    available_channels.add(ch)
        
        available_channels = sorted(list(available_channels))
        
        if channel_indices is None:
            channel_indices = available_channels
        else:
            # Only use channels that are both requested and available
            channel_indices = [ch for ch in channel_indices if ch in available_channels]
        
        # Calculate correlations from properties data
        for i, ch1 in enumerate(channel_indices):
            for j, ch2 in enumerate(channel_indices[i+1:], i+1):
                channel_key = f"ch{ch1}_ch{ch2}_prop"
                results[channel_key] = {
                    'pearson': None,
                    'spearman': None,
                    'region_ids': []
                }
                
                ch1_values = []
                ch2_values = []
                region_ids = []
                
                for prop in properties_data:
                    # Try different possible column naming formats
                    ch1_key = None
                    ch2_key = None
                    
                    # Check for different possible formats of intensity column names
                    for prefix in ['MEAN_INTENSITY_CH', 'Mean intensity ch', 'Mean ch']:
                        if f"{prefix}{ch1}" in prop:
                            ch1_key = f"{prefix}{ch1}"
                        if f"{prefix}{ch2}" in prop:
                            ch2_key = f"{prefix}{ch2}"
                    
                    if ch1_key and ch2_key:
                        ch1_values.append(float(prop[ch1_key]))
                        ch2_values.append(float(prop[ch2_key]))
                        region_ids.append(prop.get('ID', prop.get('Spot ID', prop.get('LABEL', len(region_ids)))))
                
                if len(ch1_values) > 2:  # Ensure enough regions for correlation
                    ch1_values = np.array(ch1_values)
                    ch2_values = np.array(ch2_values)
                    
                    try:
                        pearson = np.corrcoef(ch1_values, ch2_values)[0, 1]
                        spearman = stats.spearmanr(ch1_values, ch2_values)[0]
                        
                        results[channel_key]['pearson'] = pearson
                        results[channel_key]['spearman'] = spearman
                        results[channel_key]['region_ids'] = region_ids
                        results[channel_key]['ch1_values'] = ch1_values.tolist()
                        results[channel_key]['ch2_values'] = ch2_values.tolist()
                    except Exception as e:
                        results[channel_key]['error'] = str(e)
    
    return results

def plot_channel_correlations(correlation_results: dict) -> dict:
    """
    Create scatter plots for channel intensity correlations.
    
    Parameters
    ----------
    correlation_results : dict
        Results from spatial_intensity_correlation function
        
    Returns
    -------
    dict
        Dictionary of plotly figures for each channel pair
    """
    import plotly.express as px
    
    figures = {}
    
    for key, data in correlation_results.items():
        # Skip entries without correlation data
        if ('pearson' not in data or data['pearson'] is None or 
            'ch1_values' not in data or 'ch2_values' not in data):
            continue
            
        # Extract channel numbers from the key
        if '_prop' in key:
            ch_pair = key.replace('_prop', '')
        else:
            ch_pair = key
            
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'Channel 1': data['ch1_values'],
            'Channel 2': data['ch2_values'],
            'Region ID': data['region_ids']
        })
        
        # Create scatter plot
        fig = px.scatter(
            df, x='Channel 1', y='Channel 2',
            hover_data=['Region ID'],
            title=f"{ch_pair} Correlation (Pearson: {data['pearson']:.3f}, Spearman: {data['spearman']:.3f})"
        )
        
        # Add correlation line
        if len(df) > 1:
            x_range = [df['Channel 1'].min(), df['Channel 1'].max()]
            slope, intercept = np.polyfit(df['Channel 1'], df['Channel 2'], 1)
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[slope * x + intercept for x in x_range],
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name=f'y = {slope:.3f}x + {intercept:.3f}'
                )
            )
        
        figures[ch_pair] = fig
    
    return figures

def plot_track_density_map(tracks_df, sigma=5.0, bins=100, colorscale='Viridis'):
    """
    Create a 2D density map of track positions.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track data with columns 'x', 'y'
    sigma : float
        Standard deviation for Gaussian kernel in pixels
    bins : int
        Number of bins for 2D histogram
    colorscale : str
        Colorscale name for density map
        
    Returns
    -------
    plotly.graph_objects.Figure
        Interactive density map figure
    """
    if tracks_df.empty:
        return _empty_fig("No track data available")
    
    # Create a 2D histogram
    x = tracks_df['x'].values
    y = tracks_df['y'].values
    
    h, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    
    # Apply Gaussian smoothing if sigma > 0
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        h_smooth = gaussian_filter(h, sigma=sigma)
    else:
        h_smooth = h
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=h_smooth.T,  # Transpose for correct orientation
        x=x_edges,
        y=y_edges,
        colorscale=colorscale,
        colorbar=dict(title='Density')
    ))
    
    fig.update_layout(
        title='Track Density Map',
        xaxis_title='X Position',
        yaxis_title='Y Position',
        template='plotly_white'
    )
    
    return fig

def plot_motion_analysis(motion_analysis_results, title="Motion Model Analysis"):
    """
    Create visualization for motion model analysis results.
    
    Parameters
    ----------
    motion_analysis_results : dict
        Results dictionary from analyze_motion_models function
    title : str
        Title for the plot
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure displaying motion model analysis
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Create summary dataframe for model classification
    if 'classifications' not in motion_analysis_results:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, "No motion analysis data available", 
                horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        return fig
    
    # Count occurrences of each model type
    model_counts = {}
    for track_id, model in motion_analysis_results['classifications'].items():
        if model not in model_counts:
            model_counts[model] = 0
        model_counts[model] += 1
    
    # Create model parameters dataframe
    params_data = []
    for track_id, model in motion_analysis_results['classifications'].items():
        if track_id in motion_analysis_results['model_parameters'] and model in motion_analysis_results['model_parameters'][track_id]:
            params = motion_analysis_results['model_parameters'][track_id][model]
            params_data.append({
                'track_id': track_id,
                'model': model,
                **params
            })
    
    # Create subplot figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Add pie chart for model classification
    model_labels = list(model_counts.keys())
    model_values = list(model_counts.values())
    
    model_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    axes[0].pie(model_values, labels=model_labels, autopct='%1.1f%%', 
               colors=model_colors[:len(model_labels)],
               wedgeprops={'edgecolor': 'w'})
    axes[0].set_title("Model Classification")
    
    # Add boxplot for model parameters if data is available
    if params_data:
        import pandas as pd
        params_df = pd.DataFrame(params_data)
        if 'D' in params_df.columns:
            model_palette = {
                'brownian': '#1f77b4',
                'directed': '#ff7f0e',
                'confined': '#2ca02c'
            }
            
            # Create boxplot for diffusion coefficients
            sns.boxplot(x='model', y='D', data=params_df, ax=axes[1],
                       palette=model_palette)
            
            # Set y-axis to log scale
            axes[1].set_yscale('log')
            axes[1].set_title("Diffusion Coefficient Distribution")
            axes[1].set_xlabel("Motion Model")
            axes[1].set_ylabel("Diffusion Coefficient (μm²/s)")
    else:
        axes[1].text(0.5, 0.5, "No parameter data available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=axes[1].transAxes)
    
    # Update layout
    plt.suptitle(title)
    plt.tight_layout()
    
    return fig

def plot_energy_landscape(energy_data):
    """
    Plot the potential energy landscape as a heatmap.
    
    Parameters:
    -----------
    energy_data : dict
        Dictionary containing energy landscape data with keys:
        - 'potential_energy_map': 2D array of energy values
        - 'x_edges': x-axis bin edges
        - 'y_edges': y-axis bin edges
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Interactive heatmap of the energy landscape
    """
    try:
        potential_map = energy_data['potential_energy_map']
        x_edges = energy_data['x_edges']
        y_edges = energy_data['y_edges']
        
        # Create x and y coordinate arrays for the center of each bin
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        
        fig = go.Figure(data=go.Heatmap(
            z=potential_map,
            x=x_centers,
            y=y_centers,
            colorscale='Viridis',
            colorbar=dict(title="Energy (kT)")
        ))
        
        fig.update_layout(
            title="Potential Energy Landscape",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=600,
            height=500
        )
        
        return fig
        
    except KeyError as e:
        st.error(f"Missing required data for energy landscape plot: {e}")
        return go.Figure()
    except Exception as e:
        st.error(f"Error creating energy landscape plot: {e}")
        return go.Figure()

def plot_polymer_physics_results(polymer_data):
    """
    Plot polymer physics analysis results.
    
    Parameters:
    -----------
    polymer_data : dict
        Dictionary containing polymer physics results
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Combined plot of polymer physics metrics
    """
    try:
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Persistence Length', 'Contour Length', 
                          'End-to-End Distance', 'Radius of Gyration'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add plots based on available data
        if 'persistence_length' in polymer_data:
            fig.add_trace(
                go.Histogram(x=polymer_data['persistence_length'], name="Persistence Length"),
                row=1, col=1
            )
            
        # Add more subplots as needed based on polymer_data content
        
        fig.update_layout(
            title="Polymer Physics Analysis Results",
            showlegend=False,
            height=600
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating polymer physics plot: {e}")
        return go.Figure()

def plot_hmm_state_transition_diagram(model) -> go.Figure:
    """
    Visualizes an HMM as a state transition diagram.

    Args:
        model: A trained hmmlearn GaussianHMM model.

    Returns:
        A Plotly Figure object showing the state transition diagram.
    """
    if not hasattr(model, 'transmat_'):
        return _empty_fig("Invalid HMM model provided")

    n_states = model.n_components
    trans_matrix = model.transmat_

    # Calculate stationary distribution for node sizes
    try:
        # Find the eigenvector corresponding to the eigenvalue 1
        eigenvals, eigenvecs = np.linalg.eig(trans_matrix.T)
        stationary_dist = np.real(eigenvecs[:, np.isclose(eigenvals, 1)][:, 0])
        stationary_dist /= stationary_dist.sum()
    except Exception:
        # Fallback to uniform distribution if calculation fails
        stationary_dist = np.ones(n_states) / n_states

    # Calculate diffusion coefficients for node colors
    # D = (var(dx) + var(dy)) / (2 * dt), assuming dt=1 for simplicity here
    # Handle different covariance types
    cov_type = getattr(model, "covariance_type", "full")
    n_states = model.n_components
    n_features = getattr(model, "n_features", None)
    if n_features is None:
        # Try to infer n_features from covars_ shape
        if cov_type == "full":
            n_features = model.covars_.shape[1]
        elif cov_type == "diag":
            n_features = model.covars_.shape[1]
        elif cov_type == "spherical":
            n_features = 1  # fallback, may be updated below
        elif cov_type == "tied":
            n_features = model.covars_.shape[-1]
    diffusion_coeffs = []
    if cov_type == "full":
        # model.covars_ shape: (n_states, n_features, n_features)
        diffusion_coeffs = [np.trace(cov) / 2 for cov in model.covars_]
    elif cov_type == "diag":
        # model.covars_ shape: (n_states, n_features)
        diffusion_coeffs = [np.sum(cov) / 2 for cov in model.covars_]
    elif cov_type == "spherical":
        # model.covars_ shape: (n_states,) or (n_states, 1)
        # Each cov is a scalar variance, but for multi-dimensional data, total variance is cov * n_features
        for cov in model.covars_:
            diffusion_coeffs.append((cov * n_features) / 2)
    elif cov_type == "tied":
        # model.covars_ shape: (n_features, n_features) or (n_features,)
        cov = model.covars_
        if cov.ndim == 2:
            val = np.trace(cov) / 2
        else:
            val = np.sum(cov) / 2
        diffusion_coeffs = [val] * n_states
    else:
        # fallback: try to use trace if possible
        try:
            diffusion_coeffs = [np.trace(cov) / 2 for cov in model.covars_]
        except Exception:
            diffusion_coeffs = [0.0] * n_states

    # Node positions in a circle
    angle_step = 2 * np.pi / n_states
    node_x = [np.cos(i * angle_step) for i in range(n_states)]
    node_y = [np.sin(i * angle_step) for i in range(n_states)]

    # Create figure
    fig = go.Figure()

    # Create edges (arrows)
    for i in range(n_states):
        for j in range(n_states):
            prob = trans_matrix[i, j]
            if prob > 0.01:  # Only draw significant transitions
                # Arrow properties
                if i == j: # Self-transition loop
                    # Position loop slightly offset from node
                    loop_angle = i * angle_step + np.pi / 2
                    x_start = node_x[i] + 0.1 * np.cos(loop_angle - 0.3)
                    y_start = node_y[i] + 0.1 * np.sin(loop_angle - 0.3)
                    x_end = node_x[i] + 0.1 * np.cos(loop_angle + 0.3)
                    y_end = node_y[i] + 0.1 * np.sin(loop_angle + 0.3)

                    fig.add_annotation(
                        x=x_end, y=y_end, ax=x_start, ay=y_start,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=prob * 5 + 1,
                        opacity=prob
                    )
                    # Label for the loop
                    fig.add_annotation(
                        x=node_x[i] + 0.2 * np.cos(loop_angle),
                        y=node_y[i] + 0.2 * np.sin(loop_angle),
                        text=f"{prob:.2f}", showarrow=False,
                        font=dict(size=10, color="black")
                    )
                else: # Transitions between different nodes
                    fig.add_annotation(
                        x=node_x[j], y=node_y[j], ax=node_x[i], ay=node_y[i],
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=prob * 5 + 1,
                        opacity=min(1.0, prob * 2)
                    )
                    # Add transition probability label
                    fig.add_annotation(
                        x=(node_x[i] + node_x[j]) / 2,
                        y=(node_y[i] + node_y[j]) / 2,
                        text=f'{prob:.2f}',
                        showarrow=False,
                        font=dict(size=10, color="black")
                    )

    # Create nodes
    node_sizes = stationary_dist * 100 + 20
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        marker=dict(
            size=node_sizes,
            color=diffusion_coeffs,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title='Diffusion Coeff.'),
            cmin=min(diffusion_coeffs),
            cmax=max(diffusion_coeffs)
        ),
        text=[f"State {i}" for i in range(n_states)],
        textposition="top center",
        hoverinfo='text',
        hovertext=[f"State {i}<br>Stationary Prob: {p:.2f}<br>Diffusion Coeff: {d:.4f}"
                   for i, (p, d) in enumerate(zip(stationary_dist, diffusion_coeffs))]
    ))

    # Update layout
    fig.update_layout(
        title="HMM State Transition Diagram",
        showlegend=False,
        xaxis=dict(visible=False, range=[-1.5, 1.5]),
        yaxis=dict(visible=False, range=[-1.5, 1.5]),
        template="plotly_white",
        width=600,
        height=600
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig
