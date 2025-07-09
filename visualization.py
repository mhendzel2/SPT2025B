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
