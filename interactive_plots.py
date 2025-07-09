"""
Interactive Plot Components with Click Events for Track Selection
Provides Plotly-based interactive visualizations with custom data handling
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, Any, List, Optional, Tuple

def create_interactive_track_plot(tracks_df: pd.DataFrame, 
                                  color_by: str = 'track_id',
                                  max_tracks: int = 50,
                                  show_points: bool = False) -> go.Figure:
    """
    Create interactive track plot with click events and proper hover data
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with x, y, track_id columns
    color_by : str
        Column to use for color coding
    max_tracks : int
        Maximum number of tracks to display
    show_points : bool
        Whether to show individual points
        
    Returns
    -------
    go.Figure
        Interactive Plotly figure
    """
    
    # Limit tracks for performance
    track_ids = tracks_df['track_id'].unique()[:max_tracks]
    plot_data = tracks_df[tracks_df['track_id'].isin(track_ids)].copy()
    
    if plot_data.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No track data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate track statistics for each track
    track_stats = {}
    for track_id in track_ids:
        track_data = plot_data[plot_data['track_id'] == track_id]
        if len(track_data) > 0:
            track_stats[track_id] = calculate_track_statistics(track_data)
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping
    if color_by == 'track_id':
        colors = px.colors.qualitative.Set3
        color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(track_ids)}
    else:
        color_map = None
    
    # Add track lines
    for track_id in track_ids:
        track_data = plot_data[plot_data['track_id'] == track_id].sort_values('frame' if 'frame' in plot_data.columns else plot_data.index)
        
        if len(track_data) < 2:
            continue
        
        # Get track statistics
        stats = track_stats.get(track_id, {})
        
        # Prepare hover information using customdata
        customdata = np.column_stack([
            [track_id] * len(track_data),
            track_data.get('frame', range(len(track_data))).values,
            [stats.get('num_points', len(track_data))] * len(track_data),
            [stats.get('net_displacement', 0.0)] * len(track_data),
            [stats.get('max_radial_displacement', 0.0)] * len(track_data),
            [stats.get('mean_velocity', 0.0)] * len(track_data),
            [stats.get('duration', len(track_data))] * len(track_data)
        ])
        
        # Create hover template
        hover_template = (
            "<b>Track %{customdata[0]}</b><br>"
            "Position: (%{x:.2f}, %{y:.2f}) µm<br>"
            "Frame: %{customdata[1]}<br>"
            "Points: %{customdata[2]}<br>"
            "Net Displacement: %{customdata[3]:.2f} µm<br>"
            "Max Radial: %{customdata[4]:.2f} µm<br>"
            "Mean Velocity: %{customdata[5]:.3f} µm/frame<br>"
            "Duration: %{customdata[6]} frames"
            "<extra></extra>"
        )
        
        # Color for this track
        if color_map:
            line_color = color_map[track_id]
        else:
            line_color = px.colors.sequential.Viridis[int((track_id % 10) / 10 * len(px.colors.sequential.Viridis))]
        
        # Add track line
        fig.add_trace(go.Scatter(
            x=track_data['x'],
            y=track_data['y'],
            mode='lines+markers' if show_points else 'lines',
            name=f'Track {track_id}',
            line=dict(color=line_color, width=2),
            marker=dict(size=4) if show_points else None,
            customdata=customdata,
            hovertemplate=hover_template,
            showlegend=len(track_ids) <= 10  # Only show legend for small number of tracks
        ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Track Visualization - Click tracks for details",
        xaxis_title="X Position (µm)",
        yaxis_title="Y Position (µm)",
        height=600,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    # Equal aspect ratio
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    return fig

def calculate_track_statistics(track_data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for a single track
    
    Parameters
    ----------
    track_data : pd.DataFrame
        Data for a single track
        
    Returns
    -------
    Dict[str, float]
        Dictionary of track statistics
    """
    stats = {}
    
    # Basic properties
    stats['num_points'] = len(track_data)
    
    # Temporal properties
    if 'frame' in track_data.columns:
        stats['frame_start'] = float(track_data['frame'].min())
        stats['frame_end'] = float(track_data['frame'].max())
        stats['duration'] = stats['frame_end'] - stats['frame_start'] + 1
    else:
        stats['duration'] = float(len(track_data))
    
    # Spatial properties
    stats['net_displacement'] = calculate_net_displacement(track_data)
    stats['max_radial_displacement'] = calculate_max_radial_displacement(track_data)
    stats['path_length'] = calculate_path_length(track_data)
    
    # Velocity analysis
    if len(track_data) >= 2:
        velocities = calculate_step_velocities(track_data)
        stats['mean_velocity'] = float(np.mean(velocities))
        stats['velocity_std'] = float(np.std(velocities))
        stats['max_velocity'] = float(np.max(velocities))
    else:
        stats['mean_velocity'] = 0.0
        stats['velocity_std'] = 0.0
        stats['max_velocity'] = 0.0
    
    # Spatial extent
    stats['x_range'] = float(track_data['x'].max() - track_data['x'].min())
    stats['y_range'] = float(track_data['y'].max() - track_data['y'].min())
    
    # Center of mass
    stats['center_x'] = float(track_data['x'].mean())
    stats['center_y'] = float(track_data['y'].mean())
    
    # Straightness (net displacement / path length)
    if stats['path_length'] > 0:
        stats['straightness'] = stats['net_displacement'] / stats['path_length']
    else:
        stats['straightness'] = 0.0
    
    return stats

def calculate_net_displacement(track_data: pd.DataFrame) -> float:
    """
    Calculate net displacement (start-to-end distance) for a track
    
    Parameters
    ----------
    track_data : pd.DataFrame
        Track data with x, y columns
        
    Returns
    -------
    float
        Net displacement in micrometers
    """
    if len(track_data) < 2:
        return 0.0
    
    start_pos = (track_data['x'].iloc[0], track_data['y'].iloc[0])
    end_pos = (track_data['x'].iloc[-1], track_data['y'].iloc[-1])
    
    return float(np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2))

def calculate_max_radial_displacement(track_data: pd.DataFrame) -> float:
    """
    Calculate maximum radial displacement from starting position
    
    Parameters
    ----------
    track_data : pd.DataFrame
        Track data with x, y columns
        
    Returns
    -------
    float
        Maximum radial displacement in micrometers
    """
    if len(track_data) < 2:
        return 0.0
    
    start_x, start_y = track_data['x'].iloc[0], track_data['y'].iloc[0]
    radial_distances = np.sqrt((track_data['x'] - start_x)**2 + (track_data['y'] - start_y)**2)
    
    return float(radial_distances.max())

def calculate_path_length(track_data: pd.DataFrame) -> float:
    """
    Calculate total path length (sum of step distances) for a track
    
    Parameters
    ----------
    track_data : pd.DataFrame
        Track data with x, y columns
        
    Returns
    -------
    float
        Total path length in micrometers
    """
    if len(track_data) < 2:
        return 0.0
    
    # Calculate step distances
    dx = np.diff(track_data['x'].values)
    dy = np.diff(track_data['y'].values)
    step_distances = np.sqrt(dx**2 + dy**2)
    
    return float(np.sum(step_distances))

def calculate_step_velocities(track_data: pd.DataFrame) -> np.ndarray:
    """
    Calculate instantaneous velocities for a track
    
    Parameters
    ----------
    track_data : pd.DataFrame
        Track data with x, y columns
        
    Returns
    -------
    np.ndarray
        Array of step velocities
    """
    if len(track_data) < 2:
        return np.array([])
    
    # Calculate step displacements
    dx = np.diff(track_data['x'].values)
    dy = np.diff(track_data['y'].values)
    
    # Calculate step distances (velocities assuming unit time steps)
    velocities = np.sqrt(dx**2 + dy**2)
    
    return velocities

def handle_track_selection(selection_data: Dict[str, Any], tracks_df: pd.DataFrame):
    """
    Handle track selection from plot click events with improved error handling
    
    Parameters
    ----------
    selection_data : Dict[str, Any]
        Selection data from Plotly click event
    tracks_df : pd.DataFrame
        Complete track data
    """
    if not selection_data or 'points' not in selection_data:
        st.info("Click on tracks in the plot above to see detailed statistics")
        return
    
    selected_points = selection_data['points']
    
    if not selected_points:
        st.info("Click on tracks in the plot above to see detailed statistics")
        return
    
    # Extract selected track IDs from customdata
    selected_track_ids = []
    for point in selected_points:
        try:
            if 'customdata' in point and point['customdata'] is not None:
                track_id = int(point['customdata'][0])  # First element is track_id
                selected_track_ids.append(track_id)
        except (KeyError, IndexError, ValueError, TypeError):
            continue
    
    if not selected_track_ids:
        st.warning("Could not extract track information from selection")
        return
    
    # Remove duplicates while preserving order
    selected_track_ids = list(dict.fromkeys(selected_track_ids))
    
    st.subheader(f"Selected Track Details ({len(selected_track_ids)} tracks)")
    
    # Display detailed statistics for selected tracks
    display_selected_track_details(selected_track_ids, tracks_df)

def display_selected_track_details(track_ids: List[int], tracks_df: pd.DataFrame):
    """
    Display detailed statistics for selected tracks with improved layout
    
    Parameters
    ----------
    track_ids : List[int]
        List of selected track IDs
    tracks_df : pd.DataFrame
        Complete track data
    """
    # Calculate detailed statistics
    detailed_stats = []
    
    for track_id in track_ids:
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        
        if track_data.empty:
            continue
        
        stats = calculate_track_statistics(track_data)
        stats['track_id'] = track_id
        detailed_stats.append(stats)
    
    if not detailed_stats:
        st.warning("No data found for selected tracks")
        return
    
    # Convert to DataFrame for display
    stats_df = pd.DataFrame(detailed_stats)
    
    # Display summary table with improved formatting
    st.write("**Summary Statistics:**")
    display_columns = [
        'track_id', 'num_points', 'duration', 'net_displacement',
        'max_radial_displacement', 'path_length', 'mean_velocity', 
        'velocity_std', 'straightness'
    ]
    
    # Filter columns that exist and format for display
    available_columns = [col for col in display_columns if col in stats_df.columns]
    display_df = stats_df[available_columns].copy()
    
    # Round numerical columns
    numerical_columns = display_df.select_dtypes(include=[np.number]).columns
    display_df[numerical_columns] = display_df[numerical_columns].round(3)
    
    # Rename columns for better display
    column_rename = {
        'track_id': 'Track ID',
        'num_points': 'Points',
        'duration': 'Duration (frames)',
        'net_displacement': 'Net Displacement (µm)',
        'max_radial_displacement': 'Max Radial (µm)',
        'path_length': 'Path Length (µm)',
        'mean_velocity': 'Mean Velocity (µm/frame)',
        'velocity_std': 'Velocity Std (µm/frame)',
        'straightness': 'Straightness'
    }
    
    display_df = display_df.rename(columns=column_rename)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Display individual track plots for first few tracks
    if len(track_ids) <= 5:
        st.write("**Individual Track Trajectories:**")
        
        for track_id in track_ids[:5]:  # Limit to 5 tracks
            plot_individual_track(track_id, tracks_df)

def plot_individual_track(track_id: int, tracks_df: pd.DataFrame):
    """
    Plot individual track trajectory with improved visualizations
    
    Parameters
    ----------
    track_id : int
        Track ID to plot
    tracks_df : pd.DataFrame
        Complete track data
    """
    track_data = tracks_df[tracks_df['track_id'] == track_id].sort_values(
        'frame' if 'frame' in tracks_df.columns else tracks_df.index
    )
    
    if track_data.empty:
        st.warning(f"No data found for track {track_id}")
        return
    
    # Create subplot with trajectory and velocity
    col1, col2 = st.columns(2)
    
    with col1:
        # Trajectory plot
        fig_traj = go.Figure()
        
        # Main trajectory
        fig_traj.add_trace(go.Scatter(
            x=track_data['x'],
            y=track_data['y'],
            mode='lines+markers',
            name=f'Track {track_id}',
            line=dict(width=2, color='blue'),
            marker=dict(size=4, color='lightblue')
        ))
        
        # Mark start point
        fig_traj.add_trace(go.Scatter(
            x=[track_data['x'].iloc[0]],
            y=[track_data['y'].iloc[0]],
            mode='markers',
            name='Start',
            marker=dict(size=12, color='green', symbol='circle'),
            hovertemplate="Start<br>(%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
        
        # Mark end point
        fig_traj.add_trace(go.Scatter(
            x=[track_data['x'].iloc[-1]],
            y=[track_data['y'].iloc[-1]],
            mode='markers',
            name='End',
            marker=dict(size=12, color='red', symbol='square'),
            hovertemplate="End<br>(%{x:.2f}, %{y:.2f})<extra></extra>"
        ))
        
        fig_traj.update_layout(
            title=f"Track {track_id} Trajectory",
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            height=300,
            showlegend=True
        )
        
        # Equal aspect ratio
        fig_traj.update_yaxes(scaleanchor="x", scaleratio=1)
        
        st.plotly_chart(fig_traj, use_container_width=True)
    
    with col2:
        # Velocity plot
        if len(track_data) >= 2:
            velocities = calculate_step_velocities(track_data)
            
            fig_vel = go.Figure()
            
            fig_vel.add_trace(go.Scatter(
                y=velocities,
                mode='lines+markers',
                name='Velocity',
                line=dict(width=2, color='orange'),
                marker=dict(size=4),
                hovertemplate="Step %{x}<br>Velocity: %{y:.3f} µm/frame<extra></extra>"
            ))
            
            # Add mean velocity line
            mean_vel = np.mean(velocities)
            fig_vel.add_hline(
                y=mean_vel,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {mean_vel:.3f}"
            )
            
            fig_vel.update_layout(
                title=f"Track {track_id} Velocity Profile",
                xaxis_title="Step Number",
                yaxis_title="Velocity (µm/frame)",
                height=300
            )
            
            st.plotly_chart(fig_vel, use_container_width=True)
        else:
            st.info("Track too short for velocity analysis")

def create_msd_interactive_plot(msd_data: Dict[str, Any]) -> go.Figure:
    """
    Create interactive MSD plot with improved error handling and display
    
    Parameters
    ----------
    msd_data : Dict[str, Any]
        MSD data dictionary
        
    Returns
    -------
    go.Figure
        Interactive MSD plot
    """
    fig = go.Figure()
    
    # Check if data is available
    if not msd_data:
        fig.add_annotation(
            text="No MSD data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Add ensemble average if available
    if 'ensemble_msd' in msd_data and msd_data['ensemble_msd'] is not None:
        ensemble = msd_data['ensemble_msd']
        if 'lag_time' in ensemble and 'msd' in ensemble:
            fig.add_trace(go.Scatter(
                x=ensemble['lag_time'],
                y=ensemble['msd'],
                mode='lines+markers',
                name='Ensemble Average',
                line=dict(width=4, color='red'),
                marker=dict(size=6),
                hovertemplate="<b>Ensemble Average</b><br>Lag Time: %{x:.3f} s<br>MSD: %{y:.3f} µm²<extra></extra>"
            ))
    
    # Add individual track MSDs
    if 'individual_msds' in msd_data and msd_data['individual_msds']:
        individual = msd_data['individual_msds']
        
        # Show only first few tracks to avoid overcrowding
        track_ids = list(individual.keys())[:20]  # Limit to 20 tracks
        
        for track_id in track_ids:
            track_msd = individual[track_id]
            if 'lag_time' in track_msd and 'msd' in track_msd:
                fig.add_trace(go.Scatter(
                    x=track_msd['lag_time'],
                    y=track_msd['msd'],
                    mode='lines',
                    name=f'Track {track_id}',
                    opacity=0.3,
                    line=dict(width=1),
                    hovertemplate=f"<b>Track {track_id}</b><br>Lag Time: %{{x:.3f}} s<br>MSD: %{{y:.3f}} µm²<extra></extra>",
                    showlegend=False  # Don't show individual tracks in legend
                ))
    
    # Add theoretical lines if available
    if 'theoretical_fits' in msd_data and msd_data['theoretical_fits']:
        fits = msd_data['theoretical_fits']
        for fit_name, fit_data in fits.items():
            if 'lag_time' in fit_data and 'msd' in fit_data:
                fig.add_trace(go.Scatter(
                    x=fit_data['lag_time'],
                    y=fit_data['msd'],
                    mode='lines',
                    name=f'{fit_name} fit',
                    line=dict(width=3, dash='dash'),
                    hovertemplate=f"<b>{fit_name} Fit</b><br>Lag Time: %{{x:.3f}} s<br>MSD: %{{y:.3f}} µm²<extra></extra>"
                ))
    
    fig.update_layout(
        title="Interactive Mean Squared Displacement",
        xaxis_title="Lag Time (s)",
        yaxis_title="MSD (µm²)",
        height=500,
        hovermode='closest',
        plot_bgcolor='white'
    )
    
    return fig

def create_velocity_distribution_plot(tracks_df: pd.DataFrame) -> go.Figure:
    """
    Create interactive velocity distribution plot
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
        
    Returns
    -------
    go.Figure
        Interactive velocity distribution plot
    """
    fig = go.Figure()
    
    if tracks_df.empty:
        fig.add_annotation(
            text="No track data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Calculate velocities for all tracks
    all_velocities = []
    
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        if len(track_data) >= 2:
            velocities = calculate_step_velocities(track_data)
            all_velocities.extend(velocities)
    
    if not all_velocities:
        fig.add_annotation(
            text="Insufficient data for velocity analysis",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16)
        )
        return fig
    
    # Create histogram
    fig.add_trace(go.Histogram(
        x=all_velocities,
        nbinsx=50,
        name='Velocity Distribution',
        marker_color='lightblue',
        marker_line_color='blue',
        marker_line_width=1,
        hovertemplate="Velocity: %{x:.3f} µm/frame<br>Count: %{y}<extra></extra>"
    ))
    
    # Add statistics
    mean_vel = np.mean(all_velocities)
    median_vel = np.median(all_velocities)
    
    fig.add_vline(x=mean_vel, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_vel:.3f}")
    fig.add_vline(x=median_vel, line_dash="dot", line_color="green", 
                  annotation_text=f"Median: {median_vel:.3f}")
    
    fig.update_layout(
        title="Velocity Distribution",
        xaxis_title="Velocity (µm/frame)",
        yaxis_title="Count",
        height=400,
        plot_bgcolor='white'
    )
    
    return fig
    return fig
