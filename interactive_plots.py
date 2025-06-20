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
from state_manager import get_state_manager

def create_interactive_track_plot(tracks_df: pd.DataFrame, 
                                  color_by: str = 'track_id',
                                  max_tracks: int = 50,
                                  show_points: bool = False) -> go.Figure:
    """
    Create interactive track plot with click events and custom data
    
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
    
    # Calculate track statistics for hover info
    track_stats = calculate_track_stats_for_hover(plot_data)
    
    # Merge statistics with plot data
    plot_data = plot_data.merge(track_stats, on='track_id', how='left')
    
    # Create figure
    fig = go.Figure()
    
    # Color mapping
    if color_by == 'track_id':
        colors = px.colors.qualitative.Set3
        color_map = {tid: colors[i % len(colors)] for i, tid in enumerate(track_ids)}
    else:
        # Use continuous color scale for other variables
        color_values = plot_data[color_by] if color_by in plot_data.columns else plot_data['track_id']
        color_map = None
    
    # Add track lines
    for track_id in track_ids:
        track_data = plot_data[plot_data['track_id'] == track_id]
        
        if len(track_data) < 2:
            continue
        
        # Prepare custom data for hover and click events
        custom_data = prepare_custom_data(track_data)
        
        # Hover template
        hover_template = create_hover_template(track_data.iloc[0])
        
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
            customdata=custom_data,
            hovertemplate=hover_template,
            meta={'track_id': track_id, 'track_stats': track_stats.loc[track_stats['track_id'] == track_id].iloc[0].to_dict()}
        ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Track Visualization - Click tracks for details",
        xaxis_title="X Position (µm)",
        yaxis_title="Y Position (µm)",
        height=600,
        showlegend=False,
        hovermode='closest'
    )
    
    return fig

def calculate_track_stats_for_hover(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate track statistics for hover information"""
    
    stats_list = []
    
    for track_id in tracks_df['track_id'].unique():
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        
        # Basic statistics
        stats = {
            'track_id': track_id,
            'num_points': len(track_data),
            'total_displacement': calculate_total_displacement(track_data),
            'max_displacement': calculate_max_displacement(track_data),
            'mean_velocity': calculate_mean_velocity(track_data),
            'track_length': len(track_data)
        }
        
        # Add frame range if available
        if 'frame' in track_data.columns:
            stats['frame_start'] = track_data['frame'].min()
            stats['frame_end'] = track_data['frame'].max()
            stats['duration'] = stats['frame_end'] - stats['frame_start'] + 1
        
        stats_list.append(stats)
    
    return pd.DataFrame(stats_list)

def calculate_total_displacement(track_data: pd.DataFrame) -> float:
    """Calculate total displacement for a track"""
    if len(track_data) < 2:
        return 0.0
    
    start_pos = (track_data['x'].iloc[0], track_data['y'].iloc[0])
    end_pos = (track_data['x'].iloc[-1], track_data['y'].iloc[-1])
    
    return np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)

def calculate_max_displacement(track_data: pd.DataFrame) -> float:
    """Calculate maximum displacement from starting position"""
    if len(track_data) < 2:
        return 0.0
    
    start_x, start_y = track_data['x'].iloc[0], track_data['y'].iloc[0]
    displacements = np.sqrt((track_data['x'] - start_x)**2 + (track_data['y'] - start_y)**2)
    
    return displacements.max()

def calculate_mean_velocity(track_data: pd.DataFrame) -> float:
    """Calculate mean velocity for a track"""
    if len(track_data) < 2:
        return 0.0
    
    # Calculate step displacements
    dx = np.diff(track_data['x'].values)
    dy = np.diff(track_data['y'].values)
    step_distances = np.sqrt(dx**2 + dy**2)
    
    return np.mean(step_distances)

def prepare_custom_data(track_data: pd.DataFrame) -> np.ndarray:
    """Prepare custom data array for Plotly hover and click events"""
    
    custom_data = []
    
    # Vectorized preparation of custom data
    base_data = [
        track_data['track_id'].values,
        track_data.get('frame', pd.Series([0] * len(track_data))).values,
        track_data['x'].values,
        track_data['y'].values
    ]
    
    # Add additional columns if available
    for col in ['velocity', 'displacement', 'intensity']:
        if col in track_data.columns:
            base_data.append(track_data[col].values)
        else:
            base_data.append(np.full(len(track_data), None))
    
    custom_data = np.array(base_data).T.tolist()
    
    return np.array(custom_data)

def create_hover_template(track_sample: pd.Series) -> str:
    """Create hover template with track information"""
    
    template = (
        "<b>Track %{meta.track_id}</b><br>"
        "Position: (%{x:.2f}, %{y:.2f}) µm<br>"
        "Points: %{meta.track_stats.num_points}<br>"
        "Total Displacement: %{meta.track_stats.total_displacement:.2f} µm<br>"
        "Mean Velocity: %{meta.track_stats.mean_velocity:.3f} µm/frame<br>"
    )
    
    # Add frame info if available
    if 'frame' in track_sample.index:
        template += "Frame: %{customdata[1]}<br>"
    
    template += "<extra></extra>"  # Remove default trace box
    
    return template

def handle_track_selection(selection_data: Dict[str, Any], tracks_df: pd.DataFrame):
    """Handle track selection from plot click events"""
    
    if not selection_data or 'points' not in selection_data:
        return
    
    selected_points = selection_data['points']
    
    if not selected_points:
        st.info("Click on tracks in the plot above to see detailed statistics")
        return
    
    # Extract selected track IDs
    selected_track_ids = []
    for point in selected_points:
        if 'meta' in point and 'track_id' in point['meta']:
            selected_track_ids.append(point['meta']['track_id'])
    
    if not selected_track_ids:
        return
    
    # Remove duplicates while preserving order
    selected_track_ids = list(dict.fromkeys(selected_track_ids))
    
    st.subheader(f"Selected Track Details ({len(selected_track_ids)} tracks)")
    
    # Display detailed statistics for selected tracks
    display_selected_track_details(selected_track_ids, tracks_df)

def display_selected_track_details(track_ids: List[int], tracks_df: pd.DataFrame):
    """Display detailed statistics for selected tracks"""
    
    # Calculate detailed statistics
    detailed_stats = []
    
    for track_id in track_ids:
        track_data = tracks_df[tracks_df['track_id'] == track_id]
        
        if track_data.empty:
            continue
        
        stats = calculate_detailed_track_stats(track_data)
        stats['track_id'] = track_id
        detailed_stats.append(stats)
    
    if not detailed_stats:
        st.warning("No data found for selected tracks")
        return
    
    # Convert to DataFrame for display
    stats_df = pd.DataFrame(detailed_stats)
    
    # Display summary table
    st.write("**Summary Statistics:**")
    display_columns = [
        'track_id', 'num_points', 'duration', 'total_displacement',
        'max_displacement', 'mean_velocity', 'velocity_std'
    ]
    
    # Filter columns that exist
    available_columns = [col for col in display_columns if col in stats_df.columns]
    
    st.dataframe(
        stats_df[available_columns].round(3),
        use_container_width=True,
        hide_index=True
    )
    
    # Display individual track plots for first few tracks
    if len(track_ids) <= 5:
        st.write("**Individual Track Trajectories:**")
        
        for track_id in track_ids[:5]:  # Limit to 5 tracks
            plot_individual_track(track_id, tracks_df)

def calculate_detailed_track_stats(track_data: pd.DataFrame) -> Dict[str, Any]:
    """Calculate comprehensive statistics for a single track"""
    
    stats = {}
    
    # Basic properties
    stats['num_points'] = len(track_data)
    
    # Temporal properties
    if 'frame' in track_data.columns:
        stats['frame_start'] = track_data['frame'].min()
        stats['frame_end'] = track_data['frame'].max()
        stats['duration'] = stats['frame_end'] - stats['frame_start'] + 1
    else:
        stats['duration'] = len(track_data)
    
    # Spatial properties
    stats['total_displacement'] = calculate_total_displacement(track_data)
    stats['max_displacement'] = calculate_max_displacement(track_data)
    
    # Velocity analysis
    if len(track_data) >= 2:
        velocities = calculate_track_velocities(track_data)
        stats['mean_velocity'] = np.mean(velocities)
        stats['velocity_std'] = np.std(velocities)
        stats['max_velocity'] = np.max(velocities)
    else:
        stats['mean_velocity'] = 0
        stats['velocity_std'] = 0
        stats['max_velocity'] = 0
    
    # Spatial extent
    stats['x_range'] = track_data['x'].max() - track_data['x'].min()
    stats['y_range'] = track_data['y'].max() - track_data['y'].min()
    
    # Center of mass
    stats['center_x'] = track_data['x'].mean()
    stats['center_y'] = track_data['y'].mean()
    
    return stats

def calculate_track_velocities(track_data: pd.DataFrame) -> np.ndarray:
    """Calculate instantaneous velocities for a track"""
    
    if len(track_data) < 2:
        return np.array([])
    
    # Calculate step displacements
    dx = np.diff(track_data['x'].values)
    dy = np.diff(track_data['y'].values)
    
    # Calculate step distances (velocities)
    velocities = np.sqrt(dx**2 + dy**2)
    
    return velocities

def plot_individual_track(track_id: int, tracks_df: pd.DataFrame):
    """Plot individual track trajectory"""
    
    track_data = tracks_df[tracks_df['track_id'] == track_id]
    
    if track_data.empty:
        return
    
    # Create subplot with trajectory and velocity
    col1, col2 = st.columns(2)
    
    with col1:
        # Trajectory plot
        fig_traj = go.Figure()
        
        fig_traj.add_trace(go.Scatter(
            x=track_data['x'],
            y=track_data['y'],
            mode='lines+markers',
            name=f'Track {track_id}',
            line=dict(width=2),
            marker=dict(size=4)
        ))
        
        # Mark start and end points
        fig_traj.add_trace(go.Scatter(
            x=[track_data['x'].iloc[0]],
            y=[track_data['y'].iloc[0]],
            mode='markers',
            name='Start',
            marker=dict(size=10, color='green', symbol='circle')
        ))
        
        fig_traj.add_trace(go.Scatter(
            x=[track_data['x'].iloc[-1]],
            y=[track_data['y'].iloc[-1]],
            mode='markers',
            name='End',
            marker=dict(size=10, color='red', symbol='square')
        ))
        
        fig_traj.update_layout(
            title=f"Track {track_id} Trajectory",
            xaxis_title="X (µm)",
            yaxis_title="Y (µm)",
            height=300,
            showlegend=True
        )
        
        st.plotly_chart(fig_traj, use_container_width=True)
    
    with col2:
        # Velocity plot
        if len(track_data) >= 2:
            velocities = calculate_track_velocities(track_data)
            
            fig_vel = go.Figure()
            
            fig_vel.add_trace(go.Scatter(
                y=velocities,
                mode='lines+markers',
                name='Velocity',
                line=dict(width=2),
                marker=dict(size=4)
            ))
            
            fig_vel.update_layout(
                title=f"Track {track_id} Velocity",
                xaxis_title="Step",
                yaxis_title="Velocity (µm/frame)",
                height=300
            )
            
            st.plotly_chart(fig_vel, use_container_width=True)

def create_msd_interactive_plot(msd_data: Dict[str, Any]) -> go.Figure:
    """Create interactive MSD plot with track selection"""
    
    fig = go.Figure()
    
    # Add ensemble average if available
    if 'ensemble_msd' in msd_data:
        ensemble = msd_data['ensemble_msd']
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
    if 'individual_msds' in msd_data:
        individual = msd_data['individual_msds']
        
        for track_id, track_msd in individual.items():
            fig.add_trace(go.Scatter(
                x=track_msd['lag_time'],
                y=track_msd['msd'],
                mode='lines',
                name=f'Track {track_id}',
                opacity=0.3,
                line=dict(width=1),
                hovertemplate=f"<b>Track {track_id}</b><br>Lag Time: %{{x:.3f}} s<br>MSD: %{{y:.3f}} µm²<extra></extra>",
                meta={'track_id': track_id}
            ))
    
    fig.update_layout(
        title="Interactive Mean Squared Displacement",
        xaxis_title="Lag Time (s)",
        yaxis_title="MSD (µm²)",
        height=500,
        hovermode='closest'
    )
    
    return fig
