"""
Multi-channel analysis module for SPT Analysis application.
Provides tools for analyzing multiple particle tracking channels and compartment segmentation.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st
from scipy.spatial.distance import cdist
try:
    from enhanced_segmentation import classify_particles_by_contour
    ENHANCED_SEGMENTATION_AVAILABLE = True
except ImportError:
    ENHANCED_SEGMENTATION_AVAILABLE = False


def analyze_channel_colocalization(channel1_tracks: pd.DataFrame, channel2_tracks: pd.DataFrame,
                                 distance_threshold: float = 1.0, frame_tolerance: int = 1) -> Dict[str, Any]:
    """
    Analyze colocalization between two particle tracking channels.
    
    Parameters
    ----------
    channel1_tracks : pd.DataFrame
        First channel track data
    channel2_tracks : pd.DataFrame
        Second channel track data
    distance_threshold : float
        Maximum distance for colocalization (in micrometers)
    frame_tolerance : int
        Frame tolerance for temporal colocalization
        
    Returns
    -------
    Dict[str, Any]
        Colocalization analysis results
    """
    results = {
        'colocalized_pairs': [],
        'channel1_colocalized_fraction': 0,
        'channel2_colocalized_fraction': 0,
        'mean_colocalization_distance': 0,
        'temporal_overlap': [],
        'frame_by_frame_stats': []
    }
    
    # Get common frames
    frames1 = set(channel1_tracks['frame'].unique())
    frames2 = set(channel2_tracks['frame'].unique())
    common_frames = sorted(frames1.intersection(frames2))
    
    if not common_frames:
        return results
    
    colocalized_pairs = []
    frame_stats = []
    
    for frame in common_frames:
        # Get particles in this frame
        frame1_particles = channel1_tracks[channel1_tracks['frame'] == frame]
        frame2_particles = channel2_tracks[channel2_tracks['frame'] == frame]
        
        if len(frame1_particles) == 0 or len(frame2_particles) == 0:
            frame_stats.append({
                'frame': frame,
                'channel1_particles': len(frame1_particles),
                'channel2_particles': len(frame2_particles),
                'colocalized_pairs': 0,
                'mean_distance': np.nan
            })
            continue
        
        # Calculate distances between all particle pairs
        coords1 = frame1_particles[['x', 'y']].values
        coords2 = frame2_particles[['x', 'y']].values
        distances = cdist(coords1, coords2)
        
        # Find colocalized pairs
        frame_pairs = []
        frame_distances = []
        
        # Use Hungarian algorithm-like approach for optimal pairing
        min_distances = np.min(distances, axis=1)
        for i, min_dist in enumerate(min_distances):
            if min_dist <= distance_threshold:
                j = np.argmin(distances[i, :])
                
                # Check if this is the best match for particle j in channel 2
                if i == np.argmin(distances[:, j]):
                    frame_pairs.append({
                        'frame': frame,
                        'channel1_particle_id': frame1_particles.iloc[i]['track_id'],
                        'channel2_particle_id': frame2_particles.iloc[j]['track_id'],
                        'distance': min_dist,
                        'channel1_x': frame1_particles.iloc[i]['x'],
                        'channel1_y': frame1_particles.iloc[i]['y'],
                        'channel2_x': frame2_particles.iloc[j]['x'],
                        'channel2_y': frame2_particles.iloc[j]['y']
                    })
                    frame_distances.append(min_dist)
        
        colocalized_pairs.extend(frame_pairs)
        
        frame_stats.append({
            'frame': frame,
            'channel1_particles': len(frame1_particles),
            'channel2_particles': len(frame2_particles),
            'colocalized_pairs': len(frame_pairs),
            'mean_distance': np.mean(frame_distances) if frame_distances else np.nan
        })
    
    # Calculate overall statistics
    total_channel1_particles = len(channel1_tracks['track_id'].unique())
    total_channel2_particles = len(channel2_tracks['track_id'].unique())
    
    if colocalized_pairs:
        colocalized_ch1_tracks = set([pair['channel1_particle_id'] for pair in colocalized_pairs])
        colocalized_ch2_tracks = set([pair['channel2_particle_id'] for pair in colocalized_pairs])
        
        results['channel1_colocalized_fraction'] = len(colocalized_ch1_tracks) / total_channel1_particles
        results['channel2_colocalized_fraction'] = len(colocalized_ch2_tracks) / total_channel2_particles
        results['mean_colocalization_distance'] = np.mean([pair['distance'] for pair in colocalized_pairs])
    
    results['colocalized_pairs'] = colocalized_pairs
    results['frame_by_frame_stats'] = frame_stats
    
    return results

def analyze_compartment_occupancy(tracks_df: pd.DataFrame, compartments: List[Dict[str, Any]],
                                pixel_size: float = 1.0) -> Dict[str, Any]:
    """
    Analyze how particles occupy different compartments over time.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data
    compartments : List[Dict[str, Any]]
        List of segmented compartments
    pixel_size : float
        Pixel size for coordinate conversion
        
    Returns
    -------
    Dict[str, Any]
        Compartment occupancy analysis results
    """
    if not compartments:
        return {'error': 'No compartments provided'}
    
    results = {
        'track_compartment_assignments': [],
        'compartment_statistics': [],
        'transition_matrix': None,
        'dwell_times': []
    }
    
    # Convert track coordinates
    tracks_um = tracks_df.copy()
    tracks_um['x_um'] = tracks_df['x'] * pixel_size
    tracks_um['y_um'] = tracks_df['y'] * pixel_size
    
    # Assign each track point to a compartment using vectorized operations
    tracks_enhanced = tracks_um.copy()
    tracks_enhanced['compartment'] = 'outside'
    
    for comp in compartments:
        bbox = comp['bbox_um']
        # Create boolean mask for points within this compartment
        in_compartment = (
            (tracks_enhanced['x_um'] >= bbox['x1']) & 
            (tracks_enhanced['x_um'] <= bbox['x2']) &
            (tracks_enhanced['y_um'] >= bbox['y1']) & 
            (tracks_enhanced['y_um'] <= bbox['y2'])
        )
        tracks_enhanced.loc[in_compartment, 'compartment'] = comp['id']
    
    tracks_enhanced = tracks_enhanced.to_dict('records')
    
    tracks_enhanced_df = pd.DataFrame(tracks_enhanced)
    
    # Calculate compartment statistics
    compartment_stats = []
    for comp in compartments:
        comp_tracks = tracks_enhanced_df[tracks_enhanced_df['compartment'] == comp['id']]
        unique_tracks = comp_tracks['track_id'].nunique()
        total_detections = len(comp_tracks)
        
        if total_detections > 0:
            mean_dwell_time = comp_tracks.groupby('track_id').size().mean()
        else:
            mean_dwell_time = 0
        
        compartment_stats.append({
            'compartment_id': comp['id'],
            'unique_tracks': unique_tracks,
            'total_detections': total_detections,
            'mean_dwell_time': mean_dwell_time,
            'area_um2': comp['area_um2'],
            'density_tracks_per_um2': unique_tracks / comp['area_um2'] if comp['area_um2'] > 0 else 0
        })
    
    # Analyze transitions between compartments
    transition_data = []
    for track_id in tracks_enhanced_df['track_id'].unique():
        track_data = tracks_enhanced_df[tracks_enhanced_df['track_id'] == track_id].sort_values('frame')
        compartments_visited = track_data['compartment'].tolist()
        
        for i in range(len(compartments_visited) - 1):
            from_comp = compartments_visited[i]
            to_comp = compartments_visited[i + 1]
            if from_comp != to_comp:
                transition_data.append({
                    'track_id': track_id,
                    'from_compartment': from_comp,
                    'to_compartment': to_comp,
                    'frame': track_data.iloc[i + 1]['frame']
                })
    
    results['track_compartment_assignments'] = tracks_enhanced
    results['compartment_statistics'] = compartment_stats
    results['transitions'] = transition_data
    
    return results

def compare_channel_dynamics(channel1_results: Dict[str, Any], channel2_results: Dict[str, Any],
                           channel1_name: str = "Channel 1", channel2_name: str = "Channel 2") -> Dict[str, Any]:
    """
    Compare dynamics between two particle tracking channels.
    
    Parameters
    ----------
    channel1_results : Dict[str, Any]
        Analysis results from first channel
    channel2_results : Dict[str, Any]
        Analysis results from second channel
    channel1_name : str
        Name of first channel
    channel2_name : str
        Name of second channel
        
    Returns
    -------
    Dict[str, Any]
        Comparative analysis results
    """
    comparison = {
        'diffusion_comparison': {},
        'motion_comparison': {},
        'statistical_tests': {}
    }
    
    # Compare diffusion coefficients
    if 'diffusion' in channel1_results and 'diffusion' in channel2_results:
        ch1_diffusion = channel1_results['diffusion']
        ch2_diffusion = channel2_results['diffusion']
        
        if ('track_results' in ch1_diffusion and 'track_results' in ch2_diffusion and
            ch1_diffusion['track_results'] is not None and ch2_diffusion['track_results'] is not None):
            
            ch1_D = ch1_diffusion['track_results']['diffusion_coefficient'].dropna()
            ch2_D = ch2_diffusion['track_results']['diffusion_coefficient'].dropna()
            
            comparison['diffusion_comparison'] = {
                f'{channel1_name}_mean_D': ch1_D.mean() if len(ch1_D) > 0 else np.nan,
                f'{channel2_name}_mean_D': ch2_D.mean() if len(ch2_D) > 0 else np.nan,
                f'{channel1_name}_median_D': ch1_D.median() if len(ch1_D) > 0 else np.nan,
                f'{channel2_name}_median_D': ch2_D.median() if len(ch2_D) > 0 else np.nan,
                f'{channel1_name}_std_D': ch1_D.std() if len(ch1_D) > 0 else np.nan,
                f'{channel2_name}_std_D': ch2_D.std() if len(ch2_D) > 0 else np.nan
            }
    
    # Compare motion characteristics
    if 'motion' in channel1_results and 'motion' in channel2_results:
        ch1_motion = channel1_results['motion']
        ch2_motion = channel2_results['motion']
        
        if ('track_results' in ch1_motion and 'track_results' in ch2_motion and
            ch1_motion['track_results'] is not None and ch2_motion['track_results'] is not None):
            
            ch1_speed = ch1_motion['track_results']['mean_speed'].dropna()
            ch2_speed = ch2_motion['track_results']['mean_speed'].dropna()
            
            comparison['motion_comparison'] = {
                f'{channel1_name}_mean_speed': ch1_speed.mean() if len(ch1_speed) > 0 else np.nan,
                f'{channel2_name}_mean_speed': ch2_speed.mean() if len(ch2_speed) > 0 else np.nan,
                f'{channel1_name}_median_speed': ch1_speed.median() if len(ch1_speed) > 0 else np.nan,
                f'{channel2_name}_median_speed': ch2_speed.median() if len(ch2_speed) > 0 else np.nan
            }
    
    return comparison

def create_multi_channel_visualization(channel1_tracks: pd.DataFrame, channel2_tracks: pd.DataFrame,
                                     compartments: Optional[List[Dict[str, Any]]] = None,
                                     channel1_name: str = "Channel 1", channel2_name: str = "Channel 2",
                                     channel1_color: str = "#FF4B4B", channel2_color: str = "#4B70FF") -> go.Figure:
    """
    Create a visualization showing both channels and compartments.
    
    Parameters
    ----------
    channel1_tracks : pd.DataFrame
        First channel track data
    channel2_tracks : pd.DataFrame
        Second channel track data
    compartments : List[Dict[str, Any]], optional
        List of segmented compartments
    channel1_name : str
        Name of first channel
    channel2_name : str
        Name of second channel
    channel1_color : str
        Color for first channel
    channel2_color : str
        Color for second channel
        
    Returns
    -------
    go.Figure
        Multi-channel visualization
    """
    fig = go.Figure()
    
    # Plot channel 1 tracks
    for track_id in channel1_tracks['track_id'].unique():
        track_data = channel1_tracks[channel1_tracks['track_id'] == track_id]
        fig.add_trace(go.Scatter(
            x=track_data['x'],
            y=track_data['y'],
            mode='lines+markers',
            name=f'{channel1_name} Track {track_id}',
            line=dict(color=channel1_color, width=2),
            marker=dict(color=channel1_color, size=4),
            showlegend=track_id == channel1_tracks['track_id'].iloc[0]  # Only show legend for first track
        ))
    
    # Plot channel 2 tracks
    for track_id in channel2_tracks['track_id'].unique():
        track_data = channel2_tracks[channel2_tracks['track_id'] == track_id]
        fig.add_trace(go.Scatter(
            x=track_data['x'],
            y=track_data['y'],
            mode='lines+markers',
            name=f'{channel2_name} Track {track_id}',
            line=dict(color=channel2_color, width=2),
            marker=dict(color=channel2_color, size=4),
            showlegend=track_id == channel2_tracks['track_id'].iloc[0]  # Only show legend for first track
        ))
    
    # Plot compartment boundaries
    if compartments:
        for comp in compartments:
            if comp['contour_pixels']:
                contour = comp['contour_pixels']
                x_coords = [pt[0] for pt in contour] + [contour[0][0]]  # Close the contour
                y_coords = [pt[1] for pt in contour] + [contour[0][1]]
                
                fig.add_trace(go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='lines',
                    name=f'Compartment {comp["id"]}',
                    line=dict(color='black', width=2, dash='dash'),
                    showlegend=comp == compartments[0]  # Only show legend for first compartment
                ))
    
    fig.update_layout(
        title='Multi-Channel Particle Tracking with Compartments',
        xaxis_title='X Position (pixels)',
        yaxis_title='Y Position (pixels)',
        template="plotly_white",
        hovermode='closest'
    )
    
    # Make axes equal scale
    fig.update_layout(
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    return fig
