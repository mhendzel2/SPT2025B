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


# ============================================================================
# Condensate Interface Analysis (PhaseMetrics-inspired)
# Based on: PhaseMetrics tools for condensate characterization (May 2024)
# ============================================================================

def analyze_condensate_interface(tracks_df: pd.DataFrame, compartments: List[Dict[str, Any]],
                                pixel_size: float = 1.0, frame_interval: float = 1.0,
                                interface_width: float = 0.5) -> Dict[str, Any]:
    """
    Analyze particle behavior at condensate boundaries (interface dynamics).
    
    Nuclear tracking often involves phase-separated condensates (nucleoli, speckles,
    Cajal bodies, PML bodies, stress granules). Traditional "in/out" classification
    ignores the critical interface region where unique physics occurs:
    - Selective permeability barriers
    - Concentration gradients
    - Surface tension effects
    - Wetting/dewetting transitions
    
    This function quantifies interface-specific properties inspired by PhaseMetrics:
    1. Partition Coefficient: Equilibrium concentration ratio (in/out)
    2. Interface Permeability: Fraction of particles that successfully cross
    3. Boundary Crossing Probability: Rate of entry/exit events
    4. Residence Time Asymmetry: Difference in dwell times inside vs outside
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Track data with columns 'track_id', 'frame', 'x', 'y'.
    compartments : List[Dict[str, Any]]
        List of compartment definitions, each with:
        - 'id': Compartment identifier
        - 'contour_pixels': List of (x, y) boundary points
        - 'area': Area in pixels² (optional, will be calculated if missing)
        - 'name': Descriptive name (e.g., 'Nucleolus', 'Speckle')
    pixel_size : float, default=1.0
        Pixel size in micrometers for spatial scaling.
    frame_interval : float, default=1.0
        Time between frames in seconds.
    interface_width : float, default=0.5
        Width of interface region in micrometers.
        Particles within this distance of boundary are considered "at interface".
    
    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'success': bool, whether analysis completed
        - 'compartment_results': List of results for each compartment with:
            - 'compartment_id': Identifier
            - 'partition_coefficient': C_in / C_out ratio
            - 'enrichment_ratio': Normalized partition coefficient
            - 'interface_permeability': Crossing success rate (0-1)
            - 'boundary_crossing_rate': Crossings per second
            - 'mean_residence_time_inside': Average dwell time (s)
            - 'mean_residence_time_outside': Average dwell time (s)
            - 'residence_asymmetry': Ratio of inside/outside residence
            - 'interface_density': Particles per µm at boundary
        - 'crossing_events': pd.DataFrame with individual crossing events
        - 'summary': Overall statistics across all compartments
    
    References
    ----------
    - PhaseMetrics: High-throughput quantification of condensate properties (bioRxiv, May 2024)
    - Material properties of condensates (Cell, 2022)
    - Interface dynamics in liquid-liquid phase separation (Nature Physics, 2021)
    
    Notes
    -----
    **Partition Coefficient (K_p):**
    Thermodynamic measure of preferential localization:
        K_p = (C_in / Area_in) / (C_out / Area_out)
    where C is particle count.
    
    Interpretation:
    - K_p > 1: Enrichment (particles prefer condensate)
    - K_p = 1: No preference (equal partitioning)
    - K_p < 1: Exclusion (particles avoid condensate)
    
    **Interface Permeability Score:**
    Measures barrier function:
        Permeability = N_cross / (N_cross + N_bounce)
    where:
    - N_cross: Tracks that successfully traverse boundary
    - N_bounce: Tracks that approach but turn back
    
    Interpretation:
    - Score → 1: Highly permeable (no barrier)
    - Score → 0: Impermeable (strong barrier)
    
    **Residence Time Asymmetry:**
    Indicates binding/entrapment:
        Asymmetry = τ_in / τ_out
    
    Interpretation:
    - Asymmetry > 1: Longer residence inside (retention)
    - Asymmetry = 1: Symmetric dynamics
    - Asymmetry < 1: Shorter residence inside (expulsion)
    
    **Interface Region:**
    Defined as annulus of width `interface_width` around boundary.
    Particles in this zone experience boundary effects:
    - Surface tension
    - Concentration gradients
    - Electrostatic barriers
    
    Examples
    --------
    >>> # Analyze nucleolar dynamics
    >>> compartments = [{
    ...     'id': 1,
    ...     'name': 'Nucleolus',
    ...     'contour_pixels': nucleolus_boundary,
    ...     'area': 1500  # pixels²
    ... }]
    >>> result = analyze_condensate_interface(tracks_df, compartments, pixel_size=0.1)
    >>> 
    >>> # Extract partition coefficient
    >>> K_p = result['compartment_results'][0]['partition_coefficient']
    >>> print(f"Nucleolar enrichment: {K_p:.2f}x")
    >>> 
    >>> # Check permeability
    >>> perm = result['compartment_results'][0]['interface_permeability']
    >>> if perm < 0.3:
    ...     print("Strong barrier detected - selective permeability")
    >>> elif perm > 0.7:
    ...     print("Weak barrier - high permeability")
    """
    try:
        from scipy.spatial import distance
        from shapely.geometry import Point, Polygon
        from shapely.ops import nearest_points
        
        # Input validation
        if not compartments:
            return {
                'success': False,
                'error': 'No compartments provided'
            }
        
        required_cols = ['track_id', 'frame', 'x', 'y']
        if not all(col in tracks_df.columns for col in required_cols):
            return {
                'success': False,
                'error': f'Missing required columns: {required_cols}'
            }
        
        # Process each compartment
        compartment_results = []
        all_crossing_events = []
        
        for comp in compartments:
            comp_id = comp.get('id', 0)
            contour = comp.get('contour_pixels', [])
            
            if not contour or len(contour) < 3:
                continue
            
            # Create polygon from contour
            try:
                polygon = Polygon(contour)
                if not polygon.is_valid:
                    polygon = polygon.buffer(0)  # Fix invalid geometries
            except Exception:
                continue
            
            # Calculate compartment area if not provided
            area_pixels = comp.get('area', polygon.area)
            area_um2 = area_pixels * (pixel_size ** 2)
            
            # Classify particle positions
            tracks_inside = []
            tracks_outside = []
            tracks_interface = []
            crossing_events = []
            
            for track_id, track_data in tracks_df.groupby('track_id'):
                track_data = track_data.sort_values('frame')
                
                # Track state over time
                states = []  # 'in', 'out', or 'interface'
                
                for idx, row in track_data.iterrows():
                    point = Point(row['x'], row['y'])
                    
                    # Check if inside compartment
                    is_inside = polygon.contains(point)
                    
                    # Calculate distance to boundary
                    boundary_dist = polygon.exterior.distance(point) * pixel_size
                    
                    if is_inside:
                        if boundary_dist <= interface_width:
                            state = 'interface_in'
                        else:
                            state = 'in'
                    else:
                        if boundary_dist <= interface_width:
                            state = 'interface_out'
                        else:
                            state = 'out'
                    
                    states.append(state)
                
                # Detect boundary crossings
                for i in range(len(states) - 1):
                    curr_state = states[i]
                    next_state = states[i + 1]
                    
                    # Crossing: transition from out → in or in → out
                    if curr_state in ['out', 'interface_out'] and next_state in ['in', 'interface_in']:
                        # Inward crossing
                        crossing_events.append({
                            'track_id': track_id,
                            'frame': track_data.iloc[i]['frame'],
                            'direction': 'inward',
                            'compartment_id': comp_id,
                            'type': 'crossing'
                        })
                    elif curr_state in ['in', 'interface_in'] and next_state in ['out', 'interface_out']:
                        # Outward crossing
                        crossing_events.append({
                            'track_id': track_id,
                            'frame': track_data.iloc[i]['frame'],
                            'direction': 'outward',
                            'compartment_id': comp_id,
                            'type': 'crossing'
                        })
                    elif curr_state == 'interface_out' and next_state == 'out':
                        # Approached but bounced back
                        crossing_events.append({
                            'track_id': track_id,
                            'frame': track_data.iloc[i]['frame'],
                            'direction': 'bounce',
                            'compartment_id': comp_id,
                            'type': 'bounce'
                        })
                
                # Classify track overall
                in_count = sum(1 for s in states if 'in' in s)
                out_count = sum(1 for s in states if 'out' in s)
                
                if in_count > out_count:
                    tracks_inside.append(track_id)
                elif out_count > 0:
                    tracks_outside.append(track_id)
            
            # Calculate metrics
            n_in = len(tracks_inside)
            n_out = len(tracks_outside)
            
            # 1. Partition Coefficient
            # Estimate "outside" area as total imaging area minus compartment
            # (This is approximate - ideally use nuclear boundary)
            total_area_estimate = area_um2 * 10  # Assume compartment is ~10% of total
            area_out = total_area_estimate - area_um2
            
            density_in = n_in / area_um2 if area_um2 > 0 else 0
            density_out = n_out / area_out if area_out > 0 else 0
            partition_coeff = density_in / density_out if density_out > 0 else float('inf')
            
            # Enrichment ratio (log-scale for plotting)
            enrichment = np.log2(partition_coeff) if partition_coeff > 0 else 0
            
            # 2. Interface Permeability
            crossings = [e for e in crossing_events if e['type'] == 'crossing']
            bounces = [e for e in crossing_events if e['type'] == 'bounce']
            
            n_crossings = len(crossings)
            n_bounces = len(bounces)
            
            permeability = n_crossings / (n_crossings + n_bounces + 1e-9)
            
            # 3. Boundary Crossing Rate
            total_time = len(tracks_df) * frame_interval  # Approximate
            crossing_rate = n_crossings / total_time if total_time > 0 else 0
            
            # 4. Residence Times (simplified - uses track lengths as proxy)
            # More sophisticated: calculate actual dwell times in each region
            residence_in = np.mean([len(tracks_df[tracks_df['track_id'] == tid]) for tid in tracks_inside]) if tracks_inside else 0
            residence_out = np.mean([len(tracks_df[tracks_df['track_id'] == tid]) for tid in tracks_outside]) if tracks_outside else 0
            
            residence_in_sec = residence_in * frame_interval
            residence_out_sec = residence_out * frame_interval
            
            residence_asymmetry = residence_in_sec / residence_out_sec if residence_out_sec > 0 else float('inf')
            
            # 5. Interface Density
            boundary_length = polygon.length * pixel_size  # Perimeter in µm
            interface_density = len(crossing_events) / boundary_length if boundary_length > 0 else 0
            
            # Compile compartment results
            comp_result = {
                'compartment_id': comp_id,
                'compartment_name': comp.get('name', f'Compartment {comp_id}'),
                'n_particles_inside': int(n_in),
                'n_particles_outside': int(n_out),
                'partition_coefficient': float(partition_coeff),
                'enrichment_ratio': float(enrichment),
                'interface_permeability': float(permeability),
                'boundary_crossing_rate': float(crossing_rate),
                'n_crossings_inward': sum(1 for e in crossings if e['direction'] == 'inward'),
                'n_crossings_outward': sum(1 for e in crossings if e['direction'] == 'outward'),
                'n_bounces': int(n_bounces),
                'mean_residence_time_inside': float(residence_in_sec),
                'mean_residence_time_outside': float(residence_out_sec),
                'residence_asymmetry': float(residence_asymmetry),
                'interface_density': float(interface_density),
                'area_um2': float(area_um2),
                'boundary_length_um': float(boundary_length)
            }
            
            compartment_results.append(comp_result)
            all_crossing_events.extend(crossing_events)
        
        if not compartment_results:
            return {
                'success': False,
                'error': 'No valid compartments analyzed'
            }
        
        # Create summary statistics
        all_K_p = [r['partition_coefficient'] for r in compartment_results if np.isfinite(r['partition_coefficient'])]
        all_perm = [r['interface_permeability'] for r in compartment_results]
        
        summary = {
            'n_compartments': len(compartment_results),
            'mean_partition_coefficient': float(np.mean(all_K_p)) if all_K_p else 0,
            'mean_permeability': float(np.mean(all_perm)) if all_perm else 0,
            'total_crossings': len([e for e in all_crossing_events if e['type'] == 'crossing']),
            'total_bounces': len([e for e in all_crossing_events if e['type'] == 'bounce']),
            'interpretation': {}
        }
        
        # Add interpretation
        mean_K_p = summary['mean_partition_coefficient']
        if mean_K_p > 2:
            summary['interpretation']['partition'] = 'Strong enrichment - particles accumulate in condensates'
        elif mean_K_p > 1.2:
            summary['interpretation']['partition'] = 'Moderate enrichment - preferential localization'
        elif mean_K_p > 0.8:
            summary['interpretation']['partition'] = 'No preference - equal partitioning'
        else:
            summary['interpretation']['partition'] = 'Exclusion - particles avoid condensates'
        
        mean_perm = summary['mean_permeability']
        if mean_perm > 0.7:
            summary['interpretation']['permeability'] = 'High permeability - weak barrier'
        elif mean_perm > 0.3:
            summary['interpretation']['permeability'] = 'Moderate permeability - selective barrier'
        else:
            summary['interpretation']['permeability'] = 'Low permeability - strong barrier'
        
        # Compile final results
        results = {
            'success': True,
            'compartment_results': compartment_results,
            'crossing_events': pd.DataFrame(all_crossing_events) if all_crossing_events else pd.DataFrame(),
            'summary': summary,
            'parameters': {
                'pixel_size': pixel_size,
                'frame_interval': frame_interval,
                'interface_width_um': interface_width
            }
        }
        
        return results
        
    except ImportError:
        return {
            'success': False,
            'error': 'shapely package required for interface analysis. Install with: pip install shapely'
        }
    except Exception as e:
        return {
            'success': False,
            'error': f'Condensate interface analysis failed: {str(e)}'
        }
