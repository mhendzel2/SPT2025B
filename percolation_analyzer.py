"""
Percolation Analysis Module for SPT2025B

Implements percolation theory analysis for understanding connectivity
and phase transitions in nuclear environments and particle tracking data.

References:
- Stauffer & Aharony (1994): "Introduction to Percolation Theory"
- Sahimi (1994): "Applications of Percolation Theory"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import label, find_objects
from scipy.optimize import curve_fit
from typing import Dict, Tuple, List, Optional
import warnings

try:
    from sklearn.cluster import DBSCAN
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    DBSCAN = None
    csr_matrix = None
    connected_components = None


class PercolationAnalyzer:
    """
    Analyze percolation properties of particle tracking data.
    
    Percolation theory characterizes phase transitions in connectivity,
    critical for understanding:
    - Chromatin network connectivity
    - Nuclear environment organization
    - Transport through heterogeneous media
    - Phase transitions in cellular structures
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1):
        """
        Initialize percolation analyzer.
        
        Parameters:
        -----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y (z optional)
        pixel_size : float
            Pixel size in micrometers (default: 0.1 μm)
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        
        # Convert pixel coordinates to physical units
        if 'x' in self.tracks_df.columns and 'y' in self.tracks_df.columns:
            self.tracks_df['x_um'] = self.tracks_df['x'] * pixel_size
            self.tracks_df['y_um'] = self.tracks_df['y'] * pixel_size
            if 'z' in self.tracks_df.columns:
                self.tracks_df['z_um'] = self.tracks_df['z'] * pixel_size
                self.is_3d = True
            else:
                self.is_3d = False
        
        # Calculate system size
        self.x_range = (self.tracks_df['x_um'].min(), self.tracks_df['x_um'].max())
        self.y_range = (self.tracks_df['y_um'].min(), self.tracks_df['y_um'].max())
        self.system_size = max(
            self.x_range[1] - self.x_range[0],
            self.y_range[1] - self.y_range[0]
        )
    
    def estimate_percolation_threshold(
        self,
        method: str = 'density',
        distance_threshold: Optional[float] = None,
        n_samples: int = 50
    ) -> Dict:
        """
        Estimate if system is above/below percolation threshold.
        
        Parameters:
        -----------
        method : str
            Method for estimation:
            - 'density': Based on particle density
            - 'connectivity': Based on network connectivity
            - 'msd_transition': Based on MSD scaling changes
        distance_threshold : float, optional
            Connection distance in μm (auto-estimated if None)
        n_samples : int
            Number of density samples for threshold estimation
        
        Returns:
        --------
        dict with keys:
            - 'is_percolating': bool
            - 'density': float (particles/μm²)
            - 'p_c_estimate': float (critical threshold)
            - 'p_c_theory': float (theoretical critical threshold)
            - 'percolation_probability': float (0-1)
            - 'confidence': str ('high', 'medium', 'low')
        """
        results = {
            'method': method,
            'is_percolating': False,
            'density': 0.0,
            'p_c_estimate': 0.0,
            'p_c_theory': 0.0,
            'percolation_probability': 0.0,
            'confidence': 'low'
        }
        
        # Theoretical percolation thresholds
        # 2D: p_c ≈ 0.5927 (site), 0.5 (bond)
        # 3D: p_c ≈ 0.3116 (site), 0.2488 (bond)
        if self.is_3d:
            results['p_c_theory'] = 0.2488  # Bond percolation 3D
        else:
            results['p_c_theory'] = 0.5  # Bond percolation 2D
        
        if method == 'density':
            # Calculate particle density
            area = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
            n_particles = len(self.tracks_df.groupby('track_id'))
            density = n_particles / area
            results['density'] = density
            
            # Estimate critical density based on theoretical percolation
            # For continuum percolation: n_c ≈ p_c / (π * r_c²)
            if distance_threshold is None:
                # Estimate from nearest neighbor distances
                nn_dist = self._estimate_nearest_neighbor_distance()
                distance_threshold = nn_dist
            
            critical_density = results['p_c_theory'] / (np.pi * distance_threshold**2)
            results['p_c_estimate'] = critical_density
            
            # Calculate percolation probability
            if density > critical_density:
                results['is_percolating'] = True
                results['percolation_probability'] = min(1.0, density / critical_density - 1.0)
                results['confidence'] = 'high' if density > 1.5 * critical_density else 'medium'
            else:
                results['percolation_probability'] = density / critical_density
                results['confidence'] = 'medium' if density > 0.5 * critical_density else 'low'
        
        elif method == 'connectivity':
            if not SKLEARN_AVAILABLE:
                warnings.warn("sklearn not available. Using simplified connectivity analysis.")
                return self._simple_connectivity_analysis(distance_threshold)
            
            # Build connectivity network
            network_results = self.analyze_connectivity_network(distance_threshold)
            
            results['density'] = network_results['density']
            results['p_c_estimate'] = network_results.get('p_c_estimate', results['p_c_theory'])
            
            # Check for spanning cluster
            if network_results['spanning_cluster']:
                results['is_percolating'] = True
                results['percolation_probability'] = 0.9
                results['confidence'] = 'high'
            else:
                # Estimate from cluster size distribution
                max_cluster_fraction = network_results['largest_cluster_size'] / network_results['num_nodes']
                results['percolation_probability'] = max_cluster_fraction
                results['is_percolating'] = max_cluster_fraction > 0.5
                results['confidence'] = 'medium'
        
        elif method == 'msd_transition':
            # Analyze MSD scaling for phase transition signatures
            # Near percolation threshold, diffusion exponent changes
            from analysis import calculate_msd
            
            msd_results = calculate_msd(self.tracks_df, self.pixel_size, frame_interval=0.1)
            
            if isinstance(msd_results, dict) and msd_results.get('success'):
                msd_df = msd_results.get('ensemble_msd')
            else:
                msd_df = msd_results
            
            if msd_df is not None and not msd_df.empty:
                # Fit MSD ~ t^alpha
                from scipy.stats import linregress
                log_t = np.log(msd_df['lag_time'].values[1:6])
                log_msd = np.log(msd_df['msd'].values[1:6])
                slope, intercept, r_value, p_value, std_err = linregress(log_t, log_msd)
                
                alpha = slope
                results['alpha_exponent'] = alpha
                
                # At percolation threshold: α ≈ 0.87 (2D), 0.72 (3D)
                alpha_c = 0.87 if not self.is_3d else 0.72
                
                if alpha > alpha_c:
                    results['is_percolating'] = True
                    results['percolation_probability'] = 0.8
                    results['confidence'] = 'medium'
                else:
                    results['percolation_probability'] = alpha / alpha_c
                    results['confidence'] = 'low'
        
        return results
    
    def analyze_connectivity_network(
        self,
        distance_threshold: Optional[float] = None,
        use_time_window: bool = True,
        time_window: int = 5
    ) -> Dict:
        """
        Build and analyze connectivity network from particle positions.
        
        Two particles are connected if their distance is below threshold.
        
        Parameters:
        -----------
        distance_threshold : float, optional
            Connection distance in μm (auto-estimated if None)
        use_time_window : bool
            If True, only connect particles within time_window frames
        time_window : int
            Number of frames for temporal connectivity
        
        Returns:
        --------
        dict with keys:
            - 'num_nodes': int
            - 'num_edges': int
            - 'num_clusters': int
            - 'largest_cluster_size': int
            - 'spanning_cluster': bool
            - 'cluster_size_distribution': array
            - 'density': float
            - 'average_degree': float
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available. Using simplified analysis.")
            return self._simple_connectivity_analysis(distance_threshold)
        
        # Auto-estimate distance threshold
        if distance_threshold is None:
            distance_threshold = self._estimate_nearest_neighbor_distance() * 2.0
        
        # Get time-averaged positions for each track
        track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean',
            'frame': 'mean'
        }).reset_index()
        
        n_nodes = len(track_positions)
        
        if n_nodes < 2:
            return {
                'num_nodes': n_nodes,
                'num_edges': 0,
                'num_clusters': 1,
                'largest_cluster_size': n_nodes,
                'spanning_cluster': False,
                'cluster_size_distribution': np.array([n_nodes]),
                'density': 0.0,
                'average_degree': 0.0
            }
        
        # Build distance matrix
        positions = track_positions[['x_um', 'y_um']].values
        distances = squareform(pdist(positions, metric='euclidean'))
        
        # Apply time window if requested
        if use_time_window:
            time_matrix = np.abs(
                track_positions['frame'].values[:, np.newaxis] -
                track_positions['frame'].values[np.newaxis, :]
            )
            valid_connections = (distances < distance_threshold) & (time_matrix < time_window)
        else:
            valid_connections = distances < distance_threshold
        
        # Build adjacency matrix (sparse for efficiency)
        adjacency = csr_matrix(valid_connections.astype(int))
        
        # Find connected components
        n_clusters, labels = connected_components(adjacency, directed=False)
        
        # Analyze cluster sizes
        cluster_sizes = np.bincount(labels)
        largest_cluster_size = np.max(cluster_sizes)
        
        # Check for spanning cluster (touches opposite boundaries)
        spanning = self._check_spanning_cluster(track_positions, labels)
        
        # Calculate network properties
        num_edges = np.sum(valid_connections) // 2  # Undirected graph
        average_degree = 2 * num_edges / n_nodes if n_nodes > 0 else 0
        
        # Calculate density
        area = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
        density = n_nodes / area
        
        return {
            'num_nodes': n_nodes,
            'num_edges': num_edges,
            'num_clusters': n_clusters,
            'largest_cluster_size': int(largest_cluster_size),
            'spanning_cluster': spanning,
            'cluster_size_distribution': cluster_sizes,
            'density': density,
            'average_degree': average_degree,
            'distance_threshold': distance_threshold,
            'labels': labels
        }
    
    def calculate_cluster_size_distribution(
        self,
        distance_threshold: Optional[float] = None,
        normalize: bool = True
    ) -> Dict:
        """
        Calculate distribution of cluster sizes.
        
        Near percolation threshold: n(s) ~ s^(-τ) where τ ≈ 2.05 (2D), 2.18 (3D)
        
        Parameters:
        -----------
        distance_threshold : float, optional
            Connection distance in μm
        normalize : bool
            If True, normalize distribution to probability
        
        Returns:
        --------
        dict with keys:
            - 'cluster_sizes': array of unique sizes
            - 'counts': array of counts for each size
            - 'probabilities': normalized probabilities
            - 'tau_exponent': power-law exponent (if fitted)
            - 'r_squared': goodness of fit
        """
        # Get network analysis
        network = self.analyze_connectivity_network(distance_threshold)
        
        cluster_sizes = network['cluster_size_distribution']
        
        # Get unique sizes and counts
        unique_sizes, counts = np.unique(cluster_sizes, return_counts=True)
        
        # Remove size = 0 if present
        mask = unique_sizes > 0
        unique_sizes = unique_sizes[mask]
        counts = counts[mask]
        
        # Normalize to probabilities
        probabilities = counts / np.sum(counts) if normalize else counts
        
        # Fit power law: n(s) ~ s^(-tau)
        tau_exponent = np.nan
        r_squared = 0.0
        
        if len(unique_sizes) > 3:
            try:
                # Fit in log-log space
                from scipy.stats import linregress
                
                # Use only sizes > 1 for better fit
                mask_fit = unique_sizes > 1
                if np.sum(mask_fit) > 2:
                    log_s = np.log(unique_sizes[mask_fit])
                    log_n = np.log(counts[mask_fit])
                    
                    slope, intercept, r_value, p_value, std_err = linregress(log_s, log_n)
                    tau_exponent = -slope  # Convert to positive exponent
                    r_squared = r_value**2
            except Exception as e:
                warnings.warn(f"Power-law fitting failed: {e}")
        
        return {
            'cluster_sizes': unique_sizes,
            'counts': counts,
            'probabilities': probabilities,
            'tau_exponent': tau_exponent,
            'r_squared': r_squared,
            'theoretical_tau_2d': 2.05,
            'theoretical_tau_3d': 2.18
        }
    
    def detect_spanning_cluster(
        self,
        distance_threshold: Optional[float] = None,
        direction: str = 'both'
    ) -> Dict:
        """
        Detect if a spanning cluster exists (percolating cluster).
        
        A spanning cluster connects opposite boundaries of the system.
        
        Parameters:
        -----------
        distance_threshold : float, optional
            Connection distance in μm
        direction : str
            'horizontal', 'vertical', or 'both'
        
        Returns:
        --------
        dict with keys:
            - 'has_spanning_cluster': bool
            - 'spanning_horizontal': bool
            - 'spanning_vertical': bool
            - 'spanning_cluster_size': int
            - 'spanning_cluster_fraction': float
        """
        network = self.analyze_connectivity_network(distance_threshold)
        
        if network['num_nodes'] == 0:
            return {
                'has_spanning_cluster': False,
                'spanning_horizontal': False,
                'spanning_vertical': False,
                'spanning_cluster_size': 0,
                'spanning_cluster_fraction': 0.0
            }
        
        # Get track positions with cluster labels
        track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean'
        }).reset_index()
        
        labels = network['labels']
        track_positions['cluster'] = labels
        
        # Check for spanning clusters
        spanning_h = False
        spanning_v = False
        spanning_cluster_id = -1
        
        for cluster_id in np.unique(labels):
            cluster_mask = track_positions['cluster'] == cluster_id
            cluster_x = track_positions.loc[cluster_mask, 'x_um']
            cluster_y = track_positions.loc[cluster_mask, 'y_um']
            
            # Check horizontal spanning
            x_span = cluster_x.max() - cluster_x.min()
            h_spans = x_span > 0.8 * (self.x_range[1] - self.x_range[0])
            
            # Check vertical spanning
            y_span = cluster_y.max() - cluster_y.min()
            v_spans = y_span > 0.8 * (self.y_range[1] - self.y_range[0])
            
            if h_spans or v_spans:
                spanning_h = spanning_h or h_spans
                spanning_v = spanning_v or v_spans
                spanning_cluster_id = cluster_id
        
        has_spanning = spanning_h or spanning_v
        
        spanning_size = 0
        if spanning_cluster_id >= 0:
            spanning_size = np.sum(labels == spanning_cluster_id)
        
        return {
            'has_spanning_cluster': has_spanning,
            'spanning_horizontal': spanning_h,
            'spanning_vertical': spanning_v,
            'spanning_cluster_size': int(spanning_size),
            'spanning_cluster_fraction': spanning_size / network['num_nodes']
        }
    
    def visualize_percolation_map(
        self,
        distance_threshold: Optional[float] = None,
        show_connections: bool = True,
        color_by: str = 'cluster'
    ) -> go.Figure:
        """
        Create visualization of percolation network.
        
        Parameters:
        -----------
        distance_threshold : float, optional
            Connection distance in μm
        show_connections : bool
            If True, draw edges between connected nodes
        color_by : str
            'cluster' or 'degree'
        
        Returns:
        --------
        Plotly figure object
        """
        network = self.analyze_connectivity_network(distance_threshold)
        
        # Get track positions
        track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean'
        }).reset_index()
        
        labels = network['labels']
        track_positions['cluster'] = labels
        
        # Create figure
        fig = go.Figure()
        
        # Draw connections if requested
        if show_connections and network['num_edges'] > 0:
            # Build connection list
            positions = track_positions[['x_um', 'y_um']].values
            distances = squareform(pdist(positions, metric='euclidean'))
            connections = distances < network['distance_threshold']
            
            # Draw edges (sample if too many)
            max_edges = 1000
            edge_indices = np.argwhere(np.triu(connections, k=1))
            
            if len(edge_indices) > max_edges:
                edge_indices = edge_indices[
                    np.random.choice(len(edge_indices), max_edges, replace=False)
                ]
            
            for i, j in edge_indices:
                fig.add_trace(go.Scatter(
                    x=[positions[i, 0], positions[j, 0], None],
                    y=[positions[i, 1], positions[j, 1], None],
                    mode='lines',
                    line=dict(color='lightgray', width=0.5),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Color nodes
        if color_by == 'cluster':
            color_values = labels
            colorbar_title = "Cluster ID"
        else:  # degree
            # Calculate degree for each node
            positions = track_positions[['x_um', 'y_um']].values
            distances = squareform(pdist(positions, metric='euclidean'))
            connections = distances < network['distance_threshold']
            degrees = np.sum(connections, axis=1)
            color_values = degrees
            colorbar_title = "Node Degree"
        
        # Plot nodes
        fig.add_trace(go.Scatter(
            x=track_positions['x_um'],
            y=track_positions['y_um'],
            mode='markers',
            marker=dict(
                size=8,
                color=color_values,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=colorbar_title),
                line=dict(width=1, color='white')
            ),
            text=[f"Track {tid}<br>Cluster {cid}" 
                  for tid, cid in zip(track_positions['track_id'], labels)],
            hovertemplate='%{text}<br>x=%{x:.2f} μm<br>y=%{y:.2f} μm<extra></extra>',
            name='Particles'
        ))
        
        # Update layout
        fig.update_layout(
            title=f"Percolation Network (d={network['distance_threshold']:.2f} μm)<br>" +
                  f"Clusters: {network['num_clusters']}, Largest: {network['largest_cluster_size']} nodes",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=800,
            showlegend=False,
            hovermode='closest',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        return fig
    
    def _estimate_nearest_neighbor_distance(self) -> float:
        """Estimate typical nearest neighbor distance."""
        track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean'
        }).reset_index()
        
        positions = track_positions[['x_um', 'y_um']].values
        
        if len(positions) < 2:
            return 1.0  # Default
        
        distances = squareform(pdist(positions, metric='euclidean'))
        np.fill_diagonal(distances, np.inf)  # Exclude self
        
        nn_distances = np.min(distances, axis=1)
        return np.median(nn_distances)
    
    def _check_spanning_cluster(
        self,
        track_positions: pd.DataFrame,
        labels: np.ndarray
    ) -> bool:
        """Check if largest cluster spans the system."""
        largest_cluster = np.argmax(np.bincount(labels))
        cluster_mask = labels == largest_cluster
        
        if np.sum(cluster_mask) < 3:
            return False
        
        cluster_x = track_positions.loc[cluster_mask, 'x_um']
        cluster_y = track_positions.loc[cluster_mask, 'y_um']
        
        x_span = cluster_x.max() - cluster_x.min()
        y_span = cluster_y.max() - cluster_y.min()
        
        # Spanning if cluster extends across >80% of system in either direction
        x_spanning = x_span > 0.8 * (self.x_range[1] - self.x_range[0])
        y_spanning = y_span > 0.8 * (self.y_range[1] - self.y_range[0])
        
        return x_spanning or y_spanning
    
    def _simple_connectivity_analysis(self, distance_threshold: Optional[float]) -> Dict:
        """Simplified connectivity analysis when sklearn unavailable."""
        if distance_threshold is None:
            distance_threshold = self._estimate_nearest_neighbor_distance() * 2.0
        
        track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean'
        }).reset_index()
        
        n_nodes = len(track_positions)
        area = (self.x_range[1] - self.x_range[0]) * (self.y_range[1] - self.y_range[0])
        density = n_nodes / area
        
        return {
            'num_nodes': n_nodes,
            'num_edges': 0,
            'num_clusters': n_nodes,
            'largest_cluster_size': 1,
            'spanning_cluster': False,
            'cluster_size_distribution': np.ones(n_nodes),
            'density': density,
            'average_degree': 0.0,
            'distance_threshold': distance_threshold
        }
