"""
Chromosome Territory Mapping Module for SPT2025B

Automated detection of chromosome territories from particle tracking data.
Identifies spatial domains and analyzes inter- vs intra-territory diffusion.

References:
- Cremer & Cremer (2010): "Chromosome Territories"
- Bolzer et al. (2005): "Three-dimensional maps of all chromosomes"
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
from typing import Dict, List, Optional, Tuple
import warnings

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.mixture import GaussianMixture
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ChromosomeTerritoryMapper:
    """
    Detect and map chromosome territories from particle tracking data.
    
    Chromosome territories are spatial domains where chromatin from
    specific chromosomes preferentially locates.
    """
    
    def __init__(self, tracks_df: pd.DataFrame, pixel_size: float = 0.1):
        """
        Initialize territory mapper.
        
        Parameters:
        -----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, frame, x, y
        pixel_size : float
            Pixel size in micrometers
        """
        self.tracks_df = tracks_df.copy()
        self.pixel_size = pixel_size
        
        # Convert to physical units
        self.tracks_df['x_um'] = self.tracks_df['x'] * pixel_size
        self.tracks_df['y_um'] = self.tracks_df['y'] * pixel_size
        
        # Calculate time-averaged positions
        self.track_positions = self.tracks_df.groupby('track_id').agg({
            'x_um': 'mean',
            'y_um': 'mean',
            'frame': 'mean'
        }).reset_index()
    
    def detect_territories(
        self,
        method: str = 'density',
        n_territories: Optional[int] = None,
        density_threshold: float = 0.5
    ) -> Dict:
        """
        Detect chromosome territories.
        
        Parameters:
        -----------
        method : str
            Detection method:
            - 'density': Density-based clustering (DBSCAN)
            - 'kmeans': K-means clustering
            - 'gmm': Gaussian Mixture Model
        n_territories : int, optional
            Number of territories (required for kmeans/gmm)
        density_threshold : float
            Density threshold for DBSCAN (particles/μm²)
        
        Returns:
        --------
        dict with keys:
            - 'n_territories': int
            - 'territory_labels': array
            - 'territory_centers': array (N, 2)
            - 'territory_sizes': array
            - 'boundary_map': 2D array
        """
        if not SKLEARN_AVAILABLE:
            warnings.warn("sklearn not available. Using simple grid-based method.")
            return self._simple_territory_detection(n_territories or 4)
        
        positions = self.track_positions[['x_um', 'y_um']].values
        
        if method == 'density':
            # DBSCAN clustering
            eps = np.sqrt(1.0 / (density_threshold * np.pi))  # Convert density to distance
            clusterer = DBSCAN(eps=eps, min_samples=5)
            labels = clusterer.fit_predict(positions)
            
        elif method == 'kmeans':
            if n_territories is None:
                n_territories = self._estimate_n_territories(positions)
            
            clusterer = KMeans(n_clusters=n_territories, random_state=42, n_init=10)
            labels = clusterer.fit_predict(positions)
            
        elif method == 'gmm':
            if n_territories is None:
                n_territories = self._estimate_n_territories(positions)
            
            gmm = GaussianMixture(n_components=n_territories, random_state=42)
            labels = gmm.fit_predict(positions)
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Calculate territory properties
        unique_labels = np.unique(labels)
        n_territories_found = len(unique_labels[unique_labels >= 0])  # Exclude noise (-1)
        
        territory_centers = []
        territory_sizes = []
        
        for label in unique_labels:
            if label == -1:  # Noise in DBSCAN
                continue
            
            mask = labels == label
            territory_positions = positions[mask]
            
            center = np.mean(territory_positions, axis=0)
            size = len(territory_positions)
            
            territory_centers.append(center)
            territory_sizes.append(size)
        
        territory_centers = np.array(territory_centers)
        territory_sizes = np.array(territory_sizes)
        
        # Create boundary map
        boundary_map = self._create_boundary_map(positions, labels)
        
        # Store labels in track_positions
        self.track_positions['territory'] = labels
        
        return {
            'n_territories': n_territories_found,
            'territory_labels': labels,
            'territory_centers': territory_centers,
            'territory_sizes': territory_sizes,
            'boundary_map': boundary_map,
            'method': method
        }
    
    def analyze_territory_diffusion(self, territory_labels: np.ndarray) -> Dict:
        """
        Analyze diffusion within and between territories.
        
        Parameters:
        -----------
        territory_labels : array
            Territory labels for each track
        
        Returns:
        --------
        dict with keys:
            - 'intra_territory_diffusion': dict {territory_id: D_intra}
            - 'inter_territory_diffusion': float (average D_inter)
            - 'diffusion_ratio': float (D_inter / D_intra)
            - 'territory_crossing_events': int
        """
        self.track_positions['territory'] = territory_labels
        
        intra_territory_D = {}
        inter_territory_displacements = []
        crossing_events = 0
        
        # Analyze each territory
        for territory_id in np.unique(territory_labels):
            if territory_id == -1:  # Skip noise
                continue
            
            # Get tracks in this territory
            territory_tracks = self.track_positions[self.track_positions['territory'] == territory_id]['track_id'].values
            
            if len(territory_tracks) < 2:
                continue
            
            # Calculate displacements within territory
            intra_displacements = []
            
            for track_id in territory_tracks:
                track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
                
                if len(track) < 2:
                    continue
                
                dx = np.diff(track['x_um'].values)
                dy = np.diff(track['y_um'].values)
                displacements_sq = dx**2 + dy**2
                
                intra_displacements.extend(displacements_sq)
            
            if len(intra_displacements) > 0:
                # Estimate D from <r²> = 4*D*t (2D)
                mean_disp_sq = np.mean(intra_displacements)
                D_intra = mean_disp_sq / (4.0 * self.tracks_df['frame'].diff().mean() * 0.1)  # Assuming 0.1s frame interval
                intra_territory_D[int(territory_id)] = D_intra
        
        # Analyze inter-territory motion
        # Look for tracks that cross territory boundaries
        for track_id in self.tracks_df['track_id'].unique():
            track = self.tracks_df[self.tracks_df['track_id'] == track_id].sort_values('frame')
            
            if len(track) < 2:
                continue
            
            # Get territory labels along track
            track_territories = []
            for _, point in track.iterrows():
                # Find nearest labeled track position
                distances = np.sqrt(
                    (self.track_positions['x_um'] - point['x_um'])**2 +
                    (self.track_positions['y_um'] - point['y_um'])**2
                )
                nearest_idx = distances.idxmin()
                track_territories.append(self.track_positions.loc[nearest_idx, 'territory'])
            
            # Count transitions
            track_territories = np.array(track_territories)
            transitions = np.sum(np.diff(track_territories) != 0)
            crossing_events += transitions
            
            # Calculate displacements across boundaries
            for i in range(len(track) - 1):
                if track_territories[i] != track_territories[i+1] and track_territories[i] >= 0 and track_territories[i+1] >= 0:
                    dx = track['x_um'].iloc[i+1] - track['x_um'].iloc[i]
                    dy = track['y_um'].iloc[i+1] - track['y_um'].iloc[i]
                    inter_territory_displacements.append(dx**2 + dy**2)
        
        # Calculate inter-territory diffusion
        if len(inter_territory_displacements) > 0:
            mean_inter_disp_sq = np.mean(inter_territory_displacements)
            D_inter = mean_inter_disp_sq / (4.0 * 0.1)
        else:
            D_inter = 0.0
        
        # Calculate average intra-territory diffusion
        D_intra_mean = np.mean(list(intra_territory_D.values())) if len(intra_territory_D) > 0 else 0.0
        
        diffusion_ratio = D_inter / D_intra_mean if D_intra_mean > 0 else 0.0
        
        return {
            'intra_territory_diffusion': intra_territory_D,
            'inter_territory_diffusion': D_inter,
            'mean_intra_diffusion': D_intra_mean,
            'diffusion_ratio': diffusion_ratio,
            'territory_crossing_events': crossing_events,
            'interpretation': self._interpret_diffusion_ratio(diffusion_ratio)
        }
    
    def _interpret_diffusion_ratio(self, ratio: float) -> str:
        """Interpret diffusion ratio."""
        if ratio < 0.5:
            return "Strong compartmentalization: inter-territory diffusion highly restricted"
        elif ratio < 0.8:
            return "Moderate compartmentalization: some boundary barriers"
        elif ratio < 1.2:
            return "Weak compartmentalization: similar intra/inter diffusion"
        else:
            return "No compartmentalization: enhanced inter-territory mobility"
    
    def estimate_territory_volumes(
        self,
        territory_labels: np.ndarray,
        convex_hull: bool = True
    ) -> Dict:
        """
        Estimate territory volumes (areas in 2D).
        
        Parameters:
        -----------
        territory_labels : array
            Territory labels
        convex_hull : bool
            If True, use convex hull; otherwise use bounding box
        
        Returns:
        --------
        dict with keys:
            - 'territory_areas': dict {territory_id: area (μm²)}
            - 'total_area': float
            - 'density_map': 2D array
        """
        from scipy.spatial import ConvexHull
        
        positions = self.track_positions[['x_um', 'y_um']].values
        territory_areas = {}
        
        for territory_id in np.unique(territory_labels):
            if territory_id == -1:
                continue
            
            mask = territory_labels == territory_id
            territory_positions = positions[mask]
            
            if len(territory_positions) < 3:
                continue
            
            if convex_hull:
                try:
                    hull = ConvexHull(territory_positions)
                    area = hull.volume  # In 2D, volume is actually area
                except Exception:
                    # Fallback to bounding box
                    x_range = territory_positions[:, 0].max() - territory_positions[:, 0].min()
                    y_range = territory_positions[:, 1].max() - territory_positions[:, 1].min()
                    area = x_range * y_range
            else:
                x_range = territory_positions[:, 0].max() - territory_positions[:, 0].min()
                y_range = territory_positions[:, 1].max() - territory_positions[:, 1].min()
                area = x_range * y_range
            
            territory_areas[int(territory_id)] = area
        
        total_area = sum(territory_areas.values())
        
        # Create density map
        density_map = self._create_density_map(positions, territory_labels)
        
        return {
            'territory_areas': territory_areas,
            'total_area': total_area,
            'mean_area': np.mean(list(territory_areas.values())) if len(territory_areas) > 0 else 0.0,
            'density_map': density_map
        }
    
    def visualize_territories(self, territory_labels: np.ndarray) -> go.Figure:
        """
        Visualize detected territories.
        
        Creates scatter plot with territories colored differently.
        """
        positions = self.track_positions[['x_um', 'y_um']].values
        
        fig = go.Figure()
        
        # Plot each territory
        for territory_id in np.unique(territory_labels):
            mask = territory_labels == territory_id
            territory_positions = positions[mask]
            
            if territory_id == -1:
                # Noise points
                label = "Noise"
                color = 'gray'
            else:
                label = f"Territory {territory_id}"
                color = None  # Auto-assign
            
            fig.add_trace(go.Scatter(
                x=territory_positions[:, 0],
                y=territory_positions[:, 1],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=color,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=f'{label}<br>x=%{{x:.2f}} μm<br>y=%{{y:.2f}} μm<extra></extra>'
            ))
        
        fig.update_layout(
            title=f"Chromosome Territory Map<br>{len(np.unique(territory_labels[territory_labels >= 0]))} territories detected",
            xaxis_title="X Position (μm)",
            yaxis_title="Y Position (μm)",
            width=800,
            height=800,
            showlegend=True,
            hovermode='closest',
            yaxis=dict(scaleanchor="x", scaleratio=1)
        )
        
        return fig
    
    def _estimate_n_territories(self, positions: np.ndarray) -> int:
        """Estimate optimal number of territories using elbow method."""
        from sklearn.metrics import silhouette_score
        
        max_k = min(10, len(positions) // 10)
        
        if max_k < 2:
            return 2
        
        silhouette_scores = []
        k_values = range(2, max_k + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(positions)
            score = silhouette_score(positions, labels)
            silhouette_scores.append(score)
        
        # Choose k with highest silhouette score
        best_k = k_values[np.argmax(silhouette_scores)]
        
        return best_k
    
    def _create_boundary_map(self, positions: np.ndarray, labels: np.ndarray, resolution: int = 50) -> np.ndarray:
        """Create 2D map showing territory boundaries."""
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        x_edges = np.linspace(x_min, x_max, resolution + 1)
        y_edges = np.linspace(y_min, y_max, resolution + 1)
        
        boundary_map = np.zeros((resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                x_center = (x_edges[i] + x_edges[i+1]) / 2
                y_center = (y_edges[j] + y_edges[j+1]) / 2
                
                # Find nearest particles
                distances = np.sqrt(
                    (positions[:, 0] - x_center)**2 +
                    (positions[:, 1] - y_center)**2
                )
                
                nearest_k = min(5, len(distances))
                nearest_indices = np.argpartition(distances, nearest_k)[:nearest_k]
                nearest_labels = labels[nearest_indices]
                
                # Boundary if multiple territories nearby
                unique_labels = len(np.unique(nearest_labels[nearest_labels >= 0]))
                boundary_map[j, i] = unique_labels
        
        return boundary_map
    
    def _create_density_map(self, positions: np.ndarray, labels: np.ndarray, resolution: int = 50) -> np.ndarray:
        """Create density map of particle positions."""
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        density_map, x_edges, y_edges = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=resolution,
            range=[[x_min, x_max], [y_min, y_max]]
        )
        
        # Smooth with Gaussian filter
        density_map = gaussian_filter(density_map, sigma=1.0)
        
        return density_map.T
    
    def _simple_territory_detection(self, n_territories: int) -> Dict:
        """Simple grid-based territory detection when sklearn unavailable."""
        positions = self.track_positions[['x_um', 'y_um']].values
        
        # Divide space into grid
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Create sqrt(n) x sqrt(n) grid
        grid_size = int(np.ceil(np.sqrt(n_territories)))
        x_bins = np.linspace(x_min, x_max, grid_size + 1)
        y_bins = np.linspace(y_min, y_max, grid_size + 1)
        
        # Assign labels based on grid position
        labels = np.zeros(len(positions), dtype=int)
        
        for idx, pos in enumerate(positions):
            x_idx = np.digitize(pos[0], x_bins) - 1
            y_idx = np.digitize(pos[1], y_bins) - 1
            
            x_idx = np.clip(x_idx, 0, grid_size - 1)
            y_idx = np.clip(y_idx, 0, grid_size - 1)
            
            labels[idx] = y_idx * grid_size + x_idx
        
        # Calculate centers and sizes
        unique_labels = np.unique(labels)
        territory_centers = []
        territory_sizes = []
        
        for label in unique_labels:
            mask = labels == label
            center = np.mean(positions[mask], axis=0)
            size = np.sum(mask)
            
            territory_centers.append(center)
            territory_sizes.append(size)
        
        return {
            'n_territories': len(unique_labels),
            'territory_labels': labels,
            'territory_centers': np.array(territory_centers),
            'territory_sizes': np.array(territory_sizes),
            'boundary_map': np.zeros((10, 10)),
            'method': 'simple_grid'
        }
