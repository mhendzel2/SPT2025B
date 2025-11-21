"""
Subpopulation Analysis Module

Detects and characterizes heterogeneous subpopulations within experimental groups
based on single-particle tracking data. Uses clustering and statistical approaches
to identify distinct subgroups (e.g., different cell cycle stages, metabolic states).

Author: SPT2025B
Date: November 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallbacks
try:
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Subpopulation analysis will use basic methods only.")

try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Some statistical tests will be unavailable.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("plotly not available. Visualizations will be limited.")


@dataclass
class SubpopulationConfig:
    """Configuration for subpopulation analysis."""
    min_tracks_per_cell: int = 10  # Minimum tracks to consider a cell
    min_cell_sample_size: int = 5  # Minimum cells needed for clustering
    clustering_methods: List[str] = None  # ['kmeans', 'gmm', 'hierarchical', 'dbscan']
    n_clusters_range: Tuple[int, int] = (2, 5)  # Range of cluster numbers to test
    features_to_use: List[str] = None  # Which features to use for clustering
    use_pca: bool = True  # Whether to use PCA for dimensionality reduction
    pca_variance_threshold: float = 0.95  # Variance explained by PCA
    significance_level: float = 0.05  # For statistical tests
    
    def __post_init__(self):
        if self.clustering_methods is None:
            self.clustering_methods = ['kmeans', 'gmm', 'hierarchical']
        if self.features_to_use is None:
            self.features_to_use = [
                'mean_diffusion_coefficient',
                'median_displacement',
                'mean_alpha',
                'confinement_radius',
                'mean_velocity',
                'track_length_mean'
            ]


class SubpopulationAnalyzer:
    """
    Analyzes tracking data to detect and characterize subpopulations within groups.
    
    This class implements multiple clustering algorithms and validation metrics to
    identify heterogeneous subgroups that may represent different biological states
    (e.g., cell cycle stages, metabolic conditions, treatment responses).
    """
    
    def __init__(self, config: Optional[SubpopulationConfig] = None):
        """
        Initialize the subpopulation analyzer.
        
        Parameters
        ----------
        config : SubpopulationConfig, optional
            Configuration object. If None, uses default settings.
        """
        self.config = config or SubpopulationConfig()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.pca = None
        self.results = {}
        
    def aggregate_tracks_by_cell(
        self,
        tracks_df: pd.DataFrame,
        cell_id_column: str = 'cell_id'
    ) -> pd.DataFrame:
        """
        Aggregate tracking data at the single-cell level.
        
        Computes per-cell statistics from individual track measurements.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Track data with columns: track_id, cell_id, x, y, frame, etc.
        cell_id_column : str
            Name of column containing cell identifiers
            
        Returns
        -------
        pd.DataFrame
            Cell-level aggregated features
        """
        if cell_id_column not in tracks_df.columns:
            logger.error(f"Column '{cell_id_column}' not found. Available: {tracks_df.columns.tolist()}")
            raise ValueError(f"Cell identifier column '{cell_id_column}' not found in data")
        
        # Group by cell and aggregate
        cell_features = []
        
        for cell_id, cell_data in tracks_df.groupby(cell_id_column):
            # Skip cells with too few tracks
            n_tracks = cell_data['track_id'].nunique()
            if n_tracks < self.config.min_tracks_per_cell:
                continue
            
            # Calculate cell-level features
            features = {
                'cell_id': cell_id,
                'n_tracks': n_tracks,
                'n_observations': len(cell_data)
            }
            
            # Diffusion-related features
            if 'diffusion_coefficient' in cell_data.columns:
                features['mean_diffusion_coefficient'] = cell_data['diffusion_coefficient'].mean()
                features['std_diffusion_coefficient'] = cell_data['diffusion_coefficient'].std()
                features['cv_diffusion_coefficient'] = (
                    features['std_diffusion_coefficient'] / features['mean_diffusion_coefficient']
                    if features['mean_diffusion_coefficient'] > 0 else 0
                )
            
            # Displacement features
            if 'x' in cell_data.columns and 'y' in cell_data.columns:
                displacements = np.sqrt(
                    cell_data.groupby('track_id').apply(
                        lambda g: np.sum(np.diff(g['x'])**2 + np.diff(g['y'])**2)
                    )
                )
                features['mean_displacement'] = displacements.mean()
                features['median_displacement'] = displacements.median()
                features['std_displacement'] = displacements.std()
            
            # Anomalous diffusion exponent
            if 'alpha' in cell_data.columns:
                features['mean_alpha'] = cell_data['alpha'].mean()
                features['std_alpha'] = cell_data['alpha'].std()
            
            # Confinement features
            if 'confinement_radius' in cell_data.columns:
                confined = cell_data[cell_data['confinement_radius'].notna()]
                if len(confined) > 0:
                    features['fraction_confined'] = len(confined) / len(cell_data)
                    features['mean_confinement_radius'] = confined['confinement_radius'].mean()
                else:
                    features['fraction_confined'] = 0
                    features['mean_confinement_radius'] = np.nan
            
            # Velocity features
            if 'velocity' in cell_data.columns:
                features['mean_velocity'] = cell_data['velocity'].mean()
                features['max_velocity'] = cell_data['velocity'].max()
                features['std_velocity'] = cell_data['velocity'].std()
            
            # Track length statistics
            track_lengths = cell_data.groupby('track_id').size()
            features['track_length_mean'] = track_lengths.mean()
            features['track_length_median'] = track_lengths.median()
            features['track_length_std'] = track_lengths.std()
            
            # Spatial features (if available)
            if 'x' in cell_data.columns and 'y' in cell_data.columns:
                features['spatial_extent_x'] = cell_data['x'].max() - cell_data['x'].min()
                features['spatial_extent_y'] = cell_data['y'].max() - cell_data['y'].min()
                features['center_x'] = cell_data['x'].mean()
                features['center_y'] = cell_data['y'].mean()
            
            cell_features.append(features)
        
        if not cell_features:
            raise ValueError(
                f"No cells have at least {self.config.min_tracks_per_cell} tracks. "
                f"Consider lowering min_tracks_per_cell."
            )
        
        cell_df = pd.DataFrame(cell_features)
        logger.info(f"Aggregated {len(cell_df)} cells from {len(tracks_df)} track observations")
        
        return cell_df
    
    def select_clustering_features(self, cell_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select and prepare features for clustering.
        
        Parameters
        ----------
        cell_df : pd.DataFrame
            Cell-level aggregated data
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            Feature matrix and list of selected feature names
        """
        # Filter to only use features that exist and are specified in config
        available_features = [f for f in self.config.features_to_use if f in cell_df.columns]
        
        if not available_features:
            # Fallback to any numeric columns except identifiers
            exclude_cols = ['cell_id', 'n_tracks', 'n_observations', 'center_x', 'center_y']
            available_features = [
                col for col in cell_df.select_dtypes(include=[np.number]).columns
                if col not in exclude_cols
            ]
        
        logger.info(f"Using features for clustering: {available_features}")
        
        # Extract feature matrix
        feature_matrix = cell_df[available_features].copy()
        
        # Handle missing values
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        # Remove infinite values
        feature_matrix = feature_matrix.replace([np.inf, -np.inf], np.nan)
        feature_matrix = feature_matrix.fillna(feature_matrix.median())
        
        return feature_matrix, available_features
    
    def determine_optimal_clusters(
        self,
        features_scaled: np.ndarray,
        method: str = 'kmeans'
    ) -> Dict[str, Any]:
        """
        Determine optimal number of clusters using multiple metrics.
        
        Parameters
        ----------
        features_scaled : np.ndarray
            Scaled feature matrix
        method : str
            Clustering method ('kmeans', 'gmm', 'hierarchical')
            
        Returns
        -------
        dict
            Contains optimal_k, scores for each k, and method used
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("sklearn not available, defaulting to 2 clusters")
            return {'optimal_k': 2, 'method': method, 'reason': 'sklearn unavailable'}
        
        n_min, n_max = self.config.n_clusters_range
        n_samples = features_scaled.shape[0]
        
        # Adjust range if sample size is small
        n_max = min(n_max, n_samples - 1)
        if n_max < n_min:
            return {'optimal_k': 2, 'method': method, 'reason': 'insufficient samples'}
        
        scores = {
            'k_values': [],
            'silhouette': [],
            'davies_bouldin': [],
            'calinski_harabasz': [],
            'bic': [] if method == 'gmm' else None,
            'aic': [] if method == 'gmm' else None
        }
        
        for k in range(n_min, n_max + 1):
            try:
                # Fit clustering model
                if method == 'kmeans':
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(features_scaled)
                elif method == 'gmm':
                    model = GaussianMixture(n_components=k, random_state=42)
                    labels = model.fit_predict(features_scaled)
                    scores['bic'].append(model.bic(features_scaled))
                    scores['aic'].append(model.aic(features_scaled))
                elif method == 'hierarchical':
                    model = AgglomerativeClustering(n_clusters=k)
                    labels = model.fit_predict(features_scaled)
                else:
                    logger.warning(f"Unknown method: {method}, defaulting to kmeans")
                    model = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = model.fit_predict(features_scaled)
                
                # Calculate validation metrics
                if len(np.unique(labels)) > 1:  # Need at least 2 clusters for these metrics
                    scores['k_values'].append(k)
                    scores['silhouette'].append(silhouette_score(features_scaled, labels))
                    scores['davies_bouldin'].append(davies_bouldin_score(features_scaled, labels))
                    scores['calinski_harabasz'].append(calinski_harabasz_score(features_scaled, labels))
                
            except Exception as e:
                logger.warning(f"Failed to evaluate k={k}: {str(e)}")
                continue
        
        # Determine optimal k
        optimal_k = n_min
        if scores['k_values']:
            # Use silhouette score (higher is better)
            optimal_idx = np.argmax(scores['silhouette'])
            optimal_k = scores['k_values'][optimal_idx]
            
            # For GMM, also consider BIC (lower is better)
            if method == 'gmm' and scores['bic']:
                bic_optimal_idx = np.argmin(scores['bic'])
                # If BIC and silhouette disagree significantly, prefer BIC
                if abs(bic_optimal_idx - optimal_idx) > 1:
                    optimal_k = scores['k_values'][bic_optimal_idx]
        
        return {
            'optimal_k': optimal_k,
            'scores': scores,
            'method': method
        }
    
    def perform_clustering(
        self,
        features_scaled: np.ndarray,
        method: str,
        n_clusters: int
    ) -> Tuple[np.ndarray, Any]:
        """
        Perform clustering with specified method and number of clusters.
        
        Parameters
        ----------
        features_scaled : np.ndarray
            Scaled feature matrix
        method : str
            Clustering method
        n_clusters : int
            Number of clusters
            
        Returns
        -------
        Tuple[np.ndarray, model]
            Cluster labels and fitted model
        """
        if not SKLEARN_AVAILABLE:
            # Fallback: simple split at median
            median_vals = np.median(features_scaled, axis=0)
            labels = (features_scaled[:, 0] > median_vals[0]).astype(int)
            return labels, None
        
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        elif method == 'gmm':
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == 'dbscan':
            # DBSCAN doesn't take n_clusters; use eps estimation
            from sklearn.neighbors import NearestNeighbors
            neighbors = NearestNeighbors(n_neighbors=5)
            neighbors_fit = neighbors.fit(features_scaled)
            distances, indices = neighbors_fit.kneighbors(features_scaled)
            distances = np.sort(distances[:, -1])
            eps = np.percentile(distances, 90)
            model = DBSCAN(eps=eps, min_samples=3)
        else:
            logger.warning(f"Unknown method {method}, using kmeans")
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        
        labels = model.fit_predict(features_scaled)
        
        return labels, model
    
    def characterize_subpopulations(
        self,
        cell_df: pd.DataFrame,
        labels: np.ndarray,
        features_used: List[str]
    ) -> Dict[str, Any]:
        """
        Characterize each identified subpopulation.
        
        Parameters
        ----------
        cell_df : pd.DataFrame
            Cell-level data
        labels : np.ndarray
            Cluster assignments
        features_used : List[str]
            Feature names used for clustering
            
        Returns
        -------
        dict
            Characteristics of each subpopulation
        """
        cell_df = cell_df.copy()
        cell_df['subpopulation'] = labels
        
        subpop_characteristics = {}
        
        for subpop_id in np.unique(labels):
            if subpop_id == -1:  # DBSCAN noise points
                continue
            
            subpop_data = cell_df[cell_df['subpopulation'] == subpop_id]
            n_cells = len(subpop_data)
            
            characteristics = {
                'subpopulation_id': int(subpop_id),
                'n_cells': n_cells,
                'fraction_of_total': n_cells / len(cell_df),
                'feature_means': {},
                'feature_std': {},
                'feature_medians': {}
            }
            
            # Characterize each feature
            for feature in features_used:
                if feature in subpop_data.columns:
                    characteristics['feature_means'][feature] = float(subpop_data[feature].mean())
                    characteristics['feature_std'][feature] = float(subpop_data[feature].std())
                    characteristics['feature_medians'][feature] = float(subpop_data[feature].median())
            
            # Statistical comparison to other subpopulations
            if SCIPY_AVAILABLE and len(np.unique(labels)) == 2:
                # For binary classification, perform statistical tests
                other_data = cell_df[cell_df['subpopulation'] != subpop_id]
                characteristics['statistical_tests'] = {}
                
                for feature in features_used[:3]:  # Test top 3 features
                    if feature in cell_df.columns:
                        try:
                            stat, pval = stats.mannwhitneyu(
                                subpop_data[feature].dropna(),
                                other_data[feature].dropna(),
                                alternative='two-sided'
                            )
                            characteristics['statistical_tests'][feature] = {
                                'statistic': float(stat),
                                'p_value': float(pval),
                                'significant': pval < self.config.significance_level
                            }
                        except Exception as e:
                            logger.warning(f"Failed statistical test for {feature}: {str(e)}")
            
            subpop_characteristics[f"subpop_{subpop_id}"] = characteristics
        
        return subpop_characteristics
    
    def analyze_group(
        self,
        tracks_df: pd.DataFrame,
        group_name: str,
        cell_id_column: str = 'cell_id'
    ) -> Dict[str, Any]:
        """
        Perform complete subpopulation analysis for a single group.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame
            Tracking data for this group
        group_name : str
            Name of the experimental group
        cell_id_column : str
            Column name for cell identifiers
            
        Returns
        -------
        dict
            Complete analysis results including clusters and characterizations
        """
        logger.info(f"Analyzing subpopulations for group: {group_name}")
        
        results = {
            'group_name': group_name,
            'success': False,
            'n_cells_total': 0,
            'subpopulations_detected': False
        }
        
        try:
            # Step 1: Aggregate to cell level
            cell_df = self.aggregate_tracks_by_cell(tracks_df, cell_id_column)
            results['n_cells_total'] = len(cell_df)
            
            if len(cell_df) < self.config.min_cell_sample_size:
                results['error'] = f"Insufficient cells (n={len(cell_df)}, minimum={self.config.min_cell_sample_size})"
                return results
            
            # Step 2: Select and prepare features
            feature_matrix, features_used = self.select_clustering_features(cell_df)
            results['features_used'] = features_used
            
            # Step 3: Scale features
            if SKLEARN_AVAILABLE and self.scaler is not None:
                features_scaled = self.scaler.fit_transform(feature_matrix)
            else:
                # Simple standardization
                features_scaled = (feature_matrix - feature_matrix.mean()) / feature_matrix.std()
                features_scaled = features_scaled.values
            
            # Step 4: Optional PCA
            if self.config.use_pca and SKLEARN_AVAILABLE:
                self.pca = PCA(n_components=self.config.pca_variance_threshold, random_state=42)
                features_scaled = self.pca.fit_transform(features_scaled)
                results['pca_components'] = self.pca.n_components_
                results['pca_explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
            
            # Step 5: Determine optimal number of clusters
            best_results = {}
            for method in self.config.clustering_methods:
                if method == 'dbscan':
                    # DBSCAN determines clusters automatically
                    continue
                
                opt_result = self.determine_optimal_clusters(features_scaled, method)
                best_results[method] = opt_result
            
            # Choose best method (highest average silhouette score)
            if best_results:
                best_method = max(
                    best_results.keys(),
                    key=lambda m: np.mean(best_results[m]['scores']['silhouette'])
                    if best_results[m]['scores']['silhouette'] else 0
                )
                optimal_k = best_results[best_method]['optimal_k']
            else:
                best_method = 'kmeans'
                optimal_k = 2
            
            results['clustering_method'] = best_method
            results['n_subpopulations'] = optimal_k
            results['optimization_results'] = best_results
            
            # Step 6: Perform final clustering
            labels, model = self.perform_clustering(features_scaled, best_method, optimal_k)
            results['cluster_labels'] = labels.tolist()
            results['cluster_model'] = model
            
            # Step 7: Characterize subpopulations
            subpop_chars = self.characterize_subpopulations(cell_df, labels, features_used)
            results['subpopulation_characteristics'] = subpop_chars
            
            # Step 8: Store cell-level data with assignments
            cell_df['subpopulation'] = labels
            results['cell_level_data'] = cell_df
            
            # Check if subpopulations are meaningfully different
            if len(np.unique(labels[labels != -1])) > 1:
                results['subpopulations_detected'] = True
            else:
                results['subpopulations_detected'] = False
                results['note'] = "No distinct subpopulations detected; group appears homogeneous"
            
            results['success'] = True
            logger.info(f"Completed analysis for {group_name}: {optimal_k} subpopulations detected")
            
        except Exception as e:
            results['error'] = str(e)
            logger.error(f"Error in subpopulation analysis for {group_name}: {str(e)}")
            import traceback
            results['traceback'] = traceback.format_exc()
        
        return results
    
    def compare_groups_with_subpopulations(
        self,
        group_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Compare groups considering their subpopulation structure.
        
        Parameters
        ----------
        group_results : dict
            Results from analyze_group for each experimental group
            
        Returns
        -------
        dict
            Comparative analysis considering subpopulation heterogeneity
        """
        comparison = {
            'n_groups': len(group_results),
            'groups_with_subpopulations': [],
            'homogeneous_groups': [],
            'summary': {}
        }
        
        for group_name, results in group_results.items():
            if results.get('subpopulations_detected'):
                comparison['groups_with_subpopulations'].append(group_name)
            else:
                comparison['homogeneous_groups'].append(group_name)
        
        # Summary statistics
        comparison['summary'] = {
            'total_groups': len(group_results),
            'heterogeneous_groups': len(comparison['groups_with_subpopulations']),
            'homogeneous_groups': len(comparison['homogeneous_groups']),
            'recommendation': self._generate_recommendation(group_results)
        }
        
        return comparison
    
    def _generate_recommendation(self, group_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate analysis recommendation based on findings."""
        heterogeneous = sum(1 for r in group_results.values() if r.get('subpopulations_detected'))
        total = len(group_results)
        
        if heterogeneous == 0:
            return "All groups appear homogeneous. Proceed with standard group comparisons."
        elif heterogeneous == total:
            return "All groups show subpopulations. Recommend subgroup-level comparisons."
        else:
            return f"{heterogeneous}/{total} groups show subpopulations. Consider mixed analysis approach."


def create_subpopulation_visualizations(
    results: Dict[str, Any],
    group_name: str
) -> Dict[str, Any]:
    """
    Create comprehensive visualizations for subpopulation analysis.
    
    Parameters
    ----------
    results : dict
        Results from SubpopulationAnalyzer.analyze_group
    group_name : str
        Name of the group
        
    Returns
    -------
    dict
        Dictionary of plotly figures
    """
    if not PLOTLY_AVAILABLE:
        return {'error': 'Plotly not available for visualizations'}
    
    figures = {}
    
    if not results.get('success'):
        return figures
    
    cell_df = results.get('cell_level_data')
    if cell_df is None or 'subpopulation' not in cell_df.columns:
        return figures
    
    # 1. Feature distributions by subpopulation
    features_used = results.get('features_used', [])
    for feature in features_used[:4]:  # Show top 4 features
        if feature in cell_df.columns:
            fig = px.box(
                cell_df,
                x='subpopulation',
                y=feature,
                color='subpopulation',
                title=f"{feature} by Subpopulation - {group_name}",
                labels={'subpopulation': 'Subpopulation', feature: feature.replace('_', ' ').title()}
            )
            figures[f'{feature}_distribution'] = fig
    
    # 2. PCA visualization (if PCA was used)
    if 'pca_components' in results and len(features_used) >= 2:
        # We'd need to store PCA-transformed coordinates for this
        pass
    
    # 3. Subpopulation size pie chart
    subpop_counts = cell_df['subpopulation'].value_counts()
    fig = go.Figure(data=[go.Pie(
        labels=[f"Subpop {i}" for i in subpop_counts.index],
        values=subpop_counts.values,
        hole=0.3
    )])
    fig.update_layout(title=f"Subpopulation Distribution - {group_name}")
    figures['subpopulation_sizes'] = fig
    
    # 4. Feature correlation heatmap
    if len(features_used) > 1:
        feature_cols = [f for f in features_used if f in cell_df.columns]
        if feature_cols:
            corr_matrix = cell_df[feature_cols].corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='RdBu',
                zmid=0
            ))
            fig.update_layout(title=f"Feature Correlations - {group_name}")
            figures['feature_correlations'] = fig
    
    return figures
