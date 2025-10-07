"""
MD-SPT Comparison Framework
Compare molecular dynamics simulation trajectories with experimental SPT data.
Includes statistical tests, compartment-specific analysis, and visualization.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from dataclasses import dataclass
import warnings

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Visualization features disabled.")

# Import analysis functions
try:
    from analysis import calculate_msd, analyze_diffusion
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False
    warnings.warn("Analysis module not available. Some features disabled.")


# ==================== DATA CLASSES ====================

@dataclass
class ComparisonMetrics:
    """Statistical comparison metrics"""
    diffusion_coeff_md: float
    diffusion_coeff_spt: float
    diffusion_ratio: float  # MD/SPT
    p_value_diff: float  # Statistical test p-value
    msd_correlation: float  # Correlation between MD and SPT MSD curves
    track_length_md: float
    track_length_spt: float
    n_tracks_md: int
    n_tracks_spt: int


@dataclass
class CompartmentComparison:
    """Compartment-specific comparison"""
    compartment: str
    md_diffusion: float
    spt_diffusion: float
    md_confinement: float
    spt_confinement: float
    p_value: float
    n_tracks_md: int
    n_tracks_spt: int


# ==================== DIFFUSION ANALYSIS ====================

def calculate_diffusion_coefficient(tracks_df: pd.DataFrame, pixel_size: float = 0.1,
                                    frame_interval: float = 0.1, method: str = 'msd') -> Tuple[float, np.ndarray]:
    """
    Calculate diffusion coefficient from tracks.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks with columns [track_id, frame, x, y]
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    method : str
        'msd' or 'variance' method
        
    Returns
    -------
    D : float
        Diffusion coefficient in μm²/s
    msd_curve : np.ndarray
        MSD values for each lag time
    """
    track_ids = tracks_df['track_id'].unique()
    
    if method == 'msd':
        # MSD-based estimation
        max_lag = 50
        msds = []
        
        for lag in range(1, max_lag + 1):
            lag_msds = []
            for track_id in track_ids:
                track = tracks_df[tracks_df['track_id'] == track_id].copy()
                if len(track) <= lag:
                    continue
                
                coords = track[['x', 'y']].values * pixel_size
                disps = coords[lag:] - coords[:-lag]
                msd = np.mean(np.sum(disps**2, axis=1))
                lag_msds.append(msd)
            
            if lag_msds:
                msds.append(np.mean(lag_msds))
            else:
                msds.append(np.nan)
        
        msds = np.array(msds)
        lags = np.arange(1, max_lag + 1) * frame_interval
        
        # Linear fit to first 5 points: MSD = 4Dt for 2D
        valid = ~np.isnan(msds[:5])
        if valid.sum() >= 3:
            D_fit = np.polyfit(lags[:5][valid], msds[:5][valid], 1)
            D = D_fit[0] / 4.0  # 2D diffusion
        else:
            D = np.nan
        
        return D, msds
    
    elif method == 'variance':
        # Variance-based estimation
        all_disps = []
        for track_id in track_ids:
            track = tracks_df[tracks_df['track_id'] == track_id].copy()
            if len(track) < 2:
                continue
            coords = track[['x', 'y']].values * pixel_size
            disps = np.diff(coords, axis=0)
            all_disps.append(disps)
        
        if all_disps:
            all_disps = np.vstack(all_disps)
            variance = np.mean(np.sum(all_disps**2, axis=1))
            D = variance / (4 * frame_interval)
            return D, np.array([variance])
        else:
            return np.nan, np.array([])
    
    else:
        raise ValueError(f"Unknown method: {method}")


def calculate_confinement_metrics(tracks_df: pd.DataFrame, pixel_size: float = 0.1) -> Dict[str, float]:
    """
    Calculate confinement metrics for tracks.
    
    Returns
    -------
    dict
        'radius_gyration': mean radius of gyration
        'confinement_ratio': ratio of Rg to max distance from centroid
        'asphericity': shape anisotropy measure
    """
    track_ids = tracks_df['track_id'].unique()
    
    rg_values = []
    conf_ratios = []
    asphericities = []
    
    for track_id in track_ids:
        track = tracks_df[tracks_df['track_id'] == track_id].copy()
        if len(track) < 5:
            continue
        
        coords = track[['x', 'y']].values * pixel_size
        centroid = np.mean(coords, axis=0)
        
        # Radius of gyration
        rg = np.sqrt(np.mean(np.sum((coords - centroid)**2, axis=1)))
        rg_values.append(rg)
        
        # Confinement ratio
        max_dist = np.max(np.linalg.norm(coords - centroid, axis=1))
        conf_ratio = rg / (max_dist + 1e-10)
        conf_ratios.append(conf_ratio)
        
        # Asphericity
        centered = coords - centroid
        gyration_tensor = np.dot(centered.T, centered) / len(coords)
        eigenvalues = np.linalg.eigvalsh(gyration_tensor)
        if eigenvalues[1] + eigenvalues[0] > 0:
            asphericity = (eigenvalues[1] - eigenvalues[0])**2 / (eigenvalues[1] + eigenvalues[0])**2
            asphericities.append(asphericity)
    
    return {
        'radius_gyration': np.mean(rg_values) if rg_values else np.nan,
        'confinement_ratio': np.mean(conf_ratios) if conf_ratios else np.nan,
        'asphericity': np.mean(asphericities) if asphericities else np.nan
    }


# ==================== STATISTICAL TESTS ====================

def compare_diffusion_coefficients(D_md: float, D_spt: float,
                                   tracks_md: pd.DataFrame, tracks_spt: pd.DataFrame,
                                   pixel_size: float = 0.1,
                                   frame_interval: float = 0.1) -> Dict[str, Any]:
    """
    Statistical comparison of diffusion coefficients.
    
    Uses bootstrap resampling to estimate confidence intervals and
    performs hypothesis testing.
    
    Returns
    -------
    dict
        'diffusion_md': MD diffusion coefficient
        'diffusion_spt': SPT diffusion coefficient
        'ratio': MD/SPT ratio
        'p_value': p-value from bootstrap test
        'ci_md': 95% CI for MD
        'ci_spt': 95% CI for SPT
        'significant': whether difference is significant (p < 0.05)
    """
    # Bootstrap resampling
    n_bootstrap = 1000
    md_bootstrap = []
    spt_bootstrap = []
    
    track_ids_md = tracks_md['track_id'].unique()
    track_ids_spt = tracks_spt['track_id'].unique()
    
    rng = np.random.default_rng(42)
    
    for _ in range(n_bootstrap):
        # Resample tracks with replacement
        sampled_md = rng.choice(track_ids_md, size=len(track_ids_md), replace=True)
        sampled_spt = rng.choice(track_ids_spt, size=len(track_ids_spt), replace=True)
        
        # Calculate D for resampled tracks
        md_sample = tracks_md[tracks_md['track_id'].isin(sampled_md)]
        spt_sample = tracks_spt[tracks_spt['track_id'].isin(sampled_spt)]
        
        D_md_boot, _ = calculate_diffusion_coefficient(md_sample, pixel_size, frame_interval)
        D_spt_boot, _ = calculate_diffusion_coefficient(spt_sample, pixel_size, frame_interval)
        
        if not np.isnan(D_md_boot):
            md_bootstrap.append(D_md_boot)
        if not np.isnan(D_spt_boot):
            spt_bootstrap.append(D_spt_boot)
    
    md_bootstrap = np.array(md_bootstrap)
    spt_bootstrap = np.array(spt_bootstrap)
    
    # Calculate confidence intervals
    ci_md = (np.percentile(md_bootstrap, 2.5), np.percentile(md_bootstrap, 97.5))
    ci_spt = (np.percentile(spt_bootstrap, 2.5), np.percentile(spt_bootstrap, 97.5))
    
    # Hypothesis test: are the distributions different?
    # Use Mann-Whitney U test (non-parametric)
    if len(md_bootstrap) > 0 and len(spt_bootstrap) > 0:
        statistic, p_value = stats.mannwhitneyu(md_bootstrap, spt_bootstrap, alternative='two-sided')
    else:
        p_value = np.nan
    
    return {
        'diffusion_md': D_md,
        'diffusion_spt': D_spt,
        'ratio': D_md / D_spt if D_spt != 0 else np.inf,
        'p_value': p_value,
        'ci_md': ci_md,
        'ci_spt': ci_spt,
        'significant': p_value < 0.05 if not np.isnan(p_value) else False,
        'bootstrap_md': md_bootstrap,
        'bootstrap_spt': spt_bootstrap
    }


def compare_msd_curves(msd_md: np.ndarray, msd_spt: np.ndarray) -> Dict[str, float]:
    """
    Compare MSD curves between MD and SPT.
    
    Returns
    -------
    dict
        'correlation': Pearson correlation
        'rmse': Root mean squared error
        'mae': Mean absolute error
    """
    # Match lengths
    min_len = min(len(msd_md), len(msd_spt))
    msd_md_trim = msd_md[:min_len]
    msd_spt_trim = msd_spt[:min_len]
    
    # Remove NaN values
    valid = ~(np.isnan(msd_md_trim) | np.isnan(msd_spt_trim))
    msd_md_valid = msd_md_trim[valid]
    msd_spt_valid = msd_spt_trim[valid]
    
    if len(msd_md_valid) < 3:
        return {'correlation': np.nan, 'rmse': np.nan, 'mae': np.nan}
    
    # Calculate metrics
    correlation, _ = stats.pearsonr(msd_md_valid, msd_spt_valid)
    rmse = np.sqrt(np.mean((msd_md_valid - msd_spt_valid)**2))
    mae = np.mean(np.abs(msd_md_valid - msd_spt_valid))
    
    return {
        'correlation': correlation,
        'rmse': rmse,
        'mae': mae
    }


# ==================== COMPARTMENT ANALYSIS ====================

def compare_by_compartment(tracks_md: pd.DataFrame, tracks_spt: pd.DataFrame,
                          pixel_size: float = 0.1,
                          frame_interval: float = 0.1) -> List[CompartmentComparison]:
    """
    Compare MD and SPT trajectories by nuclear compartment.
    
    Both DataFrames must have a 'compartment' column.
    
    Returns
    -------
    list of CompartmentComparison
        Comparison for each compartment
    """
    if 'compartment' not in tracks_md.columns or 'compartment' not in tracks_spt.columns:
        warnings.warn("Compartment column not found in both datasets. Skipping compartment analysis.")
        return []
    
    # Get unique compartments
    compartments_md = set(tracks_md['compartment'].unique())
    compartments_spt = set(tracks_spt['compartment'].unique())
    common_compartments = compartments_md.intersection(compartments_spt)
    
    results = []
    
    for comp in common_compartments:
        # Filter tracks for this compartment
        md_comp = tracks_md[tracks_md['compartment'] == comp]
        spt_comp = tracks_spt[tracks_spt['compartment'] == comp]
        
        if len(md_comp) < 10 or len(spt_comp) < 10:
            continue  # Not enough data
        
        # Calculate diffusion coefficients
        D_md, _ = calculate_diffusion_coefficient(md_comp, pixel_size, frame_interval)
        D_spt, _ = calculate_diffusion_coefficient(spt_comp, pixel_size, frame_interval)
        
        # Calculate confinement
        conf_md = calculate_confinement_metrics(md_comp, pixel_size)
        conf_spt = calculate_confinement_metrics(spt_comp, pixel_size)
        
        # Statistical test
        comparison = compare_diffusion_coefficients(
            D_md, D_spt, md_comp, spt_comp, pixel_size, frame_interval
        )
        
        results.append(CompartmentComparison(
            compartment=comp,
            md_diffusion=D_md,
            spt_diffusion=D_spt,
            md_confinement=conf_md['radius_gyration'],
            spt_confinement=conf_spt['radius_gyration'],
            p_value=comparison['p_value'],
            n_tracks_md=len(md_comp['track_id'].unique()),
            n_tracks_spt=len(spt_comp['track_id'].unique())
        ))
    
    return results


# ==================== VISUALIZATION ====================

def plot_diffusion_comparison(comparison_result: Dict[str, Any],
                             compartment: str = "Overall") -> go.Figure:
    """
    Create visualization of diffusion coefficient comparison.
    
    Shows:
    - Bar plot of MD vs SPT diffusion coefficients
    - Error bars from bootstrap confidence intervals
    - Statistical significance indicator
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required for visualization")
    
    D_md = comparison_result['diffusion_md']
    D_spt = comparison_result['diffusion_spt']
    ci_md = comparison_result['ci_md']
    ci_spt = comparison_result['ci_spt']
    p_value = comparison_result['p_value']
    
    fig = go.Figure()
    
    # MD bar
    fig.add_trace(go.Bar(
        x=['MD Simulation'],
        y=[D_md],
        name='MD Simulation',
        marker_color='#3498db',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[ci_md[1] - D_md],
            arrayminus=[D_md - ci_md[0]]
        )
    ))
    
    # SPT bar
    fig.add_trace(go.Bar(
        x=['Experimental SPT'],
        y=[D_spt],
        name='Experimental SPT',
        marker_color='#e74c3c',
        error_y=dict(
            type='data',
            symmetric=False,
            array=[ci_spt[1] - D_spt],
            arrayminus=[D_spt - ci_spt[0]]
        )
    ))
    
    # Add significance indicator
    sig_text = f"p = {p_value:.4f}" if not np.isnan(p_value) else "p = N/A"
    if comparison_result.get('significant', False):
        sig_text += " *"
    
    fig.add_annotation(
        x=0.5, y=max(D_md, D_spt) * 1.1,
        text=sig_text,
        showarrow=False,
        font=dict(size=12)
    )
    
    fig.update_layout(
        title=f"Diffusion Coefficient Comparison - {compartment}",
        yaxis_title="Diffusion Coefficient (μm²/s)",
        showlegend=True,
        template='plotly_white'
    )
    
    return fig


def plot_msd_comparison(msd_md: np.ndarray, msd_spt: np.ndarray,
                       lag_times: np.ndarray,
                       title: str = "MSD Comparison") -> go.Figure:
    """
    Plot MSD curves for MD and SPT data.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required for visualization")
    
    fig = go.Figure()
    
    # MD MSD
    fig.add_trace(go.Scatter(
        x=lag_times,
        y=msd_md,
        mode='lines+markers',
        name='MD Simulation',
        line=dict(color='#3498db', width=2),
        marker=dict(size=6)
    ))
    
    # SPT MSD
    fig.add_trace(go.Scatter(
        x=lag_times,
        y=msd_spt,
        mode='lines+markers',
        name='Experimental SPT',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        marker=dict(size=6, symbol='square')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Lag Time (s)",
        yaxis_title="MSD (μm²)",
        template='plotly_white',
        showlegend=True
    )
    
    return fig


def plot_compartment_comparison(compartment_results: List[CompartmentComparison]) -> go.Figure:
    """
    Plot compartment-specific comparison.
    """
    if not PLOTLY_AVAILABLE:
        raise RuntimeError("Plotly required for visualization")
    
    compartments = [c.compartment for c in compartment_results]
    md_values = [c.md_diffusion for c in compartment_results]
    spt_values = [c.spt_diffusion for c in compartment_results]
    
    fig = go.Figure()
    
    # MD bars
    fig.add_trace(go.Bar(
        x=compartments,
        y=md_values,
        name='MD Simulation',
        marker_color='#3498db'
    ))
    
    # SPT bars
    fig.add_trace(go.Bar(
        x=compartments,
        y=spt_values,
        name='Experimental SPT',
        marker_color='#e74c3c'
    ))
    
    # Add significance markers
    for i, comp_result in enumerate(compartment_results):
        if comp_result.p_value < 0.05:
            max_y = max(comp_result.md_diffusion, comp_result.spt_diffusion)
            fig.add_annotation(
                x=comp_result.compartment,
                y=max_y * 1.05,
                text="*",
                showarrow=False,
                font=dict(size=16, color='red')
            )
    
    fig.update_layout(
        title="Diffusion Coefficients by Compartment",
        xaxis_title="Compartment",
        yaxis_title="Diffusion Coefficient (μm²/s)",
        barmode='group',
        template='plotly_white'
    )
    
    return fig


# ==================== HIGH-LEVEL API ====================

def compare_md_with_spt(tracks_md: pd.DataFrame, tracks_spt: pd.DataFrame,
                       pixel_size: float = 0.1, frame_interval: float = 0.1,
                       analyze_compartments: bool = True) -> Dict[str, Any]:
    """
    Comprehensive comparison of MD simulation with experimental SPT data.
    
    Parameters
    ----------
    tracks_md : pd.DataFrame
        MD simulation tracks
    tracks_spt : pd.DataFrame
        Experimental SPT tracks
    pixel_size : float
        Pixel size in μm
    frame_interval : float
        Frame interval in seconds
    analyze_compartments : bool
        Whether to perform compartment-specific analysis
        
    Returns
    -------
    dict
        Comprehensive comparison results including:
        - Overall diffusion comparison
        - MSD curve comparison
        - Confinement metrics
        - Compartment-specific results (if available)
        - Visualization figures
    """
    results = {
        'success': True,
        'parameters': {
            'pixel_size': pixel_size,
            'frame_interval': frame_interval,
            'n_tracks_md': len(tracks_md['track_id'].unique()),
            'n_tracks_spt': len(tracks_spt['track_id'].unique())
        }
    }
    
    # Calculate overall diffusion coefficients
    D_md, msd_md = calculate_diffusion_coefficient(tracks_md, pixel_size, frame_interval)
    D_spt, msd_spt = calculate_diffusion_coefficient(tracks_spt, pixel_size, frame_interval)
    
    results['diffusion_md'] = D_md
    results['diffusion_spt'] = D_spt
    results['diffusion_ratio'] = D_md / D_spt if D_spt != 0 else np.inf
    
    # Statistical comparison
    diff_comparison = compare_diffusion_coefficients(
        D_md, D_spt, tracks_md, tracks_spt, pixel_size, frame_interval
    )
    results['statistical_test'] = diff_comparison
    
    # MSD comparison
    msd_comparison = compare_msd_curves(msd_md, msd_spt)
    results['msd_comparison'] = msd_comparison
    results['msd_md'] = msd_md
    results['msd_spt'] = msd_spt
    
    # Confinement metrics
    conf_md = calculate_confinement_metrics(tracks_md, pixel_size)
    conf_spt = calculate_confinement_metrics(tracks_spt, pixel_size)
    results['confinement_md'] = conf_md
    results['confinement_spt'] = conf_spt
    
    # Compartment analysis (if requested and available)
    if analyze_compartments and 'compartment' in tracks_md.columns and 'compartment' in tracks_spt.columns:
        compartment_results = compare_by_compartment(tracks_md, tracks_spt, pixel_size, frame_interval)
        results['compartment_comparison'] = compartment_results
    
    # Generate visualizations
    if PLOTLY_AVAILABLE:
        lag_times = np.arange(1, len(msd_md) + 1) * frame_interval
        
        results['figures'] = {
            'diffusion_comparison': plot_diffusion_comparison(diff_comparison),
            'msd_comparison': plot_msd_comparison(msd_md, msd_spt, lag_times)
        }
        
        if analyze_compartments and 'compartment_comparison' in results:
            if len(results['compartment_comparison']) > 0:
                results['figures']['compartment_comparison'] = plot_compartment_comparison(
                    results['compartment_comparison']
                )
    
    # Summary statistics
    results['summary'] = {
        'diffusion_agreement': 'Good' if abs(D_md - D_spt) / D_spt < 0.2 else 'Poor',
        'msd_correlation': msd_comparison['correlation'],
        'statistically_different': diff_comparison['significant'],
        'recommendation': _generate_recommendation(results)
    }
    
    return results


def _generate_recommendation(results: Dict[str, Any]) -> str:
    """Generate interpretation and recommendation based on results"""
    diff_ratio = results['diffusion_ratio']
    significant = results['statistical_test']['significant']
    correlation = results['msd_comparison']['correlation']
    
    if abs(diff_ratio - 1.0) < 0.2 and not significant and correlation > 0.8:
        return "Excellent agreement between MD simulation and experimental data. Model parameters are well-calibrated."
    elif abs(diff_ratio - 1.0) < 0.5 and correlation > 0.6:
        return "Good agreement. Minor parameter adjustments may improve fit."
    elif diff_ratio > 1.5:
        return "MD simulation shows faster diffusion than experiment. Consider increasing viscosity or crowding in model."
    elif diff_ratio < 0.7:
        return "MD simulation shows slower diffusion than experiment. Consider reducing viscosity or crowding in model."
    else:
        return "Moderate agreement. Review model assumptions and compartment properties."
