"""
Batch Report Generator Enhancements

This module extends the enhanced_report_generator with:
1. Missing advanced biophysical analyses (FBM, TAMSD/EAMSD, NGP, van Hove, VACF, turning angles, ergodicity)
2. Advanced microrheology metrics (frequency sweeps, viscoelasticity, power-law rheology)
3. Statistical group comparison tools (t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis)
4. Effect size calculations and multiple testing corrections
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway, kruskal
import warnings

try:
    from advanced_biophysical_metrics import (
        AdvancedMetricsAnalyzer, MetricConfig,
        fit_fbm_model, analyze_turning_angles, calculate_ergodicity_breaking
    )
    ADVANCED_METRICS_AVAILABLE = True
except ImportError:
    ADVANCED_METRICS_AVAILABLE = False

try:
    from rheology import MicrorheologyAnalyzer, create_rheology_plots
    RHEOLOGY_AVAILABLE = True
except ImportError:
    RHEOLOGY_AVAILABLE = False

try:
    from analysis import calculate_msd
    MSD_AVAILABLE = True
except ImportError:
    MSD_AVAILABLE = False


class AdvancedBiophysicalReportExtension:
    """
    Extension for EnhancedSPTReportGenerator adding advanced biophysical analyses.
    """
    
    @staticmethod
    def analyze_fbm_ensemble(tracks_df: pd.DataFrame, pixel_size: float = 1.0, 
                            frame_interval: float = 1.0) -> Dict[str, Any]:
        """
        Fractional Brownian Motion analysis for all tracks.
        
        Returns
        -------
        Dict with:
            - success: bool
            - hurst_values: List of H values per track
            - diffusion_values: List of D values per track
            - summary: Statistics on H and D
            - data: DataFrame with track-level results
        """
        if not ADVANCED_METRICS_AVAILABLE:
            return {
                'success': False,
                'error': 'Advanced biophysical metrics module not available'
            }
        
        try:
            results = []
            for track_id in tracks_df['track_id'].unique():
                track_data = tracks_df[tracks_df['track_id'] == track_id].copy()
                fbm_result = fit_fbm_model(track_data, pixel_size, frame_interval)
                
                results.append({
                    'track_id': track_id,
                    'H': fbm_result.get('H', np.nan),
                    'D': fbm_result.get('D', np.nan),
                    'track_length': len(track_data)
                })
            
            results_df = pd.DataFrame(results)
            valid_h = results_df['H'].dropna()
            valid_d = results_df['D'].dropna()
            
            summary = {
                'n_tracks': len(results_df),
                'n_valid': len(valid_h),
                'H_mean': float(valid_h.mean()) if len(valid_h) > 0 else np.nan,
                'H_std': float(valid_h.std()) if len(valid_h) > 0 else np.nan,
                'H_median': float(valid_h.median()) if len(valid_h) > 0 else np.nan,
                'D_mean': float(valid_d.mean()) if len(valid_d) > 0 else np.nan,
                'D_std': float(valid_d.std()) if len(valid_d) > 0 else np.nan,
                'D_median': float(valid_d.median()) if len(valid_d) > 0 else np.nan
            }
            
            return {
                'success': True,
                'hurst_values': valid_h.tolist(),
                'diffusion_values': valid_d.tolist(),
                'summary': summary,
                'data': results_df
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def plot_fbm_results(result: Dict) -> go.Figure:
        """Create visualization for FBM analysis results."""
        if not result.get('success', False):
            return None
        
        data = result.get('data')
        if data is None or data.empty:
            return None
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Hurst Exponent Distribution', 'Diffusion Coefficient Distribution')
        )
        
        # Hurst exponent histogram
        valid_h = data['H'].dropna()
        if len(valid_h) > 0:
            fig.add_trace(
                go.Histogram(x=valid_h, name='H', nbinsx=30, marker_color='steelblue'),
                row=1, col=1
            )
            # Add reference lines
            fig.add_vline(x=0.5, line_dash="dash", line_color="red", 
                         annotation_text="H=0.5 (Brownian)", row=1, col=1)
        
        # Diffusion coefficient histogram
        valid_d = data['D'].dropna()
        if len(valid_d) > 0:
            fig.add_trace(
                go.Histogram(x=valid_d, name='D', nbinsx=30, marker_color='coral'),
                row=1, col=2
            )
        
        fig.update_xaxes(title_text="Hurst Exponent", row=1, col=1)
        fig.update_xaxes(title_text="Diffusion Coefficient (μm²/s)", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        
        fig.update_layout(
            height=400,
            showlegend=False,
            title_text="Fractional Brownian Motion Analysis"
        )
        
        return fig
    
    @staticmethod
    def analyze_advanced_metrics_ensemble(tracks_df: pd.DataFrame, pixel_size: float = 1.0,
                                         frame_interval: float = 1.0, 
                                         max_lag: int = 20) -> Dict[str, Any]:
        """
        Comprehensive advanced metrics analysis using AdvancedMetricsAnalyzer.
        
        Returns TAMSD, EAMSD, ergodicity, NGP, van Hove, VACF, turning angles, and Hurst exponent.
        """
        if not ADVANCED_METRICS_AVAILABLE:
            return {
                'success': False,
                'error': 'Advanced biophysical metrics module not available'
            }
        
        try:
            config = MetricConfig(
                pixel_size=pixel_size,
                frame_interval=frame_interval,
                max_lag=max_lag,
                n_bootstrap=100
            )
            
            analyzer = AdvancedMetricsAnalyzer(tracks_df, config)
            results = analyzer.compute_all()
            
            return results
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    @staticmethod
    def plot_advanced_metrics(result: Dict) -> Dict[str, go.Figure]:
        """Create comprehensive visualizations for advanced metrics."""
        if not result.get('success', False):
            return {}
        
        figures = {}
        
        # 1. TAMSD vs EAMSD
        if 'tamsd' in result and 'eamsd' in result:
            tamsd_df = result['tamsd']
            eamsd_df = result['eamsd']
            
            fig = go.Figure()
            
            # Plot individual TAMSD curves (sample)
            track_ids = tamsd_df['track_id'].unique()
            sample_tracks = np.random.choice(track_ids, size=min(10, len(track_ids)), replace=False)
            
            for tid in sample_tracks:
                track_tamsd = tamsd_df[tamsd_df['track_id'] == tid]
                fig.add_trace(go.Scatter(
                    x=track_tamsd['tau_s'],
                    y=track_tamsd['tamsd'],
                    mode='lines',
                    name=f'Track {tid}',
                    line=dict(width=1),
                    opacity=0.5,
                    showlegend=False
                ))
            
            # Plot EAMSD
            fig.add_trace(go.Scatter(
                x=eamsd_df['tau_s'],
                y=eamsd_df['eamsd'],
                mode='lines+markers',
                name='EAMSD',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            fig.update_xaxes(title_text="Lag Time (s)", type='log')
            fig.update_yaxes(title_text="MSD (μm²)", type='log')
            fig.update_layout(
                title="Time-Averaged MSD (TAMSD) vs Ensemble-Averaged MSD (EAMSD)",
                height=500
            )
            
            figures['tamsd_eamsd'] = fig
        
        # 2. Ergodicity Breaking
        if 'ergodicity' in result:
            erg_df = result['ergodicity']
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('EB Ratio', 'EB Parameter')
            )
            
            fig.add_trace(go.Scatter(
                x=erg_df['tau_s'],
                y=erg_df['EB_ratio'],
                mode='lines+markers',
                name='EB Ratio',
                marker=dict(size=8, color='steelblue')
            ), row=1, col=1)
            
            # Add reference line at 1 (ergodic)
            fig.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=erg_df['tau_s'],
                y=erg_df['EB_parameter'],
                mode='lines+markers',
                name='EB Parameter',
                marker=dict(size=8, color='coral')
            ), row=1, col=2)
            
            fig.update_xaxes(title_text="Lag Time (s)", type='log', row=1, col=1)
            fig.update_xaxes(title_text="Lag Time (s)", type='log', row=1, col=2)
            fig.update_yaxes(title_text="EB Ratio", row=1, col=1)
            fig.update_yaxes(title_text="EB Parameter", row=1, col=2)
            
            fig.update_layout(
                title="Ergodicity Breaking Analysis",
                height=400,
                showlegend=False
            )
            
            figures['ergodicity'] = fig
        
        # 3. NGP (Non-Gaussian Parameter)
        if 'ngp' in result:
            ngp_df = result['ngp']
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=ngp_df['tau_s'],
                y=ngp_df['NGP_1D'],
                mode='lines+markers',
                name='NGP 1D',
                marker=dict(size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=ngp_df['tau_s'],
                y=ngp_df['NGP_2D'],
                mode='lines+markers',
                name='NGP 2D',
                marker=dict(size=8)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="red", 
                         annotation_text="NGP=0 (Gaussian)")
            
            fig.update_xaxes(title_text="Lag Time (s)", type='log')
            fig.update_yaxes(title_text="Non-Gaussian Parameter")
            fig.update_layout(
                title="Non-Gaussian Parameter vs Lag Time",
                height=400
            )
            
            figures['ngp'] = fig
        
        # 4. VACF (Velocity Autocorrelation Function)
        if 'vacf' in result:
            vacf_df = result['vacf']
            
            fig = go.Figure()
            
            # Handle both 'VACF' (from advanced_biophysical_metrics.py) and 'vacf' column names
            vacf_col = 'VACF' if 'VACF' in vacf_df.columns else 'vacf'
            
            fig.add_trace(go.Scatter(
                x=vacf_df['lag'],
                y=vacf_df[vacf_col],
                mode='lines+markers',
                marker=dict(size=8, color='steelblue'),
                line=dict(width=2)
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            
            fig.update_xaxes(title_text="Lag (frames)")
            fig.update_yaxes(title_text="VACF (normalized)")
            fig.update_layout(
                title="Velocity Autocorrelation Function",
                height=400
            )
            
            figures['vacf'] = fig
        
        # 5. Turning Angles
        if 'turning_angles' in result:
            ang_df = result['turning_angles']
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=np.degrees(ang_df['angle_rad']),
                nbinsx=60,
                name='Turning Angles',
                marker_color='steelblue'
            ))
            
            fig.update_xaxes(title_text="Turning Angle (degrees)")
            fig.update_yaxes(title_text="Count")
            fig.update_layout(
                title="Distribution of Turning Angles",
                height=400
            )
            
            figures['turning_angles'] = fig
        
        return figures


class StatisticalComparisonTools:
    """
    Statistical comparison tools for batch analysis across experimental groups.
    """
    
    @staticmethod
    def compare_groups_parametric(groups: Dict[str, List[float]], 
                                  metric_name: str = "Metric") -> Dict[str, Any]:
        """
        Perform parametric statistical tests (t-test or ANOVA) on groups.
        
        Parameters
        ----------
        groups : Dict[str, List[float]]
            Dictionary mapping group names to values
        metric_name : str
            Name of the metric being compared
            
        Returns
        -------
        Dict with test results, p-values, and effect sizes
        """
        group_names = list(groups.keys())
        group_values = [np.array(groups[name]) for name in group_names]
        
        # Remove NaN values
        group_values = [vals[~np.isnan(vals)] for vals in group_values]
        group_names = [name for name, vals in zip(group_names, group_values) if len(vals) > 0]
        group_values = [vals for vals in group_values if len(vals) > 0]
        
        if len(group_values) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 groups with valid data'
            }
        
        result = {
            'success': True,
            'metric_name': metric_name,
            'group_names': group_names,
            'n_groups': len(group_values)
        }
        
        # Calculate descriptive statistics
        result['descriptive'] = {}
        for name, vals in zip(group_names, group_values):
            result['descriptive'][name] = {
                'n': len(vals),
                'mean': float(np.mean(vals)),
                'std': float(np.std(vals, ddof=1)),
                'median': float(np.median(vals)),
                'sem': float(stats.sem(vals))
            }
        
        # Perform appropriate test
        if len(group_values) == 2:
            # Two-sample t-test
            t_stat, p_value = ttest_ind(group_values[0], group_values[1], equal_var=False)
            
            # Cohen's d effect size
            mean1, mean2 = np.mean(group_values[0]), np.mean(group_values[1])
            std1, std2 = np.std(group_values[0], ddof=1), np.std(group_values[1], ddof=1)
            n1, n2 = len(group_values[0]), len(group_values[1])
            pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1 + n2 - 2))
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else np.nan
            
            result['test'] = 't-test (Welch)'
            result['t_statistic'] = float(t_stat)
            result['p_value'] = float(p_value)
            result['effect_size'] = {
                'cohens_d': float(cohens_d),
                'interpretation': StatisticalComparisonTools._interpret_cohens_d(cohens_d)
            }
            
        else:
            # One-way ANOVA
            f_stat, p_value = f_oneway(*group_values)
            
            # Eta-squared effect size
            grand_mean = np.mean(np.concatenate(group_values))
            ss_between = sum(len(vals) * (np.mean(vals) - grand_mean)**2 for vals in group_values)
            ss_total = sum((vals - grand_mean)**2 for vals in np.concatenate(group_values))
            eta_squared = ss_between / ss_total if ss_total > 0 else np.nan
            
            result['test'] = 'One-Way ANOVA'
            result['f_statistic'] = float(f_stat)
            result['p_value'] = float(p_value)
            result['effect_size'] = {
                'eta_squared': float(eta_squared),
                'interpretation': StatisticalComparisonTools._interpret_eta_squared(eta_squared)
            }
        
        # Add significance interpretation
        result['significant'] = p_value < 0.05
        result['significance_level'] = StatisticalComparisonTools._get_significance_level(p_value)
        
        return result
    
    @staticmethod
    def compare_groups_nonparametric(groups: Dict[str, List[float]], 
                                    metric_name: str = "Metric") -> Dict[str, Any]:
        """
        Perform non-parametric statistical tests (Mann-Whitney or Kruskal-Wallis) on groups.
        """
        group_names = list(groups.keys())
        group_values = [np.array(groups[name]) for name in group_names]
        
        # Remove NaN values
        group_values = [vals[~np.isnan(vals)] for vals in group_values]
        group_names = [name for name, vals in zip(group_names, group_values) if len(vals) > 0]
        group_values = [vals for vals in group_values if len(vals) > 0]
        
        if len(group_values) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 groups with valid data'
            }
        
        result = {
            'success': True,
            'metric_name': metric_name,
            'group_names': group_names,
            'n_groups': len(group_values)
        }
        
        # Calculate descriptive statistics
        result['descriptive'] = {}
        for name, vals in zip(group_names, group_values):
            result['descriptive'][name] = {
                'n': len(vals),
                'median': float(np.median(vals)),
                'q25': float(np.percentile(vals, 25)),
                'q75': float(np.percentile(vals, 75)),
                'iqr': float(np.percentile(vals, 75) - np.percentile(vals, 25))
            }
        
        # Perform appropriate test
        if len(group_values) == 2:
            # Mann-Whitney U test
            u_stat, p_value = mannwhitneyu(group_values[0], group_values[1], alternative='two-sided')
            
            # Rank-biserial correlation effect size
            n1, n2 = len(group_values[0]), len(group_values[1])
            r = 1 - (2*u_stat) / (n1 * n2)
            
            result['test'] = 'Mann-Whitney U'
            result['u_statistic'] = float(u_stat)
            result['p_value'] = float(p_value)
            result['effect_size'] = {
                'rank_biserial_r': float(r),
                'interpretation': StatisticalComparisonTools._interpret_rank_biserial(r)
            }
            
        else:
            # Kruskal-Wallis H test
            h_stat, p_value = kruskal(*group_values)
            
            # Epsilon-squared effect size
            n_total = sum(len(vals) for vals in group_values)
            epsilon_squared = (h_stat - len(group_values) + 1) / (n_total - len(group_values))
            
            result['test'] = 'Kruskal-Wallis H'
            result['h_statistic'] = float(h_stat)
            result['p_value'] = float(p_value)
            result['effect_size'] = {
                'epsilon_squared': float(epsilon_squared),
                'interpretation': StatisticalComparisonTools._interpret_epsilon_squared(epsilon_squared)
            }
        
        # Add significance interpretation
        result['significant'] = p_value < 0.05
        result['significance_level'] = StatisticalComparisonTools._get_significance_level(p_value)
        
        return result
    
    @staticmethod
    def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
        """
        Apply Bonferroni correction for multiple comparisons.
        
        Returns corrected alpha and which tests remain significant.
        """
        n_tests = len(p_values)
        corrected_alpha = alpha / n_tests if n_tests > 0 else alpha
        
        return {
            'n_tests': n_tests,
            'original_alpha': alpha,
            'corrected_alpha': corrected_alpha,
            'significant_tests': [i for i, p in enumerate(p_values) if p < corrected_alpha],
            'n_significant': sum(1 for p in p_values if p < corrected_alpha)
        }
    
    @staticmethod
    def plot_group_comparison(comparison_result: Dict, groups_data: Dict[str, List[float]]) -> go.Figure:
        """Create visualization for group comparison results."""
        if not comparison_result.get('success', False):
            return None
        
        group_names = comparison_result['group_names']
        metric_name = comparison_result.get('metric_name', 'Metric')
        
        # Create box plot with individual points
        fig = go.Figure()
        
        for name in group_names:
            values = groups_data[name]
            values = np.array(values)
            values = values[~np.isnan(values)]
            
            # Box plot
            fig.add_trace(go.Box(
                y=values,
                name=name,
                boxmean='sd',
                marker_color='steelblue'
            ))
            
            # Individual points (jittered)
            jitter = np.random.normal(0, 0.05, size=len(values))
            x_positions = [name] * len(values)
            
            fig.add_trace(go.Scatter(
                x=x_positions,
                y=values,
                mode='markers',
                marker=dict(size=6, color='rgba(0,0,0,0.3)'),
                showlegend=False,
                hoverinfo='y'
            ))
        
        # Add statistics annotation
        p_value = comparison_result['p_value']
        test_name = comparison_result['test']
        sig_level = comparison_result['significance_level']
        
        annotation_text = f"{test_name}<br>p = {p_value:.4f} {sig_level}"
        
        if 'effect_size' in comparison_result:
            effect = comparison_result['effect_size']
            effect_str = next(iter(effect.keys()))
            effect_val = effect[effect_str]
            if isinstance(effect_val, (int, float)):
                annotation_text += f"<br>Effect: {effect_val:.3f} ({effect.get('interpretation', '')})"
        
        fig.add_annotation(
            text=annotation_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            align="left"
        )
        
        fig.update_layout(
            title=f"Group Comparison: {metric_name}",
            yaxis_title=metric_name,
            xaxis_title="Group",
            height=500,
            showlegend=False
        )
        
        return fig
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_eta_squared(eta2: float) -> str:
        """Interpret eta-squared effect size."""
        if eta2 < 0.01:
            return "Negligible"
        elif eta2 < 0.06:
            return "Small"
        elif eta2 < 0.14:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_rank_biserial(r: float) -> str:
        """Interpret rank-biserial correlation."""
        abs_r = abs(r)
        if abs_r < 0.1:
            return "Negligible"
        elif abs_r < 0.3:
            return "Small"
        elif abs_r < 0.5:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_epsilon_squared(eps2: float) -> str:
        """Interpret epsilon-squared effect size."""
        if eps2 < 0.01:
            return "Negligible"
        elif eps2 < 0.08:
            return "Small"
        elif eps2 < 0.26:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _get_significance_level(p: float) -> str:
        """Get significance level string."""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return "ns"


# Export all analysis functions for integration
__all__ = [
    'AdvancedBiophysicalReportExtension',
    'StatisticalComparisonTools',
    'ADVANCED_METRICS_AVAILABLE',
    'RHEOLOGY_AVAILABLE',
    'MSD_AVAILABLE'
]
