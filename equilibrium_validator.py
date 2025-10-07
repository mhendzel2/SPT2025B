"""
Equilibrium Validity Detection Module

Multi-test system to detect when GSER assumptions are violated:
1. VACF symmetry check (thermal equilibrium implies VACF is symmetric)
2. 1P-2P microrheology agreement (homogeneous + equilibrium)
3. AFM/OT concordance (NOT IMPLEMENTED - excluded per user request)

NOTE: AFM/optical tweezer cross-validation module was intentionally excluded
from this implementation per user requirements. Users who wish to cross-validate
SPT-derived rheology with active rheometry (AFM, optical tweezers) should use
external tools or manual comparison.

CRITICAL: GSER assumes thermal equilibrium and passive diffusion.
Active stresses (motors, flows) violate these assumptions and lead to
misinterpretation of G*(omega).

Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


class EquilibriumValidator:
    """
    Multi-test system for equilibrium validity.
    
    Prevents systematic misinterpretation of non-equilibrium systems.
    """
    
    def __init__(self):
        """Initialize validator with thresholds."""
        self.thresholds = {
            'vacf_symmetry_min': 0.8,  # >0.8 = symmetric
            '1p_2p_agreement_range': (0.8, 1.2),  # G_2P/G_1P should be 0.8-1.2
            'afm_agreement_range': (0.8, 1.2)  # SPT/AFM should be 0.8-1.2
        }
    
    
    def check_vacf_symmetry(self, vacf_df: pd.DataFrame) -> Dict:
        """
        Thermal equilibrium implies VACF(-tau) = VACF(+tau).
        
        For equilibrium systems, the velocity autocorrelation function
        should be symmetric in time. Asymmetry indicates:
        - Active driving forces
        - Non-thermal noise
        - Memory effects from non-equilibrium processes
        
        Parameters
        ----------
        vacf_df : pd.DataFrame
            VACF data with columns: lag_time, vacf
            Should include both positive and negative lags
        
        Returns
        -------
        Dict
            {
                'symmetry_score': float (0-1, 1=perfect),
                'valid': bool (symmetry > 0.8),
                'interpretation': str,
                'method': 'VACF_symmetry'
            }
        """
        if 'lag_time' not in vacf_df.columns or 'vacf' not in vacf_df.columns:
            return {
                'success': False,
                'error': 'VACF dataframe must have lag_time and vacf columns',
                'method': 'VACF_symmetry'
            }
        
        # Find positive and negative lags
        positive_lags = vacf_df[vacf_df['lag_time'] > 0].copy()
        negative_lags = vacf_df[vacf_df['lag_time'] < 0].copy()
        
        if len(positive_lags) == 0 or len(negative_lags) == 0:
            return {
                'success': False,
                'error': 'VACF must include both positive and negative lags',
                'method': 'VACF_symmetry',
                'recommendation': 'Compute VACF with symmetric lag range'
            }
        
        # Make negative lags positive for comparison
        negative_lags['lag_time'] = -negative_lags['lag_time']
        
        # Interpolate to common lag times
        common_lags = np.linspace(
            max(positive_lags['lag_time'].min(), negative_lags['lag_time'].min()),
            min(positive_lags['lag_time'].max(), negative_lags['lag_time'].max()),
            50
        )
        
        vacf_positive = np.interp(common_lags, 
                                 positive_lags['lag_time'].values,
                                 positive_lags['vacf'].values)
        vacf_negative = np.interp(common_lags,
                                 negative_lags['lag_time'].values,
                                 negative_lags['vacf'].values)
        
        # Calculate symmetry score
        # Score = 1 - mean(|VACF(+) - VACF(-)|) / mean(|VACF|)
        diff = np.abs(vacf_positive - vacf_negative)
        mean_abs_vacf = np.mean(np.abs(vacf_positive) + np.abs(vacf_negative)) / 2
        
        if mean_abs_vacf > 0:
            symmetry_score = 1.0 - np.mean(diff) / mean_abs_vacf
            symmetry_score = np.clip(symmetry_score, 0.0, 1.0)
        else:
            symmetry_score = 0.0
        
        # Interpretation
        threshold = self.thresholds['vacf_symmetry_min']
        valid = symmetry_score >= threshold
        
        if symmetry_score > 0.9:
            interpretation = 'Excellent symmetry - thermal equilibrium likely'
        elif symmetry_score > threshold:
            interpretation = 'Good symmetry - equilibrium assumption reasonable'
        elif symmetry_score > 0.6:
            interpretation = 'Moderate asymmetry - caution advised, possible active driving'
        else:
            interpretation = 'Strong asymmetry - active drive or non-equilibrium detected'
        
        return {
            'success': True,
            'symmetry_score': symmetry_score,
            'valid': valid,
            'interpretation': interpretation,
            'method': 'VACF_symmetry',
            'threshold': threshold,
            'n_lags_compared': len(common_lags)
        }
    
    
    def check_1p_2p_agreement(self, one_point_G: Dict, 
                             two_point_G: Dict,
                             frequency_overlap: Optional[np.ndarray] = None) -> Dict:
        """
        1P and 2P microrheology should agree if homogeneous + equilibrium.
        
        One-point (1P) measures local rheology around single particle.
        Two-point (2P) measures bulk-like rheology via pair correlations.
        
        Large discrepancy indicates:
        - Local adhesion (1P < 2P)
        - Active stress gradients (1P != 2P)
        - Spatial heterogeneity
        
        Parameters
        ----------
        one_point_G : Dict
            1P results with keys: 'frequencies_hz', 'g_prime_pa', 'g_double_prime_pa'
        two_point_G : Dict
            2P results with same keys
        frequency_overlap : np.ndarray, optional
            Specific frequencies to compare. If None, auto-detect overlap.
        
        Returns
        -------
        Dict
            {
                'agreement_ratio': float (G_2P / G_1P),
                'valid': bool (0.8 < ratio < 1.2),
                'interpretation': str,
                'method': '1P_2P_agreement'
            }
        """
        # Check data structure
        required_keys = ['frequencies_hz', 'g_prime_pa']
        for key in required_keys:
            if key not in one_point_G or key not in two_point_G:
                return {
                    'success': False,
                    'error': f'Missing required key: {key}',
                    'method': '1P_2P_agreement'
                }
        
        # Find frequency overlap
        freq_1p = np.array(one_point_G['frequencies_hz'])
        freq_2p = np.array(two_point_G['frequencies_hz'])
        g_prime_1p = np.array(one_point_G['g_prime_pa'])
        g_prime_2p = np.array(two_point_G['g_prime_pa'])
        
        # Find common frequency range
        freq_min = max(freq_1p.min(), freq_2p.min())
        freq_max = min(freq_1p.max(), freq_2p.max())
        
        if freq_min >= freq_max:
            return {
                'success': False,
                'error': 'No frequency overlap between 1P and 2P data',
                'method': '1P_2P_agreement',
                'recommendation': 'Ensure 1P and 2P analyses use same frequency range'
            }
        
        # Interpolate to common frequencies
        if frequency_overlap is None:
            common_freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), 10)
        else:
            common_freqs = frequency_overlap[
                (frequency_overlap >= freq_min) & (frequency_overlap <= freq_max)
            ]
        
        if len(common_freqs) == 0:
            return {
                'success': False,
                'error': 'No valid frequencies for comparison',
                'method': '1P_2P_agreement'
            }
        
        # Interpolate G'
        g1p_interp = np.interp(common_freqs, freq_1p, g_prime_1p)
        g2p_interp = np.interp(common_freqs, freq_2p, g_prime_2p)
        
        # Calculate agreement ratio
        # Use geometric mean to handle both directions
        with np.errstate(divide='ignore', invalid='ignore'):
            ratios = g2p_interp / g1p_interp
        
        # Remove invalid ratios
        valid_mask = np.isfinite(ratios) & (ratios > 0)
        if np.sum(valid_mask) == 0:
            return {
                'success': False,
                'error': 'No valid ratios computed',
                'method': '1P_2P_agreement'
            }
        
        ratios = ratios[valid_mask]
        agreement_ratio = np.median(ratios)  # Use median for robustness
        ratio_std = np.std(ratios)
        
        # Check validity
        min_ratio, max_ratio = self.thresholds['1p_2p_agreement_range']
        valid = (agreement_ratio >= min_ratio) and (agreement_ratio <= max_ratio)
        
        # Interpretation
        if 0.9 <= agreement_ratio <= 1.1:
            interpretation = 'Excellent agreement - homogeneous equilibrium system'
        elif valid:
            interpretation = 'Good agreement - equilibrium assumption reasonable'
        elif agreement_ratio < min_ratio:
            interpretation = (
                'Weak agreement (2P < 1P) - possible local adhesion or confinement '
                'affecting 1P measurements'
            )
        else:  # agreement_ratio > max_ratio
            interpretation = (
                'Weak agreement (2P > 1P) - possible active stress or '
                'long-range elastic network not captured by 1P'
            )
        
        return {
            'success': True,
            'agreement_ratio': agreement_ratio,
            'ratio_std': ratio_std,
            'ratio_range': (np.min(ratios), np.max(ratios)),
            'valid': valid,
            'interpretation': interpretation,
            'method': '1P_2P_agreement',
            'threshold_range': (min_ratio, max_ratio),
            'n_frequencies_compared': len(ratios),
            'frequency_range_hz': (common_freqs.min(), common_freqs.max())
        }
    
    
    def generate_equilibrium_badge(self, validation_results: Dict) -> Dict:
        """
        Generate composite equilibrium validity badge.
        
        Parameters
        ----------
        validation_results : Dict
            Results from generate_validity_report()
        
        Returns
        -------
        Dict
            {
                'badge_html': HTML string for display,
                'badge_emoji': emoji version,
                'badge_text': plain text,
                'color': hex color code,
                'level': 'valid', 'caution', or 'invalid'
            }
        """
        tests_passed = validation_results.get('tests_passed', 0)
        tests_total = validation_results.get('tests_total', 3)
        overall_valid = validation_results.get('overall_valid', False)
        
        # Determine badge level
        if tests_passed == tests_total and overall_valid:
            level = 'valid'
            emoji = '🟢'
            text = 'Equilibrium Valid'
            color = '#28a745'  # Green
        elif tests_passed >= tests_total - 1:
            level = 'caution'
            emoji = '🟡'
            text = 'Caution: Mild Deviation'
            color = '#ffc107'  # Yellow
        else:
            level = 'invalid'
            emoji = '🔴'
            text = 'Non-Equilibrium Detected'
            color = '#dc3545'  # Red
        
        # HTML badge
        badge_html = f"""
        <div style="display: inline-block; padding: 8px 16px; 
                    background-color: {color}; color: white; 
                    border-radius: 8px; font-weight: bold; 
                    font-size: 14px; margin: 4px;">
            {emoji} {text}
            <br>
            <span style="font-size: 12px; font-weight: normal;">
                Tests passed: {tests_passed}/{tests_total}
            </span>
        </div>
        """
        
        return {
            'badge_html': badge_html,
            'badge_emoji': f'{emoji} {text}',
            'badge_text': f'{text} ({tests_passed}/{tests_total} tests)',
            'color': color,
            'level': level,
            'tests_passed': tests_passed,
            'tests_total': tests_total
        }
    
    
    def generate_validity_report(self, tracks_df: Optional[pd.DataFrame] = None,
                                 rheology_results: Optional[Dict] = None,
                                 vacf_df: Optional[pd.DataFrame] = None,
                                 one_point_G: Optional[Dict] = None,
                                 two_point_G: Optional[Dict] = None) -> Dict:
        """
        Run all available checks and return composite badge.
        
        Parameters
        ----------
        tracks_df : pd.DataFrame, optional
            Tracking data for VACF computation
        rheology_results : Dict, optional
            Rheology results with 1P data
        vacf_df : pd.DataFrame, optional
            Pre-computed VACF data
        one_point_G : Dict, optional
            One-point microrheology results
        two_point_G : Dict, optional
            Two-point microrheology results
        
        Returns
        -------
        Dict
            {
                'overall_valid': bool,
                'tests_passed': int,
                'tests_total': int,
                'badge': emoji badge string,
                'test_results': {
                    'vacf_symmetry': {...},
                    '1p_2p_agreement': {...}
                },
                'recommendations': list of strings
            }
        """
        test_results = {}
        tests_passed = 0
        tests_total = 0
        recommendations = []
        
        # Test 1: VACF symmetry
        if vacf_df is not None:
            vacf_result = self.check_vacf_symmetry(vacf_df)
            test_results['vacf_symmetry'] = vacf_result
            tests_total += 1
            if vacf_result.get('success', False) and vacf_result.get('valid', False):
                tests_passed += 1
            elif vacf_result.get('success', False) and not vacf_result.get('valid', False):
                recommendations.append(
                    'VACF asymmetry detected - check for active forces (motors, flow) '
                    'or consider reporting moduli as "apparent"'
                )
        
        # Test 2: 1P-2P agreement
        if one_point_G is not None and two_point_G is not None:
            agreement_result = self.check_1p_2p_agreement(one_point_G, two_point_G)
            test_results['1p_2p_agreement'] = agreement_result
            tests_total += 1
            if agreement_result.get('success', False) and agreement_result.get('valid', False):
                tests_passed += 1
            elif agreement_result.get('success', False):
                ratio = agreement_result.get('agreement_ratio', 1.0)
                if ratio < 0.8:
                    recommendations.append(
                        '1P < 2P: Possible local confinement or adhesion. '
                        'Single particles may not represent bulk properties.'
                    )
                else:
                    recommendations.append(
                        '1P > 2P: Possible active stress or elastic network. '
                        'Bulk properties may differ from single-particle measurements.'
                    )
        
        # Overall assessment
        overall_valid = (tests_passed == tests_total) if tests_total > 0 else False
        
        # General recommendations
        if not overall_valid and tests_total > 0:
            recommendations.append(
                'Consider cross-validating with AFM or optical tweezer measurements'
            )
            recommendations.append(
                'Label rheological moduli as "apparent" rather than "equilibrium" in publications'
            )
        
        if tests_total == 0:
            recommendations.append(
                'No equilibrium tests could be performed. '
                'Provide VACF data and/or two-point rheology for validation.'
            )
        
        # Generate badge
        validation_summary = {
            'overall_valid': overall_valid,
            'tests_passed': tests_passed,
            'tests_total': tests_total
        }
        badge_info = self.generate_equilibrium_badge(validation_summary)
        
        return {
            'success': True,
            'overall_valid': overall_valid,
            'tests_passed': tests_passed,
            'tests_total': tests_total,
            'badge': badge_info['badge_emoji'],
            'badge_html': badge_info['badge_html'],
            'badge_level': badge_info['level'],
            'test_results': test_results,
            'recommendations': recommendations if recommendations else ['No issues detected - equilibrium valid']
        }


def quick_equilibrium_check(vacf_df: pd.DataFrame = None,
                            one_point_g_prime: np.ndarray = None,
                            two_point_g_prime: np.ndarray = None) -> str:
    """
    Quick equilibrium check with simple output.
    
    Parameters
    ----------
    vacf_df : pd.DataFrame, optional
        VACF data
    one_point_g_prime : np.ndarray, optional
        1P G' values
    two_point_g_prime : np.ndarray, optional
        2P G' values
    
    Returns
    -------
    str
        Quick assessment message
    """
    validator = EquilibriumValidator()
    
    # Prepare data
    one_point_G = None
    two_point_G = None
    
    if one_point_g_prime is not None and two_point_g_prime is not None:
        freq = np.logspace(-1, 2, len(one_point_g_prime))
        one_point_G = {'frequencies_hz': freq, 'g_prime_pa': one_point_g_prime}
        two_point_G = {'frequencies_hz': freq, 'g_prime_pa': two_point_g_prime}
    
    # Run validation
    report = validator.generate_validity_report(
        vacf_df=vacf_df,
        one_point_G=one_point_G,
        two_point_G=two_point_G
    )
    
    badge = report['badge']
    tests = f"{report['tests_passed']}/{report['tests_total']}"
    recs = '\n  - '.join(report['recommendations'])
    
    return f"{badge}\nTests passed: {tests}\nRecommendations:\n  - {recs}"
