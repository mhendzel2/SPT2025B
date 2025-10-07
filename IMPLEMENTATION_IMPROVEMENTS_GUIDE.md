# Quick Implementation Guide for Code Review Recommendations

**Date**: October 6, 2025  
**Purpose**: Prioritized improvements for 2025 feature modules  
**Effort**: 2-4 days for medium priority items

---

## Medium Priority Fixes (Recommended Before v1.0 Release)

### 1. biased_inference.py: Fisher Information Uncertainties

**Current**: Simplified uncertainty estimation
```python
# Line ~295
D_std = D_mle * 0.1 / np.sqrt(N)  # Rough estimate
```

**Improved**:
```python
def _compute_fisher_information_matrix(self, params, displacements, dt, 
                                       exposure_time, localization_error):
    """
    Compute Fisher information matrix for Cramér-Rao bound.
    
    Reference: Berglund (2010) Physical Review E, Supplemental Material
    
    Returns
    -------
    fim : np.ndarray
        Fisher information matrix, shape (n_params, n_params)
    """
    D, alpha = params if len(params) == 2 else (params[0], 1.0)
    N = len(displacements)
    d = displacements.shape[1]
    
    # Motion blur factor
    R = (exposure_time / dt)**2 / 3 if exposure_time > 0 else 0
    
    # Variance
    var = 2 * D * (dt**alpha) * (1 - R) + 2 * localization_error**2
    
    # Partial derivatives of log-likelihood
    sum_r2 = np.sum(displacements**2)
    
    # ∂²L/∂D² (second derivative wrt D)
    d2L_dD2 = -N * d / var**2 + sum_r2 / var**3
    
    if len(params) == 2:
        # ∂²L/∂α²
        d2L_dalpha2 = -N * d * (np.log(dt))**2 + sum_r2 * (np.log(dt))**2 / var**2
        
        # ∂²L/∂D∂α (cross term)
        d2L_dDdalpha = sum_r2 * np.log(dt) / var**2
        
        fim = -np.array([
            [d2L_dD2, d2L_dDdalpha],
            [d2L_dDdalpha, d2L_dalpha2]
        ])
    else:
        fim = -np.array([[d2L_dD2]])
    
    return fim


def mle_with_blur(self, track, dt, exposure_time, localization_error, 
                  dimensions=2, alpha_fixed=None):
    """... existing docstring ..."""
    
    # ... existing optimization code ...
    
    # After successful optimization:
    try:
        fim = self._compute_fisher_information_matrix(
            result.x, displacements, dt, exposure_time, localization_error
        )
        
        # Cramér-Rao bound: Var(θ) ≥ [FIM]^(-1)
        cov_matrix = np.linalg.inv(fim)
        
        if alpha_fixed is None:
            D_std = np.sqrt(cov_matrix[0, 0])
            alpha_std = np.sqrt(cov_matrix[1, 1])
        else:
            D_std = np.sqrt(cov_matrix[0, 0])
            alpha_std = 0.0
            
    except np.linalg.LinAlgError:
        # Fallback to simple estimate if FIM is singular
        D_std = D_mle * 0.1 / np.sqrt(N)
        alpha_std = 0.05 / np.sqrt(N) if alpha_fixed is None else 0.0
    
    return {
        'success': True,
        'D': D_mle,
        'D_std': D_std,
        'alpha': alpha_mle,
        'alpha_std': alpha_std,
        'fisher_information_matrix': fim,  # For advanced users
        # ... rest of return dict ...
    }
```

---

### 2. acquisition_advisor.py: Sub-Resolution Diffusion Warning

**Add to `recommend_framerate` method**:

```python
def recommend_framerate(self, D_expected: float, 
                       localization_precision: float,
                       motion_type: str = 'normal_diffusion') -> Dict:
    """... existing docstring ..."""
    
    # ... existing code ...
    
    # NEW: Check for sub-resolution diffusion
    displacement_per_frame = np.sqrt(2 * D_expected * optimal_dt)
    
    if displacement_per_frame < localization_precision:
        warnings.append({
            'severity': 'high',
            'message': (
                f'Sub-resolution diffusion: Expected displacement '
                f'({displacement_per_frame*1000:.1f} nm) < localization precision '
                f'({localization_precision*1000:.1f} nm). '
                f'Particle tracks will be dominated by noise.'
            ),
            'recommendation': (
                'Consider: (1) Longer frame intervals (lower framerate), '
                '(2) Higher magnification, or (3) Ensemble averaging methods (DDM/RICS)'
            )
        })
    
    return {
        'recommended_dt': optimal_dt,
        'recommended_framerate_hz': framerate_hz,
        'recommended_exposure': exposure_time,
        'warnings': warnings,
        'displacement_per_frame_um': displacement_per_frame,  # NEW
        # ... rest of return dict ...
    }
```

---

### 3. equilibrium_validator.py: Document AFM Exclusion

**Update docstring in `generate_validity_report`**:

```python
def generate_validity_report(self, tracks_df=None, rheology_results=None,
                            vacf_df=None, one_point_G=None, 
                            two_point_G=None, afm_data=None) -> Dict:
    """
    Run all available checks and return composite badge.
    
    NOTE: AFM/OT concordance check is optional and excluded by default
    per user requirements (2025-10-06 implementation scope). Can be
    added later via `check_afm_concordance()` method when AFM import
    module is implemented.
    
    Parameters
    ----------
    ... existing parameters ...
    afm_data : pd.DataFrame, optional
        AFM frequency sweep data (not yet supported, reserved for future)
    
    Returns
    -------
    Dict
        {
            'overall_valid': bool,
            'tests_passed': int,
            'tests_total': int,  # Currently 2 (VACF + 1P-2P), will be 3 when AFM added
            'badge': emoji badge string,
            'test_results': {...},
            'recommendations': [...]
        }
    """
    
    # ... existing implementation ...
    
    # TODO (future): Add AFM concordance when AFM import module is ready
    # if afm_data is not None:
    #     afm_result = self.check_afm_concordance(one_point_G, afm_data)
    #     test_results['afm_concordance'] = afm_result
    #     tests_total += 1
    #     if afm_result.get('success') and afm_result.get('valid'):
    #         tests_passed += 1
```

---

### 4. ddm_analyzer.py: Background Subtraction

**Add to `compute_image_structure_function`**:

```python
def compute_image_structure_function(self, image_stack, lag_frames=None,
                                     q_range_um_inv=None,
                                     subtract_background: bool = True) -> Dict:
    """
    Compute image structure function D(q, τ).
    
    Parameters
    ----------
    ... existing parameters ...
    subtract_background : bool, default=True
        Remove static background features (recommended for microscopy data)
    
    ... existing docstring ...
    """
    
    # Validation
    if image_stack.ndim != 3:
        return {'success': False, 'error': '...'}
    
    # NEW: Background subtraction
    if subtract_background:
        # Temporal median filter removes static features
        background = np.median(image_stack, axis=0)
        image_stack_corrected = image_stack - background[None, :, :]
        
        # Normalize to prevent negative intensities
        image_stack_corrected -= image_stack_corrected.min()
    else:
        image_stack_corrected = image_stack
    
    n_frames, height, width = image_stack_corrected.shape
    
    # ... rest of implementation using image_stack_corrected ...
    
    return {
        'success': True,
        'q_values_um_inv': q_values_um_inv,
        'lag_times_s': lag_times_s,
        'D_q_tau': D_q_tau,
        'background_subtracted': subtract_background,  # NEW
        'background_intensity': background.mean() if subtract_background else None,  # NEW
        # ... rest of return dict ...
    }
```

---

### 5. microsecond_sampling.py: Single-Pass MSD Calculation

**Replace two-pass algorithm in `calculate_msd_irregular`**:

```python
def calculate_msd_irregular(self, track_df, pixel_size=1.0, 
                           max_lag_s=None, n_lag_bins=30) -> Dict:
    """... existing docstring ..."""
    
    # ... existing setup code ...
    
    # Initialize with Welford's online algorithm for variance
    msd_values = np.zeros(n_lag_bins)
    msd_m2 = np.zeros(n_lag_bins)  # Sum of squared deviations
    n_observations = np.zeros(n_lag_bins, dtype=int)
    
    # SINGLE PASS: Compute mean and variance simultaneously
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            lag = times[j] - times[i]
            
            # Find appropriate bin
            bin_idx = np.searchsorted(lag_bins[:-1], lag, side='right') - 1
            
            if 0 <= bin_idx < n_lag_bins:
                displacement = positions[j] - positions[i]
                squared_disp = np.sum(displacement**2)
                
                # Welford's algorithm for online mean and variance
                n = n_observations[bin_idx]
                delta = squared_disp - msd_values[bin_idx]
                msd_values[bin_idx] += delta / (n + 1)
                delta2 = squared_disp - msd_values[bin_idx]
                msd_m2[bin_idx] += delta * delta2
                
                n_observations[bin_idx] += 1
    
    # Compute standard deviation
    valid = n_observations > 1
    msd_std_values = np.zeros(n_lag_bins)
    msd_std_values[valid] = np.sqrt(msd_m2[valid] / (n_observations[valid] - 1))
    
    return {
        'success': True,
        'lag_times_s': lag_centers,
        'msd_um2': msd_values,
        'msd_std': msd_std_values,
        'n_observations': n_observations,
        'algorithm': 'welford_single_pass',  # NEW
        # ... rest of return dict ...
    }
```

---

## Testing Recommendations

### Synthetic Test Suite Structure

Create `tests/test_2025_features.py`:

```python
import pytest
import numpy as np
import pandas as pd
from biased_inference import BiasedInferenceCorrector
from acquisition_advisor import AcquisitionAdvisor
from equilibrium_validator import EquilibriumValidator
from ddm_analyzer import DDMAnalyzer
from ihmm_blur_analysis import iHMMBlurAnalyzer
from microsecond_sampling import IrregularSamplingHandler


class TestBiasedInference:
    """Test CVE/MLE estimators."""
    
    def generate_brownian_track(self, N=100, D=0.1, dt=0.1, 
                                sigma_loc=0.03, dimensions=2):
        """Generate synthetic Brownian motion."""
        positions = np.zeros((N, dimensions))
        for i in range(1, N):
            # True displacement
            dr = np.random.randn(dimensions) * np.sqrt(2 * D * dt)
            # Add localization noise
            noise = np.random.randn(dimensions) * sigma_loc
            positions[i] = positions[i-1] + dr + noise
        return positions
    
    def test_cve_on_brownian_motion(self):
        """CVE should recover known D."""
        D_true = 0.1  # μm²/s
        dt = 0.1
        sigma_loc = 0.03
        
        corrector = BiasedInferenceCorrector()
        track = self.generate_brownian_track(N=50, D=D_true, dt=dt, 
                                             sigma_loc=sigma_loc)
        
        result = corrector.cve_estimator(track, dt, sigma_loc)
        
        assert result['success']
        # Allow 20% error for short tracks
        assert abs(result['D'] - D_true) / D_true < 0.2
        assert result['method'] == 'CVE'
        assert result['localization_corrected']
    
    def test_mle_with_blur_correction(self):
        """MLE should handle motion blur."""
        D_true = 0.1
        dt = 0.1
        exposure = 0.08  # 80% of frame time
        sigma_loc = 0.03
        
        corrector = BiasedInferenceCorrector()
        track = self.generate_brownian_track(N=50, D=D_true, dt=dt, 
                                             sigma_loc=sigma_loc)
        
        result = corrector.mle_with_blur(track, dt, exposure, sigma_loc)
        
        assert result['success']
        assert abs(result['D'] - D_true) / D_true < 0.3  # Allow 30% for MLE
        assert result['blur_corrected']
        assert result['exposure_time'] == exposure
    
    def test_auto_selection(self):
        """Test method auto-selection logic."""
        corrector = BiasedInferenceCorrector()
        
        # Very short track → MLE
        short_track = np.random.randn(15, 2)
        method = corrector.select_estimator(short_track, dt=0.1, 
                                           localization_error=0.03)
        assert method == 'MLE'
        
        # Long track, good SNR → MSD
        long_track = np.random.randn(100, 2) * 0.5  # High displacement
        method = corrector.select_estimator(long_track, dt=0.1, 
                                           localization_error=0.03,
                                           snr=15)
        assert method == 'MSD'
        
        # Medium track, poor SNR → CVE
        medium_track = np.random.randn(40, 2) * 0.1
        method = corrector.select_estimator(medium_track, dt=0.1,
                                           localization_error=0.03,
                                           snr=3)
        assert method == 'CVE'


class TestAcquisitionAdvisor:
    """Test frame rate optimization."""
    
    def test_optimal_dt_formula(self):
        """Verify Weimann 2024 formula."""
        advisor = AcquisitionAdvisor()
        
        D = 0.1  # μm²/s
        sigma_loc = 0.03  # μm
        
        result = advisor.recommend_framerate(D, sigma_loc)
        
        # For normal diffusion: dt_opt = 2 * σ² / D
        expected_dt = 2 * sigma_loc**2 / D
        
        assert result['success']
        assert abs(result['recommended_dt'] - expected_dt) < 0.01
        assert result['recommended_exposure'] < result['recommended_dt']
    
    def test_sub_resolution_warning(self):
        """Should warn when displacement < precision."""
        advisor = AcquisitionAdvisor()
        
        D_slow = 0.001  # Very slow diffusion
        sigma_loc = 0.05  # Large localization error
        
        result = advisor.recommend_framerate(D_slow, sigma_loc)
        
        # Check for sub-resolution warning
        warnings = [w['severity'] for w in result['warnings'] 
                   if 'sub-resolution' in w['message'].lower()]
        assert len(warnings) > 0


class TestEquilibriumValidator:
    """Test GSER validity checks."""
    
    def test_vacf_symmetry_equilibrium(self):
        """Symmetric VACF should pass."""
        validator = EquilibriumValidator()
        
        # Create symmetric VACF
        lags = np.linspace(-10, 10, 21)
        vacf_sym = np.exp(-np.abs(lags))  # Symmetric exponential
        
        vacf_df = pd.DataFrame({'lag_time': lags, 'vacf': vacf_sym})
        
        result = validator.check_vacf_symmetry(vacf_df)
        
        assert result['success']
        assert result['valid']
        assert result['symmetry_score'] > 0.95
    
    def test_1p_2p_agreement(self):
        """Should detect agreement/disagreement."""
        validator = EquilibriumValidator()
        
        freq = np.logspace(-1, 2, 20)
        g_prime_1p = freq**0.5 * 10  # Power law
        g_prime_2p = freq**0.5 * 10  # Identical
        
        one_point = {'frequencies_hz': freq, 'g_prime_pa': g_prime_1p}
        two_point = {'frequencies_hz': freq, 'g_prime_pa': g_prime_2p}
        
        result = validator.check_1p_2p_agreement(one_point, two_point)
        
        assert result['success']
        assert result['valid']
        assert abs(result['agreement_ratio'] - 1.0) < 0.1


# Add more test classes for DDM, iHMM, microsecond sampling...


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Integration Priority

1. **Fix medium-priority issues** (2 days)
   - Fisher information uncertainties
   - Sub-resolution warnings
   - Background subtraction
   - Single-pass MSD

2. **Create test suite** (2 days)
   - Synthetic data generators
   - 10 validation scenarios
   - CI/CD integration

3. **Integrate into app** (3 days)
   - Register in report generator
   - Add UI controls
   - Create visualizations

4. **Documentation** (2 days)
   - User guides
   - API reference
   - Usage examples

**Total**: ~9 days to production-ready v1.0

---

## Validation Checklist

Before marking as complete:

- [ ] All 5 medium-priority fixes implemented
- [ ] Test suite covers 10 synthetic scenarios
- [ ] All tests pass (pytest)
- [ ] Integrated into enhanced_report_generator.py
- [ ] UI controls added to app.py
- [ ] Documentation complete
- [ ] Performance benchmarked (no regressions)
- [ ] User acceptance testing on real data

---

**Document Version**: 1.0  
**Last Updated**: October 6, 2025
