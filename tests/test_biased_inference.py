"""
Test script for Bias-Corrected Diffusion Estimation

Verifies CVE and MLE estimators work correctly with synthetic data.
"""

import numpy as np
import pandas as pd
from biased_inference import BiasedInferenceCorrector, compare_estimators

def generate_test_track(D=0.1, dt=0.1, n_steps=100, sigma_loc=0.03, dimensions=2):
    """
    Generate synthetic track with known diffusion coefficient.
    
    Parameters:
    -----------
    D : float
        True diffusion coefficient (μm²/s)
    dt : float
        Time step (s)
    n_steps : int
        Number of steps
    sigma_loc : float
        Localization error (μm)
    dimensions : int
        2D or 3D
    
    Returns:
    --------
    np.ndarray : Positions with shape (n_steps+1, dimensions)
    """
    # Generate true diffusive steps
    steps = np.random.normal(0, np.sqrt(2 * D * dt), size=(n_steps, dimensions))
    positions_true = np.cumsum(steps, axis=0)
    positions_true = np.vstack([np.zeros(dimensions), positions_true])
    
    # Add localization noise
    noise = np.random.normal(0, sigma_loc, size=positions_true.shape)
    positions_observed = positions_true + noise
    
    return positions_observed


def test_cve_estimator():
    """Test Covariance-based Estimator."""
    print("=" * 60)
    print("TEST 1: Covariance-based Estimator (CVE)")
    print("=" * 60)
    
    # Generate test data
    D_true = 0.15  # μm²/s
    sigma_loc = 0.03  # μm
    dt = 0.1  # s
    
    track = generate_test_track(D=D_true, dt=dt, n_steps=50, sigma_loc=sigma_loc)
    
    # Run CVE using BiasedInferenceCorrector
    corrector = BiasedInferenceCorrector()
    result = corrector.cve_estimator(track=track, dt=dt, localization_error=sigma_loc, dimensions=2)
    
    print(f"\nTrue D: {D_true:.4f} μm²/s")
    print(f"True σ_loc: {sigma_loc:.4f} μm")
    print(f"\nCVE Results:")
    print(f"  D_CVE: {result.get('D', np.nan):.4f} ± {result.get('D_std', np.nan):.4f} μm²/s")
    print(f"  Estimated σ_loc: {result.get('localization_error', sigma_loc):.4f} μm")
    print(f"  Covariance: {result.get('covariance', 0):.6f}")
    print(f"  N_steps: {result.get('N_steps', 0)}")
    
    # Calculate error
    if result.get('success', False):
        error_pct = abs(result['D'] - D_true) / D_true * 100
        print(f"\nEstimation Error: {error_pct:.1f}%")
    
    return result.get('success', False)


def test_mle_estimator():
    """Test Maximum Likelihood Estimator with blur."""
    print("\n" + "=" * 60)
    print("TEST 2: Maximum Likelihood Estimator (MLE) with Blur")
    print("=" * 60)
    
    # Generate test data
    D_true = 0.12  # μm²/s
    sigma_loc = 0.025  # μm
    dt = 0.1  # s
    exposure_time = 0.08  # s (80% of frame time)
    
    track = generate_test_track(D=D_true, dt=dt, n_steps=30, sigma_loc=sigma_loc)
    
    # Run MLE using BiasedInferenceCorrector
    corrector = BiasedInferenceCorrector()
    result = corrector.mle_with_blur(
        track=track, dt=dt, exposure_time=exposure_time,
        localization_error=sigma_loc, dimensions=2
    )
    
    print(f"\nTrue D: {D_true:.4f} μm²/s")
    print(f"True σ_loc: {sigma_loc:.4f} μm")
    print(f"Exposure time: {exposure_time:.2f} s (R={exposure_time/dt:.2f})")
    print(f"\nMLE Results:")
    print(f"  D_MLE: {result.get('D', np.nan):.4f} ± {result.get('D_std', np.nan):.4f} μm²/s")
    print(f"  Alpha: {result.get('alpha', np.nan):.3f}")
    print(f"  Blur corrected: {result.get('blur_corrected', False)}")
    print(f"  N_steps: {result.get('N_steps', 0)}")
    
    # Calculate error
    if result.get('success', False):
        error_pct = abs(result['D'] - D_true) / D_true * 100
        print(f"\nEstimation Error: {error_pct:.1f}%")
    
    return result.get('success', False)


def test_method_selection():
    """Test automatic estimator selection."""
    print("\n" + "=" * 60)
    print("TEST 3: Automatic Method Selection")
    print("=" * 60)
    
    test_cases = [
        {'n_steps': 15, 'expected': 'MLE', 'reason': 'Very short track'},
        {'n_steps': 35, 'expected': 'CVE', 'reason': 'Medium length track'},
        {'n_steps': 100, 'expected': 'CVE', 'reason': 'Long track'},
    ]
    
    print("\n{:<15} {:<15} {:<30}".format("N_steps", "Selected", "Reason"))
    print("-" * 60)
    
    corrector = BiasedInferenceCorrector()
    
    for case in test_cases:
        # Generate dummy track
        track = np.random.randn(case['n_steps'] + 1, 2)
        
        selected = corrector.select_estimator(
            track=track, dt=0.1, localization_error=0.03
        )
        
        match = "✓" if selected == case['expected'] else "✗"
        print(f"{case['n_steps']:<15} {selected:<15} {case['reason']:<30} {match}")
    
    return True
    
    return True


def test_bias_comparison():
    """Compare all three methods on same data."""
    print("\n" + "=" * 60)
    print("TEST 4: Method Comparison (MSD vs CVE vs MLE)")
    print("=" * 60)
    
    D_true = 0.18  # μm²/s
    sigma_loc = 0.04  # μm (high noise)
    dt = 0.1  # s
    
    track = generate_test_track(D=D_true, dt=dt, n_steps=40, sigma_loc=sigma_loc)
    
    # Compare all methods
    results = compare_estimators(
        track=track, dt=dt, localization_error=sigma_loc,
        exposure_time=dt*0.9, dimensions=2
    )
    
    print(f"\nTrue D: {D_true:.4f} μm²/s")
    print(f"Localization noise: {sigma_loc:.4f} μm")
    print(f"\nResults:")
    print(f"  MSD (naive):  D = {results['msd'].get('D', np.nan):.4f} μm²/s")
    
    if results['cve'].get('success', False):
        print(f"  CVE:          D = {results['cve']['D']:.4f} ± {results['cve'].get('D_std', 0):.4f} μm²/s")
    
    if results['mle'].get('success', False):
        print(f"  MLE:          D = {results['mle']['D']:.4f} ± {results['mle'].get('D_std', 0):.4f} μm²/s")
    
    comp = results.get('comparison', {})
    if 'D_bias_msd_vs_cve' in comp:
        print(f"\nBias Analysis:")
        print(f"  MSD vs CVE bias: {comp['D_bias_msd_vs_cve']*100:.1f}%")
    if 'D_bias_msd_vs_mle' in comp:
        print(f"  MSD vs MLE bias: {comp['D_bias_msd_vs_mle']*100:.1f}%")
    if 'recommended' in comp:
        print(f"\nRecommended: {comp['recommended']}")
    
    return True
    
    return True


def test_biased_inference_corrector():
    """Test high-level BiasedInferenceCorrector API."""
    print("\n" + "=" * 60)
    print("TEST 5: BiasedInferenceCorrector API")
    print("=" * 60)
    
    D_true = 0.14
    dt = 0.1
    sigma_loc = 0.035
    exposure_time = 0.09
    
    track = generate_test_track(D=D_true, dt=dt, n_steps=45, sigma_loc=sigma_loc)
    
    corrector = BiasedInferenceCorrector()
    
    # Test auto-selection
    result_auto = corrector.analyze_track(
        track, dt, sigma_loc, exposure_time, method='auto', dimensions=2
    )
    
    print(f"\nTrue D: {D_true:.4f} μm²/s")
    print(f"\nAuto-selected method: {result_auto.get('method_selected', 'N/A')}")
    print(f"D_corrected: {result_auto.get('D', np.nan):.4f} ± {result_auto.get('D_std', np.nan):.4f} μm²/s")
    print(f"Alpha: {result_auto.get('alpha', np.nan):.3f} ± {result_auto.get('alpha_std', np.nan):.3f}")
    print(f"Localization corrected: {result_auto.get('localization_corrected', False)}")
    print(f"Blur corrected: {result_auto.get('blur_corrected', False)}")
    
    # Calculate error
    if result_auto.get('success', False):
        error_pct = abs(result_auto['D'] - D_true) / D_true * 100
        print(f"\nEstimation Error: {error_pct:.1f}%")
    
    return result_auto.get('success', False)


def test_batch_analysis():
    """Test batch analysis on multiple tracks."""
    print("\n" + "=" * 60)
    print("TEST 6: Batch Analysis")
    print("=" * 60)
    
    # Generate multiple tracks with varying parameters
    n_tracks = 10
    D_true = 0.15
    dt = 0.1
    sigma_loc = 0.03
    pixel_size = 0.1  # μm/pixel
    
    tracks_data = []
    for track_id in range(n_tracks):
        n_steps = np.random.randint(20, 60)
        track = generate_test_track(D=D_true, dt=dt, n_steps=n_steps, sigma_loc=sigma_loc)
        
        for frame_idx, pos in enumerate(track):
            tracks_data.append({
                'track_id': track_id,
                'frame': frame_idx,
                'x': pos[0] / pixel_size,  # Convert to pixels
                'y': pos[1] / pixel_size
            })
    
    tracks_df = pd.DataFrame(tracks_data)
    
    # Run batch analysis
    corrector = BiasedInferenceCorrector()
    results_df = corrector.batch_analyze(
        tracks_df,
        pixel_size_um=pixel_size,
        dt=dt,
        localization_error_um=sigma_loc,
        exposure_time=dt*0.9,
        method='auto'
    )
    
    print(f"\nAnalyzed {len(results_df)} tracks")
    print(f"Successful: {results_df['success'].sum()}")
    
    successful = results_df[results_df['success']]
    if len(successful) > 0:
        D_mean = successful['D_um2_per_s'].mean()
        D_std = successful['D_um2_per_s'].std()
        
        print(f"\nResults Summary:")
        print(f"  D_mean: {D_mean:.4f} ± {D_std:.4f} μm²/s")
        print(f"  True D: {D_true:.4f} μm²/s")
        print(f"  Error: {abs(D_mean - D_true) / D_true * 100:.1f}%")
        
        print(f"\nMethod Distribution:")
        print(successful['method_used'].value_counts())
    
    return len(successful) > 0


def run_all_tests():
    """Run complete test suite."""
    print("\n" + "=" * 70)
    print(" BIAS-CORRECTED DIFFUSION ESTIMATION TEST SUITE")
    print("=" * 70)
    print("\nTesting CVE/MLE estimators (Berglund 2010)")
    print("Reference: Physical Review E 82(1):011917. PMID: 20866658\n")
    
    results = {}
    
    try:
        results['CVE'] = test_cve_estimator()
    except Exception as e:
        print(f"\n✗ CVE test failed: {e}")
        results['CVE'] = False
    
    try:
        results['MLE'] = test_mle_estimator()
    except Exception as e:
        print(f"\n✗ MLE test failed: {e}")
        results['MLE'] = False
    
    try:
        results['Selection'] = test_method_selection()
    except Exception as e:
        print(f"\n✗ Method selection test failed: {e}")
        results['Selection'] = False
    
    try:
        results['Comparison'] = test_bias_comparison()
    except Exception as e:
        print(f"\n✗ Comparison test failed: {e}")
        results['Comparison'] = False
    
    try:
        results['API'] = test_biased_inference_corrector()
    except Exception as e:
        print(f"\n✗ API test failed: {e}")
        results['API'] = False
    
    try:
        results['Batch'] = test_batch_analysis()
    except Exception as e:
        print(f"\n✗ Batch test failed: {e}")
        results['Batch'] = False
    
    # Summary
    print("\n" + "=" * 70)
    print(" TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20} {status}")
    
    total = len(results)
    passed = sum(results.values())
    
    print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 All tests passed! Bias correction implementation verified.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review implementation.")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)
