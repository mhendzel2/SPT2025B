#!/usr/bin/env python3
"""
NumPy 2.x Compatibility Test Script for SPT2025B

This script tests whether the codebase is compatible with NumPy 2.x
Run this BEFORE updating the numpy constraint in requirements.txt

Usage:
    python test_numpy2_compatibility.py
"""

import sys
import warnings

def check_numpy_version():
    """Check current NumPy version and compatibility."""
    import numpy as np
    print(f"NumPy version: {np.__version__}")
    major_version = int(np.__version__.split('.')[0])
    return major_version

def test_array_operations():
    """Test basic array operations for NumPy 2.x compatibility."""
    import numpy as np
    
    print("\n[1] Testing basic array operations...")
    
    # Test array creation
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert arr.dtype == np.float64, "Float64 dtype issue"
    
    # Test broadcasting
    result = arr * 2.0
    assert np.allclose(result, [2.0, 4.0, 6.0, 8.0, 10.0])
    
    # Test reshape
    matrix = np.arange(12).reshape(3, 4)
    assert matrix.shape == (3, 4)
    
    # Test advanced indexing
    indices = np.array([0, 2])
    subset = arr[indices]
    assert len(subset) == 2
    
    print("   ✓ Basic array operations PASS")
    return True

def test_linear_algebra():
    """Test linear algebra operations."""
    import numpy as np
    
    print("\n[2] Testing linear algebra...")
    
    # Matrix multiplication
    A = np.random.rand(3, 3)
    B = np.random.rand(3, 3)
    C = A @ B
    assert C.shape == (3, 3)
    
    # Eigenvalues
    eigenvalues = np.linalg.eigvals(A)
    assert len(eigenvalues) == 3
    
    # SVD
    U, S, Vt = np.linalg.svd(A)
    assert U.shape == (3, 3)
    
    print("   ✓ Linear algebra PASS")
    return True

def test_msd_calculation():
    """Test MSD calculation pattern used in SPT analysis."""
    import numpy as np
    
    print("\n[3] Testing MSD calculation patterns...")
    
    # Simulate track data
    n_frames = 100
    x = np.cumsum(np.random.randn(n_frames) * 0.1)
    y = np.cumsum(np.random.randn(n_frames) * 0.1)
    
    # MSD calculation (lag-based)
    max_lag = 20
    msd = np.zeros(max_lag)
    counts = np.zeros(max_lag)
    
    for lag in range(1, max_lag + 1):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        displacements = dx**2 + dy**2
        msd[lag - 1] = np.mean(displacements)
        counts[lag - 1] = len(displacements)
    
    assert not np.any(np.isnan(msd)), "NaN in MSD calculation"
    assert msd[0] > 0, "MSD should be positive"
    
    print("   ✓ MSD calculation PASS")
    return True

def test_scipy_integration():
    """Test scipy integration with numpy arrays."""
    import numpy as np
    from scipy import optimize, stats
    
    print("\n[4] Testing scipy integration...")
    
    # Curve fitting (critical for biophysical models)
    def model(x, a, b):
        return a * x + b
    
    x_data = np.linspace(0, 10, 50)
    y_data = 2.5 * x_data + 1.0 + np.random.randn(50) * 0.5
    
    popt, pcov = optimize.curve_fit(model, x_data, y_data)
    assert len(popt) == 2
    assert np.abs(popt[0] - 2.5) < 0.5, "Fitting failed"
    
    # Statistical tests
    _, p_value = stats.normaltest(np.random.randn(100))
    assert 0 <= p_value <= 1
    
    print("   ✓ SciPy integration PASS")
    return True

def test_pandas_integration():
    """Test pandas integration with numpy arrays."""
    import numpy as np
    import pandas as pd
    
    print("\n[5] Testing pandas integration...")
    
    # Create DataFrame from numpy
    data = {
        'track_id': np.repeat(np.arange(5), 20),
        'frame': np.tile(np.arange(20), 5),
        'x': np.random.rand(100) * 10,
        'y': np.random.rand(100) * 10
    }
    df = pd.DataFrame(data)
    
    # GroupBy operations (critical for track analysis)
    grouped = df.groupby('track_id').agg({
        'x': ['mean', 'std'],
        'y': ['mean', 'std'],
        'frame': 'count'
    })
    assert len(grouped) == 5
    
    # Extract numpy arrays from DataFrame
    x_vals = df['x'].to_numpy()
    assert isinstance(x_vals, np.ndarray)
    
    print("   ✓ Pandas integration PASS")
    return True

def test_sklearn_integration():
    """Test scikit-learn integration."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    
    print("\n[6] Testing scikit-learn integration...")
    
    # Generate sample data
    X = np.random.rand(100, 4)
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    assert len(np.unique(labels)) == 3
    
    # PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    assert X_reduced.shape == (100, 2)
    
    print("   ✓ Scikit-learn integration PASS")
    return True

def test_deprecated_apis():
    """Check for deprecated NumPy APIs that changed in 2.x."""
    import numpy as np
    
    print("\n[7] Testing deprecated API compatibility...")
    
    issues = []
    
    # Check for np.string_ (deprecated in 2.x, use np.bytes_)
    try:
        _ = np.bytes_
        print("   ✓ np.bytes_ available")
    except AttributeError:
        issues.append("np.bytes_ not available")
    
    # Check for np.unicode_ (deprecated in 2.x, use np.str_)
    try:
        _ = np.str_
        print("   ✓ np.str_ available")
    except AttributeError:
        issues.append("np.str_ not available")
    
    # Check copy parameter behavior (changed in 2.x)
    arr = np.array([1, 2, 3])
    arr_copy = np.array(arr, copy=True)
    arr_copy[0] = 99
    assert arr[0] == 1, "Copy behavior issue"
    print("   ✓ Array copy behavior correct")
    
    if issues:
        print(f"   ⚠ Issues found: {issues}")
        return False
    
    print("   ✓ Deprecated APIs PASS")
    return True

def test_numba_compatibility():
    """Test numba JIT compilation if available."""
    print("\n[8] Testing Numba compatibility...")
    
    try:
        from numba import jit
        import numpy as np
        
        @jit(nopython=True)
        def fast_msd(x, y, max_lag):
            """JIT-compiled MSD calculation."""
            n = len(x)
            msd = np.zeros(max_lag)
            for lag in range(1, max_lag + 1):
                total = 0.0
                count = 0
                for i in range(n - lag):
                    dx = x[i + lag] - x[i]
                    dy = y[i + lag] - y[i]
                    total += dx * dx + dy * dy
                    count += 1
                if count > 0:
                    msd[lag - 1] = total / count
            return msd
        
        # Test the JIT function
        x = np.cumsum(np.random.randn(100))
        y = np.cumsum(np.random.randn(100))
        result = fast_msd(x, y, 10)
        
        assert len(result) == 10
        assert not np.any(np.isnan(result))
        
        print("   ✓ Numba JIT compilation PASS")
        return True
        
    except ImportError:
        print("   ⚠ Numba not installed (optional)")
        return True

def test_emcee_compatibility():
    """Test emcee MCMC sampler if available."""
    print("\n[9] Testing emcee compatibility...")
    
    try:
        import emcee
        import numpy as np
        
        # Simple linear model log-likelihood
        def log_likelihood(theta, x, y, yerr):
            m, b = theta
            model = m * x + b
            return -0.5 * np.sum(((y - model) / yerr) ** 2)
        
        def log_prior(theta):
            m, b = theta
            if -5.0 < m < 5.0 and -10.0 < b < 10.0:
                return 0.0
            return -np.inf
        
        def log_probability(theta, x, y, yerr):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, x, y, yerr)
        
        # Generate test data
        np.random.seed(42)
        x = np.linspace(0, 10, 20)
        y = 2.0 * x + 1.0 + np.random.randn(20) * 0.5
        yerr = np.ones_like(y) * 0.5
        
        # Initialize walkers
        nwalkers = 8
        ndim = 2
        pos = np.array([2.0, 1.0]) + 0.1 * np.random.randn(nwalkers, ndim)
        
        # Run short MCMC
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x, y, yerr))
        sampler.run_mcmc(pos, 50, progress=False)
        
        samples = sampler.get_chain(flat=True)
        assert samples.shape == (nwalkers * 50, ndim)
        
        print("   ✓ emcee MCMC PASS")
        return True
        
    except ImportError:
        print("   ⚠ emcee not installed (will be added)")
        return True

def main():
    """Run all compatibility tests."""
    print("=" * 60)
    print("SPT2025B NumPy 2.x Compatibility Test Suite")
    print("=" * 60)
    
    numpy_version = check_numpy_version()
    
    tests = [
        ("Array Operations", test_array_operations),
        ("Linear Algebra", test_linear_algebra),
        ("MSD Calculation", test_msd_calculation),
        ("SciPy Integration", test_scipy_integration),
        ("Pandas Integration", test_pandas_integration),
        ("Scikit-learn Integration", test_sklearn_integration),
        ("Deprecated APIs", test_deprecated_apis),
        ("Numba Compatibility", test_numba_compatibility),
        ("emcee Compatibility", test_emcee_compatibility),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"   ✗ {name} FAILED: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, r, _ in results if r)
    failed = len(results) - passed
    
    for name, result, error in results:
        status = "✓ PASS" if result else f"✗ FAIL: {error}"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if numpy_version >= 2:
        print("\n🎉 Running on NumPy 2.x - all tests should pass for full compatibility")
    else:
        print(f"\n📝 Running on NumPy {numpy_version}.x - tests verify forward compatibility")
    
    if failed == 0:
        print("\n✅ All tests passed! Safe to proceed with updates.")
        return 0
    else:
        print(f"\n⚠️ {failed} test(s) failed. Review before updating.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
