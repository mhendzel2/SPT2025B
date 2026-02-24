"""
Advanced Statistical Tests Module
Provides rigorous statistical validation for SPT analysis results.

Features:
- Goodness-of-fit tests for model fitting
- Bootstrap confidence intervals
- Non-parametric hypothesis tests
- Model selection criteria (AIC, BIC)
- Parameter uncertainty estimation
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
import warnings

try:
    from scipy import stats
    from scipy.optimize import curve_fit, minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available. Some statistical tests will be limited.")

try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Standardized multivariate tests will be limited.")

try:
    import ot
    POT_AVAILABLE = True
except ImportError:
    POT_AVAILABLE = False
    warnings.warn("POT not available. Wasserstein population statistics will be disabled.")


# ==================== GOODNESS-OF-FIT TESTS ====================

def chi_squared_goodness_of_fit(observed: np.ndarray, 
                               expected: np.ndarray,
                               n_params: int = 0) -> Dict[str, Any]:
    """
    Perform chi-squared goodness-of-fit test.
    
    Tests whether observed data matches expected distribution.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    expected : np.ndarray
        Expected values from model
    n_params : int
        Number of fitted parameters (for degrees of freedom)
        
    Returns
    -------
    dict
        chi2 statistic, p-value, degrees of freedom, conclusion
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    # Remove zeros and ensure positive values
    mask = (observed > 0) & (expected > 0)
    obs = observed[mask]
    exp = expected[mask]
    
    if len(obs) < n_params + 1:
        return {
            'success': False,
            'error': 'Insufficient data points for test'
        }
    
    # Chi-squared statistic
    chi2_stat = np.sum((obs - exp)**2 / exp)
    
    # Degrees of freedom
    dof = len(obs) - n_params - 1
    
    # P-value
    p_value = 1 - stats.chi2.cdf(chi2_stat, dof)
    
    # Conclusion
    significant = p_value < 0.05
    
    return {
        'success': True,
        'test': 'chi_squared',
        'statistic': chi2_stat,
        'p_value': p_value,
        'dof': dof,
        'significant': significant,
        'conclusion': 'Poor fit (p < 0.05)' if significant else 'Acceptable fit (p >= 0.05)',
        'reduced_chi2': chi2_stat / dof
    }


def kolmogorov_smirnov_test(observed: np.ndarray,
                           expected_cdf: Callable,
                           alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Perform Kolmogorov-Smirnov goodness-of-fit test.
    
    Tests whether observed data follows expected distribution.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    expected_cdf : callable
        Cumulative distribution function to test against
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    dict
        KS statistic, p-value, conclusion
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    # Perform KS test
    ks_stat, p_value = stats.kstest(observed, expected_cdf, alternative=alternative)
    
    return {
        'success': True,
        'test': 'kolmogorov_smirnov',
        'statistic': ks_stat,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'conclusion': 'Reject H0 (poor fit)' if p_value < 0.05 else 'Cannot reject H0 (acceptable fit)'
    }


def anderson_darling_test(observed: np.ndarray,
                         distribution: str = 'norm') -> Dict[str, Any]:
    """
    Perform Anderson-Darling goodness-of-fit test.
    
    More sensitive than KS test, especially at tails.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    distribution : str
        Distribution to test: 'norm', 'expon', 'logistic', 'gumbel', 'extreme1'
        
    Returns
    -------
    dict
        Test results with critical values
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    result = stats.anderson(observed, dist=distribution)
    
    # Find significance level
    sig_level = 0.05
    sig_index = np.argmin(np.abs(result.significance_level - 5.0))
    critical_value = result.critical_values[sig_index]
    
    return {
        'success': True,
        'test': 'anderson_darling',
        'statistic': result.statistic,
        'critical_values': dict(zip(result.significance_level, result.critical_values)),
        'significant': result.statistic > critical_value,
        'conclusion': f'Reject H0 at 5% level' if result.statistic > critical_value else 'Cannot reject H0'
    }


def residual_analysis(observed: np.ndarray,
                     predicted: np.ndarray) -> Dict[str, Any]:
    """
    Analyze residuals for model fit quality.
    
    Calculates residual statistics and tests for normality, autocorrelation.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed values
    predicted : np.ndarray
        Model predictions
        
    Returns
    -------
    dict
        Residual statistics and diagnostic tests
    """
    residuals = observed - predicted
    
    # Basic statistics
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Standardized residuals
    standardized = residuals / (std_residual + 1e-10)
    
    # R-squared
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((observed - np.mean(observed))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Adjusted R-squared (assuming 1 parameter for simplicity)
    n = len(observed)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - 2) if n > 2 else r_squared
    
    # RMSE and MAE
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    results = {
        'success': True,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'rmse': rmse,
        'mae': mae,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'max_absolute_residual': np.max(np.abs(residuals)),
        'residuals': residuals,
        'standardized_residuals': standardized
    }
    
    # Test for normality of residuals
    if SCIPY_AVAILABLE and len(residuals) >= 8:
        shapiro_stat, shapiro_p = stats.shapiro(residuals)
        results['normality_test'] = {
            'test': 'shapiro_wilk',
            'statistic': shapiro_stat,
            'p_value': shapiro_p,
            'normal': shapiro_p >= 0.05
        }
        
        # Test for autocorrelation (Durbin-Watson)
        if len(residuals) > 1:
            diff_residuals = np.diff(residuals)
            dw_stat = np.sum(diff_residuals**2) / np.sum(residuals**2)
            results['durbin_watson'] = {
                'statistic': dw_stat,
                'interpretation': 'No autocorrelation' if 1.5 <= dw_stat <= 2.5 else 'Autocorrelation present'
            }
    
    return results


# ==================== MODEL SELECTION CRITERIA ====================

def calculate_aic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Calculate Akaike Information Criterion (AIC).
    
    AIC = 2k - 2ln(L)
    Lower AIC indicates better model.
    
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of parameters
    n_samples : int
        Number of data points
        
    Returns
    -------
    float
        AIC value
    """
    aic = 2 * n_params - 2 * log_likelihood
    
    # Corrected AIC for small sample sizes
    if n_samples / n_params < 40:
        aicc = aic + (2 * n_params * (n_params + 1)) / (n_samples - n_params - 1)
        return aicc
    
    return aic


def calculate_bic(log_likelihood: float, n_params: int, n_samples: int) -> float:
    """
    Calculate Bayesian Information Criterion (BIC).
    
    BIC = k*ln(n) - 2ln(L)
    Lower BIC indicates better model.
    
    Parameters
    ----------
    log_likelihood : float
        Log-likelihood of the model
    n_params : int
        Number of parameters
    n_samples : int
        Number of data points
        
    Returns
    -------
    float
        BIC value
    """
    return n_params * np.log(n_samples) - 2 * log_likelihood


def compare_models(models: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple models using AIC and BIC.
    
    Parameters
    ----------
    models : list of dict
        Each dict contains: 'name', 'log_likelihood', 'n_params', 'n_samples'
        
    Returns
    -------
    pd.DataFrame
        Model comparison table with AIC, BIC, and rankings
    """
    results = []
    
    for model in models:
        aic = calculate_aic(model['log_likelihood'], model['n_params'], model['n_samples'])
        bic = calculate_bic(model['log_likelihood'], model['n_params'], model['n_samples'])
        
        results.append({
            'model': model['name'],
            'log_likelihood': model['log_likelihood'],
            'n_params': model['n_params'],
            'AIC': aic,
            'BIC': bic
        })
    
    df = pd.DataFrame(results)
    
    # Calculate delta AIC and BIC
    df['delta_AIC'] = df['AIC'] - df['AIC'].min()
    df['delta_BIC'] = df['BIC'] - df['BIC'].min()
    
    # Akaike weights
    df['AIC_weight'] = np.exp(-0.5 * df['delta_AIC'])
    df['AIC_weight'] /= df['AIC_weight'].sum()
    
    # Sort by AIC
    df = df.sort_values('AIC').reset_index(drop=True)
    df['rank'] = df.index + 1
    
    return df


# ==================== BOOTSTRAP METHODS ====================

def bootstrap_confidence_interval(data: np.ndarray,
                                 statistic_func: Callable,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95,
                                 random_seed: int = 42) -> Dict[str, Any]:
    """
    Calculate bootstrap confidence interval for any statistic.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    statistic_func : callable
        Function that computes statistic from data
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95%)
    random_seed : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Point estimate, confidence interval, bootstrap distribution
    """
    rng = np.random.default_rng(random_seed)
    
    # Calculate point estimate
    point_estimate = statistic_func(data)
    
    # Bootstrap resampling
    bootstrap_estimates = []
    
    for _ in range(n_bootstrap):
        # Resample with replacement
        resampled = rng.choice(data, size=len(data), replace=True)
        bootstrap_estimates.append(statistic_func(resampled))
    
    bootstrap_estimates = np.array(bootstrap_estimates)
    
    # Calculate confidence interval (percentile method)
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    ci_lower = np.percentile(bootstrap_estimates, lower_percentile)
    ci_upper = np.percentile(bootstrap_estimates, upper_percentile)
    
    # Bootstrap standard error
    se = np.std(bootstrap_estimates)
    
    return {
        'success': True,
        'point_estimate': point_estimate,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'confidence_level': confidence_level,
        'standard_error': se,
        'bootstrap_distribution': bootstrap_estimates,
        'n_bootstrap': n_bootstrap
    }


def bootstrap_parameter_uncertainty(x_data: np.ndarray,
                                   y_data: np.ndarray,
                                   fit_func: Callable,
                                   initial_params: np.ndarray,
                                   n_bootstrap: int = 1000,
                                   confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Estimate parameter uncertainty using bootstrap for curve fitting.
    
    Parameters
    ----------
    x_data : np.ndarray
        Independent variable
    y_data : np.ndarray
        Dependent variable
    fit_func : callable
        Fitting function f(x, *params)
    initial_params : np.ndarray
        Initial parameter guess
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level
        
    Returns
    -------
    dict
        Parameter estimates with confidence intervals
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    # Fit original data
    try:
        params_orig, _ = curve_fit(fit_func, x_data, y_data, p0=initial_params, maxfev=5000)
    except Exception as e:
        return {'success': False, 'error': f'Original fit failed: {str(e)}'}
    
    # Bootstrap resampling
    n_params = len(params_orig)
    bootstrap_params = np.zeros((n_bootstrap, n_params))
    
    rng = np.random.default_rng(42)
    
    successful_fits = 0
    for i in range(n_bootstrap):
        # Resample data points with replacement
        indices = rng.choice(len(x_data), size=len(x_data), replace=True)
        x_boot = x_data[indices]
        y_boot = y_data[indices]
        
        try:
            params_boot, _ = curve_fit(fit_func, x_boot, y_boot, p0=initial_params, maxfev=5000)
            bootstrap_params[successful_fits] = params_boot
            successful_fits += 1
        except:
            continue
    
    if successful_fits < n_bootstrap * 0.5:
        return {'success': False, 'error': 'Too many bootstrap fits failed'}
    
    # Trim to successful fits
    bootstrap_params = bootstrap_params[:successful_fits]
    
    # Calculate confidence intervals for each parameter
    alpha = 1 - confidence_level
    lower_percentile = 100 * alpha / 2
    upper_percentile = 100 * (1 - alpha / 2)
    
    results = {
        'success': True,
        'n_successful_fits': successful_fits,
        'parameters': {}
    }
    
    for i in range(n_params):
        param_values = bootstrap_params[:, i]
        results['parameters'][f'param_{i}'] = {
            'estimate': params_orig[i],
            'ci_lower': np.percentile(param_values, lower_percentile),
            'ci_upper': np.percentile(param_values, upper_percentile),
            'std_error': np.std(param_values),
            'bootstrap_distribution': param_values
        }
    
    return results


# ==================== NON-PARAMETRIC TESTS ====================

def mann_whitney_u_test(sample1: np.ndarray,
                       sample2: np.ndarray,
                       alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Perform Mann-Whitney U test (non-parametric alternative to t-test).
    
    Tests whether two samples come from the same distribution.
    
    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Two independent samples
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    dict
        Test results
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    statistic, p_value = stats.mannwhitneyu(sample1, sample2, alternative=alternative)
    
    return {
        'success': True,
        'test': 'mann_whitney_u',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'median_1': np.median(sample1),
        'median_2': np.median(sample2),
        'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
    }


def kruskal_wallis_test(samples: List[np.ndarray]) -> Dict[str, Any]:
    """
    Perform Kruskal-Wallis H test (non-parametric ANOVA).
    
    Tests whether multiple samples come from the same distribution.
    
    Parameters
    ----------
    samples : list of np.ndarray
        Multiple independent samples
        
    Returns
    -------
    dict
        Test results
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    statistic, p_value = stats.kruskal(*samples)
    
    return {
        'success': True,
        'test': 'kruskal_wallis',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_groups': len(samples),
        'medians': [np.median(s) for s in samples],
        'conclusion': 'Groups differ significantly' if p_value < 0.05 else 'No significant difference among groups'
    }


def wilcoxon_signed_rank_test(sample1: np.ndarray,
                              sample2: np.ndarray,
                              alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test (non-parametric paired t-test).
    
    Tests whether two paired samples have the same distribution.
    
    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Paired samples
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    dict
        Test results
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    statistic, p_value = stats.wilcoxon(sample1, sample2, alternative=alternative)
    
    return {
        'success': True,
        'test': 'wilcoxon_signed_rank',
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'median_difference': np.median(sample1 - sample2),
        'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
    }


def permutation_test(sample1: np.ndarray,
                    sample2: np.ndarray,
                    statistic_func: Callable = None,
                    n_permutations: int = 10000,
                    alternative: str = 'two-sided') -> Dict[str, Any]:
    """
    Perform permutation test for comparing two samples.
    
    Non-parametric test with no distributional assumptions.
    
    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Two samples to compare
    statistic_func : callable, optional
        Function to compute test statistic (default: difference of means)
    n_permutations : int
        Number of permutations
    alternative : str
        'two-sided', 'less', or 'greater'
        
    Returns
    -------
    dict
        Test results
    """
    if statistic_func is None:
        statistic_func = lambda x, y: np.mean(x) - np.mean(y)
    
    # Observed statistic
    observed_stat = statistic_func(sample1, sample2)
    
    # Combine samples
    combined = np.concatenate([sample1, sample2])
    n1 = len(sample1)
    
    # Permutation test
    rng = np.random.default_rng(42)
    perm_stats = []
    
    for _ in range(n_permutations):
        # Shuffle and split
        shuffled = rng.permutation(combined)
        perm_sample1 = shuffled[:n1]
        perm_sample2 = shuffled[n1:]
        
        perm_stat = statistic_func(perm_sample1, perm_sample2)
        perm_stats.append(perm_stat)
    
    perm_stats = np.array(perm_stats)
    
    # Calculate p-value
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(perm_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(perm_stats <= observed_stat)
    else:
        p_value = np.nan
    
    return {
        'success': True,
        'test': 'permutation',
        'observed_statistic': observed_stat,
        'p_value': p_value,
        'n_permutations': n_permutations,
        'significant': p_value < 0.05,
        'permutation_distribution': perm_stats,
        'conclusion': 'Significant difference' if p_value < 0.05 else 'No significant difference'
    }


def compare_populations_wasserstein(
    features_A: pd.DataFrame,
    features_B: pd.DataFrame,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Compare two trajectory populations using multivariate Wasserstein distance.

    The function standardizes requested feature columns, computes the Earth
    Mover's Distance (EMD) between populations using ``ot.emd2``, and estimates
    significance with a label-permutation test (500 permutations).

    Parameters
    ----------
    features_A : pd.DataFrame
        Feature table for population A.
    features_B : pd.DataFrame
        Feature table for population B.
    feature_cols : list of str
        Columns to include in the multivariate population comparison.

    Returns
    -------
    dict
        Dictionary containing observed Wasserstein distance, permutation
        distribution summary, and empirical p-value.
    """
    if not POT_AVAILABLE:
        return {'success': False, 'error': 'POT (ot) not available'}
    if not SKLEARN_AVAILABLE:
        return {'success': False, 'error': 'scikit-learn not available'}
    if not feature_cols:
        return {'success': False, 'error': 'feature_cols cannot be empty'}

    missing_A = [col for col in feature_cols if col not in features_A.columns]
    missing_B = [col for col in feature_cols if col not in features_B.columns]
    if missing_A or missing_B:
        return {
            'success': False,
            'error': 'Missing feature columns',
            'missing_in_A': missing_A,
            'missing_in_B': missing_B
        }

    A_raw = features_A[feature_cols].copy()
    B_raw = features_B[feature_cols].copy()

    A_raw = A_raw.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')
    B_raw = B_raw.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

    if len(A_raw) < 2 or len(B_raw) < 2:
        return {
            'success': False,
            'error': 'Insufficient valid rows after filtering NaN/inf',
            'n_A': int(len(A_raw)),
            'n_B': int(len(B_raw))
        }

    combined = pd.concat([A_raw, B_raw], axis=0, ignore_index=True)
    scaler = StandardScaler()
    combined_scaled = scaler.fit_transform(combined.to_numpy(dtype=float))

    n_A = len(A_raw)
    n_B = len(B_raw)
    A_scaled = combined_scaled[:n_A]
    B_scaled = combined_scaled[n_A:]

    def _emd2_multivariate(X: np.ndarray, Y: np.ndarray) -> float:
        a = np.full(X.shape[0], 1.0 / X.shape[0], dtype=float)
        b = np.full(Y.shape[0], 1.0 / Y.shape[0], dtype=float)
        M = ot.dist(X, Y, metric='euclidean')
        return float(ot.emd2(a, b, M))

    observed_emd = _emd2_multivariate(A_scaled, B_scaled)

    rng = np.random.default_rng(42)
    permutation_dist = np.empty(500, dtype=float)
    all_scaled = np.vstack([A_scaled, B_scaled])

    for i in range(500):
        perm_idx = rng.permutation(all_scaled.shape[0])
        perm_A = all_scaled[perm_idx[:n_A]]
        perm_B = all_scaled[perm_idx[n_A:]]
        permutation_dist[i] = _emd2_multivariate(perm_A, perm_B)

    empirical_p_value = float((1.0 + np.sum(permutation_dist >= observed_emd)) / (len(permutation_dist) + 1.0))

    return {
        'success': True,
        'test': 'wasserstein_emd2_multivariate',
        'feature_cols': feature_cols,
        'n_A': int(n_A),
        'n_B': int(n_B),
        'wasserstein_distance': observed_emd,
        'p_value': empirical_p_value,
        'n_permutations': int(len(permutation_dist)),
        'significant': empirical_p_value < 0.05,
        'permutation_mean': float(np.mean(permutation_dist)),
        'permutation_std': float(np.std(permutation_dist)),
        'permutation_distribution': permutation_dist
    }


# ==================== HIGH-LEVEL API ====================

def validate_model_fit(observed: np.ndarray,
                      predicted: np.ndarray,
                      n_params: int = 1,
                      run_all_tests: bool = True) -> Dict[str, Any]:
    """
    Comprehensive model fit validation.
    
    Runs multiple goodness-of-fit tests and residual analysis.
    
    Parameters
    ----------
    observed : np.ndarray
        Observed data
    predicted : np.ndarray
        Model predictions
    n_params : int
        Number of model parameters
    run_all_tests : bool
        Whether to run all available tests
        
    Returns
    -------
    dict
        Comprehensive validation results
    """
    results = {
        'success': True,
        'n_data_points': len(observed),
        'n_parameters': n_params
    }
    
    try:
        # Residual analysis (always run)
        results['residual_analysis'] = residual_analysis(observed, predicted)
        
        if run_all_tests and SCIPY_AVAILABLE:
            # Chi-squared test
            results['chi_squared_test'] = chi_squared_goodness_of_fit(observed, predicted, n_params)
            
            # Normality test on residuals
            residuals = observed - predicted
            if len(residuals) >= 3:
                results['shapiro_test'] = {
                    'statistic': stats.shapiro(residuals)[0],
                    'p_value': stats.shapiro(residuals)[1]
                }
        
        # Overall assessment
        r_squared = results['residual_analysis']['r_squared']
        if r_squared > 0.9:
            quality = 'Excellent'
        elif r_squared > 0.7:
            quality = 'Good'
        elif r_squared > 0.5:
            quality = 'Moderate'
        else:
            quality = 'Poor'
        
        results['overall_assessment'] = {
            'quality': quality,
            'r_squared': r_squared,
            'rmse': results['residual_analysis']['rmse']
        }
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results


def compare_statistical_distributions(sample1: np.ndarray,
                                     sample2: np.ndarray,
                                     run_parametric: bool = True,
                                     run_nonparametric: bool = True) -> Dict[str, Any]:
    """
    Comprehensive comparison of two distributions.
    
    Runs both parametric and non-parametric tests.
    
    Parameters
    ----------
    sample1, sample2 : np.ndarray
        Samples to compare
    run_parametric : bool
        Run t-test and F-test
    run_nonparametric : bool
        Run Mann-Whitney and permutation tests
        
    Returns
    -------
    dict
        Comprehensive test results
    """
    if not SCIPY_AVAILABLE:
        return {'success': False, 'error': 'scipy not available'}
    
    results = {
        'success': True,
        'sample1_size': len(sample1),
        'sample2_size': len(sample2),
        'sample1_mean': np.mean(sample1),
        'sample2_mean': np.mean(sample2),
        'sample1_median': np.median(sample1),
        'sample2_median': np.median(sample2)
    }
    
    try:
        if run_parametric:
            # T-test
            t_stat, t_p = stats.ttest_ind(sample1, sample2)
            results['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < 0.05
            }
            
            # F-test for variance
            f_stat = np.var(sample1, ddof=1) / np.var(sample2, ddof=1)
            f_p = 2 * min(stats.f.cdf(f_stat, len(sample1)-1, len(sample2)-1),
                         1 - stats.f.cdf(f_stat, len(sample1)-1, len(sample2)-1))
            results['f_test'] = {
                'statistic': f_stat,
                'p_value': f_p,
                'significant': f_p < 0.05
            }
        
        if run_nonparametric:
            # Mann-Whitney U
            results['mann_whitney'] = mann_whitney_u_test(sample1, sample2)
            
            # Permutation test
            results['permutation_test'] = permutation_test(sample1, sample2)
        
        # Overall conclusion
        tests_significant = sum([
            results.get('t_test', {}).get('significant', False),
            results.get('mann_whitney', {}).get('significant', False)
        ])
        
        results['conclusion'] = {
            'tests_run': 2 if run_parametric and run_nonparametric else 1,
            'tests_significant': tests_significant,
            'overall': 'Significant difference' if tests_significant >= 1 else 'No significant difference'
        }
        
    except Exception as e:
        results['success'] = False
        results['error'] = str(e)
    
    return results
