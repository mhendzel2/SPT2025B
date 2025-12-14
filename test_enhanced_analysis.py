#!/usr/bin/env python3
"""
Enhanced SPT Analysis Demonstration
====================================
This script demonstrates the capabilities of the newly added analysis libraries:
- lmfit: Advanced curve fitting with bounds, constraints, and confidence intervals
- uncertainties: Automatic error propagation
- powerlaw: Heavy-tailed distribution testing
- arviz: Bayesian diagnostics

Run: python test_enhanced_analysis.py
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("SPT2025B Enhanced Analysis Library Demonstration")
print("=" * 70)

# Track import success
imports_status = {}

# ============================================================================
# 1. LMFIT: Advanced Curve Fitting
# ============================================================================
print("\n1. LMFIT - Advanced Curve Fitting")
print("-" * 50)

try:
    from lmfit import Model, Parameters, minimize
    from lmfit.models import PowerLawModel
    imports_status['lmfit'] = True
    
    # Create synthetic MSD data with noise
    np.random.seed(42)
    time_lags = np.linspace(0.1, 10, 50)
    true_D = 0.05  # μm²/s
    true_alpha = 0.8  # Subdiffusive
    msd_true = 4 * true_D * (time_lags ** true_alpha)
    msd_noisy = msd_true * (1 + 0.1 * np.random.randn(len(time_lags)))
    
    # Define anomalous diffusion model with lmfit
    def anomalous_msd(t, K_alpha, alpha):
        """MSD = 4*K_alpha*t^alpha for 2D"""
        return 4 * K_alpha * (t ** alpha)
    
    # Create lmfit model
    model = Model(anomalous_msd)
    
    # Create parameters with bounds and constraints
    params = model.make_params()
    params['K_alpha'].set(value=0.01, min=1e-6, max=10)
    params['alpha'].set(value=1.0, min=0.1, max=2.0)
    
    # Fit with confidence intervals
    result = model.fit(msd_noisy, params, t=time_lags)
    
    print(f"  True values:   K_α = {true_D:.4f}, α = {true_alpha:.2f}")
    print(f"  Fitted values: K_α = {result.params['K_alpha'].value:.4f} ± {result.params['K_alpha'].stderr:.4f}")
    print(f"                 α = {result.params['alpha'].value:.3f} ± {result.params['alpha'].stderr:.3f}")
    print(f"  R² = {1 - result.residual.var() / msd_noisy.var():.4f}")
    print(f"  AIC = {result.aic:.2f}, BIC = {result.bic:.2f}")
    
    # Calculate confidence intervals
    try:
        ci = result.conf_interval()
        print(f"  95% CI for α: [{ci['alpha'][1][1]:.3f}, {ci['alpha'][5][1]:.3f}]")
    except Exception:
        pass  # CI calculation can fail for some datasets
    
    print("  ✓ lmfit provides: bounds, constraints, confidence intervals, AIC/BIC")
    
except ImportError as e:
    imports_status['lmfit'] = False
    print(f"  ✗ lmfit not available: {e}")

# ============================================================================
# 2. UNCERTAINTIES: Automatic Error Propagation
# ============================================================================
print("\n2. UNCERTAINTIES - Automatic Error Propagation")
print("-" * 50)

try:
    from uncertainties import ufloat, umath
    from uncertainties.umath import sqrt, log
    imports_status['uncertainties'] = True
    
    # Example: Diffusion coefficient with uncertainty
    D_measured = ufloat(0.05, 0.003)  # D = 0.05 ± 0.003 μm²/s
    alpha_measured = ufloat(0.85, 0.02)  # α = 0.85 ± 0.02
    
    # Calculate derived quantities - errors propagate automatically!
    # 1. MSD at t=1s: MSD = 4*D*t^α
    t = 1.0
    msd_1s = 4 * D_measured * (t ** alpha_measured)
    print(f"  D = {D_measured}")
    print(f"  α = {alpha_measured}")
    print(f"  MSD(t=1s) = {msd_1s} μm²")
    
    # 2. Confinement radius from D and viscosity
    kB = 1.380649e-23  # J/K
    T = ufloat(310, 0.5)  # Temperature 310 ± 0.5 K
    eta = ufloat(0.001, 0.0001)  # Viscosity 1 ± 0.1 mPa·s
    
    # Stokes-Einstein: D = kBT / (6πηr), so r = kBT / (6πηD)
    # Convert D from μm²/s to m²/s: multiply by 1e-12
    D_m2s = D_measured * 1e-12
    r_calculated = (kB * T) / (6 * 3.14159 * eta * D_m2s)
    r_nm = r_calculated * 1e9  # Convert to nm
    print(f"  Particle radius (Stokes-Einstein): {r_nm:.1f} nm")
    
    # 3. Calculate effective spring constant from confinement
    L = ufloat(0.5, 0.05)  # Confinement length 0.5 ± 0.05 μm
    k_eff = (kB * T) / (L * 1e-6)**2  # Spring constant k = kBT/L²
    print(f"  Effective spring constant: {k_eff:.2e} N/m")
    
    print("  ✓ uncertainties automatically propagates errors through calculations")
    
except ImportError as e:
    imports_status['uncertainties'] = False
    print(f"  ✗ uncertainties not available: {e}")

# ============================================================================
# 3. POWERLAW: Heavy-Tailed Distribution Testing
# ============================================================================
print("\n3. POWERLAW - Heavy-Tailed Distribution Testing")
print("-" * 50)

try:
    import powerlaw
    imports_status['powerlaw'] = True
    
    # Generate synthetic jump displacement data
    # Simulate a mixture: mostly normal diffusion, some Lévy-like jumps
    np.random.seed(42)
    n_jumps = 5000
    normal_jumps = np.abs(np.random.normal(0, 0.1, int(n_jumps * 0.9)))  # 90% Gaussian
    levy_jumps = np.random.pareto(1.5, int(n_jumps * 0.1)) * 0.1  # 10% heavy-tailed
    all_jumps = np.concatenate([normal_jumps, levy_jumps])
    
    # Fit power law
    fit = powerlaw.Fit(all_jumps, xmin=0.1, discrete=False, verbose=False)
    
    print(f"  Fitted power law exponent (α): {fit.alpha:.2f}")
    print(f"  x_min (minimum value for power law): {fit.xmin:.4f}")
    
    # Compare to alternative distributions
    # This is the KEY test: Is this really a power law, or lognormal, or exponential?
    R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal')
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
    
    print(f"\n  Distribution comparison tests:")
    print(f"    Power law vs Lognormal: R = {R_ln:.2f}, p = {p_ln:.3f}")
    print(f"    Power law vs Exponential: R = {R_exp:.2f}, p = {p_exp:.3f}")
    print(f"    (R > 0 favors power law, R < 0 favors alternative)")
    
    # Interpretation
    if p_ln > 0.1:
        print(f"\n  Interpretation: Cannot distinguish power law from lognormal (p={p_ln:.2f})")
    elif R_ln > 0:
        print(f"\n  Interpretation: Power law preferred over lognormal")
    else:
        print(f"\n  Interpretation: Lognormal preferred over power law")
    
    print("  ✓ powerlaw tests if displacement distributions are heavy-tailed (Lévy flights)")
    
except ImportError as e:
    imports_status['powerlaw'] = False
    print(f"  ✗ powerlaw not available: {e}")

# ============================================================================
# 4. ARVIZ: Bayesian Diagnostics
# ============================================================================
print("\n4. ARVIZ - Bayesian Model Diagnostics")
print("-" * 50)

try:
    import arviz as az
    imports_status['arviz'] = True
    
    # Check if emcee is available for full demonstration
    try:
        import emcee
        
        # Simple MCMC example: estimate diffusion coefficient from noisy MSD
        np.random.seed(42)
        time_lags = np.linspace(0.1, 5, 30)
        true_D = 0.1
        msd_data = 4 * true_D * time_lags * (1 + 0.1 * np.random.randn(len(time_lags)))
        msd_err = 0.02 * msd_data
        
        def log_likelihood(theta, t, msd, msd_err):
            D = theta[0]
            if D <= 0:
                return -np.inf
            model = 4 * D * t
            return -0.5 * np.sum(((msd - model) / msd_err) ** 2)
        
        def log_prior(theta):
            D = theta[0]
            if 0.001 < D < 1.0:
                return 0.0
            return -np.inf
        
        def log_probability(theta, t, msd, msd_err):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta, t, msd, msd_err)
        
        # Run MCMC
        nwalkers, ndim = 8, 1
        p0 = np.random.uniform(0.05, 0.15, (nwalkers, ndim))
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, 
                                        args=(time_lags, msd_data, msd_err))
        sampler.run_mcmc(p0, 1000, progress=False)
        
        # Convert to ArviZ format
        samples = sampler.get_chain(discard=200, flat=True)
        data = az.from_emcee(sampler, var_names=['D'])
        
        # ArviZ diagnostics
        summary = az.summary(data, var_names=['D'])
        print(f"  MCMC Results (arviz summary):")
        print(f"    True D:      {true_D:.4f}")
        print(f"    Estimated D: {summary['mean'].values[0]:.4f}")
        print(f"    Std Dev:     {summary['sd'].values[0]:.4f}")
        print(f"    HDI 94%:     [{summary['hdi_3%'].values[0]:.4f}, {summary['hdi_97%'].values[0]:.4f}]")
        
        # Convergence diagnostics
        rhat = az.rhat(data)
        ess = az.ess(data)
        print(f"\n  Convergence diagnostics:")
        print(f"    R-hat: {rhat['D'].values:.3f} (should be ~1.0)")
        print(f"    ESS:   {ess['D'].values:.0f} effective samples")
        
        print("  ✓ arviz provides HDI intervals, R-hat, ESS, trace plots, posterior plots")
        
    except ImportError:
        print("  (emcee not available - showing arviz capabilities without MCMC)")
        print("  arviz can visualize:")
        print("    - Posterior distributions")
        print("    - Trace plots for convergence")
        print("    - Pair plots for correlations")
        print("    - Energy plots for HMC diagnostics")
    
except ImportError as e:
    imports_status['arviz'] = False
    print(f"  ✗ arviz not available: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY: Library Import Status")
print("=" * 70)

all_passed = True
for lib, status in imports_status.items():
    symbol = "✓" if status else "✗"
    if not status:
        all_passed = False
    print(f"  {symbol} {lib}")

if all_passed:
    print("\n✓ All enhanced analysis libraries are available!")
    print("\nRecommended integrations for SPT2025B:")
    print("  1. Replace scipy.optimize.curve_fit with lmfit for MSD fitting")
    print("  2. Use uncertainties for D/α values throughout the pipeline")
    print("  3. Add powerlaw test to motion classification module")
    print("  4. Use arviz for MCMC-based parameter estimation visualization")
else:
    print("\n⚠ Some libraries failed to import. Check installation.")

print("\n" + "=" * 70)
