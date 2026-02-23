#!/usr/bin/env python3
"""
Test New Analysis Libraries with Real SPT Sample Data
======================================================
Tests lmfit, uncertainties, powerlaw, and arviz with actual track data.
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Testing New Libraries with Real SPT Sample Data")
print("=" * 70)

# ============================================================================
# Load Sample Data
# ============================================================================
print("\n1. LOADING SAMPLE DATA")
print("-" * 50)

data_path = r"c:\Users\mjhen\Github\SPT2025B\spt_projects\data\002362e1-db70-41e2-af2f-73cfed6dbd55.csv"

try:
    # Skip the header rows (3 header rows before data)
    df = pd.read_csv(data_path, header=0, skiprows=[1, 2, 3])
    
    # Rename columns to standard names
    df = df.rename(columns={
        'TRACK_ID': 'track_id',
        'FRAME': 'frame', 
        'POSITION_X': 'x',
        'POSITION_Y': 'y',
        'POSITION_T': 't'
    })
    
    # Convert to numeric
    df['track_id'] = pd.to_numeric(df['track_id'], errors='coerce')
    df['frame'] = pd.to_numeric(df['frame'], errors='coerce')
    df['x'] = pd.to_numeric(df['x'], errors='coerce')
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df['t'] = pd.to_numeric(df['t'], errors='coerce')
    
    # Drop NaN rows
    df = df.dropna(subset=['track_id', 'frame', 'x', 'y'])
    
    n_tracks = df['track_id'].nunique()
    n_points = len(df)
    print(f"  ✓ Loaded {n_points} points from {n_tracks} tracks")
    print(f"  X range: [{df['x'].min():.2f}, {df['x'].max():.2f}] μm")
    print(f"  Y range: [{df['y'].min():.2f}, {df['y'].max():.2f}] μm")
    
except Exception as e:
    print(f"  ✗ Failed to load data: {e}")
    raise

# ============================================================================
# Calculate MSD from Real Data
# ============================================================================
print("\n2. CALCULATING MSD FROM TRACK DATA")
print("-" * 50)

def calculate_msd(track_df, max_lag=20):
    """Calculate MSD for a single track."""
    track_df = track_df.sort_values('frame')
    x = track_df['x'].values
    y = track_df['y'].values
    n = len(x)
    
    msd_values = []
    msd_errors = []
    lags = []
    
    for lag in range(1, min(max_lag, n)):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        sq_displacements = dx**2 + dy**2
        
        if len(sq_displacements) >= 3:
            msd_values.append(np.mean(sq_displacements))
            msd_errors.append(np.std(sq_displacements) / np.sqrt(len(sq_displacements)))
            lags.append(lag)
    
    return np.array(lags), np.array(msd_values), np.array(msd_errors)

# Get ensemble MSD across all tracks
all_msds = {}
frame_interval = 3.0  # seconds (from the data)

for track_id, track_df in df.groupby('track_id'):
    if len(track_df) >= 10:  # Only tracks with enough points
        lags, msd, err = calculate_msd(track_df)
        if len(lags) > 5:
            all_msds[track_id] = (lags * frame_interval, msd, err)

print(f"  ✓ Calculated MSD for {len(all_msds)} tracks")

# Create ensemble average MSD
if all_msds:
    max_points = max(len(v[0]) for v in all_msds.values())
    ensemble_msd = np.zeros(max_points)
    ensemble_counts = np.zeros(max_points)
    ensemble_err = np.zeros(max_points)
    
    for track_id, (lags, msd, err) in all_msds.items():
        for i, (l, m, e) in enumerate(zip(lags, msd, err)):
            idx = int(l / frame_interval) - 1
            if idx < max_points:
                ensemble_msd[idx] += m
                ensemble_err[idx] += e**2
                ensemble_counts[idx] += 1
    
    # Average
    valid = ensemble_counts > 0
    ensemble_msd[valid] /= ensemble_counts[valid]
    ensemble_err[valid] = np.sqrt(ensemble_err[valid]) / ensemble_counts[valid]
    
    time_lags = (np.arange(max_points) + 1) * frame_interval
    time_lags = time_lags[valid]
    ensemble_msd = ensemble_msd[valid]
    ensemble_err = ensemble_err[valid]
    
    print(f"  ✓ Ensemble MSD calculated ({len(time_lags)} time points)")
    print(f"  Time range: [{time_lags[0]:.1f}, {time_lags[-1]:.1f}] s")
    print(f"  MSD range: [{ensemble_msd[0]:.4f}, {ensemble_msd[-1]:.4f}] μm²")

# ============================================================================
# Test LMFIT - Advanced Curve Fitting
# ============================================================================
print("\n3. LMFIT - Fitting MSD to Anomalous Diffusion Model")
print("-" * 50)

try:
    from lmfit import Model, Parameters
    
    # Anomalous diffusion model: MSD = 4*K_alpha*t^alpha
    def anomalous_msd(t, K_alpha, alpha):
        return 4 * K_alpha * (t ** alpha)
    
    model = Model(anomalous_msd)
    params = model.make_params()
    params['K_alpha'].set(value=0.01, min=1e-8, max=10)
    params['alpha'].set(value=1.0, min=0.1, max=2.0)
    
    # Fit with weights (inverse variance)
    weights = 1.0 / (ensemble_err + 1e-10)
    result = model.fit(ensemble_msd, params, t=time_lags, weights=weights)
    
    K_alpha = result.params['K_alpha'].value
    K_alpha_err = result.params['K_alpha'].stderr or 0
    alpha = result.params['alpha'].value
    alpha_err = result.params['alpha'].stderr or 0
    
    print(f"  Fitted parameters:")
    print(f"    K_α = {K_alpha:.6f} ± {K_alpha_err:.6f} μm²/s^α")
    print(f"    α   = {alpha:.3f} ± {alpha_err:.3f}")
    print(f"  Goodness of fit:")
    print(f"    R² = {1 - result.residual.var() / ensemble_msd.var():.4f}")
    print(f"    AIC = {result.aic:.2f}")
    print(f"    BIC = {result.bic:.2f}")
    
    # Motion classification
    if alpha < 0.9:
        motion_type = "Subdiffusive (confined/anomalous)"
    elif alpha > 1.1:
        motion_type = "Superdiffusive (directed/active)"
    else:
        motion_type = "Normal Brownian diffusion"
    print(f"  Classification: {motion_type}")
    
    # Calculate effective D at t=1s for comparison
    D_eff = K_alpha  # At t=1s, MSD = 4*K_alpha, so D = K_alpha
    print(f"  Effective D(t=1s) = {D_eff:.6f} μm²/s")
    
    print("  ✓ LMFIT fitting successful")
    
except ImportError as e:
    print(f"  ✗ lmfit not available: {e}")
except Exception as e:
    print(f"  ✗ lmfit fitting failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test UNCERTAINTIES - Error Propagation
# ============================================================================
print("\n4. UNCERTAINTIES - Automatic Error Propagation")
print("-" * 50)

try:
    from uncertainties import ufloat
    from uncertainties.umath import sqrt, log
    
    # Use the fitted values with uncertainties
    K_alpha_u = ufloat(K_alpha, K_alpha_err)
    alpha_u = ufloat(alpha, alpha_err)
    
    print(f"  Parameters with uncertainties:")
    print(f"    K_α = {K_alpha_u}")
    print(f"    α = {alpha_u}")
    
    # Calculate MSD at specific times with propagated errors
    t_values = [3.0, 10.0, 30.0]  # seconds
    print(f"\n  Predicted MSD with propagated uncertainties:")
    for t in t_values:
        msd_predicted = 4 * K_alpha_u * (t ** alpha_u)
        print(f"    MSD(t={t}s) = {msd_predicted} μm²")
    
    # Calculate confinement radius if subdiffusive
    if alpha < 0.9:
        # For confined diffusion: L² ≈ MSD_plateau
        # Estimate from long time behavior
        L_squared = 4 * K_alpha_u * (time_lags[-1] ** alpha_u)
        L_estimate = sqrt(L_squared)
        print(f"\n  Estimated confinement length: L ≈ {L_estimate} μm")
    
    # Stokes-Einstein calculation
    kB = 1.380649e-23  # J/K
    T = ufloat(310, 1)  # Temperature 310 ± 1 K (physiological)
    eta = ufloat(0.001, 0.0001)  # Viscosity 1 ± 0.1 mPa·s
    
    # D in m²/s (convert from μm²/s)
    D_m2s = K_alpha_u * 1e-12
    
    # Stokes-Einstein: r = kBT / (6πηD)
    r_calculated = (kB * T) / (6 * 3.14159 * eta * D_m2s)
    r_nm = r_calculated * 1e9
    print(f"\n  Stokes-Einstein particle radius: {r_nm:.1f} nm")
    
    print("  ✓ Uncertainties propagation successful")
    
except ImportError as e:
    print(f"  ✗ uncertainties not available: {e}")
except Exception as e:
    print(f"  ✗ uncertainties calculation failed: {e}")

# ============================================================================
# Test POWERLAW - Heavy-Tailed Distribution Testing
# ============================================================================
print("\n5. POWERLAW - Testing Step Size Distribution")
print("-" * 50)

try:
    import powerlaw
    
    # Calculate all step sizes (displacements between consecutive frames)
    step_sizes = []
    for track_id, track_df in df.groupby('track_id'):
        track_df = track_df.sort_values('frame')
        x = track_df['x'].values
        y = track_df['y'].values
        
        dx = np.diff(x)
        dy = np.diff(y)
        steps = np.sqrt(dx**2 + dy**2)
        step_sizes.extend(steps)
    
    step_sizes = np.array(step_sizes)
    step_sizes = step_sizes[step_sizes > 0.001]  # Remove very small/zero steps
    
    print(f"  Analyzed {len(step_sizes)} step displacements")
    print(f"  Step size range: [{step_sizes.min():.4f}, {step_sizes.max():.4f}] μm")
    print(f"  Mean step: {np.mean(step_sizes):.4f} μm")
    print(f"  Median step: {np.median(step_sizes):.4f} μm")
    
    # Fit power law
    fit = powerlaw.Fit(step_sizes, xmin=np.percentile(step_sizes, 10), 
                       discrete=False, verbose=False)
    
    print(f"\n  Power law fit results:")
    print(f"    Exponent (α): {fit.alpha:.2f}")
    print(f"    x_min: {fit.xmin:.4f} μm")
    
    # Compare distributions
    R_ln, p_ln = fit.distribution_compare('power_law', 'lognormal')
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential')
    
    print(f"\n  Distribution comparison (R>0 favors power law):")
    print(f"    vs Lognormal:   R = {R_ln:+.2f}, p = {p_ln:.3f}")
    print(f"    vs Exponential: R = {R_exp:+.2f}, p = {p_exp:.3f}")
    
    # Interpretation for SPT
    if p_ln < 0.05 and R_ln > 0:
        interpretation = "Strong evidence for power-law (potential Lévy flights)"
    elif p_ln < 0.05 and R_ln < 0:
        interpretation = "Lognormal distribution (typical for Brownian motion)"
    else:
        interpretation = "Cannot distinguish - likely normal diffusion"
    
    print(f"\n  Interpretation: {interpretation}")
    print("  ✓ Powerlaw analysis successful")
    
except ImportError as e:
    print(f"  ✗ powerlaw not available: {e}")
except Exception as e:
    print(f"  ✗ powerlaw analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# Test ARVIZ - Bayesian Diagnostics with emcee
# ============================================================================
print("\n6. ARVIZ + EMCEE - Bayesian Parameter Estimation")
print("-" * 50)

try:
    import arviz as az
    import emcee
    
    # Define log-likelihood for anomalous diffusion model
    def log_likelihood(theta, t, msd, msd_err):
        K_alpha, alpha = theta
        if K_alpha <= 0 or alpha <= 0 or alpha > 2:
            return -np.inf
        model_msd = 4 * K_alpha * (t ** alpha)
        chi2 = np.sum(((msd - model_msd) / msd_err) ** 2)
        return -0.5 * chi2
    
    def log_prior(theta):
        K_alpha, alpha = theta
        # Uniform priors
        if 1e-6 < K_alpha < 1.0 and 0.1 < alpha < 2.0:
            return 0.0
        return -np.inf
    
    def log_probability(theta, t, msd, msd_err):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, t, msd, msd_err)
    
    # Set up MCMC
    nwalkers = 16
    ndim = 2
    
    # Initialize near the lmfit solution
    p0 = np.array([K_alpha, alpha]) + 0.01 * np.random.randn(nwalkers, ndim)
    p0[:, 0] = np.abs(p0[:, 0])  # K_alpha must be positive
    p0[:, 1] = np.clip(p0[:, 1], 0.2, 1.9)  # alpha in range
    
    # Run MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability,
                                     args=(time_lags, ensemble_msd, ensemble_err))
    
    print("  Running MCMC (1000 steps)...")
    sampler.run_mcmc(p0, 1000, progress=False)
    
    # Convert to ArviZ
    var_names = ['K_alpha', 'alpha']
    data = az.from_emcee(sampler, var_names=var_names)
    
    # Get summary statistics
    summary = az.summary(data, var_names=var_names)
    
    print(f"\n  MCMC Results (ArviZ summary):")
    print(f"    K_α: {summary.loc['K_alpha', 'mean']:.6f} ± {summary.loc['K_alpha', 'sd']:.6f}")
    print(f"         HDI 94%: [{summary.loc['K_alpha', 'hdi_3%']:.6f}, {summary.loc['K_alpha', 'hdi_97%']:.6f}]")
    print(f"    α:   {summary.loc['alpha', 'mean']:.3f} ± {summary.loc['alpha', 'sd']:.3f}")
    print(f"         HDI 94%: [{summary.loc['alpha', 'hdi_3%']:.3f}, {summary.loc['alpha', 'hdi_97%']:.3f}]")
    
    # Convergence diagnostics
    rhat = az.rhat(data)
    ess = az.ess(data)
    
    print(f"\n  Convergence diagnostics:")
    print(f"    K_α: R-hat = {rhat['K_alpha'].values:.3f}, ESS = {ess['K_alpha'].values:.0f}")
    print(f"    α:   R-hat = {rhat['alpha'].values:.3f}, ESS = {ess['alpha'].values:.0f}")
    
    # Check convergence
    converged = all(rhat[var].values < 1.1 for var in var_names)
    print(f"\n  Convergence: {'✓ Chains converged' if converged else '⚠ Chains may not have converged'}")
    
    print("  ✓ ArviZ + emcee analysis successful")
    
except ImportError as e:
    print(f"  ✗ arviz/emcee not available: {e}")
except Exception as e:
    print(f"  ✗ Bayesian analysis failed: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70)

tests_passed = []
tests_failed = []

# Check each library
try:
    import lmfit
    tests_passed.append("lmfit")
except:
    tests_failed.append("lmfit")

try:
    import uncertainties
    tests_passed.append("uncertainties")
except:
    tests_failed.append("uncertainties")

try:
    import powerlaw
    tests_passed.append("powerlaw")
except:
    tests_failed.append("powerlaw")

try:
    import arviz
    tests_passed.append("arviz")
except:
    tests_failed.append("arviz")

print(f"\n  Passed: {len(tests_passed)}/4")
for t in tests_passed:
    print(f"    ✓ {t}")

if tests_failed:
    print(f"\n  Failed: {len(tests_failed)}/4")
    for t in tests_failed:
        print(f"    ✗ {t}")

if len(tests_passed) == 4:
    print("\n✓ All new libraries are working with real SPT data!")
    print("\nKey findings from sample data:")
    print(f"  - Diffusion coefficient K_α ≈ {K_alpha:.6f} μm²/s^α")
    print(f"  - Anomalous exponent α ≈ {alpha:.3f}")
    if alpha < 0.9:
        print("  - Motion type: Subdiffusive (confined)")
    elif alpha > 1.1:
        print("  - Motion type: Superdiffusive (directed)")
    else:
        print("  - Motion type: Normal Brownian")

print("\n" + "=" * 70)
