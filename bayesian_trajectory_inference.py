"""
Bayesian Trajectory Inference with Posterior Analysis

Implements Bayesian inference methods for trajectory analysis with:
- MCMC sampling via emcee for parameter uncertainty quantification
- Posterior interval computation
- Identifiability diagnostics using ArviZ
- Trace plots and convergence diagnostics
- Posterior predictive checks

Based on recent methods from bayes_traj (JOSS 2025) and similar tools.

Author: SPT2025B Team
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

# Try importing optional dependencies
try:
    import emcee
    EMCEE_AVAILABLE = True
except ImportError:
    EMCEE_AVAILABLE = False
    warnings.warn("emcee not available. MCMC sampling will be disabled.")

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
    try:
        JAX_GPU_AVAILABLE = any(dev.platform == "gpu" for dev in jax.devices())
    except Exception:
        JAX_GPU_AVAILABLE = False
except ImportError:
    JAX_AVAILABLE = False
    JAX_GPU_AVAILABLE = False

try:
    import numpyro
    import numpyro.distributions as npdist
    from numpyro.distributions import transforms as nptransforms
    from numpyro.infer import MCMC, NUTS
    NUMPYRO_AVAILABLE = True
except ImportError:
    NUMPYRO_AVAILABLE = False

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    warnings.warn("arviz not available. Posterior diagnostics will be limited.")

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False
    warnings.warn("corner not available. Corner plots will be disabled.")


class BayesianDiffusionInference:
    """
    Bayesian inference for diffusion parameters with full posterior analysis.
    
    Provides credible intervals and diagnostics beyond point estimates.
    """
    
    def __init__(
        self,
        frame_interval: float,
        localization_error: float,
        exposure_time: Optional[float] = None,
        dimensions: int = 2
    ):
        """
        Initialize Bayesian diffusion inference.
        
        Parameters
        ----------
        frame_interval : float
            Time between frames (seconds)
        localization_error : float
            Localization precision (micrometers)
        exposure_time : float, optional
            Camera exposure time (seconds)
        dimensions : int
            Spatial dimensions (2 or 3)
        """
        self.frame_interval = frame_interval
        self.localization_error = localization_error
        self.exposure_time = exposure_time if exposure_time is not None else frame_interval
        self.dimensions = dimensions
        
        # Motion blur factor
        R = self.exposure_time / self.frame_interval
        self.blur_factor = 1.0 - (R**2) / 6.0 if R < 1 else 0.5
    
    def log_prior(self, theta: np.ndarray, prior_config: Optional[Dict] = None) -> float:
        """
        Log prior probability for parameters.
        
        Default: Weakly informative priors
        - D ~ LogNormal(mean=-2, std=2) in µm²/s
        - alpha ~ Beta(2, 2) constrained to [0.5, 2.0]
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters [D, alpha] or just [D]
        prior_config : dict, optional
            Custom prior configuration
        
        Returns
        -------
        float
            Log prior probability
        """
        if prior_config is None:
            prior_config = {
                'D_log_mean': -2.0,
                'D_log_std': 2.0,
                'alpha_a': 2.0,
                'alpha_b': 2.0,
                'alpha_min': 0.5,
                'alpha_max': 2.0
            }
        
        log_D = np.log(theta[0]) if theta[0] > 0 else -np.inf
        
        # Log-normal prior for D
        log_prior_D = -0.5 * ((log_D - prior_config['D_log_mean']) / prior_config['D_log_std'])**2
        log_prior_D -= np.log(theta[0] * prior_config['D_log_std'] * np.sqrt(2 * np.pi))
        
        if len(theta) > 1:
            # Beta prior for alpha (scaled to [alpha_min, alpha_max])
            alpha = theta[1]
            alpha_min = prior_config['alpha_min']
            alpha_max = prior_config['alpha_max']
            
            if alpha < alpha_min or alpha > alpha_max:
                return -np.inf
            
            # Transform to [0, 1]
            alpha_normalized = (alpha - alpha_min) / (alpha_max - alpha_min)
            
            from scipy.stats import beta
            log_prior_alpha = beta.logpdf(
                alpha_normalized,
                prior_config['alpha_a'],
                prior_config['alpha_b']
            )
            
            return log_prior_D + log_prior_alpha
        else:
            return log_prior_D
    
    def log_likelihood(
        self,
        theta: np.ndarray,
        displacements: np.ndarray
    ) -> float:
        """
        Log likelihood for observed displacements.
        
        P(data | D, alpha) assuming Gaussian displacements with
        variance = 2*D*dt^alpha + 2*sigma_loc^2 (per dimension)
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters [D] or [D, alpha]
        displacements : np.ndarray
            Observed displacements, shape (N, dimensions)
        
        Returns
        -------
        float
            Log likelihood
        """
        D = theta[0]
        alpha = theta[1] if len(theta) > 1 else 1.0
        
        if D <= 0 or alpha <= 0 or alpha > 2:
            return -np.inf
        
        # Expected variance per dimension
        var_expected = (
            2 * D * (self.frame_interval ** alpha) * self.blur_factor +
            2 * self.localization_error**2
        )
        
        if var_expected <= 0:
            return -np.inf
        
        # Gaussian likelihood
        # L = -N*d/2 * log(2π*var) - sum(displacements^2) / (2*var)
        N = len(displacements)
        sum_squared = np.sum(displacements**2)
        
        log_L = (
            -N * self.dimensions / 2 * np.log(2 * np.pi * var_expected) -
            sum_squared / (2 * var_expected)
        )
        
        return log_L
    
    def log_probability(
        self,
        theta: np.ndarray,
        displacements: np.ndarray,
        prior_config: Optional[Dict] = None
    ) -> float:
        """
        Log posterior probability (prior + likelihood).
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters
        displacements : np.ndarray
            Observed displacements
        prior_config : dict, optional
            Prior configuration
        
        Returns
        -------
        float
            Log posterior probability
        """
        lp = self.log_prior(theta, prior_config)
        if not np.isfinite(lp):
            return -np.inf
        
        return lp + self.log_likelihood(theta, displacements)
    
    def run_mcmc(
        self,
        displacements: np.ndarray,
        n_walkers: int = 32,
        n_steps: int = 2000,
        n_burn: int = 500,
        estimate_alpha: bool = True,
        prior_config: Optional[Dict] = None,
        progress: bool = False,
        backend: str = "auto",
        n_warmup: int = 500,
        rng_seed: int = 42,
        use_gpu: bool = True,
    ) -> Dict:
        """
        Run MCMC sampling to obtain posterior samples.
        
        Parameters
        ----------
        displacements : np.ndarray
            Observed displacements, shape (N, dimensions)
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of steps per walker
        n_burn : int
            Burn-in steps to discard
        estimate_alpha : bool
            Whether to estimate alpha or fix to 1.0
        prior_config : dict, optional
            Prior configuration
        progress : bool
            Show progress bar
        
        Returns
        -------
        Dict
            {
                'success': bool,
                'samples': np.ndarray,  # Shape (n_samples, n_params)
                'D_median': float,
                'D_std': float,
                'D_credible_interval': (lower, upper),
                'alpha_median': float (if estimate_alpha),
                'alpha_std': float (if estimate_alpha),
                'alpha_credible_interval': (lower, upper) (if estimate_alpha),
                'diagnostics': dict,
                'chain': emcee.EnsembleSampler (if available)
            }
        """
        backend_name = str(backend).strip().lower()
        if backend_name not in {"auto", "emcee", "numpyro"}:
            return {'success': False, 'error': f'Unknown backend: {backend}'}

        if prior_config is None:
            prior_config = {
                'D_log_mean': -2.0,
                'D_log_std': 2.0,
                'alpha_a': 2.0,
                'alpha_b': 2.0,
                'alpha_min': 0.5,
                'alpha_max': 2.0,
            }

        if backend_name == "auto":
            if NUMPYRO_AVAILABLE and JAX_AVAILABLE:
                backend_name = "numpyro"
            elif EMCEE_AVAILABLE:
                backend_name = "emcee"
            else:
                return {
                    'success': False,
                    'error': 'No Bayesian backend available (install emcee or numpyro+jax).'
                }

        if backend_name == "numpyro":
            if not (NUMPYRO_AVAILABLE and JAX_AVAILABLE):
                return {
                    'success': False,
                    'error': 'numpyro backend requested but numpyro/jax is unavailable.'
                }
            return self._run_numpyro(
                displacements=displacements,
                n_steps=n_steps,
                n_warmup=n_warmup,
                estimate_alpha=estimate_alpha,
                prior_config=prior_config,
                progress=progress,
                rng_seed=rng_seed,
                use_gpu=use_gpu,
            )

        if not EMCEE_AVAILABLE:
            return {
                'success': False,
                'error': 'emcee backend requested but emcee is unavailable.'
            }
        return self._run_emcee(
            displacements=displacements,
            n_walkers=n_walkers,
            n_steps=n_steps,
            n_burn=n_burn,
            estimate_alpha=estimate_alpha,
            prior_config=prior_config,
            progress=progress,
        )

    def _run_emcee(
        self,
        displacements: np.ndarray,
        n_walkers: int,
        n_steps: int,
        n_burn: int,
        estimate_alpha: bool,
        prior_config: Dict,
        progress: bool,
    ) -> Dict:
        """CPU `emcee` implementation."""
        # Initial guess from MLE
        squared_disp = np.sum(displacements**2, axis=1)
        msd_simple = np.mean(squared_disp)
        D_init = msd_simple / (2 * self.dimensions * self.frame_interval)

        # Initialize walkers
        n_dim = 2 if estimate_alpha else 1

        if estimate_alpha:
            # Small perturbations around initial guess
            pos = np.zeros((n_walkers, n_dim))
            pos[:, 0] = D_init * (1 + 0.1 * np.random.randn(n_walkers))  # D
            pos[:, 1] = 1.0 + 0.05 * np.random.randn(n_walkers)  # alpha
            pos[:, 0] = np.abs(pos[:, 0])  # Ensure D > 0
            pos[:, 1] = np.clip(pos[:, 1], 0.5, 2.0)  # Constrain alpha
        else:
            pos = D_init * (1 + 0.1 * np.random.randn(n_walkers, 1))
            pos = np.abs(pos)

        # Set up sampler
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_probability,
            args=(displacements, prior_config)
        )

        # Run MCMC
        try:
            sampler.run_mcmc(pos, n_steps, progress=progress)

            # Ensure burn-in does not discard all samples.
            effective_n_burn = min(max(int(n_burn), 0), max(int(n_steps) - 1, 0))

            # Get samples (discard burn-in)
            samples = sampler.get_chain(discard=effective_n_burn, flat=True)
            if samples.size == 0:
                return {
                    'success': False,
                    'error': (
                        f'No posterior samples after burn-in '
                        f'(n_steps={n_steps}, n_burn={effective_n_burn})'
                    ),
                }

            # Extract posteriors
            D_samples = samples[:, 0]
            D_median = np.median(D_samples)
            D_std = np.std(D_samples)
            D_ci = np.percentile(D_samples, [2.5, 97.5])  # 95% credible interval

            result = {
                'success': True,
                'samples': samples,
                'D_median': float(D_median),
                'D_std': float(D_std),
                'D_credible_interval': tuple(D_ci.tolist()),
                'D_mean': float(np.mean(D_samples)),
                'n_samples': len(samples),
                'n_walkers': n_walkers,
                'n_steps': n_steps,
                'n_burn': effective_n_burn,
                'backend': 'emcee',
                'device': 'cpu',
            }

            if estimate_alpha:
                alpha_samples = samples[:, 1]
                alpha_median = np.median(alpha_samples)
                alpha_std = np.std(alpha_samples)
                alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])

                result.update({
                    'alpha_median': float(alpha_median),
                    'alpha_std': float(alpha_std),
                    'alpha_credible_interval': tuple(alpha_ci.tolist()),
                    'alpha_mean': float(np.mean(alpha_samples))
                })

            # Diagnostics
            result['diagnostics'] = self._compute_diagnostics(sampler, effective_n_burn)
            result['sampler'] = sampler  # Keep for additional analysis

            return result

        except Exception as e:
            return {
                'success': False,
                'error': f'MCMC sampling failed: {str(e)}'
            }

    def _run_numpyro(
        self,
        displacements: np.ndarray,
        n_steps: int,
        n_warmup: int,
        estimate_alpha: bool,
        prior_config: Dict,
        progress: bool,
        rng_seed: int,
        use_gpu: bool,
    ) -> Dict:
        """JAX/NumPyro implementation; can run on GPU when available."""
        try:
            if not use_gpu:
                numpyro.set_platform("cpu")
            elif JAX_GPU_AVAILABLE:
                numpyro.set_platform("gpu")

            disp_obs = jnp.asarray(displacements, dtype=jnp.float32)

            D_log_mean = float(prior_config.get('D_log_mean', -2.0))
            D_log_std = float(prior_config.get('D_log_std', 2.0))
            alpha_a = float(prior_config.get('alpha_a', 2.0))
            alpha_b = float(prior_config.get('alpha_b', 2.0))
            alpha_min = float(prior_config.get('alpha_min', 0.5))
            alpha_max = float(prior_config.get('alpha_max', 2.0))

            frame_interval = float(self.frame_interval)
            blur_factor = float(self.blur_factor)
            localization_error = float(self.localization_error)

            def model(obs):
                D = numpyro.sample("D", npdist.LogNormal(D_log_mean, D_log_std))
                if estimate_alpha:
                    alpha = numpyro.sample(
                        "alpha",
                        npdist.TransformedDistribution(
                            npdist.Beta(alpha_a, alpha_b),
                            nptransforms.AffineTransform(
                                loc=alpha_min,
                                scale=alpha_max - alpha_min,
                            ),
                        ),
                    )
                else:
                    alpha = 1.0

                var_expected = (
                    2.0 * D * (frame_interval ** alpha) * blur_factor +
                    2.0 * (localization_error ** 2)
                )
                sigma = jnp.sqrt(jnp.maximum(var_expected, 1e-12))
                numpyro.sample("disp", npdist.Normal(0.0, sigma), obs=obs)

            num_samples = max(1, int(n_steps))
            num_warmup = min(max(int(n_warmup), 0), max(num_samples - 1, 0))

            kernel = NUTS(model, target_accept_prob=0.8)
            mcmc = MCMC(
                kernel,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=1,
                progress_bar=bool(progress),
            )
            key = jax.random.PRNGKey(int(rng_seed))
            mcmc.run(key, disp_obs)
            sample_dict = mcmc.get_samples(group_by_chain=False)

            D_samples = np.asarray(sample_dict.get("D", []), dtype=float)
            if D_samples.size == 0:
                return {'success': False, 'error': 'NumPyro produced no D samples'}

            if estimate_alpha:
                alpha_samples = np.asarray(sample_dict.get("alpha", []), dtype=float)
                if alpha_samples.size == 0:
                    return {'success': False, 'error': 'NumPyro produced no alpha samples'}
                samples = np.column_stack([D_samples, alpha_samples])
            else:
                samples = D_samples.reshape(-1, 1)

            D_median = np.median(D_samples)
            D_std = np.std(D_samples)
            D_ci = np.percentile(D_samples, [2.5, 97.5])

            result = {
                'success': True,
                'samples': samples,
                'D_median': float(D_median),
                'D_std': float(D_std),
                'D_credible_interval': tuple(D_ci.tolist()),
                'D_mean': float(np.mean(D_samples)),
                'n_samples': int(samples.shape[0]),
                'n_steps': num_samples,
                'n_burn': num_warmup,
                'backend': 'numpyro',
                'device': 'gpu' if (use_gpu and JAX_GPU_AVAILABLE) else 'cpu',
                'diagnostics': self._compute_numpyro_diagnostics(mcmc),
                'sampler': mcmc,
            }

            if estimate_alpha:
                alpha_samples = samples[:, 1]
                alpha_median = np.median(alpha_samples)
                alpha_std = np.std(alpha_samples)
                alpha_ci = np.percentile(alpha_samples, [2.5, 97.5])
                result.update({
                    'alpha_median': float(alpha_median),
                    'alpha_std': float(alpha_std),
                    'alpha_credible_interval': tuple(alpha_ci.tolist()),
                    'alpha_mean': float(np.mean(alpha_samples)),
                })

            return result
        except Exception as e:
            return {'success': False, 'error': f'NumPyro sampling failed: {str(e)}'}

    def _compute_numpyro_diagnostics(self, mcmc: Any) -> Dict[str, Any]:
        """Best-effort diagnostics for NumPyro runs."""
        diagnostics: Dict[str, Any] = {}
        try:
            extra = mcmc.get_extra_fields()
            accept_prob = extra.get('accept_prob')
            if accept_prob is not None:
                diagnostics['mean_acceptance'] = float(np.mean(np.asarray(accept_prob)))
            else:
                diagnostics['mean_acceptance'] = float('nan')

            diverging = extra.get('diverging')
            diagnostics['n_divergent'] = int(np.sum(np.asarray(diverging))) if diverging is not None else 0
            diagnostics['converged'] = diagnostics['n_divergent'] == 0
            diagnostics['n_effective'] = int(mcmc.num_samples)
        except Exception as e:
            diagnostics['error'] = str(e)
        return diagnostics
    
    def _compute_diagnostics(self, sampler: Any, n_burn: int) -> Dict:
        """
        Compute convergence diagnostics for MCMC.
        
        Parameters
        ----------
        sampler : emcee.EnsembleSampler
            MCMC sampler
        n_burn : int
            Burn-in steps
        
        Returns
        -------
        Dict
            Diagnostic metrics
        """
        diagnostics = {}
        
        try:
            # Acceptance fraction
            diagnostics['mean_acceptance'] = float(np.mean(sampler.acceptance_fraction))
            
            # Autocorrelation time (if enough samples)
            try:
                tau = sampler.get_autocorr_time(quiet=True)
                diagnostics['autocorr_time'] = tau.tolist()
                diagnostics['converged'] = np.all(sampler.iteration > 50 * tau)
            except Exception:
                diagnostics['autocorr_time'] = None
                diagnostics['converged'] = None
            
            # Effective sample size
            chain = sampler.get_chain(discard=n_burn, flat=True)
            diagnostics['n_effective'] = len(chain)
            
            # Gelman-Rubin statistic (if arviz available)
            if ARVIZ_AVAILABLE:
                try:
                    # Convert to InferenceData
                    chain_3d = sampler.get_chain(discard=n_burn)
                    param_names = ['D', 'alpha'] if chain_3d.shape[2] == 2 else ['D']
                    
                    idata = az.from_emcee(sampler, var_names=param_names)
                    rhat = az.rhat(idata)
                    diagnostics['rhat'] = {k: float(v) for k, v in rhat.items()}
                    diagnostics['converged_rhat'] = all(v < 1.1 for v in rhat.values())
                except Exception as e:
                    diagnostics['rhat'] = None
                    diagnostics['converged_rhat'] = None
        
        except Exception as e:
            diagnostics['error'] = str(e)
        
        return diagnostics
    
    def analyze_track_bayesian(
        self,
        track: np.ndarray,
        n_walkers: int = 32,
        n_steps: int = 2000,
        estimate_alpha: bool = True,
        return_samples: bool = False,
        backend: str = "auto",
        n_warmup: int = 500,
        rng_seed: int = 42,
        use_gpu: bool = True,
    ) -> Dict:
        """
        High-level API for Bayesian track analysis.
        
        Parameters
        ----------
        track : np.ndarray
            Track coordinates, shape (N, dimensions)
        n_walkers : int
            Number of MCMC walkers
        n_steps : int
            Number of MCMC steps
        estimate_alpha : bool
            Whether to estimate anomalous exponent
        return_samples : bool
            Whether to return full posterior samples
        
        Returns
        -------
        Dict
            Bayesian analysis results with credible intervals
        """
        if len(track) < 3:
            return {
                'success': False,
                'error': 'Need at least 3 positions for Bayesian analysis'
            }
        
        # Calculate displacements
        displacements = np.diff(track, axis=0)
        
        # Run MCMC
        result = self.run_mcmc(
            displacements,
            n_walkers=n_walkers,
            n_steps=n_steps,
            estimate_alpha=estimate_alpha,
            progress=False,
            backend=backend,
            n_warmup=n_warmup,
            rng_seed=rng_seed,
            use_gpu=use_gpu,
        )
        
        if not result['success']:
            return result
        
        # Add track info
        result['track_length'] = len(track)
        result['n_displacements'] = len(displacements)
        backend_used = str(result.get('backend', 'emcee')).upper()
        result['method'] = f'Bayesian_{backend_used}'
        
        # Remove samples if not requested
        if not return_samples:
            result.pop('samples', None)
            result.pop('sampler', None)
        
        return result


def create_arviz_inference_data(
    mcmc_result: Dict,
    param_names: Optional[List[str]] = None
) -> Optional[Any]:
    """
    Convert MCMC results to ArviZ InferenceData for diagnostics.
    
    Parameters
    ----------
    mcmc_result : Dict
        Result from BayesianDiffusionInference.run_mcmc()
    param_names : list of str, optional
        Parameter names
    
    Returns
    -------
    arviz.InferenceData or None
        Inference data object for diagnostics
    """
    if not ARVIZ_AVAILABLE:
        warnings.warn("arviz not available")
        return None
    
    if not mcmc_result.get('success', False):
        return None
    
    samples = mcmc_result.get('samples')
    sampler = mcmc_result.get('sampler')
    backend = str(mcmc_result.get('backend', '')).lower()

    if param_names is None and samples is not None:
        n_params = int(np.asarray(samples).shape[1]) if np.asarray(samples).ndim == 2 else 1
        param_names = ['D', 'alpha'][:n_params]
    elif param_names is None:
        param_names = ['D']

    try:
        if sampler is not None and backend in ('', 'emcee'):
            return az.from_emcee(sampler, var_names=param_names)

        if samples is None:
            return None

        samples_arr = np.asarray(samples)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(-1, 1)
        if samples_arr.ndim != 2 or samples_arr.shape[0] == 0:
            return None

        posterior = {}
        for idx, name in enumerate(param_names):
            if idx >= samples_arr.shape[1]:
                break
            posterior[name] = samples_arr[:, idx][None, :]
        return az.from_dict(posterior=posterior)
    except Exception as e:
        warnings.warn(f"Failed to create InferenceData: {e}")
        return None


def plot_posterior_diagnostics(
    mcmc_result: Dict,
    param_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> Optional[Any]:
    """
    Create diagnostic plots for MCMC results.
    
    Includes:
    - Trace plots
    - Autocorrelation plots
    - Posterior distributions
    - Rank plots
    
    Parameters
    ----------
    mcmc_result : Dict
        Result from run_mcmc()
    param_names : list of str, optional
        Parameter names
    save_path : str, optional
        Path to save figure
    
    Returns
    -------
    matplotlib.figure.Figure or None
        Diagnostic figure
    """
    if not ARVIZ_AVAILABLE:
        warnings.warn("arviz not available for diagnostics")
        return None
    
    idata = create_arviz_inference_data(mcmc_result, param_names)
    if idata is None:
        return None
    
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trace plots
        az.plot_trace(idata, axes=axes[:, 0])
        
        # Posterior distributions
        az.plot_posterior(idata, axes=axes[0, 1])
        
        # Autocorrelation
        az.plot_autocorr(idata, axes=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return fig
    
    except Exception as e:
        warnings.warn(f"Failed to create diagnostic plots: {e}")
        return None
