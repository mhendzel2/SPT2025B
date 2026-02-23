# Advanced Trajectory Analysis Features

This document describes the advanced trajectory analysis features implemented in SPT2025B, including bias-aware population inference, Bayesian posterior analysis, ML-based trajectory classification, and HMM with localization error.

## Overview

Four major advanced analysis modules have been added to SPT2025B:

1. **Spot-On Style Population Inference** - Multi-population diffusion analysis with bias correction
2. **Bayesian Posterior Analysis** - MCMC-based parameter inference with credible intervals
3. **ML Trajectory Classification** - Machine learning-based motion type classification
4. **HMM with Localization Error** - Already implemented via `ihmm_blur_analysis.py`

## 1. Bias-Aware Diffusion Population Inference (Spot-On Style)

### Overview
Based on Hansen et al. (2018) *eLife* 7:e33125, this module implements robust multi-population diffusion analysis that corrects for experimental biases.

### Key Features
- **Out-of-focus correction**: Accounts for particles diffusing out of the axial detection range
- **Motion blur correction**: Corrects for finite camera exposure time
- **Localization noise correction**: Handles Gaussian localization errors
- **Model selection**: Automatically determines optimal number of populations via BIC

### Usage

#### Python API
```python
from biased_inference import SpotOnPopulationInference
import numpy as np

# Initialize with experimental parameters
spoton = SpotOnPopulationInference(
    frame_interval=0.1,          # seconds
    pixel_size=0.1,              # micrometers
    localization_error=0.03,     # micrometers (30 nm)
    exposure_time=0.05,          # seconds
    axial_detection_range=1.0,  # micrometers
    dimensions=2
)

# Calculate jump distances from tracks
jump_distances = []
for track in tracks:
    displacements = np.diff(track, axis=0)
    jumps = np.sqrt(np.sum(displacements**2, axis=1))
    jump_distances.extend(jumps)

jump_distances = np.array(jump_distances)

# Fit 2-population model
result = spoton.fit_populations(
    jump_distances,
    n_populations=2
)

if result['success']:
    print(f"D values: {result['D_values']}")
    print(f"Fractions: {result['fractions']}")
    print(f"BIC: {result['BIC']}")

# Or use automatic model selection
selection = spoton.model_selection(
    jump_distances,
    max_populations=4
)

optimal_n = selection['optimal_n']
best_result = selection['best_result']
```

#### In Report Generator
The Spot-On analysis is available in the Enhanced Report Generator under "2025 Methods" → "Spot-On Population Inference".

### Output
- `D_values`: Diffusion coefficients for each population (µm²/s)
- `fractions`: Fraction of particles in each population
- `D_std`: Standard errors for D values
- `fraction_std`: Standard errors for fractions
- `log_likelihood`: Maximum log-likelihood
- `BIC`: Bayesian Information Criterion
- `blur_corrected`: Boolean indicating motion blur correction
- `out_of_focus_corrected`: Boolean indicating out-of-focus correction

### Required Metadata
- Frame interval (dt)
- Pixel size
- Localization error (typical: 20-50 nm for fluorescence)
- Exposure time (important when > 30% of frame interval)
- Axial detection range (for 2D imaging of 3D diffusion, typical: 0.5-2 µm)

### Limitations
- Requires accurate experimental metadata
- Assumes isotropic diffusion
- Best with > 1000 jump distances for robust inference
- Model selection can be slow for large datasets

## 2. Bayesian Trajectory Inference

### Overview
Implements full Bayesian inference for trajectory analysis using MCMC sampling (emcee). Provides posterior distributions and credible intervals rather than point estimates.

### Key Features
- **MCMC sampling**: Uses affine-invariant ensemble sampler (emcee)
- **Posterior intervals**: 95% credible intervals for all parameters
- **Convergence diagnostics**: R-hat, autocorrelation time, acceptance rates
- **ArviZ integration**: Advanced diagnostics and trace plots (optional)
- **Informative priors**: Weakly informative priors prevent unphysical values

### Usage

#### Python API
```python
from bayesian_trajectory_inference import BayesianDiffusionInference

# Initialize
bayes_inf = BayesianDiffusionInference(
    frame_interval=0.1,
    localization_error=0.03,
    exposure_time=0.05,
    dimensions=2
)

# Analyze single track
result = bayes_inf.analyze_track_bayesian(
    track,
    n_walkers=32,
    n_steps=2000,
    estimate_alpha=True,  # Estimate anomalous exponent
    return_samples=True
)

if result['success']:
    print(f"D median: {result['D_median']} µm²/s")
    print(f"95% CI: [{result['D_credible_interval'][0]:.3f}, {result['D_credible_interval'][1]:.3f}]")
    print(f"Diagnostics: {result['diagnostics']}")

# Access posterior samples
samples = result['samples']  # Shape: (n_samples, n_params)
```

#### Diagnostic Plots (requires ArviZ)
```python
from bayesian_trajectory_inference import plot_posterior_diagnostics

# Create diagnostic plots
fig = plot_posterior_diagnostics(
    result,
    param_names=['D', 'alpha'],
    save_path='diagnostics.png'
)
```

### Output
- `D_median`: Median of posterior distribution
- `D_mean`: Mean of posterior
- `D_std`: Standard deviation of posterior
- `D_credible_interval`: (lower, upper) 95% CI
- `alpha_median`, `alpha_mean`, etc.: Same for anomalous exponent (if estimated)
- `samples`: Full posterior samples (if requested)
- `diagnostics`: Dictionary with convergence metrics

### Diagnostics
- `mean_acceptance`: Should be 0.2-0.5 for good mixing
- `autocorr_time`: Autocorrelation time (lower is better)
- `rhat`: Gelman-Rubin statistic (< 1.1 indicates convergence)
- `converged`: Boolean based on diagnostics

### Installation Requirements
```bash
pip install emcee>=3.1.0  # Required
pip install arviz>=0.22.0  # Optional, for diagnostics
```

### Computational Cost
- MCMC is ~10-100x slower than MLE
- Recommended for: Important tracks, final analysis, publication
- Not recommended for: Real-time analysis, exploratory work

### Priors
Default weakly informative priors:
- D ~ LogNormal(mean=-2, std=2) in µm²/s
- alpha ~ Beta(2, 2) scaled to [0.5, 2.0]

Custom priors can be specified via `prior_config` parameter.

## 3. ML Trajectory Classification

### Overview
Machine learning-based classification of trajectories into motion types using handcrafted features and Random Forest (sklearn) or Transformer models (PyTorch, when available).

### Motion Classes
- **Brownian**: Normal diffusion
- **Confined**: Diffusion within boundaries
- **Directed**: Active transport with drift
- **Anomalous (sub)**: Subdiffusive motion (α < 1)
- **Anomalous (super)**: Superdiffusive motion (α > 1)

### Key Features
- **Synthetic pre-training**: Generates large labeled dataset with domain randomization
- **Domain randomization**: Varies D, noise, confinement size to improve generalization
- **Feature extraction**: 20 handcrafted features (MSD, straightness, kurtosis, etc.)
- **Calibrated probabilities**: Returns class probabilities, not just labels
- **sklearn fallback**: Works without PyTorch

### Usage

#### Synthetic Data Generation
```python
from transformer_trajectory_classifier import SyntheticTrajectoryGenerator

generator = SyntheticTrajectoryGenerator(
    dt=0.1,
    dimensions=2,
    randomize_params=True  # Domain randomization
)

# Generate training dataset
trajectories, labels = generator.generate_dataset(
    n_per_class=1000,
    n_steps_range=(20, 100),
    classes=['brownian', 'confined', 'directed']
)
```

#### Training and Classification
```python
from transformer_trajectory_classifier import (
    train_trajectory_classifier,
    classify_trajectories
)

# Train classifier on synthetic data
classifier, train_result = train_trajectory_classifier(
    trajectories,
    labels,
    dt=0.1,
    method='sklearn'  # or 'transformer' if PyTorch available
)

print(f"Training accuracy: {train_result['train_accuracy']}")

# Classify real tracks
predictions, probabilities = classify_trajectories(
    classifier,
    real_trajectories,
    return_proba=True
)

# predictions: ['brownian', 'directed', ...]
# probabilities: array of shape (n_tracks, n_classes)
```

#### Feature Extraction
```python
from transformer_trajectory_classifier import extract_trajectory_features

features = extract_trajectory_features(track, dt=0.1)
# Returns 20-dimensional feature vector
```

### Features Extracted
1. MSD at lags 1, 2, 3, 5
2. Anomalous exponent (α)
3. Straightness (end-to-end / path length)
4. Asymmetry (std / mean of squared displacements)
5. Kurtosis
6. Radius of gyration
7. Velocity autocorrelation
8. ... (20 total)

### Performance
- Training time: ~10 seconds for 3000 tracks (sklearn)
- Classification time: < 1 ms per track
- Typical accuracy: 85-95% on synthetic data, 70-85% on real data

### Limitations
- **Synthetic-to-real gap**: Model trained on synthetic may not transfer perfectly
- **Feature engineering**: Handcrafted features may miss subtle patterns
- **Short tracks**: Needs ≥ 20 points for reliable features
- **Mixed modes**: Struggles with tracks exhibiting multiple motion types

### Future Extensions
- Transformer encoder (when PyTorch available)
- Contrastive learning pre-training
- Fine-tuning on labeled real data
- Uncertainty quantification

## 4. HMM with Localization Error

### Overview
Already implemented via `ihmm_blur_analysis.py`. Infinite HMM with blur-aware emission models for automatic state segmentation.

### Key Features
- **Blur-aware emissions**: Accounts for motion blur and localization noise
- **Automatic state number**: Uses HDP prior to infer number of states
- **Variational Bayes**: Efficient inference via variational EM
- **Per-state D estimation**: Diffusion coefficient for each state

### Usage
See `ihmm_blur_analysis.py` documentation. Already integrated in Enhanced Report Generator under "iHMM State Segmentation".

## Integration with Report Generator

All modules are integrated into `enhanced_report_generator.py`:

### Available in Report Interface
1. **Spot-On Population Inference** (Category: 2025 Methods)
2. **Bayesian Posterior Analysis** (Category: 2025 Methods)
3. **ML Trajectory Classification** (Category: Machine Learning)
4. **iHMM State Segmentation** (Category: 2025 Methods) - already existed

### Accessing in UI
When generating reports in the Streamlit app:
1. Navigate to "Report Generation" tab
2. Expand "Advanced Analyses" or "2025 Methods" section
3. Select desired modules
4. Click "Generate Report"

### Batch Processing
All modules support batch analysis via the Enhanced Report Generator's batch mode.

## Testing

Comprehensive test suite in `test_advanced_trajectory_analysis.py`:

```bash
# Run all tests
python test_advanced_trajectory_analysis.py

# Expected output: ALL TESTS PASSED ✓
```

Tests verify:
- Bias correction accuracy
- Population inference on synthetic mixtures
- Bayesian MCMC convergence (if emcee available)
- ML classifier training and prediction

## Dependencies

### Required
- numpy >= 1.24.0
- pandas >= 2.0.0
- scipy >= 1.14.0
- scikit-learn >= 1.5.0

### Optional (for full functionality)
- emcee >= 3.1.0 (Bayesian MCMC)
- arviz >= 0.22.0 (Posterior diagnostics)
- torch >= 2.0.0 (Transformer models)
- fbm >= 0.3.0 (Fractional Brownian motion generation)

Install optional dependencies:
```bash
pip install emcee arviz
# For transformer (optional):
pip install torch torchvision
```

## References

1. **Spot-On**: Hansen et al. (2018) "Robust model-based analysis of single-particle tracking experiments with Spot-On" *eLife* 7:e33125

2. **Bayesian Trajectory Analysis**: Similar to bayes_traj (JOSS 2025) and related Bayesian SPT methods

3. **Bias Correction**: Berglund (2010) "Statistics of camera-based single-particle tracking" *Physical Review E* 82:011917

4. **iHMM**: Lindén et al. (2017) "Variational Bayesian framework for analysis of single-particle trajectories" *Nature Methods* (PMC6050756)

5. **ML Classification**: Various methods from recent SPT literature, adapted with domain randomization

## Best Practices

### When to Use Each Method

**Spot-On Population Inference**:
- Heterogeneous samples with multiple diffusing species
- Need to quantify population fractions
- Have accurate experimental metadata
- Large datasets (> 1000 jumps)

**Bayesian Posterior Analysis**:
- Need rigorous uncertainty quantification
- Publication-quality parameter estimates
- Willing to wait for MCMC (slower)
- Track-by-track analysis important

**ML Trajectory Classification**:
- Exploratory analysis
- Need fast, automated classification
- Don't know exact motion model
- Have diverse motion types

**iHMM State Segmentation**:
- Tracks exhibit switching between states
- Need to identify state transitions
- Want automatic state number determination
- Have localization noise and motion blur

### Workflow Recommendation

1. **Exploratory Phase**:
   - Use ML classification for quick overview
   - Run Spot-On for population structure
   
2. **Detailed Analysis**:
   - Use iHMM for tracks with switching
   - Apply Bayesian inference to key tracks
   
3. **Publication**:
   - Report Bayesian credible intervals
   - Include Spot-On population fractions
   - Show ML classification as supplementary

## Troubleshooting

### "emcee not available"
```bash
pip install emcee
```

### "Optimization failed" in Spot-On
- Check if jump distances are reasonable (not all zeros)
- Try reducing max_populations
- Verify metadata (frame_interval, pixel_size) are correct

### "MCMC not converging"
- Increase n_steps (try 5000)
- Check diagnostics: acceptance should be 0.2-0.5
- Verify track has enough data points (≥ 20)

### ML classification accuracy low
- Generate more synthetic training data
- Enable domain randomization
- Check that dt matches between training and test data

## Contact

For issues, questions, or contributions, please open an issue on the SPT2025B GitHub repository.

## License

These modules are part of SPT2025B and follow the same license terms.
