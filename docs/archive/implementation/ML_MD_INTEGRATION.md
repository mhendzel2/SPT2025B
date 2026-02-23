# Machine Learning & Molecular Dynamics Integration

## Overview

This document describes the new machine learning-based motion classification and molecular dynamics simulation integration features added to SPT2025B. These features enable:

1. **Automated trajectory classification** using machine learning algorithms
2. **Nuclear diffusion simulation** based on multi-compartment nuclear architecture
3. **Quantitative comparison** between experimental SPT data and MD simulations

## Features

### 1. Machine Learning Motion Classification

Automatically classify particle trajectories into distinct motion types using supervised or unsupervised learning.

#### Supported Algorithms

**Supervised Learning:**
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- LSTM Neural Networks

**Unsupervised Learning:**
- K-Means clustering
- DBSCAN (density-based clustering)

#### Feature Extraction

The classifier extracts 22 comprehensive features from each trajectory:

**MSD-Based Features (3):**
- Diffusion coefficient
- Anomalous diffusion exponent (α)
- MSD non-linearity

**Velocity Features (4):**
- Mean speed
- Speed standard deviation
- Maximum speed
- Directional persistence (velocity autocorrelation)

**Geometric Features (4):**
- Radius of gyration
- Asphericity
- Path efficiency
- Confinement ratio

**Dynamic Features (4):**
- Mean turning angle
- Turning angle standard deviation
- Fractal dimension
- Kurtosis of displacement distribution

**Statistical Features (5):**
- X/Y position autocorrelation
- Track straightness
- Bounding box aspect ratio
- Log(number of points)

#### Usage Example

```python
from ml_trajectory_classifier_enhanced import classify_motion_types
import pandas as pd

# Load your tracks
tracks_df = pd.read_csv('tracks.csv')  # Must have columns: track_id, frame, x, y

# Perform unsupervised classification
result = classify_motion_types(
    tracks_df,
    pixel_size=0.1,        # μm per pixel
    frame_interval=0.1,    # seconds
    method='unsupervised',
    model_type='kmeans'
)

print(f"Identified {result['n_classes']} motion classes")
print(f"Silhouette score: {result['clustering_results']['silhouette_score']}")

# Access predicted labels
labels = result['predicted_labels']
track_ids = result['track_ids']
```

For supervised learning (requires ground truth labels):

```python
# Your ground truth labels
labels = np.array(['confined', 'normal', 'directed', ...])

result = classify_motion_types(
    tracks_df,
    pixel_size=0.1,
    frame_interval=0.1,
    method='supervised',
    model_type='random_forest',
    labels=labels
)

print(f"Validation accuracy: {result['training_results']['accuracy']:.2%}")
```

### 2. Nuclear Diffusion Simulation

Simulate particle diffusion in a multi-compartment nuclear environment based on your [nuclear-diffusion-si repository](https://github.com/mhendzel2/nuclear-diffusion-si).

#### Nuclear Compartments

The simulator models 5 distinct nuclear compartments:

1. **Nucleoplasm** (bulk nuclear volume)
   - Viscosity: 1× water
   - Diffusion: Normal Brownian
   - Crowding: 10%

2. **Nucleolus** (ribosome assembly site)
   - Viscosity: 2× water
   - Diffusion: Anomalous subdiffusive (α=0.7)
   - Crowding: 50%

3. **Heterochromatin** (condensed chromatin)
   - Viscosity: 3× water
   - Diffusion: Anomalous subdiffusive (α=0.6)
   - Crowding: 70%
   - Pore size: 50 nm

4. **Speckles** (splicing factor domains)
   - Viscosity: 1.5× water
   - Diffusion: Anomalous subdiffusive (α=0.75)
   - Crowding: 40%

5. **Euchromatin** (active chromatin)
   - Viscosity: 1.2× water
   - Diffusion: Normal Brownian
   - Crowding: 20%

#### Physics Models

**Diffusion Models:**
- **Normal Brownian motion**: MSD ~ 4Dt
- **Continuous-Time Random Walk (CTRW)**: Power-law waiting times for subdiffusion
- **Fractional Brownian motion (fBm)**: Hurst exponent H controls persistence

**Particle Interactions:**
- Electrostatic forces
- Temporary binding to nuclear structures
- Surface chemistry effects (PEGylation reduces friction)
- Size-dependent hindrance in porous media

#### Usage Example

```python
from nuclear_diffusion_simulator import simulate_nuclear_diffusion

# Run simulation
tracks_df, summary = simulate_nuclear_diffusion(
    n_particles=100,
    particle_radius=40,      # nm (20, 40, or 100 typical)
    n_steps=1000,
    time_step=0.001,         # seconds
    temperature=310          # Kelvin (37°C)
)

print(f"Simulated {summary['n_particles']} particles for {summary['simulation_time']:.2f}s")
print(f"Compartment distribution: {summary['by_compartment']}")

# Tracks are ready for analysis
# Columns: track_id, frame, x, y, compartment
```

#### Custom Geometry from Images

You can create nuclear geometry from DNA and splicing factor fluorescence images:

```python
from nuclear_diffusion_simulator import NuclearGeometry
import numpy as np

# Load and normalize your images (0-1 range)
dna_image = load_and_normalize('dna_channel.tif')
sf_image = load_and_normalize('sf_channel.tif')

# Create geometry from images
geometry = NuclearGeometry.from_images(
    dna_image, 
    sf_image,
    dna_thresholds=(0.2, 0.5, 0.8),  # low, mid, high
    sf_thresholds=(0.2, 0.5, 0.8)
)

# Use in simulation
from nuclear_diffusion_simulator import PhysicsEngine, NuclearDiffusionSimulator, ParticleProperties

physics = PhysicsEngine(temperature=310, time_step=0.001)
simulator = NuclearDiffusionSimulator(geometry, physics)

particle_props = ParticleProperties(radius=40)
simulator.add_particles(100, particle_props)
simulator.run(1000)

tracks_df = simulator.get_tracks_dataframe()
```

### 3. MD-SPT Comparison Framework

Quantitatively compare simulated MD trajectories with experimental SPT data using statistical tests and visualization.

#### Comparison Metrics

**Diffusion Analysis:**
- Diffusion coefficient estimation
- Bootstrap confidence intervals (95%)
- Mann-Whitney U test for statistical significance

**MSD Comparison:**
- Pearson correlation between MSD curves
- Root mean squared error (RMSE)
- Mean absolute error (MAE)

**Confinement Metrics:**
- Radius of gyration
- Confinement ratio
- Asphericity

**Compartment-Specific Analysis:**
- Per-compartment diffusion coefficients
- Statistical tests for each compartment
- Particle residence time distributions

#### Usage Example

```python
from md_spt_comparison import compare_md_with_spt
from nuclear_diffusion_simulator import simulate_nuclear_diffusion
import pandas as pd

# Load experimental data
tracks_spt = pd.read_csv('experimental_tracks.csv')

# Generate comparable MD simulation
tracks_md, _ = simulate_nuclear_diffusion(
    n_particles=len(tracks_spt['track_id'].unique()),
    particle_radius=40,
    n_steps=int(tracks_spt.groupby('track_id').size().mean()),
    time_step=0.1
)

# Perform comprehensive comparison
result = compare_md_with_spt(
    tracks_md, 
    tracks_spt,
    pixel_size=0.1,
    frame_interval=0.1,
    analyze_compartments=True
)

# View results
print(f"Diffusion MD: {result['diffusion_md']:.4f} μm²/s")
print(f"Diffusion SPT: {result['diffusion_spt']:.4f} μm²/s")
print(f"Ratio (MD/SPT): {result['diffusion_ratio']:.2f}")
print(f"p-value: {result['statistical_test']['p_value']:.4f}")
print(f"MSD correlation: {result['msd_comparison']['correlation']:.3f}")

# Interpretation
print(f"\nAgreement: {result['summary']['diffusion_agreement']}")
print(f"Recommendation: {result['summary']['recommendation']}")

# Visualizations
for name, fig in result['figures'].items():
    fig.show()
```

### 4. Report Generator Integration

All new features are integrated into the Enhanced Report Generator for automated analysis workflows.

#### Available Analyses

In the report generator interface, you'll find three new analyses:

**ML Motion Classification** (Machine Learning category)
- Unsupervised clustering of trajectory types
- Feature importance visualization
- PCA projection plots

**MD Simulation Comparison** (Simulation category)
- Run nuclear diffusion simulation
- Compare with experimental data
- Statistical significance testing
- Compartment-specific analysis

**Nuclear Diffusion Simulation** (Simulation category)
- Standalone nuclear diffusion simulation
- Trajectory visualization
- MSD analysis
- Compartment distribution

#### Usage in Streamlit

```python
import streamlit as st
from enhanced_report_generator import EnhancedSPTReportGenerator

# In your Streamlit app
generator = EnhancedSPTReportGenerator()
generator.display_enhanced_analysis_interface()

# Or generate batch report programmatically
selected_analyses = [
    'basic_statistics',
    'diffusion_analysis',
    'ml_classification',
    'md_comparison',
    'nuclear_diffusion_sim'
]

generator.generate_batch_report(
    tracks_df,
    selected_analyses=selected_analyses,
    condition_name="Experimental_Condition_1"
)
```

## Installation

### Required Dependencies

The new features require additional packages:

```bash
# Machine Learning (required)
pip install scikit-learn>=1.3.0

# Deep Learning (optional, for LSTM classification)
pip install tensorflow>=2.12.0

# Already included in requirements.txt
pip install numpy pandas scipy plotly
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Verifying Installation

Run the comprehensive test suite:

```bash
python test_ml_md_integration.py
```

Expected output:
```
✓ ML Classifier tests passed!
✓ Nuclear Diffusion Simulator tests passed!
✓ MD-SPT Comparison tests passed!
✓ Report Generator Integration tests passed!
✓ ALL TESTS PASSED!
```

## Advanced Usage

### Custom Particle Properties

```python
from nuclear_diffusion_simulator import ParticleProperties, simulate_nuclear_diffusion

# Charged nanoparticle
charged_particle = ParticleProperties(
    radius=40,
    charge=5.0,                    # +5 elementary charges
    surface_chemistry='positive',
    binding_affinity=0.3           # 30% binding probability
)

# PEGylated particle (reduced friction)
pegylated_particle = ParticleProperties(
    radius=100,
    surface_chemistry='pegylated'
)
```

### Custom Medium Properties

```python
from nuclear_diffusion_simulator import MediumProperties, DiffusionModel

# Highly viscous gel
gel_medium = MediumProperties(
    viscosity=5.0,
    pore_size=30,                  # nm
    diffusion_model=DiffusionModel.ANOMALOUS_SUBDIFFUSIVE,
    alpha=0.5,
    tau=0.02
)

# Fractional Brownian motion
fbm_medium = MediumProperties(
    viscosity=1.0,
    diffusion_model=DiffusionModel.FRACTIONAL_BROWNIAN,
    alpha=0.3                      # Hurst exponent (anti-persistent)
)
```

### Feature-Based Analysis

```python
from ml_trajectory_classifier_enhanced import extract_trajectory_features
import pandas as pd

# Extract features from single track
track_df = tracks_df[tracks_df['track_id'] == 0]
features = extract_trajectory_features(track_df, pixel_size=0.1, frame_interval=0.1)

# Features include:
# [D_coeff, alpha, msd_nonlinearity, mean_speed, std_speed, ...]
print(f"Diffusion coefficient: {features[0]:.4f} μm²/s")
print(f"Anomalous exponent: {features[1]:.3f}")
print(f"Mean speed: {features[3]:.4f} μm/s")
```

## Troubleshooting

### Common Issues

**1. TensorFlow import warnings**
```
Import "tensorflow.keras.models" could not be resolved
```
- This is expected if TensorFlow is not installed
- LSTM classification requires TensorFlow
- Other classifiers work without it

**2. Low silhouette score in clustering**
```
Silhouette score: 0.15
```
- Your data may not have distinct motion classes
- Try different number of clusters
- Consider supervised learning with labeled data

**3. MD-SPT comparison shows poor agreement**
- Check pixel_size and frame_interval match experimental conditions
- Adjust simulation parameters (viscosity, crowding)
- Ensure sufficient track length and number

**4. Memory issues with large datasets**
- Reduce number of tracks or track length
- Use batch processing
- Increase system RAM allocation

## Performance Considerations

### Computational Cost

**ML Classification:**
- Feature extraction: ~1ms per track
- K-Means clustering: ~100ms for 1000 tracks
- Random Forest training: ~1s for 1000 tracks
- LSTM training: ~10s for 1000 tracks

**Nuclear Diffusion Simulation:**
- ~1ms per particle per time step
- 100 particles, 1000 steps: ~100s
- Scales linearly with particles and steps

**MD-SPT Comparison:**
- Diffusion calculation: ~10ms per dataset
- Bootstrap resampling (1000): ~10s
- Compartment analysis: +5s per compartment

### Optimization Tips

1. **Use fewer particles for prototyping**
   ```python
   tracks_md, _ = simulate_nuclear_diffusion(n_particles=20, n_steps=100)
   ```

2. **Reduce bootstrap iterations for faster testing**
   ```python
   # Edit md_spt_comparison.py line 169:
   n_bootstrap = 100  # instead of 1000
   ```

3. **Cache results for repeated analyses**
   ```python
   # Results are automatically cached in enhanced_report_generator
   ```

## Scientific Background

### Nuclear Compartments

The multi-compartment nuclear model is based on:

**Nucleolus:**
- Low DNA intensity, low splicing factor intensity
- High macromolecular crowding (~300 mg/mL)
- Anomalous subdiffusion observed experimentally

**Heterochromatin:**
- High DNA intensity, low splicing factor intensity
- Porous gel-like structure
- Strong size-dependent hindrance

**Speckles:**
- High splicing factor intensity, low DNA intensity
- Liquid-like phase-separated compartments
- Moderate subdiffusion

**Euchromatin:**
- Moderate DNA intensity
- Active transcription sites
- Near-normal diffusion

### Anomalous Diffusion Models

**CTRW (Continuous-Time Random Walk):**
- Power-law waiting time distribution: ψ(t) ~ t^(-1-α)
- Models transient binding and unbinding
- Common in crowded/structured environments

**fBm (Fractional Brownian Motion):**
- Characterized by Hurst exponent H
- H < 0.5: subdiffusive (anti-persistent)
- H = 0.5: normal diffusion
- H > 0.5: superdiffusive (persistent)

## References

### Nuclear Diffusion Model

Based on the [nuclear-diffusion-si repository](https://github.com/mhendzel2/nuclear-diffusion-si):

- Multi-compartment nuclear architecture
- DNA and splicing factor-based segmentation
- Anomalous diffusion physics (CTRW, fBm)
- Particle-environment interactions

### Key Papers

1. **Anomalous Diffusion:**
   - Höfling & Franosch (2013). "Anomalous transport in the crowded world of biological cells." *Rep. Prog. Phys.*

2. **Nuclear Organization:**
   - Bancaud et al. (2009). "Molecular crowding affects diffusion and binding of nuclear proteins." *Mol. Cell*

3. **Single Particle Tracking:**
   - Di Rienzo et al. (2014). "Probing short-range protein Brownian motion in the cytoplasm of living cells." *Nat. Commun.*

4. **Machine Learning for SPT:**
   - Granik & Weiss (2019). "Single-Particle Diffusion Characterization by Deep Learning." *Biophys. J.*

## Contributing

To extend these features:

1. **Add new ML models:**
   - Edit `ml_trajectory_classifier_enhanced.py`
   - Add to `TrajectoryClassifier._create_model()`

2. **Add new diffusion models:**
   - Edit `nuclear_diffusion_simulator.py`
   - Add to `PhysicsEngine.generate_displacement()`

3. **Add new comparison metrics:**
   - Edit `md_spt_comparison.py`
   - Add functions and update `compare_md_with_spt()`

4. **Add to report generator:**
   - Edit `enhanced_report_generator.py`
   - Register in `available_analyses` dict
   - Implement `_analyze_*` and `_plot_*` methods

## Support

For issues and questions:

1. Check this documentation
2. Run `python test_ml_md_integration.py` for diagnostics
3. Check console output for error messages
4. Review logs in `debug.log`

## License

These features are part of SPT2025B and inherit the project license.

---

**Last Updated:** January 2025  
**Version:** 1.0.0  
**Author:** SPT2025B Development Team
