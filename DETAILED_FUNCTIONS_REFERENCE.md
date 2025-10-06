# Detailed Analysis Functions Reference

## Overview
This document provides detailed information about each of the 16 analysis functions available in SPT2025B.

---

## 1. Basic Track Statistics
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_basic_statistics()`  
**Visualization**: `_plot_basic_statistics()`  
**Category**: Basic  
**Priority**: 1 (Essential)

### What It Does
Calculates fundamental track metrics including:
- Track lengths (number of frames per track)
- Total displacements (start to end distance)
- Mean velocities
- Statistical distributions of all metrics

### Output Structure
```python
{
    'success': True,
    'statistics_df': DataFrame,        # Per-track statistics
    'ensemble_statistics': {           # Aggregate statistics
        'mean_track_length': float,
        'median_displacement': float,
        'velocity_stats': dict,
        ...
    }
}
```

### Visualization
4-panel histogram showing:
- Track length distribution
- Displacement distribution
- Velocity distribution
- Summary statistics table

---

## 2. Diffusion Analysis
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_diffusion()`  
**Visualization**: `_plot_diffusion()`  
**Category**: Core Physics  
**Priority**: 2 (High)

### What It Does
Performs mean squared displacement (MSD) analysis:
- Calculates ensemble-averaged MSD
- Fits MSD to power law: MSD = 4Dt^α
- Extracts diffusion coefficients (D)
- Determines anomalous diffusion exponent (α)
- Classifies diffusion regime (normal, sub-, super-diffusive)

### Output Structure
```python
{
    'success': True,
    'msd_data': array,                 # Time lag vs MSD
    'track_results': {                 # Per-track results
        'track_id': {
            'D': float,                # Diffusion coefficient
            'alpha': float,            # Scaling exponent
            'regime': str              # Classification
        }
    },
    'ensemble_results': {              # Ensemble metrics
        'mean_D': float,
        'median_alpha': float,
        ...
    }
}
```

### Visualization
Multi-trace plot showing:
- MSD curves for each track
- Ensemble-averaged MSD
- Power law fit
- Regime classification annotations

---

## 3. Motion Classification
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_motion()`  
**Visualization**: `_plot_motion()`  
**Category**: Core Physics  
**Priority**: 2 (High)

### What It Does
Classifies particle motion into categories:
- **Brownian**: Normal random walk (α ≈ 1)
- **Subdiffusive**: Hindered motion (α < 1)
- **Superdiffusive**: Active or flow-driven (α > 1)
- **Confined**: Motion restricted to region
- **Directed**: Persistent directional movement

### Classification Criteria
- Uses MSD scaling exponent (α)
- Confinement analysis (radius of gyration)
- Directionality index
- Statistical tests for each regime

### Output Structure
```python
{
    'success': True,
    'track_results': {                 # Per-track classification
        'track_id': {
            'motion_type': str,
            'confidence': float,
            'alpha': float,
            'metrics': dict
        }
    },
    'ensemble_results': {              # Population statistics
        'motion_type_counts': dict,
        'predominant_type': str,
        ...
    }
}
```

---

## 4. Spatial Organization
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_clustering()`  
**Visualization**: `_plot_clustering()`  
**Category**: Spatial Analysis  
**Priority**: 3 (Medium)

### What It Does
Analyzes spatial distribution and clustering:
- DBSCAN clustering at each time frame
- Tracks cluster dynamics over time
- Calculates spatial correlations
- Territory/domain analysis
- Cluster persistence and merging

### Clustering Parameters
- Epsilon (ε): Maximum distance for neighborhood
- MinPts: Minimum points to form cluster
- Adaptive based on point density

### Output Structure
```python
{
    'success': True,
    'frames_analyzed': int,
    'frame_results': {                 # Frame-by-frame clusters
        'frame': {
            'n_clusters': int,
            'cluster_labels': array,
            'cluster_centers': array
        }
    },
    'cluster_dynamics': {              # Temporal evolution
        'persistent_clusters': int,
        'average_lifetime': float,
        ...
    }
}
```

---

## 5. Anomaly Detection
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_anomalies()`  
**Visualization**: `_plot_anomalies()`  
**Category**: Machine Learning  
**Priority**: 4 (Advanced)

### What It Does
Identifies outlier trajectories using ML:
- Feature extraction (MSD, velocity, curvature, etc.)
- Isolation Forest algorithm
- Statistical outlier detection
- Anomalous behavior classification

### Features Analyzed
- Diffusion coefficient
- Track length
- Step size distribution
- Turning angle statistics
- Velocity autocorrelation

### Output Structure
```python
{
    'success': True,
    'anomalous_tracks': list,          # Track IDs flagged
    'anomaly_scores': dict,            # Outlier scores per track
    'features': DataFrame,             # Feature matrix
    'classification': {
        'n_normal': int,
        'n_anomalous': int,
        'threshold': float
    }
}
```

---

## 6. Microrheology Analysis
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_microrheology()`  
**Visualization**: `_plot_microrheology()`  
**Category**: Biophysical Models  
**Priority**: 3 (Medium)

### What It Does
Calculates viscoelastic properties from particle motion:
- **Storage modulus (G')**: Elastic component
- **Loss modulus (G'')**: Viscous component
- **Complex viscosity (η*)**: Frequency-dependent
- **Tan δ**: Loss tangent (G''/G')

### Method
Uses Generalized Stokes-Einstein Relation (GSER):
```
G*(ω) = (k_B T) / (π a ⟨Δr²(1/ω)⟩ Γ(1+α(ω)))
```
where a is particle radius, ω is frequency

### Output Structure
```python
{
    'success': True,
    'frequencies': array,              # Angular frequencies
    'G_prime': array,                  # Storage modulus
    'G_double_prime': array,           # Loss modulus
    'complex_viscosity': array,        # Complex viscosity
    'tan_delta': array,                # Loss tangent
    'summary': {
        'plateau_modulus': float,
        'crossover_frequency': float,
        'material_classification': str
    }
}
```

---

## 7. Intensity Analysis
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_intensity()`  
**Visualization**: `_plot_intensity()`  
**Category**: Photophysics  
**Priority**: 3 (Medium)

### What It Does
Analyzes fluorescence intensity dynamics:
- Photobleaching kinetics
- Intensity fluctuations (CV)
- Binding/unbinding events
- Multi-channel correlation (if applicable)

### Requirements
Data must have intensity columns (e.g., `mean_intensity_ch1`)

### Output Structure
```python
{
    'success': True,
    'intensity_column': str,           # Column used
    'track_intensities': {             # Per-track data
        'track_id': {
            'mean': float,
            'std': float,
            'cv': float,              # Coefficient of variation
            'bleaching_rate': float,   # Exponential decay
            'time_series': array
        }
    },
    'ensemble_statistics': {
        'mean_intensity': float,
        'bleaching_half_life': float,
        ...
    }
}
```

---

## 8. Confinement Analysis
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_confinement()`  
**Visualization**: `_plot_confinement()`  
**Category**: Spatial Analysis  
**Priority**: 3 (Medium)

### What It Does
Detects and characterizes confined motion:
- Radius of gyration (Rg)
- Confinement ratio (Rg² / 4Dt)
- Boundary detection
- Escape event analysis
- Dwell time in confined regions

### Confinement Criteria
- Rg² / MSD ratio < threshold
- Plateau in MSD at long times
- Return probability analysis

### Output Structure
```python
{
    'success': True,
    'confined_tracks': list,           # Track IDs showing confinement
    'track_analysis': {
        'track_id': {
            'radius_of_gyration': float,
            'confinement_ratio': float,
            'is_confined': bool,
            'confinement_strength': float,
            'boundary_estimate': tuple  # (x, y, radius)
        }
    },
    'summary': {
        'fraction_confined': float,
        'mean_confinement_size': float,
        ...
    }
}
```

---

## 9. Velocity Correlation
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_velocity_correlation()`  
**Visualization**: `_plot_velocity_correlation()`  
**Category**: Core Physics  
**Priority**: 3 (Medium)

### What It Does
Analyzes velocity autocorrelation function (VACF):
- VACF(τ) = ⟨v(t)·v(t+τ)⟩
- Persistence length extraction
- Memory effects detection
- Directional correlation analysis

### Physics Interpretation
- VACF > 0: Persistent motion
- VACF < 0: Anti-persistent (corralled)
- VACF ≈ 0: Memoryless (Brownian)

### Output Structure
```python
{
    'success': True,
    'time_lags': array,                # Time lags
    'vacf': array,                     # VACF values
    'persistence_length': float,       # Extracted from decay
    'decorrelation_time': float,       # Time to VACF=0
    'memory_coefficient': float,       # Strength of memory
    'track_vacf': {                    # Per-track VACF
        'track_id': array
    }
}
```

---

## 10. Multi-Particle Interactions
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_particle_interactions()`  
**Visualization**: `_plot_particle_interactions()`  
**Category**: Advanced Statistics  
**Priority**: 4 (Advanced)

### What It Does
Analyzes interactions between particles:
- Pair correlation function g(r)
- Nearest neighbor distances
- Collective motion metrics
- Interaction networks
- Crowding effects

### Metrics Calculated
- **g(r)**: Radial distribution function
- **Coordination number**: Average neighbors
- **Interaction radius**: Distance of first peak in g(r)
- **Clustering coefficient**: Network connectivity

### Output Structure
```python
{
    'success': True,
    'distances': array,                # r values
    'pair_correlation': array,         # g(r) values
    'nearest_neighbors': {             # Per-particle NN analysis
        'track_id': {
            'nn_distance': float,
            'n_neighbors': int,
            'local_density': float
        }
    },
    'collective_metrics': {
        'coordination_number': float,
        'interaction_radius': float,
        'clustering_coefficient': float,
        ...
    }
}
```

---

## 11. Changepoint Detection
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_changepoints()`  
**Visualization**: `_plot_changepoints()`  
**Category**: Advanced Statistics  
**Priority**: 3 (Medium)  
**Requires**: `changepoint_detection` module

### What It Does
Detects regime changes in motion:
- Bayesian changepoint detection
- Switches in diffusion coefficient
- Transitions in motion type
- Temporal segmentation

### Algorithm
Uses Bayesian Online Changepoint Detection (BOCPD) or similar methods

### Output Structure
```python
{
    'success': True,
    'changepoints': {                  # Per-track changepoints
        'track_id': {
            'frames': array,           # Frame numbers of changes
            'segments': list,          # Segment properties
            'n_regimes': int
        }
    },
    'regime_analysis': {
        'predominant_n_regimes': int,
        'average_segment_length': float,
        ...
    }
}
```

---

## 12. Polymer Physics Models
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_polymer_physics()`  
**Visualization**: `_plot_polymer_physics()`  
**Category**: Biophysical Models  
**Priority**: 4 (Advanced)  
**Requires**: `biophysical_models` module

### What It Does
Fits polymer physics models to trajectory data:
- **Rouse model**: Free-draining polymer
- **Zimm model**: Polymer with hydrodynamic interactions
- Scaling exponent analysis
- Persistence length estimation

### Rouse Model
MSD(t) = (k_B T / ζ) t^(1/2) for long times

### Output Structure
```python
{
    'success': True,
    'rouse_fit': {
        'diffusion_coefficient': float,
        'friction_coefficient': float,
        'fit_quality': float
    },
    'scaling_analysis': {
        'short_time_exponent': float,
        'long_time_exponent': float,
        ...
    }
}
```

---

## 13. Energy Landscape Mapping
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_energy_landscape()`  
**Visualization**: `_plot_energy_landscape()`  
**Category**: Biophysical Models  
**Priority**: 4 (Advanced)  
**Requires**: `biophysical_models` module

### What It Does
Constructs potential energy landscape from particle density:
- 2D probability density function
- Convert to free energy: F(x,y) = -k_B T ln(ρ(x,y))
- Identify energy wells and barriers
- Measure well depths and barrier heights

### Method
Boltzmann inversion of probability distribution

### Output Structure
```python
{
    'success': True,
    'energy_landscape': array,         # 2D energy map
    'x_coords': array,
    'y_coords': array,
    'wells': list,                     # Energy minimum locations
    'barriers': list,                  # Barrier locations/heights
    'summary': {
        'n_wells': int,
        'mean_well_depth': float,
        'mean_barrier_height': float,
        ...
    }
}
```

---

## 14. Active Transport Detection
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_active_transport()`  
**Visualization**: `_plot_active_transport()`  
**Category**: Biophysical Models  
**Priority**: 4 (Advanced)  
**Requires**: `biophysical_models` module

### What It Does
Identifies directional/ballistic motion segments:
- Run length analysis
- Velocity persistence
- Motor-driven transport detection
- Directional bias quantification

### Transport Criteria
- Velocity > threshold
- Persistent direction (>2 frames)
- Displacement above diffusive expectation

### Output Structure
```python
{
    'success': True,
    'transport_segments': list,        # List of directional segments
    'track_classification': {
        'track_id': {
            'has_transport': bool,
            'n_runs': int,
            'mean_run_length': float,
            'mean_velocity': float,
            'predominant_direction': float
        }
    },
    'ensemble_metrics': {
        'fraction_with_transport': float,
        ...
    }
}
```

**Note**: May not detect transport in purely diffusive data (expected)

---

## 15. Fractional Brownian Motion (FBM)
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_fbm()`  
**Visualization**: `_plot_fbm()`  
**Category**: Biophysical Models  
**Priority**: 4 (Advanced)  
**Requires**: `batch_report_enhancements` module

### What It Does
Characterizes anomalous diffusion via Hurst exponent:
- Calculate Hurst exponent (H)
- Classify diffusion type:
  - H = 0.5: Normal Brownian
  - H < 0.5: Anti-persistent (subdiffusive)
  - H > 0.5: Persistent (superdiffusive)
- Fractal dimension analysis

### Method
R/S analysis or wavelet-based Hurst estimation

### Output Structure
```python
{
    'success': True,
    'hurst_values': {                  # Per-track Hurst
        'track_id': float
    },
    'diffusion_values': dict,          # D per track
    'summary': {
        'mean_hurst': float,
        'median_hurst': float,
        'diffusion_classification': str,
        ...
    },
    'data': {
        'hurst_distribution': array,
        'diffusion_distribution': array
    }
}
```

---

## 16. Advanced Metrics (TAMSD/EAMSD/NGP/VACF)
**Module**: `enhanced_report_generator.py`  
**Function**: `_analyze_advanced_metrics()`  
**Visualization**: `_plot_advanced_metrics()`  
**Category**: Advanced Statistics  
**Priority**: 4 (Advanced)  
**Requires**: `batch_report_enhancements` module

### What It Does
Calculates comprehensive advanced metrics:

1. **TAMSD**: Time-Averaged MSD
   - MSD averaged within each trajectory
   - Detects ergodicity breaking

2. **EAMSD**: Ensemble-Averaged MSD
   - MSD averaged across all trajectories
   - Standard MSD calculation

3. **Ergodicity Breaking Parameter**
   - EB = (⟨TAMSD²⟩ - ⟨TAMSD⟩²) / ⟨TAMSD⟩²
   - EB ≈ 0: Ergodic
   - EB > 0: Non-ergodic

4. **NGP**: Non-Gaussian Parameter
   - α₂(t) = ⟨r⁴(t)⟩ / (3⟨r²(t)⟩²) - 1
   - Measures deviation from Gaussian displacement

5. **VACF**: Velocity Autocorrelation Function
   - Already described in #9 above

6. **Turning Angles**
   - Distribution of directional changes
   - Persistence analysis

7. **Hurst Exponent**
   - Fractal analysis (see #15)

### Output Structure
```python
{
    'success': True,
    'config': dict,                    # Analysis configuration
    'summary': dict,                   # Key metrics summary
    'tamsd': {                         # Time-averaged MSD
        'time_lags': array,
        'values': array,
        'per_track': dict
    },
    'eamsd': {                         # Ensemble-averaged MSD
        'time_lags': array,
        'values': array
    },
    'ergodicity': {
        'parameter': float,
        'is_ergodic': bool,
        'per_track_variability': array
    },
    'ngp': {                           # Non-Gaussian parameter
        'time_lags': array,
        'values': array
    },
    'vacf': {                          # Velocity autocorrelation
        'time_lags': array,
        'values': array
    },
    'turning_angles': {
        'angles': array,
        'mean': float,
        'std': float
    },
    'hurst': {
        'values': dict,                # Per-track Hurst
        'mean': float
    },
    'fbm': dict                        # FBM analysis results
}
```

### Visualization
Multi-panel plot showing:
- TAMSD vs EAMSD comparison
- Ergodicity parameter vs time
- NGP vs time
- VACF decay
- Turning angle distribution
- Hurst exponent distribution

---

## Usage Examples

### Single Analysis
```python
from enhanced_report_generator import EnhancedSPTReportGenerator
import pandas as pd

# Load data
tracks_df = pd.read_csv('tracks.csv')

# Initialize generator
generator = EnhancedSPTReportGenerator()

# Run single analysis
units = {'pixel_size': 0.1, 'frame_interval': 0.1}
result = generator._analyze_diffusion(tracks_df, units)

if result['success']:
    print(f"Diffusion coefficient: {result['ensemble_results']['mean_D']}")
```

### Batch Report
```python
# Select multiple analyses
selected = [
    'basic_statistics',
    'diffusion_analysis',
    'motion_classification',
    'microrheology'
]

# Generate batch report
batch_result = generator.generate_batch_report(
    tracks_df, 
    selected, 
    'My Experiment'
)

# Access results
for analysis_key, result in batch_result['analysis_results'].items():
    print(f"{analysis_key}: {result['success']}")
```

### Interactive UI
```python
# In Streamlit app
generator.display_enhanced_analysis_interface()
```

---

## Dependencies

### Core Dependencies (All Functions)
- pandas
- numpy
- scipy
- matplotlib
- plotly

### Optional Dependencies (Specific Functions)
- scikit-learn (anomaly detection, clustering)
- changepoint_detection module (changepoint analysis)
- biophysical_models module (polymer physics, energy landscape)
- batch_report_enhancements module (FBM, advanced metrics)

---

## Performance Notes

### Computational Complexity
- Basic statistics: O(n) - Very fast
- Diffusion analysis: O(n log n) - Fast
- Motion classification: O(n log n) - Fast
- Clustering: O(n²) - Can be slow with many particles
- Anomaly detection: O(n log n) - Moderate with sklearn
- Microrheology: O(n log n) - Moderate
- Others: Generally O(n) to O(n log n)

### Memory Usage
- Most analyses: <100 MB for typical datasets
- Large dataset clustering: May require >1 GB
- Batch reports: Scales linearly with number of analyses

### Typical Execution Times
(For 100 tracks, 30 frames each)
- Basic statistics: <1 second
- Diffusion analysis: 1-2 seconds
- Motion classification: 1-2 seconds
- Microrheology: 2-3 seconds
- Advanced metrics: 3-5 seconds
- Complete batch (all 16): 30-60 seconds

---

## Troubleshooting

### Common Issues

1. **"No intensity channels found"**
   - Intensity analysis requires columns like `mean_intensity_ch1`
   - Add intensity data or skip intensity analysis

2. **"No directional motion detected"**
   - Active transport detection is working correctly
   - Purely diffusive data won't show transport (expected)

3. **Blank visualizations**
   - Some analyses produce blank plots with simple data
   - This is expected for uniform distributions or pure Brownian motion

4. **"Module not available"**
   - Some analyses require optional dependencies
   - They will be skipped if dependencies not installed

### Best Practices

1. Always check `result['success']` before using data
2. Handle missing data gracefully with try/except
3. Use appropriate pixel_size and frame_interval units
4. Start with basic analyses before running advanced ones
5. Use batch mode for multiple analyses to avoid repeated calculations

---

**Last Updated**: 2025-10-06  
**Version**: SPT2025B Current Release
