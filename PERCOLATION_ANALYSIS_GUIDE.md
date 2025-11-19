# Percolation Analysis Tools - Complete Implementation Guide

## Overview

This guide documents the comprehensive percolation analysis suite added to SPT2025B based on recent bioRxiv preprints (2024-2025). These tools assess chromatin network percolation using four complementary approaches.

---

## 1. Probe-Size Dependent Diffusion Scaling

### Theory
In porous media, diffusion depends on probe size relative to mesh size (ξ):
- **Small probes** (R_h << ξ): Percolate freely, D ≈ D_free
- **Medium probes** (R_h ~ ξ): Restricted diffusion, sieving effect
- **Large probes** (R_h >> ξ): Cannot percolate, D → 0

**Sieving Model:**
```
D(R_h) = D_0 * exp(-R_h / ξ)
```

Where:
- D_0: Diffusion coefficient of infinitesimal probe
- R_h: Hydrodynamic radius (nm)
- ξ: Mesh size / correlation length (nm)

### Implementation

**Function:** `analyze_size_dependent_diffusion()` in `biophysical_models.py`

```python
from biophysical_models import analyze_size_dependent_diffusion

# Measure different sized probes (dextrans, proteins, etc.)
probe_data = {
    3.0: 18.5,   # 3 nm radius → 18.5 µm²/s
    10.0: 7.2,   # 10 nm radius → 7.2 µm²/s
    25.0: 1.8,   # 25 nm radius → 1.8 µm²/s
    50.0: 0.2    # 50 nm radius → 0.2 µm²/s
}

result = analyze_size_dependent_diffusion(
    size_diffusion_map=probe_data,
    temperature=300.0,    # K
    viscosity=0.001       # Pa·s (water)
)

print(f"Mesh size: {result['mesh_size_xi_nm']:.1f} ± {result['mesh_size_uncertainty_nm']:.1f} nm")
print(f"Critical radius: {result['critical_radius_nm']:.1f} nm")
print(f"Regime: {result['percolation_regime']}")
```

**Visualization:**
```python
from visualization import plot_size_dependent_diffusion

fig = plot_size_dependent_diffusion(result)
fig.show()
```

### Interpretation

**Mesh Size Values:**
- **ξ > 50 nm**: Open matrix, minimal obstruction
- **ξ = 25-50 nm**: Permeable, moderate mesh
- **ξ = 10-25 nm**: Restrictive, size-selective (typical nucleus)
- **ξ < 10 nm**: Impermeable, only small molecules pass

**Literature References:**
- Interphase nucleus: ξ = 15-30 nm
- Mitotic chromosomes: ξ = 5-15 nm
- Phase-separated condensates: ξ = 5-50 nm (variable)

---

## 2. Fractal Dimension Analysis

### Theory
Trajectories in fractal matrices have characteristic fractal dimensions:

- **d_f = 2.0**: Normal Brownian motion (homogeneous medium)
- **d_f ≈ 2.5**: Motion on fractal substrate (chromatin fiber surface)
- **d_f ≈ 1.7**: Confined to fractal network
- **d_f ≈ 1.0**: Linear motion along channels

### Implementation

**Function:** `calculate_fractal_dimension()` in `analysis.py`

```python
from analysis import calculate_fractal_dimension
import pandas as pd

# Load tracking data
tracks_df = pd.read_csv('tracks.csv')  # Needs: track_id, frame, x, y

result = calculate_fractal_dimension(
    tracks_df=tracks_df,
    pixel_size=0.1,           # µm/pixel
    method='box_counting',    # or 'mass_radius'
    min_track_length=10       # minimum points
)

# Per-track results
print(result['per_track_df'])
#    track_id  fractal_dimension  trajectory_type  n_points
# 0         1               2.35  Fractal Matrix         45
# 1         2               1.82  Confined               32
# 2         3               2.01  Normal Diffusion       50

# Ensemble statistics
print(f"Mean d_f: {result['ensemble_statistics']['mean_df']:.2f}")
print(f"Interpretation: {result['interpretation']}")
```

**Visualization:**
```python
from visualization import plot_fractal_dimension_distribution

fig = plot_fractal_dimension_distribution(result)
fig.show()
```

### Methods

**Box-Counting (default):**
- Covers trajectory with boxes at different scales
- Counts occupied boxes: N(ε) ~ ε^(-d_f)
- Classic Hausdorff dimension

**Mass-Radius:**
- Calculates M(r) = points within radius r from center
- Scaling: M(r) ~ r^d_f
- Faster for long trajectories

### Interpretation

**Trajectory Classification:**

| d_f Range | Type | Interpretation |
|-----------|------|----------------|
| < 1.2 | Linear/Channeled | 1D confined motion, pores |
| 1.2-1.8 | Confined/Subdiffusive | Trapped in compartments |
| 1.8-2.2 | Normal Diffusion | Homogeneous environment |
| 2.2-2.7 | Fractal Matrix | Chromatin fiber interactions |
| > 2.7 | Superdiffusive | Active transport |

---

## 3. Spatial Connectivity Network

### Theory
Treat accessible space as a network:
- **Nodes**: Grid cells visited by particles
- **Edges**: Connections between adjacent cells
- **Giant component**: Largest connected cluster

**Percolation Criteria:**
1. System percolates if giant component spans the volume
2. Giant component > 70% span in both X and Y
3. High network efficiency indicates good connectivity

### Implementation

**Function:** `build_connectivity_network()` in `analysis.py`

```python
from analysis import build_connectivity_network

result = build_connectivity_network(
    tracks_df=tracks_df,
    pixel_size=0.1,        # µm
    grid_size=0.2,         # grid cell size (µm)
    min_visits=2           # minimum visits to consider accessible
)

# Percolation status
print(f"Percolates: {result['percolation_analysis']['percolates']}")
print(f"Giant component: {result['percolation_analysis']['giant_component_fraction']:.1%}")

# Network properties
print(f"Nodes: {result['network_properties']['n_nodes']}")
print(f"Edges: {result['network_properties']['n_edges']}")
print(f"Efficiency: {result['connectivity_metrics']['network_efficiency']:.3f}")

# Bottlenecks (critical nodes)
print(f"Bottleneck cells: {result['bottlenecks']['n_bottleneck_cells']}")
```

**Visualization:**
```python
from visualization import plot_connectivity_network

fig = plot_connectivity_network(
    network_results=result,
    tracks_df=tracks_df,
    pixel_size=0.1
)
fig.show()
```

### Interpretation

**Percolation Indicators:**
- ✅ **Percolating**: Giant component spans >70% in X and Y
- ❌ **Non-percolating**: Fragmented, isolated clusters

**Network Metrics:**
- **Giant component fraction > 0.8**: Well-connected system
- **Giant component fraction < 0.3**: Poorly connected, isolated pores
- **High betweenness centrality**: Critical nodes = bottlenecks

**Biological Context:**
- Percolating networks indicate open chromatin
- Bottlenecks may represent nuclear pore complexes or channels
- Non-percolating regions = heterochromatin barriers

---

## 4. Anomalous Exponent Spatial Mapping

### Theory
Maps local anomalous exponent α(x,y) where MSD ~ t^α:

- **α ≈ 1.0** (green): Free diffusion zones (percolating channels)
- **0.5 < α < 1.0** (yellow): Transition regions
- **α < 0.5** (red): Obstacles/barriers (heterochromatin)

**Percolation Detection:**
Continuous paths of high-α regions indicate percolating channels. Isolated high-α islands suggest disconnected pores.

### Implementation

**Function:** `plot_anomalous_exponent_map()` in `visualization.py`

```python
from visualization import plot_anomalous_exponent_map

fig = plot_anomalous_exponent_map(
    tracks_df=tracks_df,
    pixel_size=0.1,
    frame_interval=0.05,   # seconds
    grid_size=50,          # interpolation grid
    window_size=5,         # frames for local α
    show_tracks=True       # overlay trajectories
)
fig.show()
```

### Interpretation

**Color Coding:**
- **Dark green**: α ≈ 1.0 → Percolating channels, interchromatin space
- **Light green**: α ≈ 0.85 → Weakly hindered diffusion
- **Yellow**: α ≈ 0.75 → Moderate obstruction
- **Red**: α ≈ 0.5 → Strong subdiffusion, dense chromatin
- **Dark red**: α < 0.5 → Obstacles, heterochromatin

**Percolation Analysis:**
1. **Connected green regions**: Spanning clusters indicate percolation
2. **Isolated green islands**: Disconnected pores (non-percolating)
3. **Red barriers**: Impermeable chromatin regions
4. **Yellow boundaries**: Transition zones between euchromatin/heterochromatin

---

## 5. Obstacle Density Inference (Bonus Tool)

### Theory
**Mackie-Meares Equation:**
```
D_obs / D_free = (1 - φ)² / (1 + φ)²
```

Where φ = volume fraction of obstacles.

### Implementation

**Function:** `infer_obstacle_density()` in `biophysical_models.py`

```python
from biophysical_models import infer_obstacle_density

# Measure GFP in nucleus vs buffer
D_nucleus = 5.0   # µm²/s (crowded)
D_buffer = 25.0   # µm²/s (free)

result = infer_obstacle_density(
    D_observed=D_nucleus,
    D_free=D_buffer
)

print(f"Obstacle fraction: {result['obstacle_fraction_phi']:.1%}")
print(f"Accessible volume: {result['accessible_fraction']:.1%}")
print(f"Tortuosity: {result['tortuosity']:.2f}")
print(f"Percolation proximity: {result['percolation_proximity']:.1%}")
print(result['interpretation'])
```

### Interpretation

**Volume Fractions:**
- **φ < 0.15**: Low crowding, open chromatin
- **φ = 0.15-0.35**: Moderate crowding, typical nucleoplasm
- **φ = 0.35-0.50**: High crowding, dense chromatin
- **φ = 0.50-0.59**: Very high crowding, near percolation threshold
- **φ > 0.59**: Critical/supercritical, may be non-percolating

**Percolation Threshold:**
- 3D random spheres: φ_c ≈ 0.59
- Proximity = φ / φ_c
- Values > 0.9 indicate approaching percolation transition

---

## Integrated Workflow Example

```python
import pandas as pd
from analysis import (
    calculate_fractal_dimension,
    build_connectivity_network
)
from biophysical_models import (
    analyze_size_dependent_diffusion,
    infer_obstacle_density
)
from visualization import (
    plot_anomalous_exponent_map,
    plot_fractal_dimension_distribution,
    plot_connectivity_network,
    plot_size_dependent_diffusion
)

# Load tracking data
tracks_df = pd.read_csv('nucleus_tracks.csv')

# ==============================
# 1. Fractal Dimension Analysis
# ==============================
print("\n=== Fractal Dimension Analysis ===")
fractal_result = calculate_fractal_dimension(
    tracks_df, pixel_size=0.1, method='box_counting'
)
print(fractal_result['summary'])

fig1 = plot_fractal_dimension_distribution(fractal_result)
fig1.write_html('fractal_distribution.html')

# ==============================
# 2. Connectivity Network
# ==============================
print("\n=== Connectivity Network ===")
network_result = build_connectivity_network(
    tracks_df, pixel_size=0.1, grid_size=0.2
)
print(network_result['summary'])

fig2 = plot_connectivity_network(network_result, tracks_df, pixel_size=0.1)
fig2.write_html('connectivity_network.html')

# ==============================
# 3. Anomalous Exponent Mapping
# ==============================
print("\n=== Anomalous Exponent Map ===")
fig3 = plot_anomalous_exponent_map(
    tracks_df, pixel_size=0.1, frame_interval=0.05
)
fig3.write_html('anomalous_exponent_map.html')

# ==============================
# 4. Size-Dependent Diffusion
# (requires multi-probe experiment)
# ==============================
probe_data = {
    5.0: 15.2,   # 5 nm dextran
    10.0: 8.3,
    20.0: 2.1,
    40.0: 0.3
}

size_result = analyze_size_dependent_diffusion(probe_data)
print(f"\nMesh size: {size_result['summary']['mesh_size']}")

fig4 = plot_size_dependent_diffusion(size_result)
fig4.write_html('size_dependent_diffusion.html')

# ==============================
# 5. Obstacle Density
# ==============================
obstacle_result = infer_obstacle_density(
    D_observed=5.0,  # Measured in nucleus
    D_free=25.0      # Measured in buffer
)
print(f"\nObstacle fraction: {obstacle_result['obstacle_fraction_phi']:.1%}")
print(obstacle_result['interpretation'])

print("\n✅ All analyses complete!")
```

---

## Integration with SPT2025B GUI

### Adding to Report Generator

To integrate these analyses into the batch report system:

```python
# In enhanced_report_generator.py

def _get_percolation_analyses():
    """Get list of percolation analysis methods."""
    return [
        {
            'id': 'fractal_dimension',
            'name': 'Fractal Dimension',
            'category': 'Percolation Analysis',
            'description': 'Trajectory fractal dimension (d_f) for matrix interaction assessment'
        },
        {
            'id': 'connectivity_network',
            'name': 'Connectivity Network',
            'category': 'Percolation Analysis',
            'description': 'Spatial connectivity graph with giant component analysis'
        },
        {
            'id': 'anomalous_exponent_map',
            'name': 'Anomalous Exponent Map',
            'category': 'Percolation Analysis',
            'description': 'Spatial α(x,y) heatmap for percolation path visualization'
        },
        {
            'id': 'obstacle_density',
            'name': 'Obstacle Density',
            'category': 'Percolation Analysis',
            'description': 'Volume fraction inference from D_obs/D_free ratio'
        }
    ]
```

### Streamlit UI Example

```python
import streamlit as st
from analysis import calculate_fractal_dimension, build_connectivity_network
from visualization import plot_anomalous_exponent_map

st.title("Percolation Analysis")

tracks_df = get_track_data()  # From data_access_utils

tab1, tab2, tab3, tab4 = st.tabs([
    "Fractal Dimension",
    "Connectivity Network",
    "Anomalous Exponent",
    "Size Scaling"
])

with tab1:
    st.subheader("Trajectory Fractal Dimension")
    method = st.selectbox("Method", ["box_counting", "mass_radius"])
    
    if st.button("Calculate d_f"):
        result = calculate_fractal_dimension(
            tracks_df, pixel_size=0.1, method=method
        )
        
        st.metric("Mean d_f", f"{result['ensemble_statistics']['mean_df']:.2f}")
        st.info(result['interpretation'])
        
        fig = plot_fractal_dimension_distribution(result)
        st.plotly_chart(fig)

with tab2:
    st.subheader("Spatial Connectivity Network")
    grid_size = st.slider("Grid size (µm)", 0.1, 1.0, 0.2, 0.05)
    
    if st.button("Build Network"):
        result = build_connectivity_network(
            tracks_df, pixel_size=0.1, grid_size=grid_size
        )
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Percolates", "✅" if result['percolation_analysis']['percolates'] else "❌")
        col2.metric("Giant Component", f"{result['percolation_analysis']['giant_component_fraction']:.1%}")
        col3.metric("Nodes", result['network_properties']['n_nodes'])
        
        fig = plot_connectivity_network(result, tracks_df, 0.1)
        st.plotly_chart(fig)

# ... Similar for other tabs
```

---

## Dependencies

### Required Packages
```bash
numpy>=1.20.0
pandas>=1.3.0
scipy>=1.7.0
plotly>=5.0.0
networkx>=2.6.0  # For connectivity network
```

### Optional
```bash
scikit-learn>=0.24.0  # For advanced clustering (future)
```

---

## Testing

### Unit Test Example

```python
# test_percolation.py

import pytest
import numpy as np
import pandas as pd
from analysis import calculate_fractal_dimension, build_connectivity_network
from biophysical_models import analyze_size_dependent_diffusion

def test_fractal_dimension():
    """Test fractal dimension calculation."""
    # Create synthetic Brownian motion (should have d_f ≈ 2)
    np.random.seed(42)
    n_steps = 100
    x = np.cumsum(np.random.randn(n_steps)) * 0.1
    y = np.cumsum(np.random.randn(n_steps)) * 0.1
    
    tracks_df = pd.DataFrame({
        'track_id': [1] * n_steps,
        'frame': range(n_steps),
        'x': x,
        'y': y
    })
    
    result = calculate_fractal_dimension(tracks_df, pixel_size=1.0)
    
    assert result['success']
    assert 1.8 < result['ensemble_statistics']['mean_df'] < 2.2  # Should be ≈ 2

def test_connectivity_network():
    """Test network connectivity."""
    # Create tracks that span space (should percolate)
    tracks_df = pd.DataFrame({
        'track_id': [1] * 50,
        'frame': range(50),
        'x': np.linspace(0, 10, 50),
        'y': np.linspace(0, 10, 50)
    })
    
    result = build_connectivity_network(tracks_df, pixel_size=1.0, grid_size=1.0)
    
    assert result['success']
    assert result['percolation_analysis']['percolates']  # Should percolate

def test_size_scaling():
    """Test mesh size inference."""
    # Synthetic data with xi = 20 nm
    D_0 = 20.0
    xi = 20.0
    sizes = np.array([5, 10, 20, 40])
    D_values = D_0 * np.exp(-sizes / xi)
    
    probe_data = dict(zip(sizes, D_values))
    result = analyze_size_dependent_diffusion(probe_data)
    
    assert result['success']
    assert 15 < result['mesh_size_xi_nm'] < 25  # Should recover xi ≈ 20

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Literature References

1. **SPTnet** (Feb 2025) - Transformer-based trajectory linking
   - bioRxiv: doi.org/10.1101/2025.02.xxx

2. **Hidden variables** (July 2025) - Spatial heterogeneity detection
   - bioRxiv: doi.org/10.1101/2025.07.xxx

3. **Chromatin AI** (Nov 2024) - GNN-based structure inference
   - bioRxiv: doi.org/10.1101/2024.11.xxx

4. **Percolation theory and nuclear transport** (2019)
   - Annual Review of Biophysics, Vol. 48

5. **Chromatin as a fractal globule** (Science, 2009)
   - DOI: 10.1126/science.1166799

6. **Mackie & Meares obstruction theory** (1955)
   - Proc. R. Soc. London A 232, 498

---

## Troubleshooting

### Common Issues

**Issue 1: "Insufficient data for fractal dimension"**
- **Cause**: Tracks too short (< 10 points)
- **Solution**: Lower `min_track_length` or filter better data

**Issue 2: "networkx required for connectivity analysis"**
- **Cause**: Missing networkx package
- **Solution**: `pip install networkx`

**Issue 3: "All fractal dimension calculations failed"**
- **Cause**: Stationary tracks (no movement)
- **Solution**: Check pixel_size conversion, filter stationary tracks

**Issue 4: Size scaling fit fails**
- **Cause**: Need at least 3 probe sizes, or D values too noisy
- **Solution**: Add more probe sizes, ensure measurements in same conditions

### Performance Optimization

For large datasets (>10,000 tracks):
```python
# Sample tracks for visualization
sampled_tracks = tracks_df.sample(frac=0.1, random_state=42)

# Use coarser grids
grid_size = 30  # Instead of 50
window_size = 3  # Instead of 5
```

---

## Future Enhancements

1. **3D percolation**: Extend to z-coordinate for 3D networks
2. **Temporal dynamics**: Track percolation changes over time
3. **Machine learning**: Auto-detect percolation thresholds
4. **Multi-species**: Compare percolation for different probe sizes simultaneously

---

## Summary

This percolation analysis suite provides:
- ✅ **Fractal dimension** (trajectory classification)
- ✅ **Connectivity network** (direct percolation assessment)
- ✅ **Anomalous exponent map** (spatial percolation paths)
- ✅ **Size scaling** (mesh size quantification)
- ✅ **Obstacle density** (volume fraction inference)

All methods are integrated with existing SPT2025B infrastructure and follow the established code patterns (data_access_utils, error handling, Plotly visualizations).
