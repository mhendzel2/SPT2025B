# Percolation Analysis - Quick Reference Card

## 🚀 Quick Start

```python
import pandas as pd
from analysis import calculate_fractal_dimension, build_connectivity_network
from biophysical_models import analyze_size_dependent_diffusion, infer_obstacle_density
from visualization import *

# Load your tracking data
tracks_df = pd.read_csv('your_tracks.csv')  # Must have: track_id, frame, x, y
```

---

## 📊 Method 1: Fractal Dimension (d_f)

**What it tells you:** Does trajectory interact with fractal chromatin network?

```python
result = calculate_fractal_dimension(
    tracks_df, 
    pixel_size=0.1,           # µm/pixel
    method='box_counting'     # or 'mass_radius'
)

print(f"Mean d_f: {result['ensemble_statistics']['mean_df']:.2f}")
# d_f ≈ 2.0 → Normal diffusion
# d_f ≈ 2.5 → Fractal matrix (chromatin fiber)

fig = plot_fractal_dimension_distribution(result)
fig.show()
```

---

## 🌐 Method 2: Connectivity Network

**What it tells you:** Does system percolate (connected paths exist)?

```python
result = build_connectivity_network(
    tracks_df,
    pixel_size=0.1,
    grid_size=0.2,      # Grid cell size (µm)
    min_visits=2        # Min visits per cell
)

print(f"Percolates: {result['percolation_analysis']['percolates']}")
print(f"Giant component: {result['percolation_analysis']['giant_component_fraction']:.1%}")

fig = plot_connectivity_network(result, tracks_df, 0.1)
fig.show()
```

**Requires:** `pip install networkx`

---

## 🗺️ Method 3: Anomalous Exponent Map

**What it tells you:** Where are the percolating channels?

```python
fig = plot_anomalous_exponent_map(
    tracks_df,
    pixel_size=0.1,
    frame_interval=0.05,    # seconds
    grid_size=50,           # interpolation grid
    window_size=5           # frames for local α
)
fig.show()
```

**Color Code:**
- 🟢 Green (α ≈ 1.0) = Percolating channels
- 🟡 Yellow (α ≈ 0.75) = Transition zones
- 🔴 Red (α < 0.5) = Obstacles

---

## 📏 Method 4: Size-Dependent Diffusion

**What it tells you:** Mesh size (ξ) of chromatin network

```python
# Measure D for different sized probes (dextrans, proteins, etc.)
probe_data = {
    5.0: 15.2,    # 5 nm radius → 15.2 µm²/s
    10.0: 8.3,    # 10 nm
    20.0: 2.1,    # 20 nm
    40.0: 0.3     # 40 nm
}

result = analyze_size_dependent_diffusion(probe_data)
print(f"Mesh size: {result['mesh_size_xi_nm']:.1f} nm")
print(f"Critical radius: {result['critical_radius_nm']:.1f} nm")

fig = plot_size_dependent_diffusion(result)
fig.show()
```

**Typical Values:**
- Nucleus: ξ = 15-30 nm
- Mitotic chromosome: ξ = 5-15 nm
- Condensate: ξ = 5-50 nm

---

## 🧮 Bonus: Obstacle Density

**What it tells you:** Chromatin crowding level (volume fraction φ)

```python
result = infer_obstacle_density(
    D_observed=5.0,    # Measured in nucleus
    D_free=25.0        # Measured in buffer
)

print(f"Obstacle fraction: {result['obstacle_fraction_phi']:.1%}")
print(result['interpretation'])
```

**Interpretation:**
- φ < 0.15 → Open chromatin
- φ = 0.3-0.4 → Typical nucleoplasm
- φ > 0.5 → Dense heterochromatin

---

## 🔧 Troubleshooting

### "networkx required"
```bash
pip install networkx
```

### "Insufficient data"
- Need ≥10 points per track for fractal dimension
- Need ≥3 probe sizes for size scaling
- Lower `min_track_length` or filter better data

### "Curve fitting failed"
- Add more probe sizes (need ≥3)
- Check measurements are in same conditions
- Ensure D_values > 0

### Slow performance (>1000 tracks)
```python
# Sample tracks for visualization
sampled = tracks_df.sample(frac=0.1, random_state=42)

# Use coarser grids
grid_size = 30  # Instead of 50
```

---

## 📚 Full Documentation

- **Complete Guide:** `PERCOLATION_ANALYSIS_GUIDE.md` (~600 lines)
- **Implementation Details:** `PERCOLATION_IMPLEMENTATION_SUMMARY.md`
- **Changelog:** `CHANGELOG.md` (Version: 20250118)

---

## 🎯 When to Use Each Method

| Question | Method |
|----------|--------|
| "Is chromatin a fractal?" | **Fractal Dimension** |
| "Can particles percolate through?" | **Connectivity Network** |
| "Where are the channels?" | **Anomalous Exponent Map** |
| "How big are the pores?" | **Size-Dependent Diffusion** |
| "How crowded is nucleus?" | **Obstacle Density** |

---

## ⚡ Complete Workflow (1 minute)

```python
# 1. Fractal dimension
fractal = calculate_fractal_dimension(tracks_df, pixel_size=0.1)
print(f"d_f = {fractal['ensemble_statistics']['mean_df']:.2f}")

# 2. Connectivity
network = build_connectivity_network(tracks_df, pixel_size=0.1, grid_size=0.2)
print(f"Percolates: {network['percolation_analysis']['percolates']}")

# 3. Alpha map
alpha_fig = plot_anomalous_exponent_map(tracks_df, pixel_size=0.1, frame_interval=0.05)
alpha_fig.write_html('alpha_map.html')

# 4. Obstacle density (if you have D_free measurement)
obs = infer_obstacle_density(D_observed=5.0, D_free=25.0)
print(f"φ = {obs['obstacle_fraction_phi']:.1%}")

print("✅ Done!")
```

---

## 📦 Dependencies

```bash
# Already installed (in requirements.txt)
numpy pandas scipy plotly

# NEW - Install this!
pip install networkx
```

---

## 💡 Pro Tips

1. **Start with fractal dimension** - Fastest way to assess chromatin interaction
2. **Use connectivity network** - Direct percolation test (yes/no answer)
3. **Alpha maps are visual** - Best for presentations/papers
4. **Size scaling needs multi-probe data** - Requires separate experiments

---

## 🔬 Scientific Validity

All methods based on peer-reviewed literature:
- Fractal globule model (Science 2009)
- Percolation theory (Annu Rev Biophys 2019)
- Mackie-Meares obstruction (Proc Roy Soc 1955)
- Recent bioRxiv preprints (2024-2025)

---

## ✅ Output Checklist

Each method returns:
- ✅ `success` boolean (check this first!)
- ✅ Numerical results (dict/DataFrame)
- ✅ Interpretation text
- ✅ Summary statistics

All visualizations return:
- ✅ Plotly figure (`.show()` or `.write_html()`)
- ✅ Hover tooltips
- ✅ Annotations with key values

---

**Need help?** Check `PERCOLATION_ANALYSIS_GUIDE.md` for detailed examples and theory.
