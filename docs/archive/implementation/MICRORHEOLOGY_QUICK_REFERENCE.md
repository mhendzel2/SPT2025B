# Quick Reference: Advanced Microrheology Methods
**SPT2025B - Enhanced Report Generator**

---

## 1. Creep Compliance J(t)

### What it Measures
Material deformation response to constant applied stress. Shows how a material "creeps" under load.

### Key Equation
```
J(t) = <Δr²(t)> / (4·kB·T·a)
```
where:
- `<Δr²(t)>` = Mean-squared displacement at time t
- `kB` = Boltzmann constant
- `T` = Temperature (K)
- `a` = Particle radius

### Power-Law Fit
```
J(t) = J₀ · t^β
```

### Material Classification
- **β < 0.5**: Solid-like (elastic dominates)
- **0.5 ≤ β < 1.0**: Gel/viscoelastic
- **β ≥ 1.0**: Liquid-like (viscous dominates)

### Usage in Code
```python
from rheology import MicrorheologyAnalyzer

analyzer = MicrorheologyAnalyzer(
    tracks_df=tracks_df,
    pixel_size=0.1,          # μm/pixel
    frame_interval=0.1,      # s/frame
    temperature=298.15,      # K
    particle_radius=0.5      # μm
)

result = analyzer.calculate_creep_compliance()
print(f"Material type: {result['summary']['material_classification']}")
print(f"J₀ = {result['data']['fit']['J0']:.2e} Pa⁻¹")
print(f"β = {result['data']['fit']['beta']:.3f}")
```

### Interpretation
- **Low β**: Elastic gel (e.g., stiff cytoskeleton)
- **β ≈ 0.5**: Viscoelastic gel (e.g., mucus)
- **High β**: Viscous fluid (e.g., dilute polymer solution)

---

## 2. Relaxation Modulus G(t)

### What it Measures
Stress decay when material is held at constant strain. Shows how quickly material "relaxes."

### Key Equation (Approximation)
```
G(t) ≈ kB·T / (π·a·MSD(t))
```

### Exponential Decay Fit
```
G(t) = G₀ · exp(-t/τ) + G_∞
```
where:
- `G₀` = Initial modulus (Pa)
- `τ` = Relaxation time (s)
- `G_∞` = Equilibrium modulus (Pa)

### Usage in Code
```python
result = analyzer.calculate_relaxation_modulus(frequency_domain=False)

print(f"Relaxation time τ = {result['summary']['relaxation_time']:.3f} s")
print(f"Initial modulus G₀ = {result['data']['fit']['G0']:.2e} Pa")
print(f"Equilibrium modulus G_∞ = {result['data']['fit']['G_inf']:.2e} Pa")
```

### Interpretation
- **Short τ (< 0.1 s)**: Fast relaxation, fluid-like
- **Medium τ (0.1-10 s)**: Viscoelastic gel
- **Long τ (> 10 s)**: Slow relaxation, solid-like
- **High G_∞**: Permanent elastic component

---

## 3. Two-Point Microrheology

### What it Measures
Distance-dependent mechanical properties using correlated motion of particle pairs. Detects spatial heterogeneity.

### Key Concept
Particles close together experience similar microenvironment. Correlation decreases with distance.

### Correlation Function
```
C(r) = C₀ · exp(-r/ξ)
```
where:
- `ξ` = Correlation length (μm)
- `r` = Particle pair separation

### Usage in Code
```python
result = analyzer.two_point_microrheology(
    max_distance=10.0,    # μm
    distance_bins=20
)

distances = result['data']['distances']
G_prime = result['data']['G_prime']        # Storage modulus vs distance
G_double_prime = result['data']['G_double_prime']  # Loss modulus vs distance
xi = result['summary']['correlation_length']

print(f"Correlation length ξ = {xi:.2f} μm")
```

### Interpretation
- **Small ξ (< 1 μm)**: Highly heterogeneous (e.g., crosslinked network)
- **Medium ξ (1-5 μm)**: Mesoscale heterogeneity (e.g., phase-separated gel)
- **Large ξ (> 5 μm)**: Homogeneous material
- **G' increases with r**: Stiffer at longer distances
- **G'' increases with r**: More viscous at longer distances

---

## 4. Spatial Microrheology Map

### What it Measures
Local mechanical properties across entire field of view. Creates 2D maps of G', G'', and η.

### Heterogeneity Index
```
H = CV(G') = σ(G') / μ(G')
```
Coefficient of variation of storage modulus.

### Usage in Code
```python
result = analyzer.spatial_microrheology_map(
    grid_size=10,           # 10x10 grid
    min_tracks_per_bin=3    # Minimum tracks per cell
)

G_prime_map = result['data']['spatial_map']['G_prime']      # 10x10 array
G_double_prime_map = result['data']['spatial_map']['G_double_prime']
viscosity_map = result['data']['spatial_map']['viscosity']

H = result['summary']['heterogeneity_index']
print(f"Heterogeneity index H = {H:.3f}")
```

### Interpretation
- **H < 0.2**: Homogeneous material
- **0.2 ≤ H < 0.5**: Moderate heterogeneity
- **H ≥ 0.5**: Highly heterogeneous (e.g., composite material)
- **Spatial patterns**: Reveal structure (fibers, pores, domains)

---

## Complete Workflow Example

```python
import pandas as pd
from rheology import MicrorheologyAnalyzer
from enhanced_report_generator import EnhancedSPTReportGenerator

# 1. Load tracking data
tracks_df = pd.read_csv('cell_membrane_tracks.csv')

# 2. Configure units
units = {
    'pixel_size': 0.1,          # μm/pixel
    'frame_interval': 0.1,       # s/frame
    'temperature': 310.15,       # 37°C in Kelvin
    'particle_radius': 0.5       # μm (e.g., lipid probe)
}

# 3. Create analyzer
analyzer = MicrorheologyAnalyzer(
    tracks_df=tracks_df,
    **units
)

# 4. Run all 4 advanced methods
creep = analyzer.calculate_creep_compliance()
relaxation = analyzer.calculate_relaxation_modulus()
two_point = analyzer.two_point_microrheology(max_distance=8.0, distance_bins=15)
spatial = analyzer.spatial_microrheology_map(grid_size=12, min_tracks_per_bin=5)

# 5. Generate comprehensive report
generator = EnhancedSPTReportGenerator(
    tracks_df=tracks_df,
    project_name="Membrane Rheology Study",
    metadata={'cell_type': 'HeLa', 'treatment': 'control'}
)

report = generator.generate_batch_report(
    selected_analyses=[
        'creep_compliance',
        'relaxation_modulus',
        'two_point_microrheology',
        'spatial_microrheology',
        'polymer_physics',
        'energy_landscape'
    ],
    current_units=units
)

# 6. Export
generator.export_report(report, format='html', output_path='rheology_report.html')
```

---

## Comparison Table

| Method | Output | Key Metric | Best For |
|--------|--------|-----------|----------|
| **Basic Microrheology** | G'(ω), G''(ω), η*(ω) | Frequency-dependent moduli | Standard characterization |
| **Creep Compliance** | J(t) | Power-law exponent β | Material classification |
| **Relaxation Modulus** | G(t) | Relaxation time τ | Stress relaxation dynamics |
| **Two-Point** | G'(r), G''(r) | Correlation length ξ | Detecting heterogeneity |
| **Spatial Map** | G'(x,y), G''(x,y) | Heterogeneity index H | Visualizing property distribution |

---

## Troubleshooting

### Error: "Insufficient tracks for analysis"
**Solution:** Increase tracking sensitivity or use longer movies. Need ≥10 tracks with ≥20 frames each.

### Error: "Two-point analysis failed"
**Solution:** Requires multiple particles per frame. Check that tracking identifies concurrent particles.

### Warning: "Low spatial bin occupancy"
**Solution:** Reduce `grid_size` or decrease `min_tracks_per_bin` threshold.

### Result: All moduli are zero/NaN
**Solution:** Check units! Ensure `pixel_size` and `frame_interval` are correct. Wrong units → wrong MSD → wrong moduli.

---

## Physical Interpretation Guide

### Typical Values

#### Cell Cytoplasm
- G' ≈ 10-100 Pa
- G'' ≈ 5-50 Pa
- τ ≈ 0.1-1 s
- β ≈ 0.6-0.8 (viscoelastic)

#### Cell Membrane
- G' ≈ 1-10 Pa
- G'' ≈ 0.5-5 Pa
- τ ≈ 0.01-0.1 s
- β ≈ 0.7-0.9 (more fluid-like)

#### Extracellular Matrix (ECM)
- G' ≈ 100-1000 Pa
- G'' ≈ 10-100 Pa
- τ ≈ 1-100 s
- β ≈ 0.3-0.5 (gel-like)

#### Mucus
- G' ≈ 1-100 Pa
- G'' ≈ 0.5-50 Pa
- τ ≈ 0.1-10 s
- β ≈ 0.5-0.7 (viscoelastic gel)

---

## References

1. **Mason & Weitz (1995)** - "Optical Measurements of Frequency-Dependent Linear Viscoelastic Moduli of Complex Fluids"
2. **Waigh (2005)** - "Microrheology of complex fluids" (Rep. Prog. Phys.)
3. **Crocker et al. (2000)** - "Two-Point Microrheology of Inhomogeneous Soft Materials"
4. **Levine & Lubensky (2000)** - "One- and Two-Particle Microrheology" (Phys. Rev. Lett.)

---

**For full implementation details, see:**
- `rheology.py` (lines 486-1180): Method implementations
- `enhanced_report_generator.py` (lines 1360-1780): Report generator integration
- `ADVANCED_ANALYSIS_INTEGRATION_SUMMARY.md`: Complete integration documentation
