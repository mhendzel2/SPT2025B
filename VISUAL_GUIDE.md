# Visual Guide: SPT2025B Analysis Suite
**Complete Overview of Available Analyses**

---

## Analysis Architecture

```
SPT2025B Enhanced Report Generator
│
├── CORE ANALYSES (Always Available) ───────────────────────────────────────
│   │
│   ├── [1] Basic Statistics
│   │   └── Track counts, lengths, displacement/velocity distributions
│   │
│   ├── [2] Diffusion Analysis ⭐
│   │   └── MSD, D, D_app, D_eff, ensemble-averaged & time-averaged MSD
│   │
│   ├── [3] Motion Classification ⭐
│   │   └── α exponent, confined/normal/directed/super-diffusive
│   │
│   ├── [4] Spatial Organization
│   │   └── RDF g(r), nearest neighbor distances, clustering
│   │
│   ├── [5] Anomaly Detection
│   │   └── Outlier track identification, statistical anomalies
│   │
│   ├── [6] Velocity Correlation
│   │   └── VACF, persistence length, correlation time
│   │
│   ├── [7] Multi-Particle Interactions
│   │   └── Pair correlations, clustering coefficient
│   │
│   ├── [8] Confinement Analysis
│   │   └── Zone detection, escape probability, dwell times
│   │
│   └── [9] Intensity Analysis (if intensity data present)
│       └── Fluorescence dynamics, intensity-movement correlation
│
├── BIOPHYSICAL MODELS (Always Available) ──────────────────────────────────
│   │
│   ├── MICRORHEOLOGY SUITE ──────────────────────────────────────────────
│   │   │
│   │   ├── [10] Basic Microrheology
│   │   │   └── G'(ω), G''(ω), η*(ω) vs frequency
│   │   │
│   │   ├── [11] Creep Compliance ✨ NEW
│   │   │   ├── J(t) = <Δr²(t)> / (4·kB·T·a)
│   │   │   ├── Power-law fit: J(t) = J₀·t^β
│   │   │   └── Material classification: solid/gel/liquid
│   │   │
│   │   ├── [12] Relaxation Modulus ✨ NEW
│   │   │   ├── G(t) ≈ kB·T / (π·a·MSD(t))
│   │   │   ├── Exponential fit: G(t) = G₀·exp(-t/τ) + G_∞
│   │   │   └── Relaxation time τ extraction
│   │   │
│   │   ├── [13] Two-Point Microrheology ✨ NEW
│   │   │   ├── Distance-dependent G'(r) and G''(r)
│   │   │   ├── Cross-correlation C(r) = C₀·exp(-r/ξ)
│   │   │   └── Correlation length ξ detection
│   │   │
│   │   └── [14] Spatial Microrheology Map ✨ NEW
│   │       ├── 2D maps: G'(x,y), G''(x,y), η(x,y)
│   │       ├── Heterogeneity index H = σ(G')/μ(G')
│   │       └── Grid-based local property estimation
│   │
│   └── ADVANCED BIOPHYSICAL (Conditional: BIOPHYSICAL_MODELS_AVAILABLE) ──
│       │
│       ├── [15] Polymer Physics ⭐
│       │   ├── Rouse model (α = 0.5)
│       │   ├── Zimm model (α ≈ 0.6, hydrodynamic interactions)
│       │   ├── Reptation (α = 0.25 → 0.5, entanglement)
│       │   └── Fractal dimension Df
│       │
│       ├── [16] Energy Landscape ⭐
│       │   ├── Boltzmann inversion: U(r) = -kB·T·ln[P(r)]
│       │   ├── Force field: F = -∇U
│       │   ├── Energy barriers and wells
│       │   └── Transition state analysis
│       │
│       └── [17] Active Transport ⭐
│           ├── Directional motion segment detection
│           ├── Mode classification: diffusive/slow/fast/mixed
│           ├── Velocity distribution analysis
│           └── Motor-driven vs thermal motion separation
│
└── ADVANCED METRICS (Conditional: BATCH_ENHANCEMENTS_AVAILABLE) ───────────
    │
    ├── [18] Changepoint Detection
    │   └── Bayesian online changepoint, regime switching
    │
    ├── [19] Fractional Brownian Motion (FBM)
    │   └── Hurst exponent H, long-range temporal correlations
    │
    └── [20] Advanced Metrics
        └── TAMSD, EAMSD, NGP, extended VACF
```

---

## Microrheology Analysis Flow

```
INPUT: Track Data (track_id, frame, x, y)
  │
  ├─────────────────────────────────────────────────────────────┐
  │                                                               │
  ▼                                                               ▼
Calculate MSD                                            Spatial Binning
<Δr²(t)>                                                10x10 grid
  │                                                               │
  ├──────┬──────┬──────┬──────┐                                 │
  │      │      │      │      │                                  │
  ▼      ▼      ▼      ▼      ▼                                  ▼
┌─────┬─────┬─────┬─────┬─────┐                          ┌──────────┐
│Basic│Creep│Relax│Two- │Spat-│                          │Per-Bin   │
│Micro│Comp │Modu │Point│ial  │                          │MSD →     │
│rheo │     │     │Micro│Map  │                          │Local G', │
│     │     │     │     │     │                          │G'', η    │
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                          └────┬─────┘
   │     │     │     │     │                                   │
   ▼     ▼     ▼     ▼     ▼                                   ▼
G'(ω)  J(t)  G(t)  G'(r) G'(x,y)                         Heterogeneity
G''(ω) β     τ     G''(r) G''(x,y)                       Index H
η*(ω)  Material rlx  ξ    η(x,y)
       class   time  corr  spatial
                     len   pattern
   │     │     │     │     │
   └──────┴──────┴─────┴────┴────────────────────────────────────┐
                                                                   │
                                                                   ▼
                                                    Report Assembly
                                                    ├── JSON Export
                                                    ├── HTML Interactive
                                                    └── PDF Publication
```

---

## Analysis Selection Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│ QUICK SELECTION PRESETS                                                │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ ✓ Essential (3 analyses, ~2 sec)                                       │
│   └── Basic Statistics + Diffusion + Motion Classification             │
│                                                                         │
│ ✓ Standard Biophysical (6 analyses, ~20 sec)                           │
│   └── Essential + VACF + Microrheology + Polymer Physics               │
│                                                                         │
│ ✓ Advanced Microrheology (5 analyses, ~50 sec) ✨ NEW                  │
│   └── Basic Micro + Creep + Relaxation + Two-Point + Spatial           │
│                                                                         │
│ ✓ Complete Suite (17-20 analyses, ~2 min)                              │
│   └── All Available Analyses                                           │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Configuration

```
┌──────────────────────────────────────────────────────────────────────┐
│ ESSENTIAL PARAMETERS (Required for all analyses)                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   pixel_size: 0.1 μm/pixel      ← Microscope calibration             │
│   frame_interval: 0.1 s/frame   ← Acquisition settings               │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│ MICRORHEOLOGY PARAMETERS (Required for methods 10-14)                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   temperature: 298.15 K          ← Default: 25°C                     │
│   particle_radius: 0.5 μm        ← Probe particle size               │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│ ADVANCED PARAMETERS (Optional, method-specific)                      │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Two-Point:                                                          │
│   ├── max_distance: 10.0 μm      ← Maximum pair separation           │
│   └── distance_bins: 20           ← Number of distance bins          │
│                                                                       │
│   Spatial Map:                                                        │
│   ├── grid_size: 10               ← NxN spatial grid                 │
│   └── min_tracks_per_bin: 3       ← Minimum for reliable estimate   │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Output Format Structure

```
{
  'success': True,
  'data': {
    'time_lags': [0.1, 0.2, 0.3, ...],      # Raw arrays
    'creep_compliance': [1e-6, 2e-6, ...],
    'fit': {
      'J0': 1.23e-6,                         # Fit parameters
      'beta': 0.67,
      'r_squared': 0.98
    }
  },
  'summary': {
    'material_classification': 'Gel',        # Interpretation
    'characteristic_time': 0.45,
    'mean_modulus': 150.0
  },
  'units': {
    'pixel_size': 0.1,
    'frame_interval': 0.1,
    'temperature': 298.15,
    'particle_radius': 0.5
  }
}
```

---

## Visualization Outputs

```
┌──────────────────────────────────────────────────────────────────────┐
│ MICRORHEOLOGY VISUALIZATIONS                                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ [10] Basic Microrheology                                             │
│      ├── G'(ω) vs ω (log-log)                                        │
│      ├── G''(ω) vs ω (log-log)                                       │
│      └── η*(ω) vs ω (log-log)                                        │
│                                                                       │
│ [11] Creep Compliance ✨                                              │
│      ├── J(t) vs t (log-log)                                         │
│      ├── Power-law fit overlay                                       │
│      └── Material type annotation                                    │
│                                                                       │
│ [12] Relaxation Modulus ✨                                            │
│      ├── G(t) vs t (log-log)                                         │
│      ├── Exponential fit overlay                                     │
│      └── Relaxation time τ annotation                                │
│                                                                       │
│ [13] Two-Point Microrheology ✨                                       │
│      ├── G'(r) vs r (dual panel)                                     │
│      ├── G''(r) vs r                                                 │
│      └── Correlation length ξ annotation                             │
│                                                                       │
│ [14] Spatial Microrheology ✨                                         │
│      ├── G'(x,y) heatmap                                             │
│      ├── G''(x,y) heatmap                                            │
│      ├── η(x,y) heatmap                                              │
│      └── Heterogeneity index H annotation                            │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Material Classification Guide

```
┌──────────────────────────────────────────────────────────────────────┐
│ CREEP COMPLIANCE β EXPONENT → MATERIAL TYPE                          │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   0.0 ────────────────── β < 0.5 ──────────────────────────────────  │
│          SOLID-LIKE (Elastic Dominates)                              │
│          Examples: Crosslinked gels, stiff cytoskeleton              │
│                                                                       │
│   0.5 ───────────── 0.5 ≤ β < 1.0 ──────────────────────────────────  │
│          VISCOELASTIC GEL                                            │
│          Examples: Mucus, soft cytoplasm, uncrosslinked polymers     │
│                                                                       │
│   1.0 ──────────────── β ≥ 1.0 ─────────────────────────────────────  │
│          LIQUID-LIKE (Viscous Dominates)                             │
│          Examples: Dilute polymer solutions, cell membranes          │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│ RELAXATION MODULUS τ → MATERIAL DYNAMICS                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   Fast (τ < 0.1 s)      → Fluid-like, rapid stress relaxation       │
│   Medium (0.1-10 s)     → Viscoelastic gel, intermediate timescale  │
│   Slow (τ > 10 s)       → Solid-like, slow relaxation               │
│                                                                       │
├──────────────────────────────────────────────────────────────────────┤
│ SPATIAL HETEROGENEITY H → MATERIAL UNIFORMITY                        │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│   H < 0.2               → Homogeneous material                       │
│   0.2 ≤ H < 0.5         → Moderate heterogeneity                     │
│   H ≥ 0.5               → Highly heterogeneous (composite)           │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Integration Timeline

```
2025-10-06: Complete Integration
│
├── Phase 1: Core Implementation
│   ├── ✅ File size limits increased (2GB images)
│   ├── ✅ calculate_creep_compliance() (~150 lines)
│   ├── ✅ calculate_relaxation_modulus() (~150 lines)
│   ├── ✅ two_point_microrheology() (~200 lines)
│   └── ✅ spatial_microrheology_map() (~160 lines)
│
├── Phase 2: Report Generator Integration
│   ├── ✅ 4 new analyses registered
│   ├── ✅ 4 analysis wrapper functions (~190 lines)
│   ├── ✅ 4 visualization functions (~230 lines)
│   └── ✅ requirements.txt organized
│
└── Phase 3: Documentation
    ├── ✅ ADVANCED_ANALYSIS_INTEGRATION_SUMMARY.md (300+ lines)
    ├── ✅ MICRORHEOLOGY_QUICK_REFERENCE.md (200+ lines)
    ├── ✅ ANALYSIS_CATALOG.md (400+ lines)
    ├── ✅ FINAL_INTEGRATION_SUMMARY.md (200+ lines)
    └── ✅ VISUAL_GUIDE.md (this file)
```

---

## Quick Command Reference

```powershell
# Start application
streamlit run app.py --server.port 5000

# Run tests
python -m pytest tests/test_app_logic.py -k "microrheology"
python test_functionality.py
python test_comprehensive.py

# Check imports
python -c "from rheology import MicrorheologyAnalyzer; print('OK')"
python -c "from enhanced_report_generator import EnhancedSPTReportGenerator; print('OK')"

# Install dependencies
pip install -r requirements.txt
```

---

## Status Dashboard

```
┌─────────────────────────────────────────────────────────────────────┐
│ INTEGRATION STATUS                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ✅ Core Implementation Complete       (~600 lines in rheology.py)   │
│ ✅ Report Generator Integration       (~420 lines added)            │
│ ✅ Visualizations Implemented         (4 plotly functions)          │
│ ✅ Error Handling Complete            (try-except all functions)    │
│ ✅ Documentation Written              (1100+ lines)                 │
│ ✅ Compilation Errors                 (0 errors)                    │
│ ✅ Dependencies Verified              (All present)                 │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ AVAILABLE ANALYSES                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Total: 20 methods                                                    │
│ ├── Core: 9 methods                                                 │
│ ├── Biophysical Models: 5 methods (4 NEW ✨)                        │
│ ├── Advanced Biophysical: 3 methods                                 │
│ └── Advanced Metrics: 3 methods                                     │
│                                                                      │
├─────────────────────────────────────────────────────────────────────┤
│ NEW CAPABILITIES (2025-10-06)                                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ ✨ Creep Compliance J(t) - Material classification                  │
│ ✨ Relaxation Modulus G(t) - Stress relaxation dynamics             │
│ ✨ Two-Point Microrheology - Spatial heterogeneity detection        │
│ ✨ Spatial Microrheology - 2D mechanical property mapping            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Production Readiness Checklist

- [x] All 4 new methods implemented with full documentation
- [x] Integrated into report generator with analysis + visualization functions
- [x] Error handling with graceful degradation
- [x] No compilation errors or warnings
- [x] All dependencies present in requirements.txt
- [x] Follows project architecture patterns (data_access_utils, StateManager)
- [x] Compatible with 64GB+ memory systems, 2GB image files
- [x] Comprehensive documentation (4 new markdown files, 1100+ lines)
- [x] Sample usage code and examples provided
- [x] Testing instructions documented
- [x] Performance benchmarks documented
- [x] Known limitations documented

---

**STATUS: ✅ PRODUCTION READY**

All advanced biophysical models and microrheology methods are fully integrated and operational.

**Version:** SPT2025B v1.0.0  
**Date:** 2025-10-06
