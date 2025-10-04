# Quick Reference: New Features & Usage

## 🚀 Quick Start

### 1. Settings Panel (In Sidebar)
```python
# In Streamlit UI:
# 1. Expand "Settings Panel" in sidebar
# 2. Choose preset: "TIRF (100x, 1.49 NA)"
# 3. Or set custom values
# 4. Click "Apply Settings"

# In code:
from settings_panel import get_settings_panel, get_global_units

panel = get_settings_panel()
panel.apply_preset('Confocal (63x, 1.4 NA)')
units = get_global_units()  # {'pixel_size': 0.065, 'frame_interval': 0.1, ...}
```

### 2. New Analyses in Batch Reports
```python
from enhanced_report_generator import EnhancedSPTReportGenerator

generator = EnhancedSPTReportGenerator()

# Select analyses (including new ones)
analyses = [
    'basic_statistics',
    'diffusion_analysis',
    'fbm_analysis',        # NEW: Fractional Brownian Motion
    'advanced_metrics'      # NEW: TAMSD/EAMSD/NGP/VACF/etc.
]

result = generator.generate_batch_report(tracks_df, analyses, "Condition 1")
```

### 3. Statistical Group Comparison
```python
from batch_report_enhancements import StatisticalComparisonTools

groups = {
    'Control': [0.5, 0.6, 0.55, ...],
    'Treatment': [0.8, 0.9, 0.85, ...]
}

# Parametric test
result = StatisticalComparisonTools.compare_groups_parametric(groups, "Metric")
print(f"p-value: {result['p_value']}")

# Visualize
fig = StatisticalComparisonTools.plot_group_comparison(result, groups)
```

### 4. Progress Tracking (For Future Integration)
```python
from progress_utils import AnalysisProgress

progress = AnalysisProgress("MSD Calculation", total_steps=100)
for i in range(100):
    if not progress.update(i+1, f"Track {i}"):
        break  # User cancelled
progress.complete()
```

---

## 📊 New Analyses Available

### Fractional Brownian Motion (FBM)
**What it does**: Estimates Hurst exponent H for each trajectory
- H = 0.5: Normal Brownian motion
- H < 0.5: Subdiffusive (constrained)
- H > 0.5: Superdiffusive (active transport)

**Output**:
- `hurst_values`: List of H per track
- `diffusion_values`: List of D per track
- `summary`: Mean, std, median of H and D

### Advanced Metrics
**What it does**: Comprehensive biophysical characterization
- **TAMSD**: Time-averaged MSD (per trajectory)
- **EAMSD**: Ensemble-averaged MSD (population)
- **Ergodicity**: EB ratio and parameter
- **NGP**: Non-Gaussian parameter (1D & 2D)
- **van Hove**: Displacement distributions
- **VACF**: Velocity autocorrelation
- **Turning Angles**: Directional persistence
- **Hurst**: From TAMSD power-law scaling

**Output**:
- `tamsd`: DataFrame with per-track time-averaged MSD
- `eamsd`: DataFrame with ensemble-averaged MSD
- `ergodicity`: DataFrame with EB ratio & parameter vs lag
- `ngp`: DataFrame with NGP vs lag
- `vacf`: DataFrame with velocity autocorrelation
- `turning_angles`: DataFrame with angle distributions

---

## 🔬 Microscopy Presets

| Preset | Pixel Size (μm) | Frame Interval (s) | Use Case |
|--------|-----------------|-------------------|----------|
| Confocal (63x, 1.4 NA) | 0.065 | 0.1 | High-res confocal |
| Confocal (100x, 1.4 NA) | 0.041 | 0.1 | Super-res confocal |
| TIRF (100x, 1.49 NA) | 0.107 | 0.05 | TIRF microscopy, 20 Hz |
| Widefield (60x, 1.4 NA) | 0.108 | 0.1 | Standard widefield |
| Spinning Disk (60x) | 0.108 | 0.033 | Fast imaging, 30 Hz |
| Light Sheet (20x, 1.0 NA) | 0.325 | 0.05 | Large FOV, 3D imaging |

---

## 📈 Statistical Tests Available

### Parametric Tests (Assume Normal Distribution)

**Two Groups**: Welch's t-test
```python
result = StatisticalComparisonTools.compare_groups_parametric(
    {'Group A': [...], 'Group B': [...]},
    metric_name="Diffusion Coefficient"
)
# Returns: t_statistic, p_value, cohens_d
```

**Three+ Groups**: One-way ANOVA
```python
result = StatisticalComparisonTools.compare_groups_parametric(
    {'A': [...], 'B': [...], 'C': [...]},
    metric_name="Hurst Exponent"
)
# Returns: f_statistic, p_value, eta_squared
```

### Non-Parametric Tests (No Distribution Assumptions)

**Two Groups**: Mann-Whitney U
```python
result = StatisticalComparisonTools.compare_groups_nonparametric(
    {'Group A': [...], 'Group B': [...]},
    metric_name="Track Length"
)
# Returns: u_statistic, p_value, rank_biserial_r
```

**Three+ Groups**: Kruskal-Wallis H
```python
result = StatisticalComparisonTools.compare_groups_nonparametric(
    {'A': [...], 'B': [...], 'C': [...]},
    metric_name="NGP"
)
# Returns: h_statistic, p_value, epsilon_squared
```

### Effect Size Interpretations

| Effect Size | Negligible | Small | Medium | Large |
|-------------|------------|-------|--------|-------|
| Cohen's d | < 0.2 | 0.2-0.5 | 0.5-0.8 | ≥ 0.8 |
| Eta-squared | < 0.01 | 0.01-0.06 | 0.06-0.14 | ≥ 0.14 |
| Rank-biserial | < 0.1 | 0.1-0.3 | 0.3-0.5 | ≥ 0.5 |
| Epsilon-squared | < 0.01 | 0.01-0.08 | 0.08-0.26 | ≥ 0.26 |

### Multiple Testing Correction

```python
p_values = [0.001, 0.02, 0.04, 0.08, 0.5]
correction = StatisticalComparisonTools.bonferroni_correction(p_values, alpha=0.05)

print(f"Original α: {correction['original_alpha']}")       # 0.05
print(f"Corrected α: {correction['corrected_alpha']}")     # 0.01
print(f"Significant: {correction['n_significant']}/5")     # 1/5
```

---

## 🎨 Visualizations

### FBM Results
- Hurst exponent histogram
- Diffusion coefficient histogram
- Reference line at H=0.5 (Brownian)

### Advanced Metrics
- TAMSD/EAMSD curves (individual tracks + ensemble)
- Ergodicity breaking (EB ratio & parameter vs lag)
- NGP vs lag time (1D & 2D)
- VACF decay curve
- Turning angle distribution

### Statistical Comparison
- Box plots with individual data points
- Statistical annotations (test name, p-value, stars)
- Effect size with interpretation

---

## 📁 File Locations

| File | Location | Purpose |
|------|----------|---------|
| `settings_panel.py` | Root | Unified settings management |
| `progress_utils.py` | Root | Progress tracking utilities |
| `batch_report_enhancements.py` | Root | New analyses + statistics |
| `enhanced_report_generator.py` | Root | Report generation (updated) |
| `app.py` | Root | Main application (updated) |

---

## 🐛 Troubleshooting

### Issue: "Batch enhancements module not available"
**Solution**: Ensure `advanced_biophysical_metrics.py` is present in the root directory.

### Issue: Settings panel not showing
**Solution**: Check that `settings_panel.py` exists. Fallback manual controls will appear.

### Issue: New analyses not appearing
**Solution**: Restart Streamlit app: `streamlit run app.py`

### Issue: Statistical tests failing
**Solution**: Ensure groups have enough data points (n ≥ 3 per group)

---

## 📚 Documentation Files

| File | Content |
|------|---------|
| `INTEGRATION_FINAL_SUMMARY.md` | Complete integration summary |
| `BATCH_REPORT_ENHANCEMENT_SUMMARY.md` | Detailed enhancement documentation |
| `IMPLEMENTATION_COMPLETE.md` | Settings panel + progress utils guide |
| `PLACEHOLDER_ANALYSIS_AND_GUI_IMPROVEMENTS.md` | GUI analysis report |

---

## ✅ Validation

All new code validated:
```bash
python -m py_compile batch_report_enhancements.py  # ✅ PASS
python -m py_compile settings_panel.py             # ✅ PASS
python -m py_compile progress_utils.py             # ✅ PASS
```

No linting errors in VS Code.

---

**Quick Help**: For detailed usage, see `BATCH_REPORT_ENHANCEMENT_SUMMARY.md`
