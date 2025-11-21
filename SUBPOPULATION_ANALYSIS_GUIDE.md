# Subpopulation Analysis Documentation

## Overview

The **Subpopulation Analysis** feature addresses the critical issue of **cellular heterogeneity** within experimental groups. It detects and characterizes distinct subpopulations (e.g., different cell cycle stages, metabolic states) that may be masked in group-level analyses.

## When to Use

Use subpopulation analysis when:
- You observe **high variance** in tracking parameters within a group
- You suspect **mixed cell populations** (e.g., different cell cycle stages)
- You want to identify **treatment-responsive vs. non-responsive** subgroups
- Standard group comparisons show **inconclusive results**
- You need to **separate heterogeneous states** before comparing groups

## Key Features

### 1. Single-Cell Level Analysis
- Aggregates tracking data at the **individual cell level**
- Computes comprehensive features per cell:
  - Diffusion coefficients (mean, std, CV)
  - Displacement statistics
  - Anomalous diffusion exponents
  - Confinement characteristics
  - Velocity profiles
  - Track length distributions

### 2. Intelligent Clustering
Multiple algorithms to detect subpopulations:
- **K-means**: Fast, assumes spherical clusters
- **Gaussian Mixture Models (GMM)**: Probabilistic, handles elliptical clusters
- **Hierarchical Clustering**: Builds dendrogram, flexible cluster shapes
- **DBSCAN** (optional): Density-based, finds arbitrary shapes

### 3. Automatic Cluster Optimization
- Tests multiple cluster numbers (default: 2-5)
- Uses validation metrics:
  - **Silhouette Score**: Cluster separation quality
  - **Davies-Bouldin Index**: Cluster compactness
  - **Calinski-Harabasz Score**: Variance ratio
  - **BIC/AIC**: Model selection criteria (GMM only)

### 4. Dimensionality Reduction
- **PCA (Principal Component Analysis)**: Reduces feature space while preserving 95% variance
- Improves clustering performance
- Enables visualization in 2D/3D

### 5. Statistical Characterization
For each subpopulation:
- Feature means, medians, standard deviations
- Fraction of total cells
- Statistical tests (Mann-Whitney U) comparing subgroups
- Heterogeneity metrics

## Workflow

### Step 1: Data Requirements

Your tracking data **must include cell identifiers**:

```python
# Required columns:
- track_id: Unique identifier for each track
- cell_id: Identifier for the cell/nucleus
- frame: Time frame
- x, y: Spatial coordinates
```

#### If Cell IDs Are Missing:

The system offers **automatic spatial clustering**:
1. Groups nearby tracks based on spatial proximity
2. Uses DBSCAN with configurable radius
3. Assigns `cell_id` to each track group

**Configuration:**
- **Clustering Radius**: Maximum distance (pixels) for tracks to belong to same cell
  - Typical: 50-100 pixels for nuclear tracking
- **Min Tracks/Cell**: Minimum tracks to form a valid cell
  - Typical: 5-10 tracks

### Step 2: Configuration

#### Analysis Parameters:
```python
Min Tracks/Cell: 10          # Minimum tracks to include a cell
Clustering Method: "kmeans"  # Or "gmm", "hierarchical"
Max Clusters: 4              # Maximum subpopulations to test
```

#### Group Selection:
- Analyze all groups separately (recommended)
- Or select specific groups/conditions

### Step 3: Run Analysis

Click **"🔬 Detect Subpopulations"**

The analyzer will:
1. Aggregate tracks by cell
2. Extract features per cell
3. Standardize and (optionally) apply PCA
4. Test multiple cluster numbers
5. Select optimal clustering
6. Characterize each subpopulation
7. Perform statistical tests

### Step 4: Interpret Results

#### Summary Metrics:
- **Total Groups**: Number of experimental groups analyzed
- **Heterogeneous**: Groups with detected subpopulations
- **Homogeneous**: Groups appearing uniform

#### Per-Group Results:

**Metrics:**
- Total cells analyzed
- Number of subpopulations detected
- Clustering method used
- PCA variance explained

**Subpopulation Table:**
Shows for each subpopulation:
- Cell count and fraction
- Mean values of key features
- Standard deviations

**Visualizations:**
- Feature distributions by subpopulation (box plots)
- Subpopulation size pie chart
- Feature correlation heatmap
- PCA scatter plots (if applicable)

## Interpretation Guide

### Case 1: Homogeneous Group
```
Subpopulations Detected: 0
Recommendation: "Group appears homogeneous. Proceed with standard analysis."
```
**Interpretation**: All cells behave similarly. No hidden substructure.

### Case 2: Two Subpopulations
```
Subpopulations Detected: 2
Subpop 0: 60% of cells, D=0.05 μm²/s, α=0.8 (subdiffusive)
Subpop 1: 40% of cells, D=0.15 μm²/s, α=1.1 (superdiffusive)
```
**Interpretation**: Likely two distinct states:
- Could be G1 vs. S/G2 phase
- Could be active vs. inactive cells
- Could be responders vs. non-responders

**Action**: Analyze subpopulations separately. Compare treatments within each subpopulation.

### Case 3: Multiple Subpopulations (3+)
```
Subpopulations Detected: 3
```
**Interpretation**: Complex heterogeneity:
- May represent cell cycle stages (G1, S, G2/M)
- May indicate mixed differentiation states
- Could reflect spatial compartments

**Action**: 
1. Correlate with cell cycle markers if available
2. Check spatial distribution (are subpops localized?)
3. Validate with independent methods

### Case 4: All Groups Heterogeneous
```
Heterogeneous Groups: 4/4
Recommendation: "All groups show subpopulations. Recommend subgroup-level comparisons."
```
**Interpretation**: Systematic heterogeneity across all conditions.

**Action**:
1. Compare subpopulation **fractions** between treatments
2. Compare behavior **within matched subpopulations**
3. Look for treatment-induced subpopulation shifts

## Biological Examples

### Example 1: Cell Cycle Heterogeneity

**Scenario**: Tracking chromatin dynamics in asynchronous cells.

**Expected Result**:
```
2-3 Subpopulations:
- Subpop 0 (40%): Low D, α≈0.7 → G1 phase (compact chromatin)
- Subpop 1 (35%): High D, α≈0.9 → S phase (replication factories)
- Subpop 2 (25%): Medium D, α≈0.6 → G2/M (chromatin condensation)
```

**Validation**: Correlate with cell cycle markers (e.g., DNA content, cyclin levels).

### Example 2: Treatment Response Heterogeneity

**Scenario**: Drug treatment experiment shows inconclusive group-level differences.

**Expected Result**:
```
Control: 1 subpopulation (homogeneous)
Treated: 2 subpopulations
  - Subpop 0 (60%): Similar to control (non-responders)
  - Subpop 1 (40%): Significantly different (responders)
```

**Insight**: 40% response rate masked by bulk analysis.

### Example 3: Spatial Compartmentalization

**Scenario**: Nuclear periphery vs. interior tracking.

**Expected Result**:
```
2 Subpopulations:
- Subpop 0: Low D, high confinement → Peripheral chromatin
- Subpop 1: High D, low confinement → Nuclear interior
```

**Validation**: Check spatial coordinates of subpopulations.

## Advanced Usage

### Custom Feature Selection

Modify `SubpopulationConfig` to focus on specific features:

```python
config = SubpopulationConfig(
    features_to_use=[
        'mean_diffusion_coefficient',
        'mean_alpha',
        'mean_velocity'
    ]
)
```

### Multiple Clustering Methods

Compare results across methods:

```python
config = SubpopulationConfig(
    clustering_methods=['kmeans', 'gmm', 'hierarchical']
)
```

The best method is automatically selected based on silhouette scores.

### Adjusting Sensitivity

For **more subpopulations** (higher sensitivity):
```python
n_clusters_range=(2, 6)  # Test up to 6 clusters
min_tracks_per_cell=5    # Lower threshold
```

For **fewer subpopulations** (conservative):
```python
n_clusters_range=(2, 3)  # Only test 2-3 clusters
min_tracks_per_cell=20   # Higher threshold
```

## Output Files

### Cell Data CSV
**Filename**: `{GroupName}_subpopulations.csv`

**Contains**:
- Cell ID
- Subpopulation assignment
- All computed features
- Number of tracks per cell

**Use for**:
- Further analysis in R/Python
- Correlation with other cell-level measurements
- Machine learning classification

## Troubleshooting

### "No cells have at least N tracks"
**Solution**: Lower `min_tracks_per_cell` or improve tracking quality.

### "Insufficient cells for clustering"
**Solution**: Need at least 5 cells. Combine replicates or use whole-image analysis.

### "No distinct subpopulations detected"
**Interpretation**: Group is truly homogeneous OR variance is too small to cluster.
**Check**: Look at coefficient of variation (CV) for key features. CV < 0.2 suggests homogeneity.

### Unstable cluster assignments
**Solution**: 
1. Increase `min_tracks_per_cell` for more robust features
2. Use GMM instead of kmeans (more stable)
3. Check for outlier cells

## Statistical Considerations

### Sample Size Requirements
- **Minimum**: 5 cells per subpopulation
- **Recommended**: 15+ cells per subpopulation for statistical tests
- **Ideal**: 30+ cells per subpopulation for robust characterization

### Multiple Testing Correction
When comparing many features across subpopulations:
- Consider Bonferroni or FDR correction
- Focus on top discriminating features

### Validation Strategies
1. **Internal validation**: Bootstrap clustering stability
2. **External validation**: Correlate with independent markers
3. **Biological validation**: Functional assays on sorted subpopulations

## Integration with Other Analyses

### With Segmentation-Based Analysis
1. First: Use segmentation to identify nuclear regions
2. Then: Run subpopulation analysis within each region
3. Compare: Heterogeneity in nucleus vs. cytoplasm

### With Comparative Group Analysis
1. First: Detect subpopulations in each group
2. Then: Compare matched subpopulations across groups
3. Test: Are subpopulation fractions different?

### With ML Trajectory Classification
1. First: Subpopulation analysis identifies groups
2. Then: Train classifier on subpopulation labels
3. Apply: Predict subpopulation for new cells in real-time

## References

**Clustering Methods:**
- Hartigan & Wong (1979): K-means algorithm
- McLachlan & Peel (2000): Gaussian Mixture Models
- Ward (1963): Hierarchical clustering
- Ester et al. (1996): DBSCAN

**Validation Metrics:**
- Rousseeuw (1987): Silhouette coefficient
- Davies & Bouldin (1979): DB index
- Caliński & Harabasz (1974): CH index

**Biological Applications:**
- Hansen et al. (2020): Single-cell heterogeneity in chromatin dynamics
- Izeddin et al. (2014): Cell-cycle dependent diffusion
- Spahn et al. (2022): Subpopulation analysis in drug responses

## Best Practices Summary

✅ **DO**:
- Ensure sufficient cells per group (n > 10)
- Use multiple validation metrics
- Validate clusters with independent data
- Check cluster stability across runs
- Interpret in biological context

❌ **DON'T**:
- Over-interpret with small sample sizes (n < 5 cells)
- Assume clusters are biologically meaningful without validation
- Force clustering on homogeneous data
- Ignore spatial patterns in subpopulations
- Skip visualization and feature inspection

---

**For questions or issues**: Check `subpopulation_analysis.py` source code for detailed implementation and parameter descriptions.
