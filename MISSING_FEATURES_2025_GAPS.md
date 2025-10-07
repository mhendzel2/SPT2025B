# Missing Features: 2025 Cutting-Edge SPT Methods Gap Analysis

**Date**: October 6, 2025  
**SPT2025B Version**: Current state analysis  
**Reference**: Recent bioRxiv/PubMed methods (2024-2025)

---

## Executive Summary

SPT2025B has **strong foundations** in:
- ✅ Two-point microrheology (implemented Oct 2025)
- ✅ GSER-based rheology (G', G", viscosity)
- ✅ Advanced metrics (NGP, van Hove, TAMSD, VACF, Hurst)
- ✅ HMM state detection (basic)
- ✅ Track quality metrics (SNR, precision, completeness)
- ✅ Statistical validation (bootstrap, goodness-of-fit, model comparison)

**Critical gaps** preventing adoption of 2025 methods:
- ❌ **No CVE/MLE estimators** with blur/noise correction (Berglund 2010, PMC)
- ❌ **No DDM/RICS modules** for tracking-free rheology
- ❌ **No microsecond sampling support** (SpeedyTrack PMC12026894)
- ❌ **No AFM/OT calibration import** for cross-validation
- ❌ **No acquisition advisor** (frame rate/exposure optimization)
- ❌ **No iHMM with blur-aware models** (Lindén PMC6050756)
- ❌ **No deep learning trajectory inference** (SPTnet, DeepSPT)
- ⚠️ **No equilibrium validity badges** (GSER assumes thermal equilibrium)

---

## 1. HIGH PRIORITY: Model-Based Inference (MISSING)

### 1.1 CVE/MLE Estimators with Blur Correction
**Status**: ❌ **NOT IMPLEMENTED**  
**Reference**: [Berglund 2010 PubMed 20866658][10]

**Current state**:
```python
# analysis.py uses naive MSD fitting
def fit_msd_power_law(msd_df):
    # Simple log-log linear regression - no blur correction
    log_times = np.log(msd_df['lag_time_s'])
    log_msd = np.log(msd_df['msd_m2'])
    slope, intercept = np.polyfit(log_times, log_msd, 1)
```

**What's needed**:
```python
class BiasedInferenceCorrector:
    """Likelihood-based estimators for D and α with blur/noise correction."""
    
    def cve_estimator(self, track: np.ndarray, dt: float, 
                     localization_error: float) -> Dict:
        """
        Covariance-based estimator (CVE) for diffusion coefficient.
        Corrects for static localization noise.
        
        Reference: Berglund (2010) Eq. 13-15
        Returns: D, D_std, alpha, alpha_std (bias-corrected)
        """
        pass
    
    def mle_with_blur(self, track: np.ndarray, dt: float, 
                      exposure_time: float, localization_error: float) -> Dict:
        """
        Maximum likelihood estimator accounting for:
        - Motion blur from finite exposure time
        - Localization noise (Gaussian)
        - Short trajectory bias
        
        Reference: Berglund (2010) Eq. 22-27
        Uses numerical optimization (scipy.optimize.minimize)
        """
        pass
    
    def select_estimator(self, N_steps: int, SNR: float) -> str:
        """
        Auto-select CVE vs MLE vs MSD based on:
        - N_steps < 50 → Use MLE (reduces bias)
        - SNR < 5 → Use CVE (robust to noise)
        - N_steps > 100, SNR > 10 → MSD is acceptable
        """
        pass
```

**Integration points**:
- `enhanced_report_generator.py`: Add `'bias_corrected_diffusion'` analysis
- `track_quality_metrics.py`: Use CVE/MLE when quality score < 0.7
- `advanced_statistical_tests.py`: Compare MSD vs CVE vs MLE in validation

**Impact**: Reduces D/α estimation bias by 20-50% on short (<50 steps) or noisy (SNR<10) tracks.

---

### 1.2 Acquisition Advisor Widget
**Status**: ❌ **NOT IMPLEMENTED**  
**Reference**: [Optimal temporal resolution 2024 PubMed 38724858][11]

**What's needed**:
```python
class AcquisitionAdvisor:
    """Recommends frame rate and exposure from expected D, SNR, density."""
    
    def __init__(self):
        # Load bias maps from Weimann et al. 2024
        self.bias_tables = self._load_bias_maps()  # CSV from supplement
    
    def recommend_framerate(self, D_expected: float, localization_precision: float,
                           track_length: int = 50) -> Dict:
        """
        Uses 2024 bias maps to suggest optimal frame interval.
        
        Returns:
        {
            'recommended_dt': 0.05,  # seconds
            'exposure_fraction': 0.8,  # 80% of frame time
            'expected_D_bias': 0.05,  # 5% underestimation
            'expected_alpha_bias': -0.02,
            'rationale': 'Balances localization vs blur for D=0.1 μm²/s'
        }
        """
        # Lookup table or interpolation
        pass
    
    def validate_settings(self, dt_actual: float, exposure_actual: float,
                         tracks_df: pd.DataFrame) -> Dict:
        """
        Post-acquisition check: Are settings reasonable for observed D?
        Flags if dt >> optimal (undersampling) or << optimal (photobleaching).
        """
        pass
```

**UI integration** (in `app.py`):
```python
with st.expander("⚙️ Acquisition Advisor", expanded=False):
    st.info("Optimize frame rate before next experiment")
    
    expected_D = st.number_input("Expected D (μm²/s)", 0.01, 10.0, 0.1)
    precision = st.number_input("Localization precision (nm)", 10, 100, 30)
    
    advisor = AcquisitionAdvisor()
    rec = advisor.recommend_framerate(expected_D, precision/1000)
    
    st.success(f"📊 Recommended: {rec['recommended_dt']:.3f} s/frame")
    st.caption(rec['rationale'])
    
    # Post-analysis validation
    if 'tracks_data' in st.session_state:
        validation = advisor.validate_settings(frame_interval, exposure, tracks_df)
        if validation['warning']:
            st.warning(f"⚠️ {validation['message']}")
```

**Impact**: Prevents 30-50% estimation bias from suboptimal acquisition settings.

---

## 2. MEDIUM PRIORITY: State Segmentation (PARTIAL)

### 2.1 iHMM with Blur-Aware Models
**Status**: ⚠️ **BASIC HMM EXISTS, NO BLUR MODELING**  
**Current**: `hmm_analysis.py` (line 21) uses `hmmlearn` with Gaussian emissions  
**Reference**: [Variational EM for SPT PMC6050756][12]

**Current limitations**:
```python
# hmm_analysis.py - simplified model
def fit_hmm(tracks_df, n_states=3):
    model = hmm.GaussianHMM(n_components=n_states)
    # Assumes i.i.d. Gaussian noise - NO blur, NO heterogeneous errors
    model.fit(displacements)
```

**What's needed**:
```python
class iHMMWithBlur:
    """Infinite HMM with blur-aware variational EM (Lindén et al.)."""
    
    def __init__(self, tracks_df: pd.DataFrame, 
                 exposure_time: float,
                 localization_errors: pd.Series):  # Per-point σ_loc
        """
        Initialize with:
        - exposure_time: Integrates motion during acquisition
        - localization_errors: Heterogeneous σ per detection
        """
        pass
    
    def fit_variational(self, max_states: int = 10, 
                       alpha_prior: float = 1.0) -> Dict:
        """
        Variational Bayes inference with:
        - Motion blur correction (exposure > 0)
        - Heterogeneous localization errors (per-point σ)
        - Automatic state number selection (iHMM)
        
        Returns:
        {
            'n_states': 3,  # Inferred automatically
            'state_labels': array,  # Per-step assignments
            'dwell_times': DataFrame,  # With posteriors
            'transition_matrix': array,
            'diffusion_per_state': [D1, D2, D3],
            'localization_corrected': True
        }
        """
        pass
    
    def dwell_time_posterior(self, state: int) -> Dict:
        """
        Return dwell time distribution with uncertainty.
        Not just mean, but full posterior samples.
        """
        pass
```

**Integration**:
- Replace `hmm_analysis.py` or add as `ihmm_analysis.py`
- Register in `enhanced_report_generator.py` as `'ihmm_segmentation'`
- Use `track_quality_metrics.estimate_localization_precision()` for σ_loc

**Impact**: Reduces false positive state transitions by 40-60% in noisy data.

---

## 3. HIGH PRIORITY: Tracking-Free Methods (MISSING)

### 3.1 DDM (Differential Dynamic Microscopy)
**Status**: ❌ **NOT IMPLEMENTED**  
**Reference**: [DDM microrheology BioRxiv 2025.01.09.632077][4]

**Why needed**: When tracking fails (high density, low SNR), DDM bypasses trajectories.

**What's needed**:
```python
class DDMAnalyzer:
    """
    Differential Dynamic Microscopy for tracking-free G'(ω), G"(ω).
    """
    
    def __init__(self, image_stack: np.ndarray, pixel_size_um: float, 
                 frame_interval_s: float):
        """
        image_stack: (T, H, W) array
        """
        self.stack = image_stack
        self.dt = frame_interval_s
        self.px = pixel_size_um
    
    def compute_image_structure_function(self, q_range: np.ndarray) -> Dict:
        """
        Calculate D(q, Δt) = <|I(q, t+Δt) - I(q, t)|²>
        
        Steps:
        1. FFT each frame → I(q, t)
        2. For each lag Δt, compute squared difference
        3. Azimuthal average over |q|
        
        Returns:
        {
            'q_values': array,  # μm⁻¹
            'lag_times': array,  # s
            'D_qt': array,  # (len(q), len(lag_times))
            'method': 'ddm'
        }
        """
        pass
    
    def extract_rheology(self, D_qt: np.ndarray, 
                        temperature: float = 298.15,
                        particle_radius_um: float = 0.5) -> Dict:
        """
        From D(q,t), extract MSD(t) → G'(ω), G"(ω).
        
        Uses inverse problem:
        D(q,t) ∝ [1 - exp(-q²·MSD(t)/6)]
        
        Then GSER transform: MSD → G*(ω)
        
        Returns:
        {
            'frequencies_hz': array,
            'g_prime_pa': array,
            'g_double_prime_pa': array,
            'effective_msd': DataFrame,
            'method': 'ddm_gser'
        }
        """
        pass
    
    def cross_validate_spt(self, tracks_df: pd.DataFrame) -> Dict:
        """
        Compare DDM-derived G* with SPT-derived G*.
        
        Large discrepancies indicate:
        - SPT tracking bias (wrong particles, swaps)
        - Active processes (DDM more ensemble-representative)
        """
        pass
```

**Integration**:
- Add menu in `app.py`: "Tracking-Free Analysis" tab
- Auto-suggest DDM when TrackMate density > 0.3 particles/μm² (tracking unreliable)
- Register in `enhanced_report_generator.py` as `'ddm_rheology'`

**Data requirements**:
- Raw image stack (not just tracks)
- Minimum 100 frames for good statistics
- Stationary sample (no drift)

**Impact**: Enables rheology on 100x denser samples than SPT can handle.

---

### 3.2 RICS (Raster Image Correlation Spectroscopy)
**Status**: ❌ **NOT IMPLEMENTED**  
**Reference**: [RICS review 2025 PubMed 40996071][6]

**Why needed**: Provides D(x,y) spatial maps when density precludes tracking.

**What's needed**:
```python
class RICSAnalyzer:
    """
    Raster Image Correlation Spectroscopy for D(x,y) maps.
    """
    
    def __init__(self, image_stack: np.ndarray, pixel_size_um: float,
                 pixel_dwell_time_s: float, line_time_s: float):
        """
        pixel_dwell_time_s: Time per pixel in raster scan
        line_time_s: Time per line (includes flyback)
        """
        pass
    
    def compute_spatial_acf(self, roi_size: int = 32) -> Dict:
        """
        Calculate G(ξ, ψ) = <I(x,y,t)·I(x+ξ,y+ψ,t)> / <I>²
        
        ξ: spatial lag (pixels)
        ψ: line lag
        
        Returns:
        {
            'acf_2d': array,  # (2*roi_size, 2*roi_size)
            'xi_pixels': array,
            'psi_lines': array
        }
        """
        pass
    
    def fit_diffusion_model(self, acf_2d: np.ndarray) -> Dict:
        """
        Fit G(ξ, ψ) to 2D Gaussian decay → extract D.
        
        Model: G(ξ,ψ) = G0 · exp(-2·(ξ²·τ_pixel + ψ²·τ_line) / (ω₀² + 4Dt))
        
        Returns:
        {
            'D_um2_per_s': float,
            'particle_number': float,  # From G(0,0)
            'w0_um': float,  # PSF width
            'fit_quality': float
        }
        """
        pass
    
    def spatial_diffusion_map(self, grid_size_um: float = 5.0) -> Dict:
        """
        Sliding window RICS across image → D(x, y) heatmap.
        
        Returns:
        {
            'x_centers_um': array,
            'y_centers_um': array,
            'D_map': array,  # (ny, nx)
            'confidence_map': array  # Fit R²
        }
        """
        pass
```

**Integration**:
- Add to "Tracking-Free Analysis" tab in `app.py`
- Suggest when intensity fluctuations visible but tracking fails
- Cross-validate with `spatial_microrheology_map()` from `rheology.py`

**Impact**: Provides D maps in crowded environments (chromatin, membranes).

---

## 4. MEDIUM PRIORITY: Active Rheology Calibration (MISSING)

### 4.1 AFM/OT Import and Cross-Validation
**Status**: ❌ **NOT IMPLEMENTED**  
**References**:
- [AFM frequency-dependent PubMed 40631243][7]
- [TimSOM optical tweezers Nature s41565-024-01830-y][8]

**Why needed**: Detect non-equilibrium (SPT GSER assumes thermal equilibrium).

**What's needed**:
```python
class ActiveRheologyCalibrator:
    """Import AFM/OT data and cross-validate with SPT-GSER."""
    
    def import_afm_sweep(self, file_path: str) -> pd.DataFrame:
        """
        Import AFM frequency sweep (0.1-100 Hz).
        
        Expected format (CSV):
        frequency_hz, g_prime_pa, g_double_prime_pa, phase_deg
        """
        pass
    
    def import_timsom_sweep(self, file_path: str) -> pd.DataFrame:
        """
        Import TimSOM optical tweezer data.
        
        Expected format:
        frequency_hz, G_star_complex, position_nm
        """
        pass
    
    def cross_validate_spt(self, spt_rheology: Dict, 
                          afm_data: pd.DataFrame) -> Dict:
        """
        Compare SPT-derived G*(ω) with AFM G*(ω).
        
        Returns:
        {
            'frequency_overlap_hz': array,  # Common frequencies
            'spt_g_prime': array,
            'afm_g_prime': array,
            'agreement_ratio': float,  # Mean(SPT/AFM) over overlap
            'equilibrium_valid': bool,  # True if ratio ≈ 1.0
            'deviation_magnitude': float,  # |SPT - AFM| / AFM
            'interpretation': str  # "Equilibrium", "Active stress", "Non-thermal"
        }
        """
        # Flag if ratio > 2 or < 0.5 → active processes
        pass
    
    def generate_equilibrium_badge(self, validation: Dict) -> str:
        """
        Return HTML badge for reports:
        - 🟢 "Equilibrium Valid" (0.8 < ratio < 1.2)
        - 🟡 "Caution: Mild Deviation" (0.5 < ratio < 2)
        - 🔴 "Non-Equilibrium: Active Stress" (ratio > 2 or < 0.5)
        """
        pass
```

**Integration**:
- Add "Calibration" tab in `app.py`
- File uploader for AFM/OT CSV
- Overlay plots: SPT G'(ω) vs AFM G'(ω)
- Display equilibrium badge in all rheology reports

**UI mockup**:
```python
st.subheader("🔬 Active Rheology Calibration")
afm_file = st.file_uploader("Upload AFM frequency sweep (CSV)", type=['csv'])

if afm_file and 'rheology_results' in st.session_state:
    calibrator = ActiveRheologyCalibrator()
    afm_data = calibrator.import_afm_sweep(afm_file)
    
    validation = calibrator.cross_validate_spt(
        st.session_state.rheology_results, afm_data
    )
    
    badge = calibrator.generate_equilibrium_badge(validation)
    st.markdown(badge, unsafe_allow_html=True)
    
    if not validation['equilibrium_valid']:
        st.warning(f"⚠️ {validation['interpretation']}")
        st.caption("SPT-GSER assumes thermal equilibrium. "
                  "Consider reporting moduli as **apparent**.")
```

**Impact**: Prevents misinterpretation of active stresses as passive rheology.

---

## 5. LOW PRIORITY: Advanced Tracking (MISSING)

### 5.1 Microsecond Sampling Support (SpeedyTrack)
**Status**: ❌ **NOT IMPLEMENTED**  
**Reference**: [SpeedyTrack PMC12026894][1]

**Current limitation**:
```python
# constants.py assumes millisecond scale
DEFAULT_FRAME_INTERVAL = 0.1  # seconds (100 ms)
# No support for:
# - Irregular sampling (Δt varies per frame)
# - Microsecond time resolution (Δt < 1 ms)
```

**What's needed**:
```python
# In data_loader.py
def load_tracks_irregular_sampling(file_path: str) -> pd.DataFrame:
    """
    Load tracks with per-frame time stamps (not uniform Δt).
    
    Expected columns:
    track_id, x, y, time_s  # <-- time_s is absolute, not frame index
    
    Returns DataFrame with:
    track_id, frame_index, x, y, time_s, dt_s  # dt_s = time[i] - time[i-1]
    """
    pass

# In msd_calculation.py
def calculate_msd_irregular(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    MSD for irregular sampling.
    
    Cannot use fixed lag indices. Instead:
    1. Define time bins (e.g., [0-1 μs, 1-10 μs, 10-100 μs])
    2. For each pair of points, calculate Δr² and assign to time bin
    3. Average within bins
    """
    pass

# In rheology.py
class MicrorheologyAnalyzer:
    def __init__(self, ..., irregular_sampling: bool = False):
        self.irregular = irregular_sampling
    
    def calculate_complex_modulus_gser(self, ...):
        if self.irregular:
            # Use weighted GSER with variable dt
            pass
```

**UI integration**:
```python
# In app.py data loading
sampling_mode = st.radio("Sampling Mode", 
                        ["Regular (fixed Δt)", 
                         "Irregular (microsecond/SpeedyTrack)"])

if sampling_mode == "Irregular":
    st.info("🚀 Microsecond sampling mode enabled. "
            "Expects 'time_s' column in CSV (absolute time stamps).")
```

**Impact**: Enables high-frequency rheology (>1 kHz), removes motion blur bias for fast diffusers.

---

### 5.2 Deep Learning Trajectory Inference
**Status**: ❌ **NOT IMPLEMENTED**  
**References**:
- [SPTnet transformer BioRxiv 2025.02.04.636521][13]
- [DeepSPT existing work]

**Why needed**: Short/noisy tracks where classical methods fail.

**What's needed**:
```python
class DeepSPTInference:
    """Transformer-based trajectory analysis (SPTnet)."""
    
    def __init__(self, model_path: str = 'sptnet_pretrained.pth'):
        """Load pre-trained transformer model."""
        import torch
        self.model = torch.load(model_path)
    
    def predict_diffusion_state(self, track: np.ndarray) -> Dict:
        """
        End-to-end inference: track → (D, α, state_sequence).
        
        Handles:
        - Short tracks (N < 20 steps) where MSD fails
        - Noisy tracks (SNR < 5)
        - State transitions without explicit HMM
        
        Returns:
        {
            'D_um2_per_s': float,
            'alpha': float,
            'state_labels': array,  # Per-step
            'uncertainty': Dict,  # MC dropout samples
            'model': 'SPTnet'
        }
        """
        pass
    
    def predict_motion_class(self, track: np.ndarray) -> Dict:
        """
        Classify: normal/confined/directed/anomalous.
        
        Returns probabilities + uncertainty from ensemble.
        """
        pass
```

**Integration**:
- Optional analyzer in `enhanced_report_generator.py`: `'deep_spt'`
- Use when `track_quality_score < 0.5` (short/noisy)
- Requires `torch`, model weights download (~50 MB)

**Caveats**:
- Models trained on synthetic data (validate on real data)
- Requires GPU for speed (CPU fallback >10x slower)
- Provide model cards with training details

**Impact**: Recovers 30-50% more usable tracks from low-quality data.

---

## 6. CRITICAL: Equilibrium Validity Detection (MISSING)

### 6.1 Equilibrium Validity Badge System
**Status**: ❌ **NOT IMPLEMENTED**  
**Why critical**: GSER assumes thermal equilibrium. Active stresses violate this.

**What's needed**:
```python
class EquilibriumValidator:
    """Multi-test system for equilibrium validity."""
    
    def check_vacf_symmetry(self, vacf_df: pd.DataFrame) -> Dict:
        """
        Thermal equilibrium → VACF(-τ) = VACF(+τ).
        
        Returns:
        {
            'symmetry_score': float,  # 0-1, 1 = perfect
            'valid': bool,  # symmetry > 0.8
            'interpretation': 'Equilibrium' or 'Active drive detected'
        }
        """
        pass
    
    def check_1p_2p_agreement(self, one_point_G: Dict, 
                             two_point_G: Dict) -> Dict:
        """
        1P and 2P should agree if homogeneous + equilibrium.
        
        Large discrepancy → local adhesion, active stress, or heterogeneity.
        
        Returns:
        {
            'agreement_ratio': float,  # G_2P / G_1P
            'valid': bool,  # 0.8 < ratio < 1.2
            'interpretation': str
        }
        """
        pass
    
    def check_afm_concordance(self, spt_G: Dict, afm_G: Dict) -> Dict:
        """Use cross-validation from ActiveRheologyCalibrator."""
        pass
    
    def generate_validity_report(self, tracks_df: pd.DataFrame,
                                 rheology_results: Dict,
                                 afm_data: pd.DataFrame = None) -> Dict:
        """
        Run all checks and return composite badge.
        
        Returns:
        {
            'overall_valid': bool,
            'tests_passed': int,  # Out of 3
            'badge': '🟢 Equilibrium Valid' or '🔴 Non-Equilibrium',
            'recommendations': [
                'Report moduli as apparent',
                'Check for active stress (myosin, kinesin)',
                'Consider AFM validation'
            ]
        }
        """
        pass
```

**Integration**:
- Auto-run after any rheology analysis
- Display badge prominently in reports
- Add to PDF exports

**UI mockup**:
```python
# In enhanced_report_generator.py
def _plot_microrheology(self, result):
    """Add equilibrium badge to all rheology plots."""
    
    validator = EquilibriumValidator()
    validity = validator.generate_validity_report(
        self.tracks_df, result, afm_data=None
    )
    
    fig = go.Figure()
    # ... existing G' and G" traces ...
    
    # Add annotation
    fig.add_annotation(
        text=f"{validity['badge']}<br>Tests passed: {validity['tests_passed']}/3",
        xref="paper", yref="paper",
        x=0.02, y=0.98,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)"
    )
    
    return fig
```

**Impact**: Prevents systematic misinterpretation of non-equilibrium systems.

---

## 7. Implementation Roadmap

### Phase 1 (2 weeks): High Priority
1. **CVE/MLE Estimators** (`biased_inference.py`)
   - Implement CVE (2 days)
   - Implement MLE with blur (3 days)
   - Integrate into `enhanced_report_generator.py` (1 day)
   - Test on short tracks (1 day)

2. **Acquisition Advisor** (`acquisition_advisor.py`)
   - Load 2024 bias tables (1 day)
   - Implement frame rate recommendation (2 days)
   - UI integration in `app.py` (1 day)
   - Validation function (1 day)

3. **Equilibrium Validity System** (`equilibrium_validator.py`)
   - VACF symmetry check (1 day)
   - 1P-2P agreement (1 day)
   - AFM concordance (1 day)
   - Badge generation (1 day)
   - Integration in all rheology outputs (1 day)

### Phase 2 (3 weeks): Tracking-Free Methods
4. **DDM Module** (`ddm_analyzer.py`)
   - Image structure function (3 days)
   - MSD extraction (2 days)
   - GSER transform (1 day)
   - SPT cross-validation (2 days)
   - UI integration (2 days)

5. **RICS Module** (`rics_analyzer.py`)
   - Spatial ACF calculation (3 days)
   - Diffusion model fitting (2 days)
   - Spatial map generation (2 days)
   - UI integration (1 day)

### Phase 3 (2 weeks): State Segmentation
6. **iHMM with Blur** (`ihmm_analysis.py`)
   - Variational EM implementation (5 days)
   - Blur-aware emission models (3 days)
   - Dwell time posteriors (2 days)
   - Integration/testing (2 days)

### Phase 4 (3 weeks): Optional Enhancements
7. **Microsecond Sampling** (irregular Δt support)
   - Data loader modifications (2 days)
   - MSD calculation update (2 days)
   - GSER update (2 days)
   - Testing (2 days)

8. **Deep SPT** (SPTnet/DeepSPT)
   - Model integration (3 days)
   - Inference pipeline (2 days)
   - Uncertainty quantification (2 days)
   - UI integration (2 days)

9. **AFM/OT Import** (`active_rheology_calibrator.py`)
   - File parsers (2 days)
   - Cross-validation logic (2 days)
   - UI integration (1 day)

---

## 8. Dependencies to Add

```txt
# requirements.txt additions

# Phase 1
scipy>=1.11.0  # Already present - OK

# Phase 2
scikit-image>=0.21.0  # For DDM/RICS image processing
pyfftw>=0.13.0  # Fast FFT for DDM (optional, falls back to numpy)

# Phase 3
pymc>=5.0.0  # Variational inference for iHMM (alternative: use custom EM)

# Phase 4
torch>=2.0.0  # DeepSPT (optional)
torchvision>=0.15.0  # DeepSPT (optional)
```

---

## 9. Validation Strategy

### For Each New Module:
1. **Synthetic data tests** (10 scenarios):
   - Pure Brownian (D known)
   - Confined motion (α = 0.5)
   - Directed motion (α = 2.0)
   - State switching (2-3 states)
   - Short tracks (N = 10-20)
   - Noisy tracks (SNR = 3-5)
   - Irregular sampling (microsecond)
   - High density (DDM/RICS)
   - Active stress (non-equilibrium)
   - Combined challenges

2. **AnDi Challenge benchmarks** (PMC12283970):
   - Use 2024/2025 reference datasets
   - Compare D, α, changepoint detection
   - Report metrics: RMSE, recall, precision

3. **Real data validation**:
   - `Cell1_spots.csv` (existing)
   - Literature datasets with known D/G* (e.g., glycerol standards)
   - Cross-compare: MSD vs CVE vs DeepSPT
   - Cross-compare: SPT vs DDM vs AFM

---

## 10. Documentation Checklist

For each new module, create:
1. **API docs** (`NEW_MODULE_API.md`)
   - Function signatures
   - Parameter descriptions
   - Return value schemas
   - Usage examples

2. **User guide** (`NEW_MODULE_GUIDE.md`)
   - When to use
   - Interpretation of results
   - Troubleshooting
   - Comparison with existing methods

3. **Literature references** (`NEW_MODULE_REFERENCES.md`)
   - Key papers with DOIs
   - Mathematical derivations
   - Known limitations

4. **Test report** (`NEW_MODULE_TEST_RESULTS.md`)
   - Validation on synthetic data
   - Performance benchmarks
   - Known edge cases

---

## 11. What's Already Strong in SPT2025B

✅ **Excellent foundations**:
- Two-point microrheology (Oct 2025)
- GSER-based rheology with creep/relaxation
- Track quality metrics (comprehensive)
- Statistical validation (bootstrap, goodness-of-fit)
- Advanced metrics (NGP, van Hove, TAMSD, VACF, Hurst)
- Basic HMM (needs upgrade to iHMM)
- Multi-channel analysis
- Batch processing and report generation

✅ **Strong data infrastructure**:
- Multi-format loaders (MVD2, Volocity, Imaris, CSV)
- State management (`data_access_utils.py`)
- Unit conversion system
- Project management (JSON-based)

✅ **Production-ready UI**:
- Streamlit-based with modularity
- Enhanced report generator with 25+ analyses
- Publication-ready plots (Plotly)
- PDF export

---

## 12. Priority Ranking for Immediate Impact

### Must-Have (Blocks 2025 adoption):
1. **CVE/MLE estimators** → Fixes D/α bias on short tracks
2. **Equilibrium validity badges** → Prevents GSER misuse
3. **Acquisition advisor** → Optimizes future experiments

### High Value (Enables new use cases):
4. **DDM module** → Tracking-free rheology for dense samples
5. **iHMM with blur** → Better state detection
6. **AFM/OT calibration** → Cross-validation

### Nice-to-Have (Advanced users):
7. **RICS module** → Spatial D maps
8. **Microsecond sampling** → High-frequency rheology
9. **DeepSPT** → Rescue low-quality data

---

## 13. One-Page Summary for Users

**Your SPT2025B is solid for 2020-2023 methods.**

To adopt **2025 cutting-edge methods**, you need:

| **Missing Feature** | **Why You Need It** | **Impact** |
|---------------------|---------------------|------------|
| CVE/MLE estimators | Reduces D/α bias on short/noisy tracks | 20-50% accuracy gain |
| Equilibrium badges | Detects when GSER is wrong (active stress) | Prevents misinterpretation |
| Acquisition advisor | Optimizes frame rate before experiment | 30-50% bias reduction |
| DDM/RICS | Tracking-free rheology (dense samples) | 100x density increase |
| iHMM with blur | Better state segmentation | 40-60% fewer false transitions |
| AFM/OT import | Cross-validates SPT with active rheology | Confidence in G* values |
| Microsecond support | High-frequency rheology (>1 kHz) | Removes motion blur |
| DeepSPT | Analyzes short/noisy tracks | 30-50% more usable data |

**Total implementation time**: ~10 weeks (3 phases)  
**Lines of code to add**: ~3,000 (spread across 6-8 new modules)  
**New dependencies**: 3-4 Python packages  

**Biggest wins for effort**:
1. CVE/MLE (2 weeks) → immediate D/α accuracy boost
2. Equilibrium badges (1 week) → prevents systematic errors
3. DDM (3 weeks) → opens tracking-free workflows

---

## References (Quick Links)

[1]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12026894/ "SpeedyTrack microsecond SPT"
[2]: https://pubmed.ncbi.nlm.nih.gov/40691147/ "Long axial range 3D SPT"
[3]: https://pubmed.ncbi.nlm.nih.gov/39605363/ "GEM nanoparticles"
[4]: https://www.biorxiv.org/content/10.1101/2025.01.09.632077v2.full.pdf "DDM mucus microrheology"
[5]: https://pubmed.ncbi.nlm.nih.gov/38914653/ "iSCORS chromatin dynamics"
[6]: https://pubmed.ncbi.nlm.nih.gov/40996071/ "RICS review 2025"
[7]: https://pubmed.ncbi.nlm.nih.gov/40631243/ "AFM frequency-dependent microrheology"
[8]: https://www.nature.com/articles/s41565-024-01830-y "TimSOM optical tweezer"
[9]: https://www.biorxiv.org/content/10.1101/2025.04.07.647540.full.pdf "Rheo-FLUCS active rheology"
[10]: https://pubmed.ncbi.nlm.nih.gov/20866658/ "Berglund CVE/MLE"
[11]: https://pubmed.ncbi.nlm.nih.gov/38724858/ "Optimal temporal resolution 2024"
[12]: https://pmc.ncbi.nlm.nih.gov/articles/PMC6050756/ "Variational EM for SPT"
[13]: https://www.biorxiv.org/content/10.1101/2025.02.04.636521v1 "SPTnet transformer"
[14]: https://pmc.ncbi.nlm.nih.gov/articles/PMC12283970/ "AnDi Challenge benchmarks"
[15]: https://pubs.rsc.org/en/content/articlehtml/2025/sm/d4sm01390e "Two-point microrheology review"
[16]: https://pmc.ncbi.nlm.nih.gov/articles/PMC10783629/ "u-track3D"

---

**End of Gap Analysis**  
**Next Step**: Review with team and prioritize Phase 1 (CVE/MLE + Equilibrium + Advisor).
