"""Ground-truth physics validation tests for SPT analysis formulas.

These tests enforce analytical/known-reference behavior for MSD fitting,
confinement, and microrheology calculations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis import classify_motion, fit_alpha
from msd_calculation import calculate_msd, fit_msd_linear
from rheology import MicrorheologyAnalyzer


def _ensemble_msd(msd_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-track MSD into an ensemble MSD curve."""
    out = (
        msd_df.groupby("lag_time", as_index=False)
        .agg(msd=("msd", "mean"), n_points=("n_points", "sum"))
        .sort_values("lag_time")
    )
    return out


def _simulate_brownian_track(
    rng: np.random.Generator,
    n_frames: int,
    D_true: float,
    dt: float,
    sigma_loc: float,
    track_id: int = 0,
) -> pd.DataFrame:
    sigma_step = np.sqrt(2.0 * D_true * dt)
    x_true = np.cumsum(rng.normal(0.0, sigma_step, n_frames))
    y_true = np.cumsum(rng.normal(0.0, sigma_step, n_frames))

    x_obs = x_true + rng.normal(0.0, sigma_loc, n_frames)
    y_obs = y_true + rng.normal(0.0, sigma_loc, n_frames)

    return pd.DataFrame(
        {
            "track_id": track_id,
            "frame": np.arange(n_frames),
            "x": x_obs,
            "y": y_obs,
        }
    )


@pytest.fixture
def brownian_tracks() -> tuple[pd.DataFrame, float, float, float]:
    """50 Brownian tracks with known parameters (Task 5 fixture spec)."""
    rng = np.random.default_rng(42)
    D_true = 0.05
    dt = 0.1
    sigma_loc = 0.025
    n_frames = 200
    n_tracks = 50

    tracks = []
    for tid in range(n_tracks):
        tracks.append(
            _simulate_brownian_track(
                rng=rng,
                n_frames=n_frames,
                D_true=D_true,
                dt=dt,
                sigma_loc=sigma_loc,
                track_id=tid,
            )
        )

    return pd.concat(tracks, ignore_index=True), D_true, sigma_loc, dt


def test_msd_gap_rejection() -> None:
    """MSD must ignore displacement pairs that span missing frames."""
    continuous = pd.DataFrame(
        {
            "track_id": [1, 1, 1, 1],
            "frame": [0, 1, 2, 3],
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )

    gapped = pd.DataFrame(
        {
            "track_id": [2, 2, 2, 2],
            "frame": [0, 1, 3, 4],
            "x": [0.0, 1.0, 3.0, 4.0],
            "y": [0.0, 0.0, 0.0, 0.0],
        }
    )

    msd_cont = calculate_msd(continuous, max_lag=1, min_track_length=1)
    msd_gap = calculate_msd(gapped, max_lag=1, min_track_length=1)

    msd_cont_lag1 = float(msd_cont.loc[msd_cont["lag_time"] == 1.0, "msd"].iloc[0])
    msd_gap_lag1 = float(msd_gap.loc[msd_gap["lag_time"] == 1.0, "msd"].iloc[0])
    n_points_gap = int(msd_gap.loc[msd_gap["lag_time"] == 1.0, "n_points"].iloc[0])

    assert msd_cont_lag1 == pytest.approx(1.0, abs=1e-12)
    assert msd_gap_lag1 == pytest.approx(msd_cont_lag1, abs=1e-12)
    assert n_points_gap == 2


def test_msd_brownian_linear(brownian_tracks: tuple[pd.DataFrame, float, float, float]) -> None:
    """For Brownian motion, short-lag MSD slope should be 4D."""
    tracks_df, D_true, _, dt = brownian_tracks
    msd_df = calculate_msd(tracks_df, max_lag=40, frame_interval=dt, min_track_length=5)
    ens = _ensemble_msd(msd_df)

    # Fit first 20 lags for linear regime check
    n_fit = 20
    slope, _ = np.polyfit(ens["lag_time"].values[:n_fit], ens["msd"].values[:n_fit], 1)
    slope_true = 4.0 * D_true

    assert slope == pytest.approx(slope_true, rel=0.10)


def test_D_recovery_corrected_fit(brownian_tracks: tuple[pd.DataFrame, float, float, float]) -> None:
    """Offset-aware MSD fit recovers D within 15%."""
    tracks_df, D_true, _, dt = brownian_tracks
    msd_df = calculate_msd(tracks_df, max_lag=60, frame_interval=dt, min_track_length=5)
    ens = _ensemble_msd(msd_df)

    fit = fit_msd_linear(ens, track_length=200, weighted=True)
    assert float(fit["D"]) == pytest.approx(D_true, rel=0.15)


def test_sigma_loc_recovery(brownian_tracks: tuple[pd.DataFrame, float, float, float]) -> None:
    """Offset-aware MSD fit recovers localization precision within 30%."""
    tracks_df, _, sigma_true, dt = brownian_tracks
    msd_df = calculate_msd(tracks_df, max_lag=60, frame_interval=dt, min_track_length=5)
    ens = _ensemble_msd(msd_df)

    fit = fit_msd_linear(ens, track_length=200, weighted=True)
    assert float(fit["sigma_loc"]) == pytest.approx(sigma_true, rel=0.30)


def test_old_linregress_bias_upward() -> None:
    """Legacy short-lag linregress can overestimate D by >20% for noisy tracks."""
    rng = np.random.default_rng(66)
    D_true = 0.05
    dt = 0.1
    sigma_loc = 0.025

    track = _simulate_brownian_track(rng, n_frames=200, D_true=D_true, dt=dt, sigma_loc=sigma_loc)
    msd_df = calculate_msd(track, max_lag=80, frame_interval=dt, min_track_length=5)
    ens = _ensemble_msd(msd_df)

    n_fit = 200 // 3
    slope, _, _, _, _ = stats.linregress(
        ens["lag_time"].values[:n_fit],
        ens["msd"].values[:n_fit],
    )
    D_old = slope / 4.0

    assert D_old > 1.2 * D_true


@dataclass
class _ConfinementResult:
    msd_plateau: float
    estimated_L: float


def _simulate_reflecting_box(
    rng: np.random.Generator,
    n_frames: int,
    D: float,
    dt: float,
    L: float,
    track_id: int,
) -> pd.DataFrame:
    """2D Brownian track in square box [0, L] with reflecting boundaries."""
    sigma_step = np.sqrt(2.0 * D * dt)
    x = np.empty(n_frames)
    y = np.empty(n_frames)
    x[0] = L / 2.0
    y[0] = L / 2.0

    for i in range(1, n_frames):
        x_new = x[i - 1] + rng.normal(0.0, sigma_step)
        y_new = y[i - 1] + rng.normal(0.0, sigma_step)

        # Reflect as needed (can reflect multiple times on long steps)
        while x_new < 0.0 or x_new > L:
            x_new = -x_new if x_new < 0.0 else 2.0 * L - x_new
        while y_new < 0.0 or y_new > L:
            y_new = -y_new if y_new < 0.0 else 2.0 * L - y_new

        x[i] = x_new
        y[i] = y_new

    return pd.DataFrame(
        {
            "track_id": track_id,
            "frame": np.arange(n_frames),
            "x": x,
            "y": y,
        }
    )


def _estimate_confinement_from_plateau(msd_df: pd.DataFrame) -> _ConfinementResult:
    ens = _ensemble_msd(msd_df)
    n_tail = max(5, len(ens) // 5)
    plateau = float(np.mean(ens["msd"].values[-n_tail:]))
    estimated_L = float(np.sqrt(3.0 * plateau))
    return _ConfinementResult(msd_plateau=plateau, estimated_L=estimated_L)


def test_confinement_radius_formula_exact() -> None:
    L = 0.5
    msd_plateau = L**2 / 3.0
    radius_calc = np.sqrt(3.0 * msd_plateau) / 2.0
    assert abs(radius_calc - L / 2.0) < 1e-10


def test_confinement_radius_saxton_recovery() -> None:
    rng = np.random.default_rng(7)
    D = 0.01
    dt = 0.1
    L_true = 0.5

    tracks = [
        _simulate_reflecting_box(rng, n_frames=500, D=D, dt=dt, L=L_true, track_id=i)
        for i in range(30)
    ]
    tracks_df = pd.concat(tracks, ignore_index=True)

    msd_df = calculate_msd(tracks_df, max_lag=120, frame_interval=dt, min_track_length=5)
    conf = _estimate_confinement_from_plateau(msd_df)

    assert conf.estimated_L == pytest.approx(L_true, rel=0.20)


def _analytic_newtonian_msd(eta_pa_s: float, a_m: float, T_K: float, lag_times_s: np.ndarray) -> pd.DataFrame:
    kB = 1.380649e-23
    D = (kB * T_K) / (6.0 * np.pi * eta_pa_s * a_m)
    msd = 4.0 * D * lag_times_s  # 2D projected MSD
    return pd.DataFrame({"lag_time_s": lag_times_s, "msd_m2": msd})


def test_gser_newtonian() -> None:
    eta_true = 1e-3
    a_m = 500e-9
    T = 300.0

    lag_times = np.logspace(-2, 1, 120)
    msd_df = _analytic_newtonian_msd(eta_true, a_m, T, lag_times)

    analyzer = MicrorheologyAnalyzer(particle_radius_m=a_m, temperature_K=T)
    omegas = np.logspace(-1, 2, 30)

    gp = []
    gpp = []
    for omega in omegas:
        g1, g2 = analyzer.calculate_complex_modulus_gser(msd_df, omega)
        gp.append(g1)
        gpp.append(g2)

    gp = np.asarray(gp)
    gpp = np.asarray(gpp)

    assert np.nanmax(np.abs(gp)) < 0.01 * np.nanmean(np.abs(gpp))
    assert np.allclose(gpp, eta_true * omegas, rtol=0.10, atol=0.0)

    eta_eff = analyzer.calculate_effective_viscosity(msd_df)
    assert eta_eff == pytest.approx(eta_true, rel=0.05)


def test_creep_compliance_units_consistency() -> None:
    eta_true = 1e-3
    a_m = 500e-9
    T = 300.0

    lag_times = np.logspace(-2, 1, 120)
    msd_df = _analytic_newtonian_msd(eta_true, a_m, T, lag_times)

    analyzer = MicrorheologyAnalyzer(particle_radius_m=a_m, temperature_K=T)
    creep = analyzer.calculate_creep_compliance(msd_df)
    assert creep["success"]

    t = np.asarray(creep["time_s"])
    J = np.asarray(creep["creep_compliance_pa_inv"])

    # Newtonian fluid: J(t) = t / eta
    J_true = t / eta_true
    assert np.allclose(J, J_true, rtol=0.10, atol=0.0)

    # Consistency check with direct GSER estimate at matched times
    omegas = 1.0 / t
    Gmag = []
    for omega in omegas:
        gp, gpp = analyzer.calculate_complex_modulus_gser(msd_df, omega)
        Gmag.append(np.sqrt(gp**2 + gpp**2))
    Gmag = np.asarray(Gmag)
    JG = J * Gmag

    assert np.all(np.isfinite(JG))
    assert np.nanmax(JG) < 1.1


def test_msd_weighted_vs_unweighted() -> None:
    """Weighted MSD fitting should reduce D-estimate variance across simulations."""
    D_true = 0.05
    dt = 0.1
    sigma_loc = 0.03

    d_weighted = []
    d_unweighted = []

    for seed in range(100):
        rng = np.random.default_rng(seed)
        track = _simulate_brownian_track(rng, n_frames=120, D_true=D_true, dt=dt, sigma_loc=sigma_loc)
        msd_df = calculate_msd(track, max_lag=50, frame_interval=dt, min_track_length=5)
        ens = _ensemble_msd(msd_df)

        fit_w = fit_msd_linear(ens, track_length=120, weighted=True)
        fit_u = fit_msd_linear(ens, track_length=120, weighted=False)

        d_weighted.append(float(fit_w["D"]))
        d_unweighted.append(float(fit_u["D"]))

    d_weighted = np.asarray(d_weighted)
    d_unweighted = np.asarray(d_unweighted)

    var_w = float(np.var(d_weighted, ddof=1))
    var_u = float(np.var(d_unweighted, ddof=1))

    # One-sided F test: H1 var_u > var_w
    F = var_u / var_w
    p_value = 1.0 - stats.f.cdf(F, dfn=len(d_unweighted) - 1, dfd=len(d_weighted) - 1)

    assert var_w < var_u
    assert p_value < 0.05


def test_alpha_ci_normal() -> None:
    """95% CI for Brownian alpha should contain 1.0 with high coverage."""
    D_true = 0.05
    dt = 0.1
    sigma_loc = 0.02

    contains = []
    for seed in range(200):
        rng = np.random.default_rng(seed)
        track = _simulate_brownian_track(rng, n_frames=100, D_true=D_true, dt=dt, sigma_loc=sigma_loc)
        msd_df = calculate_msd(track, max_lag=30, frame_interval=dt, min_track_length=5)
        ens = _ensemble_msd(msd_df)

        alpha, alpha_se, ci95 = fit_alpha(
            ens["lag_time"].values,
            ens["msd"].values,
            ens["n_points"].values,
        )
        assert math.isfinite(alpha)
        assert math.isfinite(alpha_se)
        contains.append(ci95[0] <= 1.0 <= ci95[1])

    coverage = np.mean(contains)
    assert coverage >= 0.95


def test_alpha_motion_classification_ci() -> None:
    """Classification should be indeterminate when CI overlaps normal band."""
    motion, confidence = classify_motion(alpha=0.88, alpha_se=0.15)
    assert motion == "normal_diffusion"
    assert confidence == "indeterminate"
