from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

@dataclass
class HMMConfig:
    pixel_size: float = 1.0
    frame_interval: float = 1.0
    lag_frames: int = 1
    n_states: int = 3
    sticky_kappa: float = 10.0
    dirichlet_alpha: float = 1.0
    mu0: Tuple[float, float] = (0.0, 0.0)
    kappa0: float = 1e-3
    a0: float = 2.0   # Inverse-Gamma shape
    b0: float = 1e-3  # Inverse-Gamma scale
    max_iter: int = 50
    tol: float = 1e-4
    random_state: Optional[int] = 1234

class BayesHMMDiffusion:
    """
    MAP HMM with isotropic Gaussian emissions for 2D increments.
    State k: Δ ~ N(μ_k, σ_k^2 I); D_k = σ_k^2 / (4 * Δt * lag_frames); drift = μ_k / (Δt * lag_frames).
    """
    def __init__(self, tracks_df: pd.DataFrame, cfg: HMMConfig):
        req = {'track_id','frame','x','y'}
        miss = req - set(tracks_df.columns)
        if miss:
            raise ValueError(f"tracks_df missing columns: {miss}")
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_state)
        df = tracks_df.loc[:, ['track_id','frame','x','y']].dropna().copy()
        df['frame'] = df['frame'].astype(int)
        df.sort_values(['track_id','frame'], inplace=True)
        df['x_um'] = df['x'] * cfg.pixel_size
        df['y_um'] = df['y'] * cfg.pixel_size
        self.df = df

    def _steps(self, g: pd.DataFrame):
        L = self.cfg.lag_frames
        x = g['x_um'].values; y = g['y_um'].values; fr = g['frame'].values
        if len(x) <= L:
            return np.array([]), np.empty((0,2)), np.array([])
        dx = x[L:] - x[:-L]; dy = y[L:] - y[:-L]
        frames_mid = (fr[L:] + fr[:-L]) // 2
        X = np.stack([dx, dy], axis=1)
        return frames_mid, X, fr

    def _kmeans_init(self, X: np.ndarray, K: int):
        sp = np.linalg.norm(X, axis=1)
        # quantile-based speed clustering
        qs = np.quantile(sp, np.linspace(0.1, 0.9, K))
        z = np.argmin(np.abs(sp[:,None] - qs[None,:]), axis=1)
        mu = np.zeros((K,2)); var = np.zeros(K)
        for k in range(K):
            Xk = X[z==k]
            if Xk.size:
                mu[k] = Xk.mean(axis=0)
                var[k] = np.mean(np.sum((Xk - mu[k])**2, axis=1))/2.0
            else:
                mu[k] = 0
                var[k] = np.var(X)/2.0 + 1e-3
        return mu, np.maximum(var, 1e-6)

    def _gauss_logpdf_iso(self, X, mu, var):
        diff = X[:,None,:] - mu[None,:,:]
        q = np.sum(diff*diff, axis=2)
        return -0.5*(2*np.log(2*np.pi*var)[None,:] + q/var[None,:])

    def _forward_backward(self, logp_emit, A, pi):
        T, K = logp_emit.shape
        logA = np.log(A + 1e-300); logpi = np.log(pi + 1e-300)
        logalpha = np.zeros((T,K)); logbeta = np.zeros((T,K))
        logalpha[0] = logpi + logp_emit[0]
        for t in range(1, T):
            logalpha[t] = logp_emit[t] + np.logaddexp.reduce(logalpha[t-1][:,None] + logA, axis=0)
        logbeta[-1] = 0
        for t in range(T-2, -1, -1):
            logbeta[t] = np.logaddexp.reduce(logA + (logp_emit[t+1] + logbeta[t+1])[None,:], axis=1)
        loglik = np.logaddexp.reduce(logalpha[-1])
        loggamma = logalpha + logbeta - loglik
        gamma = np.exp(loggamma)
        xi = np.zeros((T-1,K,K))
        for t in range(T-1):
            m = logalpha[t][:,None] + logA + logp_emit[t+1][None,:] + logbeta[t+1][None,:]
            m -= np.max(m)
            xi[t] = np.exp(m) / np.sum(np.exp(m))
        return gamma, xi, float(loglik)

    def _viterbi(self, logp_emit, A, pi):
        T, K = logp_emit.shape
        logA = np.log(A + 1e-300); logpi = np.log(pi + 1e-300)
        dp = np.zeros((T,K)); ptr = np.zeros((T,K), dtype=int)
        dp[0] = logpi + logp_emit[0]
        for t in range(1, T):
            scores = dp[t-1][:,None] + logA
            ptr[t] = np.argmax(scores, axis=0)
            dp[t] = logp_emit[t] + np.max(scores, axis=0)
        z = np.zeros(T, dtype=int); z[-1] = np.argmax(dp[-1])
        for t in range(T-2, -1, -1):
            z[t] = ptr[t+1, z[t+1]]
        return z

    def _fit(self, X: np.ndarray):
        K = self.cfg.n_states
        T = X.shape[0]
        if T < K + 5:
            raise ValueError("Sequence too short for n_states.")
        mu, var = self._kmeans_init(X, K)
        A = np.full((K,K), 1.0)
        np.fill_diagonal(A, 1.0 + self.cfg.sticky_kappa)
        A /= A.sum(axis=1, keepdims=True)
        pi = np.full(K, 1.0/K)
        mu0 = np.array(self.cfg.mu0); k0 = self.cfg.kappa0
        a0 = self.cfg.a0; b0 = self.cfg.b0
        last_ll = -np.inf
        for _ in range(self.cfg.max_iter):
            logp = self._gauss_logpdf_iso(X, mu, var)
            gamma, xi, ll = self._forward_backward(logp, A, pi)
            Nk = gamma.sum(axis=0) + 1e-12
            Xi = xi.sum(axis=0)
            alpha_prior = np.full((K,K), self.cfg.dirichlet_alpha)
            for k in range(K):
                alpha_prior[k,k] += self.cfg.sticky_kappa
            A = (Xi + (alpha_prior - 1.0))
            A /= A.sum(axis=1, keepdims=True)
            pi = gamma[0] + (1.0/K)
            pi /= pi.sum()
            xbar = (gamma.T @ X) / Nk[:,None]
            mu = (k0*mu0[None,:] + Nk[:,None]*xbar)/(k0+Nk)[:,None]
            S = np.zeros(K)
            for k in range(K):
                diff = X - mu[k]
                S[k] = np.sum(gamma[:,k]*np.sum(diff*diff, axis=1))
            aN = a0 + Nk  # dimension=2 → adds Nk
            bN = b0 + 0.5*S + 0.5*(k0*Nk/(k0+Nk))*np.sum((xbar - mu0[None,:])**2, axis=1)
            var = bN / (aN + 1.0)
            var = np.clip(var, 1e-9, None)
            if abs(ll - last_ll) < self.cfg.tol:
                break
            last_ll = ll
        z = self._viterbi(logp, A, pi)
        return {'mu': mu, 'var': var, 'A': A, 'pi': pi, 'z': z, 'loglik': last_ll}

    def segment_all(self) -> Dict[str, object]:
        dtL = self.cfg.frame_interval * self.cfg.lag_frames
        out = {'success': True, 'tracks': {}, 'config': self.cfg.__dict__}
        for tid, g in self.df.groupby('track_id'):
            frames_mid, X, _ = self._steps(g)
            if X.shape[0] == 0:
                out['tracks'][tid] = {'segments': pd.DataFrame(), 'states': pd.DataFrame(), 'A': None}
                continue
            model = self._fit(X)
            z = model['z']
            segs = []
            s = 0
            for t in range(1, len(z)):
                if z[t] != z[t-1]:
                    segs.append((s, t-1)); s = t
            segs.append((s, len(z)-1))
            seg_rows = []
            for (i,j) in segs:
                k = int(z[i])
                mu_k = model['mu'][k]; var_k = model['var'][k]
                Dk = var_k / (4.0 * dtL)
                vx, vy = mu_k / dtL
                seg_rows.append({
                    'start_frame': int(frames_mid[i]),
                    'end_frame': int(frames_mid[j]),
                    'state': k,
                    'n_steps': int(j-i+1),
                    'D_state': float(Dk),
                    'vx_um_s': float(vx),
                    'vy_um_s': float(vy),
                    'speed_um_s': float(np.hypot(vx, vy))
                })
            seg_df = pd.DataFrame(seg_rows)
            state_rows = []
            for k in range(self.cfg.n_states):
                mu_k = model['mu'][k]; var_k = model['var'][k]
                Dk = var_k / (4.0 * dtL)
                state_rows.append({
                    'state': k,
                    'D_state': float(Dk),
                    'vx_um_s': float(mu_k[0]/dtL),
                    'vy_um_s': float(mu_k[1]/dtL),
                    'speed_um_s': float(np.hypot(mu_k[0], mu_k[1])/dtL)
                })
            out['tracks'][tid] = {'segments': seg_df,
                                  'states': pd.DataFrame(state_rows),
                                  'A': model['A']}
        return out
