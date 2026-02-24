import numpy as np

import bayesian_trajectory_inference as bti


def _inference() -> bti.BayesianDiffusionInference:
    return bti.BayesianDiffusionInference(
        frame_interval=0.1,
        localization_error=0.03,
        exposure_time=0.09,
        dimensions=2,
    )


def test_run_mcmc_rejects_unknown_backend():
    inf = _inference()
    displacements = np.random.default_rng(0).normal(size=(32, 2))
    result = inf.run_mcmc(displacements, n_steps=20, backend="bogus")
    assert not result["success"]
    assert "Unknown backend" in result["error"]


def test_run_mcmc_auto_fails_when_no_backends(monkeypatch):
    monkeypatch.setattr(bti, "EMCEE_AVAILABLE", False)
    monkeypatch.setattr(bti, "NUMPYRO_AVAILABLE", False)
    monkeypatch.setattr(bti, "JAX_AVAILABLE", False)

    inf = _inference()
    displacements = np.random.default_rng(1).normal(size=(16, 2))
    result = inf.run_mcmc(displacements, n_steps=20, backend="auto")
    assert not result["success"]
    assert "No Bayesian backend available" in result["error"]


def test_analyze_track_uses_requested_backend(monkeypatch):
    inf = _inference()

    def fake_run_mcmc(self, displacements, **kwargs):
        assert kwargs["backend"] == "numpyro"
        return {
            "success": True,
            "backend": "numpyro",
            "samples": np.ones((4, 1)),
            "D_median": 0.1,
            "D_std": 0.01,
            "D_credible_interval": (0.08, 0.12),
            "D_mean": 0.1,
            "n_samples": 4,
            "diagnostics": {"mean_acceptance": 0.9},
            "sampler": object(),
        }

    monkeypatch.setattr(bti.BayesianDiffusionInference, "run_mcmc", fake_run_mcmc)

    track = np.cumsum(np.random.default_rng(2).normal(scale=0.1, size=(12, 2)), axis=0)
    result = inf.analyze_track_bayesian(
        track,
        n_steps=20,
        backend="numpyro",
        return_samples=False,
    )

    assert result["success"]
    assert result["method"] == "Bayesian_NUMPYRO"
    assert "samples" not in result
