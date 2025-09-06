"""
Adapter for integrating the external simulator 'nuclear-diffusion-si' with SPT2025B.

This module provides a thin wrapper that maps our app's mask + parameters to the
external simulator and converts its output into a DataFrame compatible with our UI.

Notes:
- The repository URL provided could not be validated from this environment.
- This adapter will try (in order):
  1) Import a Python package/module if installed.
  2) Use a local repo path (if provided) to run a CLI/entry script.
  3) Optionally fall back to the built-in DiffusionSimulator when allowed.

Usage in the app:
- Select 'Nuclear Diffusion SI' in the Simulation page and provide parameters.
- If the external simulator isn't available, a clear error is shown. An optional fallback
  path will run our built-in simulator instead to keep the UI working.
"""
from __future__ import annotations

import os
import sys
import json
import tempfile
import subprocess
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

try:
    # Optional: if the external package can be pip-installed and imported
    import nuclear_diffusion_si  # type: ignore
    HAS_PACKAGE = True
except Exception:
    nuclear_diffusion_si = None
    HAS_PACKAGE = False


def is_available(repo_path: Optional[str] = None) -> bool:
    """Return True if the external simulator seems available.

    Checks python package import or existence of a plausible repo path.
    """
    if HAS_PACKAGE:
        return True
    if repo_path and os.path.exists(repo_path):
        return True
    return False


def _write_mask_tmp(mask: np.ndarray, dirpath: str) -> str:
    """Persist mask to a temporary npy file and return path."""
    out_path = os.path.join(dirpath, "mask.npy")
    np.save(out_path, mask)
    return out_path


def _run_via_cli(repo_path: str, mask_path: str, params: Dict[str, Any]) -> Tuple[int, str, str]:
    """Attempt to run a hypothetical CLI from the repo.

    This is a placeholder. Adjust 'cli_module' and arguments to the actual interface
    once known. Returns (returncode, stdout, stderr).
    """
    # Guess a runner script; users can adapt this path as needed
    candidate = os.path.join(repo_path, "run.py")
    if not os.path.exists(candidate):
        # try python -m package style if user has it on PYTHONPATH
        cmd = [sys.executable, "-m", "nuclear_diffusion_si", "--mask", mask_path, "--params", json.dumps(params)]
    else:
        cmd = [sys.executable, candidate, "--mask", mask_path, "--params", json.dumps(params)]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, text=True)
        return proc.returncode, proc.stdout, proc.stderr
    except Exception as e:
        return 1, "", f"Failed to execute external simulator: {e}"


def run_simulation(
    mask: np.ndarray,
    params: Dict[str, Any],
    repo_path: Optional[str] = None,
    allow_builtin_fallback: bool = True,
) -> pd.DataFrame:
    """Run the external nuclear-diffusion-si simulator or fall back.

    Inputs
    - mask: 2D integer labels or binary mask; non-zero means allowed region (or regions by ID)
    - params: keys may include D (diffusion coeff), dt, steps, n_particles, seed, etc.
    - repo_path: local path to the git repo (if not installed as a package)
    - allow_builtin_fallback: when True, use our built-in simulator if external is unavailable

    Output
    - DataFrame with columns [track_id, frame, x, y, z] at minimum, in nm or um per your convention.
    """
    # Normalize mask to 2D
    if mask.ndim == 3:
        # Take a middle slice or first slice as 2D boundary for simplicity
        mask2d = mask[:, :, 0]
    else:
        mask2d = mask

    # Try Python package first
    if HAS_PACKAGE and nuclear_diffusion_si is not None:
        # Placeholder: invoke the package's API appropriately here
        raise RuntimeError(
            "nuclear_diffusion_si package detected, but adapter needs the package API to be wired. "
            "Please provide entrypoint details to complete integration."
        )

    # Try CLI via repo_path
    if repo_path and os.path.exists(repo_path):
        with tempfile.TemporaryDirectory() as tmp:
            mask_path = _write_mask_tmp(mask2d, tmp)
            code, out, err = _run_via_cli(repo_path, mask_path, params)
            if code == 0 and out:
                try:
                    data = json.loads(out)
                    df = pd.DataFrame(data)
                    # Ensure required columns
                    required = {"track_id", "frame", "x", "y"}
                    if not required.issubset(df.columns):
                        raise ValueError(f"External output missing required columns: {required}")
                    return df
                except Exception as e:
                    raise RuntimeError(f"External simulator returned invalid JSON: {e}\nSTDERR: {err}")
            raise RuntimeError(f"External simulator failed (code={code}). STDERR: {err}")

    # Built-in fallback
    if allow_builtin_fallback:
        # Use our internal simulator with coarse mapping from params
        from diffusion_simulator import DiffusionSimulator
        D = float(params.get("D", 1.0))
        steps = int(params.get("steps", 1000))
        n_particles = int(params.get("n_particles", 10))
        # Map D to a mobility scale heuristically
        mobility = max(0.1, min(10.0, D))
        sim = DiffusionSimulator()
        # Load mask as boundary/regions: non-zero is in-region
        sim.boundary_map = (mask2d == 0).astype(np.uint8)
        sim.region_map = mask2d.astype(np.uint8)
        sim.run_multi_particle_simulation([10.0], mobility=mobility, num_steps=steps, num_particles_per_size=n_particles)
        return sim.convert_all_to_tracks_format()

    raise RuntimeError(
        "nuclear-diffusion-si not available. Provide repo_path or install the package, "
        "or enable built-in fallback."
    )
