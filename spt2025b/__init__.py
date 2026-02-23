"""SPT2025B package facade."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("spt-analysis-2025b")
except PackageNotFoundError:
    __version__ = "0.1.0"

__all__ = ["__version__"]

try:
    from spt2025b.core.analysis import analyze_diffusion, analyze_motion
    from spt2025b.core.msd import calculate_msd, fit_msd_linear

    __all__.extend(
        [
            "calculate_msd",
            "fit_msd_linear",
            "analyze_diffusion",
            "analyze_motion",
        ]
    )
except Exception:
    # Keep package importable even when optional analysis dependencies are absent.
    pass
