"""Shared plotting and report-validation helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


FORBIDDEN_REPORT_STRINGS = (
    "Plotting failed:",
    "Visualization failed:",
    "Error creating loop extrusion visualization",
    "Traceback (most recent call last):",
)


def safe_hover_data(df, desired_cols: Sequence[str]) -> list[str]:
    """Return only hover columns present in the dataframe."""
    return [col for col in desired_cols if col in getattr(df, "columns", ())]


def nonempty_array(a) -> bool:
    """True when `a` is not None and has at least one element."""
    return a is not None and np.size(a) > 0


def find_report_health_issues(html_text: str) -> list[str]:
    """List forbidden strings found in report HTML."""
    return [needle for needle in FORBIDDEN_REPORT_STRINGS if needle in html_text]


def assert_report_health(html_text: str, source: str = "report HTML") -> None:
    """Raise AssertionError when report HTML contains known failure markers."""
    issues = find_report_health_issues(html_text)
    if issues:
        found = ", ".join(repr(item) for item in issues)
        raise AssertionError(f"{source} contains failure markers: {found}")


def assert_report_health_file(path: str | Path) -> None:
    """Read an HTML report file and validate that no failure markers exist."""
    report_path = Path(path)
    html_text = report_path.read_text(encoding="utf-8")
    assert_report_health(html_text, source=str(report_path))


def _main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate generated report HTML health.")
    parser.add_argument("html_files", nargs="+", help="Report HTML files to validate")
    args = parser.parse_args(list(argv) if argv is not None else None)

    for html_file in args.html_files:
        assert_report_health_file(html_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
