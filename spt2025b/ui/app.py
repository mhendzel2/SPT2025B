"""Streamlit app entry point wrapper.

This module imports the legacy root-level app script to preserve behavior while
transitioning to package-based imports.
"""

from app import *  # noqa: F401,F403
