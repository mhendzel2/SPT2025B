"""
Simulation Page Module

Provides access to particle tracking simulation tools.
"""

import streamlit as st
from page_modules import register_page

# Import the existing simulation module
try:
    from simulation import show_simulation_page
    SIMULATION_AVAILABLE = True
except ImportError:
    SIMULATION_AVAILABLE = False


@register_page("Simulation")
def render():
    """
    Render the simulation page.
    
    This page provides tools for simulating particle trajectories with various
    motion models (Brownian, confined, directed, anomalous diffusion).
    """
    if not SIMULATION_AVAILABLE:
        st.error("Simulation module is not available.")
        st.info("The simulation features require the simulation.py module.")
        return
    
    try:
        show_simulation_page()
    except Exception as e:
        st.error(f"Error in simulation page: {e}")
        import traceback
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
