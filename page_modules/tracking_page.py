"""
Tracking Page Module

Particle detection and linking functionality.
"""

import streamlit as st
from page_modules import register_page


@register_page("Tracking")
def render():
    """
    Render the tracking page.
    
    Features:
    - Particle detection
    - Particle linking
    - Track results and visualization
    """
    st.title("Particle Detection and Tracking")
    
    # Check if image data exists
    if 'image_data' not in st.session_state or st.session_state.image_data is None:
        st.warning("No image data loaded. Please upload images first.")
        
        from ui_components import navigate_to
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
        return
    
    # Create tabs
    tabs = st.tabs(["Particle Detection", "Particle Linking", "Track Results"])
    
    with tabs[0]:
        st.header("Particle Detection")
        st.info("Detect particles in microscopy images")
        st.write("Detection methods: Gaussian fitting, watershed, LoG")
        
    with tabs[1]:
        st.header("Particle Linking")
        st.info("Link detected particles into tracks")
        st.write("Linking algorithms: Simple, LAP, trackpy")
        
    with tabs[2]:
        st.header("Track Results")
        st.info("View and analyze detected tracks")
