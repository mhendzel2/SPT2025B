"""
Page Modules - SPT2025B
Manages page registration and loading for the multi-page application
"""

from typing import Dict, Callable
import streamlit as st

# Page registry - maps page names to their render functions
_PAGE_REGISTRY: Dict[str, Callable] = {}

def register_page(name: str):
    """
    Decorator to register a page module's render function.
    
    Usage:
        @register_page("Home")
        def render():
            st.title("Home Page")
            # ... page content ...
    """
    def decorator(render_func: Callable):
        _PAGE_REGISTRY[name] = render_func
        return render_func
    return decorator

def load_page(page_name: str):
    """
    Load and render the specified page.
    
    Parameters
    ----------
    page_name : str
        Name of the page to load (e.g., "Home", "Data Loading")
    """
    # Import all page modules to register them
    try:
        from page_modules import (
            home_page,
            simulation_page,
            project_management_page,
            data_loading_page,
            image_processing_page,
            tracking_page,
            analysis_page,
            visualization_page,
            advanced_analysis_page,
            ai_anomaly_detection_page,
            report_generation_page,
            md_integration_page
        )
    except ImportError as e:
        st.error(f"Error importing page modules: {e}")
        st.info("Some page modules may not be implemented yet.")
        import traceback
        st.code(traceback.format_exc())
    
    # Get the render function for this page
    render_func = _PAGE_REGISTRY.get(page_name)
    
    if render_func is None:
        # Fallback if page not found
        st.error(f"Page '{page_name}' not found.")
        st.info("Available pages: " + ", ".join(_PAGE_REGISTRY.keys()))
        
        # Show default home content
        st.title("SPT2025B Analysis Platform")
        st.write("Please select a page from the sidebar.")
    else:
        # Render the page
        try:
            render_func()
        except Exception as e:
            st.error(f"Error rendering page '{page_name}': {str(e)}")
            st.exception(e)

def get_available_pages():
    """
    Get list of all registered pages.
    
    Returns
    -------
    list
        List of page names
    """
    return list(_PAGE_REGISTRY.keys())
