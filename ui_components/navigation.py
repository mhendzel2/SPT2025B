"""
Navigation UI Components
Handles sidebar navigation and page routing
"""

import streamlit as st
from typing import List, Optional

def navigate_to(page: str):
    """
    Navigate to a specific page.
    Sets the active page in session state.
    
    Parameters
    ----------
    page : str
        Name of the page to navigate to
    """
    st.session_state.active_page = page
    # Note: Streamlit automatically reruns after callback

def load_page_content(page_name: str):
    """
    Load and render a page by name.
    
    Parameters
    ----------
    page_name : str
        Name of the page to load
    """
    from page_modules import load_page
    load_page(page_name)

def setup_sidebar() -> str:
    """
    Set up the sidebar navigation menu.
    
    Returns
    -------
    str
        The currently selected page name
    """
    st.sidebar.title("SPT Analysis")
    
    # Get active page from session state, default to Home
    if 'active_page' not in st.session_state:
        st.session_state.active_page = "Home"
    
    # Main navigation menu
    pages = [
        "Home",
        "Project Management",
        "Data Loading",
        "Image Processing",
        "Tracking",
        "Analysis",
        "Visualization",
        "Advanced Analysis",
        "AI Anomaly Detection",
        "Report Generation",
        "MD Integration",
        "Simulation"
    ]
    
    # Get current page index
    try:
        current_index = pages.index(st.session_state.active_page)
    except ValueError:
        current_index = 0
        st.session_state.active_page = pages[0]
    
    nav_option = st.sidebar.radio(
        "Navigation",
        pages,
        index=current_index,
        key="navigation_radio"
    )
    
    # Update session state if selection changed
    if nav_option != st.session_state.active_page:
        st.session_state.active_page = nav_option
    
    return st.session_state.active_page

def create_navigation_button(label: str, target_page: str, key: Optional[str] = None):
    """
    Create a button that navigates to another page when clicked.
    
    Parameters
    ----------
    label : str
        Button label text
    target_page : str
        Page to navigate to
    key : str, optional
        Unique key for the button widget
    """
    return st.button(
        label,
        on_click=navigate_to,
        args=(target_page,),
        key=key
    )

def create_page_header(title: str, description: Optional[str] = None):
    """
    Create a standard page header with title and optional description.
    
    Parameters
    ----------
    title : str
        Page title
    description : str, optional
        Page description/subtitle
    """
    st.title(title)
    if description:
        st.markdown(f"*{description}*")
        st.divider()
