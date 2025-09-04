
import streamlit as st
import pandas as pd

def analysis_search_filter(analyses):
    """
    Adds a search bar and category filters for analyses.

    Args:
        analyses (pd.DataFrame): DataFrame with analysis names and categories.

    Returns:
        pd.DataFrame: Filtered DataFrame of analyses.
    """
    st.sidebar.header("Filter Analyses")
    search_query = st.sidebar.text_input("Search Analyses", "")
    
    all_categories = analyses['category'].unique()
    selected_categories = st.sidebar.multiselect("Filter by Category", all_categories, default=all_categories)
    
    filtered_analyses = analyses[
        analyses['name'].str.contains(search_query, case=False) &
        analyses['category'].isin(selected_categories)
    ]
    
    return filtered_analyses

def show_contextual_help(context):
    """
    Displays contextual help in the sidebar.
    """
    help_content = {
        "default": "Hover over any option to see detailed help. For more information, consult the documentation.",
        "segmentation": "Segmentation settings control how particles are detected. Adjust the parameters to optimize detection for your specific imaging data.",
        "tracking": "Tracking parameters link detected particles across frames. Key parameters include search range and memory.",
        "analysis": "Select the analyses you want to perform. You can select one of our presets, or create your own."
    }
    st.sidebar.expander("Help", expanded=True).info(help_content.get(context, help_content["default"]))

def save_custom_preset(selected_analyses):
    """
    Saves the current selection of analyses as a custom preset.
    """
    preset_name = st.text_input("Preset Name")
    if st.button("Save Preset") and preset_name:
        # In a real app, this would be saved to a file
        st.session_state.setdefault('custom_presets', {})[preset_name] = selected_analyses
        st.success(f"Preset '{preset_name}' saved!")

def select_roi(image):
    """
    Allows the user to select a rectangular ROI.
    """
    st.subheader("Select Region of Interest (ROI)")
    col1, col2 = st.columns(2)
    x = col1.number_input("ROI X", 0, image.shape[1], 0)
    y = col1.number_input("ROI Y", 0, image.shape[0], 0)
    width = col2.number_input("ROI Width", 1, image.shape[1], image.shape[1])
    height = col2.number_input("ROI Height", 1, image.shape[0], image.shape[0])
    return x, y, width, height
