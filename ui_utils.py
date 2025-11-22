import streamlit as st
import base64
import os
import pandas as pd
import numpy as np
import plotly.io as pio
from datetime import datetime
from state_manager import get_state_manager

def apply_custom_css():
    """Apply custom CSS to the Streamlit app."""
    st.markdown("""
        <style>
        .main {
            padding-top: 2rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #ffffff;
            border-bottom: 2px solid #4e8cff;
        }
        div[data-testid="stExpander"] div[role="button"] p {
            font-size: 1.1rem;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)

def display_sidebar_status(tracks_data=None, image_data=None):
    """Display data status in the sidebar."""
    with st.sidebar.expander("Data Status", expanded=True):
        if tracks_data is not None:
            try:
                if isinstance(tracks_data, pd.DataFrame) and not tracks_data.empty:
                    if 'track_id' in tracks_data.columns:
                        n_tracks = tracks_data['track_id'].nunique()
                        n_points = len(tracks_data)
                        st.success(f"Tracks loaded: {n_tracks}")
                        st.caption(f"Total points: {n_points}")
                    else:
                        st.warning("Data loaded but no track_id column")
                else:
                    st.warning("Track data format issue")
            except Exception as e:
                st.error(f"Error checking tracks: {str(e)}")
        else:
            st.info("No track data loaded")

        if image_data is not None:
            try:
                if isinstance(image_data, list):
                    n_frames = len(image_data)
                elif hasattr(image_data, 'shape'):
                    n_frames = image_data.shape[0] if len(image_data.shape) > 2 else 1
                else:
                    n_frames = 0
                st.success(f"Images loaded: {n_frames} frames")
            except Exception:
                st.success("Images loaded")
        else:
            st.info("No image data loaded")

def generate_batch_html_report(report_results, condition_datasets, pixel_size, frame_interval, interactive=True):
    """
    Generate HTML report for batch condition analysis.
    """
    import html

    parts = []
    parts.append("<!DOCTYPE html><html><head><meta charset='utf-8'>")
    parts.append("<title>SPT Batch Analysis Report</title>")
    parts.append("<meta name='viewport' content='width=device-width, initial-scale=1'>")

    # Enhanced CSS styling
    parts.append("""
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-left: 4px solid #3498db; padding-left: 10px; }
        h3 { color: #7f8c8d; }
        .metadata { background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metadata ul { list-style: none; padding: 0; }
        .metadata li { padding: 5px 0; }
        .condition-section { margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 5px; background: #fafafa; }
        .metrics-table { width: 100%; border-collapse: collapse; margin: 15px 0; }
        .metrics-table th { background: #3498db; color: white; padding: 10px; text-align: left; }
        .metrics-table td { padding: 8px; border-bottom: 1px solid #ddd; }
        .metrics-table tr:hover { background: #f0f0f0; }
        .figure { margin: 20px 0; text-align: center; }
        .figure img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }
        .success { color: #27ae60; font-weight: bold; }
        .error { color: #e74c3c; font-weight: bold; }
        .summary-card { display: inline-block; margin: 10px; padding: 15px; background: white; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); min-width: 200px; }
        .summary-card h4 { margin: 0 0 10px 0; color: #3498db; }
        .summary-card .value { font-size: 24px; font-weight: bold; color: #2c3e50; }
        .comparison-section { background: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }
        code { background: #f7f7f7; padding: 2px 6px; border-radius: 3px; font-family: 'Courier New', monospace; }
    </style>
    """)

    if interactive:
        parts.append("<script src='https://cdn.plot.ly/plotly-2.18.0.min.js'></script>")

    parts.append("</head><body><div class='container'>")

    # Header
    parts.append(f"<h1>📊 SPT Batch Analysis Report</h1>")

    # Metadata section
    parts.append("<div class='metadata'>")
    parts.append("<h3>Report Metadata</h3>")
    parts.append("<ul>")
    parts.append(f"<li><b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</li>")
    parts.append(f"<li><b>Pixel Size:</b> {pixel_size} µm</li>")
    parts.append(f"<li><b>Frame Interval:</b> {frame_interval} s</li>")
    parts.append(f"<li><b>Number of Conditions:</b> {len(condition_datasets)}</li>")
    parts.append(f"<li><b>Report Type:</b> {'Interactive' if interactive else 'Static'} HTML</li>")
    parts.append("</ul>")
    parts.append("</div>")

    # Summary overview
    parts.append("<h2>📈 Summary Overview</h2>")
    parts.append("<div>")
    for cond_name, tracks_df in condition_datasets.items():
        n_tracks = tracks_df['track_id'].nunique() if 'track_id' in tracks_df.columns else 0
        n_points = len(tracks_df)
        parts.append(f"""
        <div class='summary-card'>
            <h4>{html.escape(cond_name)}</h4>
            <div><b>Tracks:</b> <span class='value'>{n_tracks}</span></div>
            <div><b>Data Points:</b> {n_points:,}</div>
        </div>
        """)
    parts.append("</div><div style='clear:both;'></div>")

    # Individual condition results
    parts.append("<h2>🔬 Condition Analysis Results</h2>")
    for cond_name, cond_result in report_results.get('conditions', {}).items():
        parts.append(f"<div class='condition-section'>")
        parts.append(f"<h3>Condition: {html.escape(cond_name)}</h3>")

        if cond_result.get('success', False):
            parts.append(f"<p class='success'>✅ Analysis completed successfully</p>")
            parts.append(f"<p><b>Analyses performed:</b> {len(cond_result.get('analysis_results', {}))}</p>")
            parts.append(f"<p><b>Figures generated:</b> {len(cond_result.get('figures', {}))}</p>")

            # Display figures
            for analysis_key, fig in cond_result.get('figures', {}).items():
                if fig:
                    parts.append(f"<div class='figure'>")
                    parts.append(f"<h4>{html.escape(analysis_key.replace('_', ' ').title())}</h4>")
                    try:
                        if interactive:
                            # Include interactive Plotly figure
                            fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
                            parts.append(fig_html)
                        else:
                            # Convert to static image
                            img_bytes = pio.to_image(fig, format='png', width=1000, height=600)
                            b64 = base64.b64encode(img_bytes).decode('utf-8')
                            parts.append(f"<img src='data:image/png;base64,{b64}' alt='{analysis_key}'>")
                    except Exception as e:
                        parts.append(f"<p class='error'>Error rendering figure: {html.escape(str(e))}</p>")
                    parts.append("</div>")
        else:
            parts.append(f"<p class='error'>❌ Analysis failed: {html.escape(cond_result.get('error', 'Unknown error'))}</p>")

        parts.append("</div>")

    # Comparison results
    comparisons = report_results.get('comparisons', {})
    if comparisons and comparisons.get('success', False) and len(condition_datasets) >= 2:
        parts.append("<div class='comparison-section'>")
        parts.append("<h2>📊 Statistical Comparisons</h2>")

        # Metrics table
        if 'metrics' in comparisons:
            parts.append("<h3>Summary Metrics by Condition</h3>")
            parts.append("<table class='metrics-table'>")
            parts.append("<tr><th>Condition</th><th>Mean Track Length</th><th>Mean Displacement (µm)</th><th>Mean Velocity (µm/s)</th></tr>")
            for cond_name, metrics in comparisons['metrics'].items():
                parts.append(f"<tr>")
                parts.append(f"<td><b>{html.escape(cond_name)}</b></td>")
                parts.append(f"<td>{metrics.get('mean_track_length', 0):.2f}</td>")
                parts.append(f"<td>{metrics.get('mean_displacement', 0):.4f}</td>")
                parts.append(f"<td>{metrics.get('mean_velocity', 0):.4f}</td>")
                parts.append(f"</tr>")
            parts.append("</table>")

        # Statistical tests
        if 'statistical_tests' in comparisons and comparisons['statistical_tests']:
            parts.append("<h3>Pairwise Statistical Tests</h3>")
            for comparison, tests in comparisons['statistical_tests'].items():
                parts.append(f"<h4>{html.escape(comparison)}</h4>")
                parts.append("<table class='metrics-table'>")
                parts.append("<tr><th>Metric</th><th>t-test p-value</th><th>Mann-Whitney p-value</th><th>Significant?</th></tr>")
                for metric, test_results in tests.items():
                    t_test_p = test_results.get('t_test', {}).get('p_value', 'N/A')
                    mw_p = test_results.get('mann_whitney', {}).get('p_value', 'N/A')
                    significant = test_results.get('significant', False)
                    sig_text = "✅ Yes" if significant else "❌ No"

                    parts.append(f"<tr>")
                    parts.append(f"<td>{html.escape(metric.replace('_', ' ').title())}</td>")
                    parts.append(f"<td>{t_test_p if isinstance(t_test_p, str) else f'{t_test_p:.4f}'}</td>")
                    parts.append(f"<td>{mw_p if isinstance(mw_p, str) else f'{mw_p:.4f}'}</td>")
                    parts.append(f"<td>{sig_text}</td>")
                    parts.append(f"</tr>")
                parts.append("</table>")

        # Comparison figures
        if 'figures' in comparisons and comparisons['figures'].get('comparison_boxplots'):
            parts.append("<h3>Comparison Visualizations</h3>")
            parts.append("<div class='figure'>")
            try:
                fig = comparisons['figures']['comparison_boxplots']
                if interactive:
                    fig_html = pio.to_html(fig, include_plotlyjs=False, full_html=False)
                    parts.append(fig_html)
                else:
                    img_bytes = pio.to_image(fig, format='png', width=1200, height=800)
                    b64 = base64.b64encode(img_bytes).decode('utf-8')
                    parts.append(f"<img src='data:image/png;base64,{b64}' alt='Comparison Boxplots'>")
            except Exception as e:
                parts.append(f"<p class='error'>Error rendering comparison figure: {html.escape(str(e))}</p>")
            parts.append("</div>")

        parts.append("</div>")

    # Footer
    parts.append("<hr style='margin-top: 40px;'>")
    parts.append("<p style='text-align: center; color: #7f8c8d;'>")
    parts.append("Generated by SPT2025B Analysis Platform | ")
    parts.append(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    parts.append("</p>")

    parts.append("</div></body></html>")

    html_str = "".join(parts)
    return html_str.encode('utf-8')

def navigate_to(page):
    """Navigate to a different page."""
    st.session_state.active_page = page

def initialize_mask_tracking():
    """Initialize mask tracking in session state."""
    if 'available_masks' not in st.session_state:
        st.session_state.available_masks = {}
    if 'mask_metadata' not in st.session_state:
        st.session_state.mask_metadata = {}

def store_mask(mask_name: str, mask_data: np.ndarray, mask_type: str, description: str = ""):
    """Store a generated mask for later use in analysis."""
    initialize_mask_tracking()

    st.session_state.available_masks[mask_name] = mask_data
    st.session_state.mask_metadata[mask_name] = {
        'type': mask_type,
        'description': description,
        'shape': mask_data.shape,
        'classes': np.unique(mask_data).tolist(),
        'n_classes': len(np.unique(mask_data))
    }

def get_available_masks():
    """Get dictionary of available masks for analysis."""
    initialize_mask_tracking()
    return st.session_state.available_masks

def create_mask_selection_ui(analysis_type: str = ""):
    """Create UI for selecting segmentation method and analyzing all classes from that method."""
    available_masks = get_available_masks()

    if not available_masks:
        st.info("No masks available. Generate masks in the Image Processing tab first.")
        return None, [], None

    st.markdown("#### Region-Based Analysis")

    # Analysis region selection
    analysis_region = st.radio(
        "Analysis Region",
        ["Whole Image", "Segmentation-Based Analysis"],
        help="Choose whether to analyze the entire image or use segmentation regions",
        key=f"analysis_region_{analysis_type}"
    )

    if analysis_region == "Whole Image":
        return None, [], None

    st.subheader("Segmentation Method Selection")

    # Categorize available masks by type
    simple_masks = []
    two_step_masks = []
    density_masks = []

    for mask_name in available_masks.keys():
        mask_metadata = st.session_state.mask_metadata[mask_name]
        mask_type = mask_metadata.get('type', 'unknown').lower()

        if 'density' in mask_type or 'nuclear density' in mask_type:
            density_masks.append(mask_name)
        elif mask_metadata['n_classes'] >= 3:
            two_step_masks.append(mask_name)
        else:
            simple_masks.append(mask_name)

    # Select segmentation method
    segmentation_methods = []
    if simple_masks:
        segmentation_methods.append("Simple Segmentation (Binary)")
    if two_step_masks:
        segmentation_methods.append("Two-Step Segmentation (3 Classes)")
    if density_masks:
        segmentation_methods.append("Nuclear Density Mapping")

    if not segmentation_methods:
        st.error("No compatible segmentation masks found.")
        return None, [], None

    selected_method = st.selectbox(
        "Choose Segmentation Method",
        segmentation_methods,
        help="Select the type of segmentation to use for analysis. All classes from this method will be analyzed.",
        key=f"segmentation_method_{analysis_type}"
    )

    # Select specific mask based on method
    if selected_method == "Simple Segmentation (Binary)":
        available_for_method = simple_masks
        expected_classes = [0, 1]  # Background, Nucleus
        class_names = ["Background", "Nucleus"]
    elif selected_method == "Two-Step Segmentation (3 Classes)":
        available_for_method = two_step_masks

        # Add analysis options for Two-Step Segmentation
        st.subheader("Two-Step Segmentation Analysis Options")
        analysis_option = st.selectbox(
            "Select 2-Step Segmentation Analysis Method",
            [
                "Analyze all three classes separately",
                "Analyze classes separately, then combine Class 1 and 2"
            ],
            help="Choose how to analyze the three segmentation classes",
            key=f"two_step_analysis_option_{analysis_type}"
        )

        if analysis_option == "Analyze all three classes separately":
            expected_classes = [0, 1, 2]  # Background, Class 1, Class 2
            class_names = ["Background", "Class 1", "Class 2"]
        else:  # "Analyze classes separately, then combine Class 1 and 2"
            expected_classes = [0, 1, 2]  # Still analyze all classes first
            class_names = ["Background", "Class 1", "Class 2", "Combined Class 1+2"]
    else:  # Nuclear Density Mapping
        available_for_method = density_masks
        expected_classes = [0, 1, 2]  # Background, Low Density, High Density
        class_names = ["Background", "Low Density", "High Density"]

    # Automatically select the first available mask for the method
    if not available_for_method:
        st.error(f"No masks available for {selected_method}")
        return None, [], None

    selected_mask = available_for_method[0]  # Use the first (and typically only) available mask

    # Only show dropdown if multiple masks are available for this method
    if len(available_for_method) > 1:
        selected_mask = st.selectbox(
            f"Select {selected_method} Mask",
            available_for_method,
            help=f"Choose the specific mask for {selected_method.lower()}",
            key=f"specific_mask_{analysis_type}"
        )
    else:
        # Show which mask is being used automatically
        st.info(f"Using mask: **{selected_mask}**")

    # Show mask information
    mask_metadata = st.session_state.mask_metadata[selected_mask]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Type:** {mask_metadata['type']}")
    with col2:
        st.write(f"**Classes:** {mask_metadata['n_classes']}")
    with col3:
        st.write(f"**Shape:** {mask_metadata['shape']}")

    if mask_metadata['description']:
        st.write(f"**Description:** {mask_metadata['description']}")

    # Display analysis plan
    st.info(f"Analysis will be performed separately for each class: {', '.join(class_names)}")

    # Return all classes for comprehensive analysis
    return [selected_mask], {selected_mask: expected_classes}, selected_method

def handle_track_upload(uploaded_file):
    """
    Wrapper for file upload -> load -> persist.
    Call wherever the upload widget processes a new file.
    """
    if not uploaded_file:
        return None
    import io
    from data_loader import load_tracks_file

    name = getattr(uploaded_file, "name", "uploaded_tracks")
    path_hint = name

    data_bytes = uploaded_file.read()
    bio = io.BytesIO(data_bytes)

    if name.lower().endswith((".xls", ".xlsx")):
        import pandas as pd
        df = pd.read_excel(bio, engine="openpyxl")
    else:
        import pandas as pd
        df = pd.read_csv(bio)

    state_manager = get_state_manager()
    df_clean = load_tracks_file(path_hint, persist=True, state_manager=state_manager, raw_df=df)
    return df_clean
