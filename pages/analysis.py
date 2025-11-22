import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from ui_utils import get_available_masks
from utils import calculate_track_statistics, get_global_pixel_size, get_global_frame_interval, sync_global_parameters, create_analysis_record
from analysis import (
    analyze_diffusion, analyze_motion, analyze_clustering, analyze_dwell_time,
    load_precalculated_dwell_events, analyze_boundary_crossing, analyze_gel_structure,
    analyze_diffusion_population, analyze_active_transport, analyze_crowding, analyze_polymer_physics
)
from visualization import plot_tracks, plot_tracks_3d, plot_track_statistics, plot_motion_analysis, plot_diffusion_coefficients
from segmentation import convert_compartments_to_boundary_crossing_format
from multi_channel_analysis import MultiChannelAnalyzer
from data_access_utils import get_track_data, get_units
from logic import apply_mask_to_tracks
from skimage import measure
from biophysics_tab import show_biophysical_models, show_advanced_biophysical_metrics

# Import optional modules with error handling
try:
    from intensity_analysis import extract_intensity_channels
    INTENSITY_ANALYSIS_AVAILABLE = True
except ImportError:
    INTENSITY_ANALYSIS_AVAILABLE = False

try:
    from correlative_analysis import CorrelativeAnalyzer
    CORRELATIVE_ANALYSIS_AVAILABLE = True
except ImportError:
    CORRELATIVE_ANALYSIS_AVAILABLE = False

def get_current_units():
    """Get the currently set unit values for use in all analyses."""
    return {
        'pixel_size': st.session_state.get('current_pixel_size', 0.1),
        'frame_interval': st.session_state.get('current_frame_interval', 0.1)
    }

def show_analysis_page():
    st.title("Track Analysis")

    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        if st.button("Go to Data Loading"):
            st.session_state.active_page = "Data Loading"
            st.rerun()
    else:
        tabs = st.tabs([
            "Overview",
            "Diffusion Analysis",
            "Motion Analysis",
            "Clustering Analysis",
            "Dwell Time Analysis",
            "Boundary Crossing Analysis",
            "Multi-Channel Analysis",
            "Advanced Analysis"
        ])

        with tabs[0]:
            st.header("Track Overview")
            st.subheader("Track Statistics")
            if not hasattr(st.session_state, 'track_statistics') or st.session_state.track_statistics is None:
                with st.spinner("Calculating track statistics..."):
                    st.session_state.track_statistics = calculate_track_statistics(st.session_state.tracks_data)

            st.dataframe(st.session_state.track_statistics)

            st.subheader("Track Visualization")
            viz_type = st.radio("Visualization Type", ["2D Tracks", "3D Tracks (time as Z)", "Statistics"])

            if viz_type == "2D Tracks":
                color_by = st.selectbox("Color tracks by", ["track_id", "track_length", "mean_speed", "straightness"])
                if color_by != "track_id" and color_by in st.session_state.track_statistics.columns:
                    color_map = st.session_state.track_statistics.set_index('track_id')[color_by].to_dict()
                    temp_df = st.session_state.tracks_data.copy()
                    temp_df[color_by] = temp_df['track_id'].map(color_map)
                    fig = plot_tracks(temp_df, color_by=color_by)
                else:
                    fig = plot_tracks(st.session_state.tracks_data, color_by=color_by)
                st.plotly_chart(fig, use_container_width=True)

            elif viz_type == "3D Tracks (time as Z)":
                fig = plot_tracks_3d(st.session_state.tracks_data)
                st.plotly_chart(fig, use_container_width=True)

            else:
                if st.session_state.track_statistics is not None:
                    figs = plot_track_statistics(st.session_state.track_statistics)
                    for name, fig in figs.items():
                        st.subheader(name.replace("_", " ").title())
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No track statistics available.")

        with tabs[1]:
            st.header("Diffusion Analysis")
            analysis_type = st.radio(
                "Analysis Type",
                ["Whole Image Analysis", "Segmentation-Based Analysis", "Subpopulation Analysis (by Cell)"],
                help="Choose analysis approach: whole image, segmentation-based, or detect subpopulations within groups"
            )

            selected_mask = None
            selected_classes = None
            segmentation_method = None
            tracks_with_classes = None
            analysis_option = None

            # Initialize selected_mask at top level scope
            selected_mask = None

            if analysis_type == "Segmentation-Based Analysis":
                available_masks = get_available_masks()
                if not available_masks:
                    st.error("No segmentation masks available. Please create masks in the Image Processing tab first.")
                else:
                    simple_masks = []
                    two_step_masks = []
                    density_masks = []
                    for mask_name in available_masks.keys():
                        if mask_name in st.session_state.mask_metadata:
                            mask_meta = st.session_state.mask_metadata[mask_name]
                            mask_type = mask_meta.get('type', 'unknown').lower()
                            if 'density' in mask_type or 'nuclear_density' in mask_type:
                                density_masks.append(mask_name)
                            elif mask_meta.get('n_classes', 2) >= 3:
                                two_step_masks.append(mask_name)
                            else:
                                simple_masks.append(mask_name)

                    segmentation_options = []
                    if simple_masks: segmentation_options.append("Simple Segmentation (Binary)")
                    if two_step_masks: segmentation_options.append("Two-Step Segmentation (3 Classes)")
                    if density_masks: segmentation_options.append("Nuclear Density Mapping")

                    if segmentation_options:
                        segmentation_method = st.selectbox("Choose Segmentation Method", segmentation_options)

                        if segmentation_method == "Simple Segmentation (Binary)":
                            if simple_masks:
                                selected_mask = simple_masks[0] if len(simple_masks) == 1 else st.selectbox("Select Mask", simple_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, [0, 1])
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        class_name = ["Background", "Nucleus"][int(class_id)]
                                        st.write(f"- {class_name}: {count} track points")

                        elif segmentation_method == "Two-Step Segmentation (3 Classes)":
                            if two_step_masks:
                                selected_mask = two_step_masks[0] if len(two_step_masks) == 1 else st.selectbox("Select Mask", two_step_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                analysis_option = st.selectbox(
                                    "Two-Step Analysis Method",
                                    ["Analyze all three classes separately", "Analyze classes separately, then combine Class 1 and 2"]
                                )
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, [0, 1, 2])
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        class_name = ["Background", "Class 1", "Class 2"][int(class_id)]
                                        st.write(f"- {class_name}: {count} track points")

                        elif segmentation_method == "Nuclear Density Mapping":
                            if density_masks:
                                selected_mask = density_masks[0] if len(density_masks) == 1 else st.selectbox("Select Mask", density_masks)
                                st.info(f"Using mask: **{selected_mask}**")
                                mask_metadata = st.session_state.mask_metadata[selected_mask]
                                n_classes = mask_metadata['n_classes']
                                class_list = list(range(n_classes))
                                tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, class_list)
                                if 'class' in tracks_with_classes.columns:
                                    class_summary = tracks_with_classes['class'].value_counts().sort_index()
                                    st.write("**Track Distribution:**")
                                    for class_id, count in class_summary.items():
                                        if int(class_id) == 0: class_name = "Background"
                                        elif int(class_id) == 1: class_name = "Low Density"
                                        elif int(class_id) == 2: class_name = "High Density"
                                        else: class_name = f"Class {int(class_id)}"
                                        st.write(f"- {class_name}: {count} track points")
                    else:
                        st.error("No compatible segmentation masks found.")

            if selected_mask:
                mask_name = selected_mask[0] if isinstance(selected_mask, list) else selected_mask
                st.info(f"Analysis will be performed on mask: {mask_name}")
                if selected_classes:
                    mask_classes = selected_classes.get(mask_name, []) if isinstance(selected_classes, dict) else selected_classes
                    st.info(f"Using mask classes: {', '.join(map(str, mask_classes))}")

            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                max_lag = st.slider("Maximum Lag Time (frames)", 5, 50, 20)
                sync_global_parameters()
                pixel_size = get_global_pixel_size()
                frame_interval = get_global_frame_interval()
                st.info(f"Using global settings: Pixel Size = {pixel_size:.3f} µm, Frame Interval = {frame_interval:.3f} s")
                st.info("To change these values, use the Image Settings tab in Data Loading section.")

            with col2:
                min_track_length = st.slider("Minimum Track Length", 5, 50, 10)
                fit_method = st.selectbox("Fitting Method", ["linear", "weighted", "nonlinear"])
                analysis_options = st.multiselect(
                    "Analysis Options",
                    ["Anomalous Diffusion", "Confined Diffusion"],
                    default=["Anomalous Diffusion", "Confined Diffusion"]
                )

            if st.button("Run Diffusion Analysis"):
                with st.spinner("Running diffusion analysis..."):
                    try:
                        analyze_anomalous = "Anomalous Diffusion" in analysis_options
                        check_confinement = "Confined Diffusion" in analysis_options

                        if analysis_type == "Whole Image Analysis":
                            st.subheader("Whole Image Diffusion Analysis")
                            result = analyze_diffusion(
                                st.session_state.tracks_data,
                                max_lag=max_lag,
                                pixel_size=pixel_size,
                                frame_interval=frame_interval,
                                min_track_length=min_track_length,
                                fit_method=fit_method,
                                analyze_anomalous=analyze_anomalous,
                                check_confinement=check_confinement
                            )
                            if result.get('success', False):
                                diffusion_results = result['result']
                                diffusion_results['success'] = True
                            else:
                                diffusion_results = {'success': False, 'error': result.get('error', 'Unknown error')}
                        elif analysis_type == "Segmentation-Based Analysis" and selected_mask and segmentation_method:
                            st.subheader(f"Class-Based Diffusion Analysis - {segmentation_method}")
                            if tracks_with_classes is not None and 'class' in tracks_with_classes.columns:
                                if segmentation_method == "Simple Segmentation (Binary)":
                                    class_names = {0: "Background", 1: "Nucleus"}
                                    mask_classes = [0, 1]
                                elif segmentation_method == "Two-Step Segmentation (3 Classes)":
                                    class_names = {0: "Background", 1: "Class 1", 2: "Class 2"}
                                    mask_classes = [0, 1, 2]
                                else:
                                    unique_classes = sorted(tracks_with_classes['class'].unique())
                                    mask_classes = unique_classes
                                    class_names = {}
                                    for class_id in unique_classes:
                                        if int(class_id) == 0: class_names[class_id] = "Background"
                                        elif int(class_id) == 1: class_names[class_id] = "Low Density"
                                        elif int(class_id) == 2: class_names[class_id] = "High Density"
                                        else: class_names[class_id] = f"Class {int(class_id)}"
                                diffusion_results = {}
                                for class_id in mask_classes:
                                    class_tracks = tracks_with_classes[tracks_with_classes['class'] == class_id]
                                    if len(class_tracks) < min_track_length:
                                        st.warning(f"Insufficient tracks for {class_names.get(class_id, f'Class {class_id}')}")
                                        continue
                                    st.write(f"**Analyzing {class_names.get(class_id, f'Class {class_id}')}** ({len(class_tracks)} track points)")
                                    class_result = analyze_diffusion(
                                        class_tracks,
                                        max_lag=max_lag,
                                        pixel_size=pixel_size,
                                        frame_interval=frame_interval,
                                        min_track_length=min_track_length,
                                        fit_method=fit_method,
                                        analyze_anomalous=analyze_anomalous,
                                        check_confinement=check_confinement
                                    )
                                    if class_result.get('success', False):
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')] = class_result['result']
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')]['success'] = True
                                    else:
                                        diffusion_results[class_names.get(class_id, f'Class {class_id}')] = {
                                            'success': False,
                                            'error': class_result.get('error', 'Unknown error')
                                        }
                                if (segmentation_method == "Two-Step Segmentation (3 Classes)" and
                                    analysis_option == "Analyze classes separately, then combine Class 1 and 2"):
                                    st.write("**Performing Combined Class 1+2 Analysis**")
                                    combined_tracks = tracks_with_classes[tracks_with_classes['class'].isin([1, 2])]
                                    if len(combined_tracks) >= min_track_length:
                                        combined_result = analyze_diffusion(
                                            combined_tracks,
                                            max_lag=max_lag,
                                            pixel_size=pixel_size,
                                            frame_interval=frame_interval,
                                            min_track_length=min_track_length,
                                            fit_method=fit_method,
                                            analyze_anomalous=analyze_anomalous,
                                            check_confinement=check_confinement
                                        )
                                        if combined_result.get('success', False):
                                            diffusion_results["Combined Class 1+2"] = combined_result['result']
                                            diffusion_results["Combined Class 1+2"]['success'] = True
                                        else:
                                            diffusion_results["Combined Class 1+2"] = {'success': False, 'error': combined_result.get('error', 'Unknown error')}
                                    else:
                                        st.warning("Insufficient tracks for combined Class 1+2 analysis")
                            else:
                                st.error("Track classification failed. Please select a segmentation method first.")
                                diffusion_results = {"error": "Classification failed"}
                        else:
                            st.error("Please select analysis type and configure segmentation if needed.")
                            diffusion_results = {"error": "Invalid configuration"}

                        st.session_state.analysis_results["diffusion"] = diffusion_results
                        create_analysis_record(
                            name="Diffusion Analysis",
                            analysis_type="diffusion",
                            parameters={
                                "max_lag": max_lag,
                                "pixel_size": pixel_size,
                                "frame_interval": frame_interval,
                                "min_track_length": min_track_length,
                                "fit_method": fit_method,
                                "analyze_anomalous": analyze_anomalous,
                                "check_confinement": check_confinement
                            }
                        )
                        st.success("Diffusion analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running diffusion analysis: {str(e)}")

            if "diffusion" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["diffusion"]
                if isinstance(results, dict) and any(key in ["Background", "Nucleus", "Class 1", "Class 2", "Combined Class 1+2", "Low Density", "High Density"] for key in results.keys()):
                    st.subheader("Class-Based Diffusion Analysis Results")
                    for class_name, class_results in results.items():
                        st.write(f"### {class_name}")
                        if "error" in class_results:
                            st.error(f"{class_name}: {class_results['error']}")
                            continue
                        if class_results.get("success", False):
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if "diffusion_coefficient" in class_results:
                                    st.metric("Diffusion Coefficient", f"{class_results['diffusion_coefficient']:.4f}")
                                if "n_tracks" in class_results:
                                    st.metric("Number of Tracks", class_results["n_tracks"])
                            with col2:
                                if "alpha" in class_results:
                                    st.metric("Anomalous Exponent (α)", f"{class_results['alpha']:.3f}")
                                if "confinement_radius" in class_results:
                                    st.metric("Confinement Radius", f"{class_results['confinement_radius']:.2f}")
                            with col3:
                                if "fitting_quality" in class_results:
                                    st.metric("R² (Fit Quality)", f"{class_results['fitting_quality']:.3f}")
                            if "msd_data" in class_results and not class_results["msd_data"].empty:
                                st.write(f"**{class_name} - Mean Squared Displacement**")
                                fig = px.scatter(class_results["msd_data"], x="lag_time", y="msd", title=f"MSD vs Lag Time - {class_name}", labels={"lag_time": "Lag Time (s)", "msd": "MSD (μm²)"})
                                st.plotly_chart(fig, use_container_width=True)
                        st.write("---")
                    st.subheader("Class Comparison")
                    comparison_data = []
                    for class_name, class_results in results.items():
                        if class_results.get("success", False) and "diffusion_coefficient" in class_results:
                            comparison_data.append({
                                "Class": class_name,
                                "Diffusion Coefficient": class_results["diffusion_coefficient"],
                                "Alpha (α)": class_results.get("alpha", "N/A"),
                                "Number of Tracks": class_results.get("n_tracks", "N/A"),
                                "R²": class_results.get("fitting_quality", "N/A")
                            })
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df)
                        fig = px.bar(comparison_df, x="Class", y="Diffusion Coefficient", title="Diffusion Coefficient by Class")
                        st.plotly_chart(fig, use_container_width=True)
                elif results.get("success", False):
                    st.subheader("Diffusion Analysis Results")
                    display_results = results.get('result', results) if 'result' in results else results
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if 'ensemble_results' in display_results and 'mean_diffusion_coefficient' in display_results['ensemble_results']:
                            st.metric("Diffusion Coefficient", f"{display_results['ensemble_results']['mean_diffusion_coefficient']:.4f}")
                        elif 'diffusion_coefficient' in display_results:
                            st.metric("Diffusion Coefficient", f"{display_results['diffusion_coefficient']:.4f}")
                        else:
                            st.metric("Diffusion Coefficient", "N/A")
                    with col2:
                        if 'ensemble_results' in display_results and 'mean_alpha' in display_results['ensemble_results']:
                            st.metric("Anomalous Exponent (α)", f"{display_results['ensemble_results']['mean_alpha']:.3f}")
                        elif 'alpha' in display_results:
                            st.metric("Anomalous Exponent (α)", f"{display_results['alpha']:.3f}")
                        else:
                            st.metric("Anomalous Exponent (α)", "N/A")
                    with col3:
                        if 'fitting_quality' in display_results:
                            st.metric("R² (Fit Quality)", f"{display_results['fitting_quality']:.3f}")
                        else:
                            st.metric("R² (Fit Quality)", "N/A")

                    if "msd_data" in display_results and display_results["msd_data"] is not None:
                        st.subheader("Mean Squared Displacement")
                        st.dataframe(display_results["msd_data"].head())
                        msd_fig = px.scatter(display_results['msd_data'], x="lag_time", y="msd", title="MSD vs Lag Time", labels={"lag_time": "Lag Time (s)", "msd": "MSD (μm²)"})
                        msd_fig.update_traces(mode='lines+markers')
                        st.plotly_chart(msd_fig, use_container_width=True)

                    if "track_results" in display_results and display_results["track_results"] is not None:
                        st.subheader("Diffusion Results by Track")
                        if isinstance(display_results["track_results"], pd.DataFrame) and not display_results["track_results"].empty:
                            st.dataframe(display_results["track_results"])
                            diff_fig = plot_diffusion_coefficients(display_results)
                            st.plotly_chart(diff_fig, use_container_width=True)
                else:
                    st.warning(f"Analysis was not successful: {results.get('error', 'Unknown error')}")

            elif analysis_type == "Subpopulation Analysis (by Cell)":
                st.subheader("🔬 Single-Cell Subpopulation Analysis")
                # ... (Rest of subpopulation logic would go here, simplified for now)
                st.info("Subpopulation analysis is available in the Project Management module or specialized analysis scripts.")

        with tabs[2]:
            st.header("Motion Analysis")
            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Window Size (frames)", 3, 20, 5)
                units = get_current_units()
                pixel_size = st.number_input("Pixel Size (µm)", 0.01, 10.0, units['pixel_size'], 0.01, key="motion_pixel_size")
                frame_interval = st.number_input("Frame Interval (s)", 0.001, 60.0, units['frame_interval'], 0.001, key="motion_frame_interval")
            with col2:
                min_track_length = st.slider("Minimum Track Length", 5, 50, 10, key="motion_min_track_length")
                motion_classification = st.selectbox("Motion Classification", ["none", "basic", "advanced"])
                analysis_options = st.multiselect("Analysis Options", ["Velocity Autocorrelation", "Directional Persistence"], default=["Velocity Autocorrelation", "Directional Persistence"])

            if st.button("Run Motion Analysis"):
                with st.spinner("Running motion analysis..."):
                    try:
                        motion_results = analyze_motion(
                            st.session_state.tracks_data,
                            window_size=window_size,
                            analyze_velocity_autocorr="Velocity Autocorrelation" in analysis_options,
                            analyze_persistence="Directional Persistence" in analysis_options,
                            motion_classification=motion_classification,
                            min_track_length=min_track_length,
                            pixel_size=pixel_size,
                            frame_interval=frame_interval
                        )
                        st.session_state.analysis_results["motion"] = motion_results
                        create_analysis_record("Motion Analysis", "motion", {
                            "window_size": window_size, "pixel_size": pixel_size, "frame_interval": frame_interval,
                            "min_track_length": min_track_length, "motion_classification": motion_classification
                        })
                        st.success("Motion analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running motion analysis: {str(e)}")

            if "motion" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["motion"]
                st.subheader("Motion Analysis Results")
                if 'track_results' in results:
                    st.dataframe(results['track_results'])
                    try:
                        motion_vis = plot_motion_analysis(results)
                        from matplotlib.figure import Figure as MplFigure
                        if isinstance(motion_vis, MplFigure):
                            st.pyplot(motion_vis, use_container_width=True)
                        elif isinstance(motion_vis, dict):
                            for plot_name, fig in motion_vis.items():
                                if fig and plot_name != "empty":
                                    st.subheader(plot_name.replace("_", " ").title())
                                    st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not generate motion visualizations: {str(e)}")

        with tabs[3]:
            st.header("Clustering Analysis")
            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                clustering_method = st.selectbox("Clustering Method", ["DBSCAN", "OPTICS", "Hierarchical", "Density-based"])
                epsilon = st.slider("Epsilon (µm)", 0.1, 10.0, 0.5, 0.1)
                min_samples = st.slider("Minimum Samples", 2, 20, 3)
            with col2:
                units = get_current_units()
                pixel_size = st.number_input("Pixel Size (µm)", 0.01, 10.0, units['pixel_size'], 0.01, key="clustering_pixel_size")
                analysis_options = st.multiselect("Analysis Options", ["Track Clusters", "Analyze Dynamics"], default=["Track Clusters"])

            if st.button("Run Clustering Analysis"):
                with st.spinner("Running clustering analysis..."):
                    try:
                        clustering_results = analyze_clustering(
                            st.session_state.tracks_data,
                            method=clustering_method,
                            epsilon=epsilon,
                            min_samples=min_samples,
                            track_clusters="Track Clusters" in analysis_options,
                            analyze_dynamics="Analyze Dynamics" in analysis_options,
                            pixel_size=pixel_size
                        )
                        st.session_state.analysis_results["clustering"] = clustering_results
                        create_analysis_record("Clustering Analysis", "clustering", {
                            "method": clustering_method, "epsilon": epsilon, "min_samples": min_samples, "pixel_size": pixel_size
                        })
                        st.success("Clustering analysis completed successfully!")
                    except Exception as e:
                        st.error(f"Error running clustering analysis: {str(e)}")

            if "clustering" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["clustering"]
                st.subheader("Cluster Statistics")
                if 'frame_results' in results and results['frame_results']:
                    for frame_result in results['frame_results']:
                        if isinstance(frame_result, dict) and 'cluster_stats' in frame_result and not frame_result['cluster_stats'].empty:
                            st.dataframe(frame_result['cluster_stats'])
                            break

                if 'cluster_tracks' in results and not results['cluster_tracks'].empty:
                    fig = px.scatter(results['cluster_tracks'], x='x', y='y', color='cluster_id',
                                   title="Spatial Clustering of Tracks", labels={'x': 'X Position', 'y': 'Y Position'})
                    fig.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig, use_container_width=True)

        with tabs[4]:
            st.header("Dwell Time Analysis")
            if 'tracks_data' in st.session_state and st.session_state.tracks_data is not None:
                dwell_cols = ['dwell_time', 'dwell_frames', 'start_frame', 'end_frame']
                available_dwell_cols = [col for col in dwell_cols if col in st.session_state.tracks_data.columns]

                if len(available_dwell_cols) >= 2:
                    st.success("📊 Detected pre-calculated dwell event data!")
                    if st.button("Load Pre-calculated Dwell Events"):
                        with st.spinner("Loading dwell event statistics..."):
                            try:
                                frame_interval = st.session_state.get('frame_interval', 0.1)
                                dwell_results = load_precalculated_dwell_events(st.session_state.tracks_data, frame_interval=frame_interval)
                                if dwell_results.get('success'):
                                    st.session_state.analysis_results["dwell_time"] = dwell_results
                                    st.success("Loaded dwell events!")
                                else:
                                    st.error(f"Failed to load: {dwell_results.get('error')}")
                            except Exception as e:
                                st.error(f"Error loading: {str(e)}")

            st.subheader("Analysis Parameters")
            col1, col2 = st.columns(2)
            with col1:
                threshold_distance = st.slider("Threshold Distance (µm)", 0.1, 5.0, 0.5, 0.1)
                min_dwell_frames = st.slider("Minimum Dwell Frames", 2, 20, 3)
            with col2:
                units = get_current_units()
                pixel_size = st.number_input("Pixel Size (µm)", 0.01, 10.0, units['pixel_size'], 0.01, key="dwell_pixel_size")
                frame_interval = st.number_input("Frame Interval (s)", 0.001, 60.0, units['frame_interval'], 0.001, key="dwell_frame_interval")
                use_regions = st.checkbox("Define Regions of Interest", value=False)

            regions = None
            if use_regions:
                num_regions = st.slider("Number of Regions", 1, 10, 1)
                regions = []
                for i in range(num_regions):
                    with st.expander(f"Region {i+1}"):
                        c1, c2, c3 = st.columns(3)
                        x = c1.number_input(f"X center (Region {i+1})", value=0.0)
                        y = c2.number_input(f"Y center (Region {i+1})", value=0.0)
                        r = c3.number_input(f"Radius (Region {i+1})", value=1.0)
                        regions.append({'x': x, 'y': y, 'radius': r})

            if st.button("Run Dwell Time Analysis"):
                with st.spinner("Running dwell time analysis..."):
                    try:
                        dwell_results = analyze_dwell_time(
                            st.session_state.tracks_data,
                            regions=regions,
                            threshold_distance=threshold_distance,
                            min_dwell_frames=min_dwell_frames,
                            pixel_size=pixel_size,
                            frame_interval=frame_interval
                        )
                        st.session_state.analysis_results["dwell_time"] = dwell_results
                        st.success("Dwell time analysis completed!")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if "dwell_time" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["dwell_time"]
                if 'dwell_events' in results:
                    st.dataframe(results['dwell_events'])
                if 'dwell_stats' in results:
                    for k, v in results['dwell_stats'].items():
                        st.text(f"{k}: {v}")

        with tabs[5]:
            st.header("Boundary Crossing Analysis")
            if st.button("Analyze Boundary Crossings"):
                with st.spinner("Analyzing boundary crossings..."):
                    try:
                        tracks_with_classes = apply_mask_to_tracks(st.session_state.tracks_data, selected_mask, [])
                        track_lengths = tracks_with_classes.groupby('track_id').size()
                        valid_tracks = track_lengths[track_lengths >= 5].index
                        filtered_tracks = tracks_with_classes[tracks_with_classes['track_id'].isin(valid_tracks)]

                        compartments_for_conversion = []
                        for class_id in filtered_tracks['class'].unique():
                            if class_id != 'none':
                                if selected_mask and selected_mask[0] in st.session_state.available_masks:
                                    mask_data = st.session_state.available_masks[selected_mask[0]]
                                    comp_mask = (mask_data == class_id)
                                    props = measure.regionprops(comp_mask.astype(int))
                                else:
                                    props = []
                                for i, prop in enumerate(props):
                                    min_row, min_col, max_row, max_col = prop.bbox
                                    compartments_for_conversion.append({
                                        'id': f'class_{class_id}_region_{i}',
                                        'bbox_um': {
                                            'x1': min_col * get_global_pixel_size(),
                                            'y1': min_row * get_global_pixel_size(),
                                            'x2': max_col * get_global_pixel_size(),
                                            'y2': max_row * get_global_pixel_size()
                                         }
                                    })
                        boundaries = convert_compartments_to_boundary_crossing_format(compartments_for_conversion)
                        boundary_stats = analyze_boundary_crossing(
                            filtered_tracks,
                            boundaries=boundaries,
                            pixel_size=get_global_pixel_size(),
                            frame_interval=get_global_frame_interval(),
                            min_track_length=5
                        )
                        if "error" not in boundary_stats:
                            st.session_state.analysis_results["boundary_crossing"] = boundary_stats
                            st.success("Boundary crossing analysis completed!")
                        else:
                            st.error(boundary_stats["error"])
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

            if "boundary_crossing" in st.session_state.analysis_results:
                results = st.session_state.analysis_results["boundary_crossing"]
                st.metric("Total Crossings", results["total_crossings"])
                if results["class_transitions"]:
                    transitions_data = [{'From': t.split('->')[0], 'To': t.split('->')[1], 'Count': c, 'Transition': t}
                                      for t, c in results["class_transitions"].items()]
                    st.dataframe(pd.DataFrame(transitions_data))

        with tabs[6]:
            st.header("Multi-Channel Analysis")
            if CORRELATIVE_ANALYSIS_AVAILABLE:
                existing_channels = None
                if st.session_state.tracks_data is not None:
                    existing_channels = extract_intensity_channels(st.session_state.tracks_data)

                if existing_channels and len(existing_channels) >= 2:
                    st.success(f"Detected {len(existing_channels)} intensity channels!")
                    data_source = st.radio("Data Source", ["Use Existing Intensity Data", "Upload Separate File"])

                    if data_source == "Use Existing Intensity Data":
                        primary = st.selectbox("Primary Channel", list(existing_channels.keys()))
                        secondary = st.selectbox("Secondary Channel", [c for c in existing_channels.keys() if c != primary])

                        if st.button("Run Multi-Channel Analysis"):
                            try:
                                primary_data = st.session_state.tracks_data.copy()
                                primary_data['intensity'] = primary_data[existing_channels[primary][0]]
                                channel2_data = st.session_state.tracks_data[['track_id', 'frame', 'x', 'y']].copy()
                                channel2_data['intensity'] = st.session_state.tracks_data[existing_channels[secondary][0]]

                                analyzer = MultiChannelAnalyzer()
                                analyzer.add_channel(primary_data, primary)
                                analyzer.add_channel(channel2_data, secondary)

                                coloc_results = analyzer.calculate_colocalization_statistics(primary, secondary, distance_threshold=2.0)
                                st.session_state.multi_channel_results = coloc_results
                                st.success("Analysis completed!")
                                st.metric("Colocalization Events", coloc_results['n_colocalization_events'])
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
                else:
                    st.info("Load data with multiple channels or upload separate files.")
            else:
                st.warning("Multi-Channel Analysis module not available.")

        with tabs[7]:
            st.header("Advanced Analysis")
            adv_tabs = st.tabs([
                "Gel Structure", "Diffusion Population", "Active Transport",
                "Boundary Crossing", "Crowding", "Polymer Physics"
            ])

            with adv_tabs[0]:
                st.header("Gel Structure")
                if st.button("Run Gel Analysis"):
                    with st.spinner("Running..."):
                        try:
                            gel_res = analyze_gel_structure(st.session_state.tracks_data, pixel_size=get_global_pixel_size(), frame_interval=get_global_frame_interval())
                            st.session_state.analysis_results["gel_structure"] = gel_res
                            st.success("Done!")
                        except Exception as e:
                            st.error(f"Error: {e}")

            with adv_tabs[5]:
                st.header("Polymer Physics")
                if st.button("Run Polymer Analysis"):
                    with st.spinner("Running..."):
                        try:
                            poly_res = analyze_polymer_physics(st.session_state.tracks_data, pixel_size=get_global_pixel_size(), frame_interval=get_global_frame_interval())
                            if poly_res.get('success'):
                                st.session_state.analysis_results['polymer_physics'] = poly_res
                                st.success("Done!")
                                st.metric("Scaling Exponent (α)", f"{poly_res['scaling_exponent']:.3f}")
                                st.metric("Regime", poly_res['regime'])
                            else:
                                st.error(poly_res.get('error'))
                        except Exception as e:
                            st.error(f"Error: {e}")
