import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from visualization import plot_tracks, plot_tracks_3d
from trajectory_heatmap import create_streamlit_heatmap_interface

def show_visualization_page():
    st.title("Visualization Tools")

    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        if st.button("Go to Data Loading"):
            st.session_state.active_page = "Data Loading"
            st.rerun()
    else:
        viz_tabs = st.tabs([
            "Track Visualization",
            "Diffusion Visualization",
            "Motion Visualization",
            "3D Visualization",
            "Trajectory Heatmaps",
            "Custom Visualization"
        ])

        with viz_tabs[0]:
            st.header("Track Visualization")
            col1, col2 = st.columns(2)
            with col1:
                color_by = st.selectbox("Color By", ["track_id", "frame", "x", "y", "Quality"])
                colormap = st.selectbox("Color Map", ["viridis", "plasma", "inferno", "magma", "cividis", "turbo", "rainbow"])
            with col2:
                max_tracks = st.slider("Max Tracks to Display", 1, min(100, st.session_state.tracks_data['track_id'].nunique()), 20)
                plot_type = st.selectbox("Plot Type", ["plotly", "matplotlib"])

            if max_tracks < st.session_state.tracks_data['track_id'].nunique():
                unique_tracks = st.session_state.tracks_data['track_id'].unique()[:max_tracks]
                filtered_tracks = st.session_state.tracks_data[st.session_state.tracks_data['track_id'].isin(unique_tracks)]
            else:
                filtered_tracks = st.session_state.tracks_data

            with st.spinner("Generating track visualization..."):
                try:
                    fig = plot_tracks(filtered_tracks, color_by=color_by, colormap=colormap, plot_type=plot_type)
                    if plot_type == "plotly":
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error generating track visualization: {str(e)}")

        with viz_tabs[1]:
            st.header("Diffusion Visualization")
            if "diffusion" not in st.session_state.analysis_results:
                st.warning("No diffusion analysis results available. Please run diffusion analysis first.")
            else:
                viz_type = st.radio("Visualization Type", ["MSD Curves", "Diffusion Coefficients", "Anomalous Exponents", "Spatial Map"])
                diffusion_results = st.session_state.analysis_results["diffusion"]

                if viz_type == "MSD Curves":
                    if 'msd_data' in diffusion_results and isinstance(diffusion_results['msd_data'], pd.DataFrame) and not diffusion_results['msd_data'].empty:
                        lag_column = 'lag_time' if 'lag_time' in diffusion_results['msd_data'].columns else 'lag'
                        fig = px.scatter(diffusion_results['msd_data'], x=lag_column, y='msd', color='track_id',
                                       labels={lag_column: 'Lag Time (frames)', 'msd': 'MSD (µm²)'},
                                       title='Mean Squared Displacement Curves')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No MSD data available.")

                elif viz_type == "Diffusion Coefficients":
                    if 'track_results' in diffusion_results:
                        fig = px.histogram(diffusion_results['track_results'], x='diffusion_coeff', title='Distribution of Diffusion Coefficients', nbins=20)
                        st.plotly_chart(fig, use_container_width=True)

                elif viz_type == "Spatial Map":
                    if 'track_results' in diffusion_results:
                        track_positions = st.session_state.tracks_data.groupby('track_id')[['x', 'y']].mean().reset_index()
                        merged_data = pd.merge(track_positions, diffusion_results['track_results'], on='track_id')
                        fig = px.scatter(merged_data, x='x', y='y', color='diffusion_coeff', title='Spatial Map of Diffusion Coefficients')
                        fig.update_yaxes(autorange="reversed")
                        st.plotly_chart(fig, use_container_width=True)

        with viz_tabs[3]:
            st.header("3D Visualization")
            with st.spinner("Generating 3D visualization..."):
                try:
                    fig = plot_tracks_3d(filtered_tracks, color_by=color_by, colormap=colormap)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating 3D visualization: {str(e)}")

        with viz_tabs[4]:
            create_streamlit_heatmap_interface()

        with viz_tabs[5]:
            st.header("Custom Visualization")
            x_axis = st.selectbox("X-axis", ["track_id", "frame", "x", "y", "Quality"])
            y_axis = st.selectbox("Y-axis", ["y", "x", "frame", "track_id", "Quality"])
            plot_kind = st.selectbox("Plot Type", ["scatter", "line", "histogram", "box", "violin", "heatmap"])

            if st.button("Generate Plot"):
                try:
                    if plot_kind == "scatter":
                        fig = px.scatter(st.session_state.tracks_data, x=x_axis, y=y_axis)
                    elif plot_kind == "histogram":
                        fig = px.histogram(st.session_state.tracks_data, x=x_axis)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")
