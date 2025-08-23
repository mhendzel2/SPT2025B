import streamlit as st
import pandas as pd
import numpy as np
import io

try:
    from . import analysis
    from . import data_loader
except (ImportError, SystemError):
    import analysis
    import data_loader

def show():
    """
    Renders the interactive parameter tuning dashboard for DBSCAN.
    """
    st.title("Interactive Parameter Tuning for DBSCAN")

    # --- Data Loading ---
    st.sidebar.header("Data Source")
    uploaded_file = st.sidebar.file_uploader("Upload Tracks File (CSV or Excel)", type=['csv', 'xlsx', 'xls'])

    tracks_df = None
    if uploaded_file is not None:
        try:
            # Create a mock file object that the loader can handle
            file_like_object = io.BytesIO(uploaded_file.getvalue())
            file_like_object.name = uploaded_file.name
            tracks_df = data_loader.load_tracks_file(file_like_object)
            st.sidebar.success(f"Loaded {len(tracks_df)} data points from {uploaded_file.name}")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            st.error(f"Could not load the uploaded file. Please ensure it is a valid tracks file. Error: {e}")
            return
    else:
        st.info("Please upload a tracks file to begin.")
        return

    st.header("DBSCAN Clustering Parameters")

    # --- Parameter Sliders ---
    pixel_size = st.sidebar.number_input("Pixel Size (µm/pixel)", value=1.0, min_value=0.01, step=0.01)

    epsilon_pixels = st.slider("Epsilon (in pixels)", 0.1, 20.0, 5.0, 0.1,
                               help="The maximum distance between two samples for one to be considered as in the neighborhood of the other.")
    min_samples = st.slider("Minimum Samples (to form a cluster)", 1, 20, 5, 1,
                            help="The number of samples in a neighborhood for a point to be considered as a core point.")

    epsilon_um = epsilon_pixels * pixel_size

    # --- Frame Selection ---
    if 'frame' in tracks_df.columns:
        frames = sorted(tracks_df['frame'].unique())
        selected_frame = st.select_slider("Select Frame to Analyze", options=frames, value=frames[0])
        frame_data = tracks_df[tracks_df['frame'] == selected_frame].copy()
    else:
        st.warning("Frame information not found. Analyzing all points together.")
        frame_data = tracks_df.copy()
        selected_frame = "All"


    # --- Analysis and Visualization ---
    if st.button("Run DBSCAN Analysis"):
        if len(frame_data) < min_samples:
            st.warning(f"Not enough points ({len(frame_data)}) in frame {selected_frame} for the given min_samples ({min_samples}).")
            return

        with st.spinner(f"Running DBSCAN on frame {selected_frame}..."):
            from sklearn.cluster import DBSCAN
            import plotly.express as px

            coords = frame_data[['x', 'y']].values
            db = DBSCAN(eps=epsilon_pixels, min_samples=min_samples).fit(coords)
            labels = db.labels_

            frame_data['cluster'] = labels

            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise_points = np.sum(labels == -1)

            st.success(f"Found {n_clusters} clusters and {noise_points} noise points.")

            fig = px.scatter(
                frame_data,
                x="x",
                y="y",
                color="cluster",
                title=f"DBSCAN Clustering on Frame {selected_frame}",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={'cluster': 'Cluster ID'}
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    show()
