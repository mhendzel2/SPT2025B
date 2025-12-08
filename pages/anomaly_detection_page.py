import streamlit as st
from anomaly_detection import AnomalyDetector
from anomaly_visualization import AnomalyVisualizer
from ui_utils import navigate_to

def show_anomaly_detection_page():
    st.title("AI-Powered Anomaly Detection")

    if st.session_state.tracks_data is None:
        st.warning("No track data loaded. Please upload track data first.")
        st.button("Go to Data Loading", on_click=navigate_to, args=("Data Loading",))
        return

    st.write("Detect unusual particle behaviors using advanced machine learning algorithms.")

    # Parameters for anomaly detection
    st.subheader("Detection Parameters")

    col1, col2 = st.columns(2)
    with col1:
        contamination = st.slider("Expected Anomaly Rate (%)", 1.0, 20.0, 5.0, 0.5) / 100.0
        z_threshold = st.number_input("Velocity Change Threshold", 1.0, 5.0, 3.0, 0.1)
        expansion_threshold = st.number_input("Confinement Violation Threshold", 1.0, 5.0, 2.0, 0.1)
    with col2:
        reversal_threshold = st.number_input("Directional Change Threshold (radians)", 1.0, 3.14, 2.5, 0.1)
        clustering_eps = st.number_input("Spatial Clustering Parameter", 1.0, 20.0, 5.0, 1.0)
        min_track_length = st.slider("Minimum Track Length", 5, 50, 10)

    if st.button("Run Anomaly Detection"):
        with st.spinner("Analyzing particle behavior patterns..."):
            try:
                detector = AnomalyDetector(contamination=contamination)
                filtered_tracks = st.session_state.tracks_data.groupby('track_id').filter(lambda x: len(x) >= min_track_length)
                if len(filtered_tracks) > 0:
                    anomaly_results = detector.comprehensive_anomaly_detection(filtered_tracks)
                    st.session_state.analysis_results["anomaly_detection"] = {
                        'results': anomaly_results,
                        'detector': detector,
                        'parameters': {
                            'contamination': contamination,
                            'z_threshold': z_threshold,
                            'expansion_threshold': expansion_threshold,
                            'reversal_threshold': reversal_threshold,
                            'clustering_eps': clustering_eps,
                            'min_track_length': min_track_length
                        }
                    }
                    st.success("Anomaly detection completed successfully!")
                else:
                    st.warning("No tracks meet the minimum length requirement for analysis.")
            except Exception as e:
                st.error(f"Error during anomaly detection: {str(e)}")

    if "anomaly_detection" in st.session_state.analysis_results:
        results = st.session_state.analysis_results["anomaly_detection"]
        anomaly_results = results['results']
        visualizer = AnomalyVisualizer()
        filtered_tracks = st.session_state.tracks_data.groupby('track_id').filter(lambda x: len(x) >= results['parameters']['min_track_length'])
        visualizer.create_anomaly_dashboard(filtered_tracks, anomaly_results)
