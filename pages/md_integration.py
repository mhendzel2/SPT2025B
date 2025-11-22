import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from md_integration import load_md_file
from visualization import plot_tracks
from utils import calculate_track_statistics

def show_md_integration_page():
    st.title("Molecular Dynamics Integration")

    md_tabs = st.tabs(["Load Simulation", "Analyze Simulation", "Compare with SPT"])

    with md_tabs[0]:
        st.header("Load Simulation")
        md_file = st.file_uploader("Upload MD file", type=["gro", "pdb", "xtc", "dcd", "trr", "csv", "xyz"])
        if md_file:
            try:
                md_sim = load_md_file(md_file)
                st.session_state.md_simulation = md_sim
                st.success("Loaded MD simulation.")

                if md_sim.trajectory is not None:
                    if st.button("Convert to SPT Format"):
                        md_tracks = md_sim.convert_to_tracks_format(list(range(min(10, md_sim.trajectory.shape[1]))))
                        st.session_state.md_tracks = md_tracks
                        st.success("Converted to tracks.")
                        st.dataframe(md_tracks.head())
            except Exception as e:
                st.error(f"Error: {e}")

    with md_tabs[1]:
        st.header("Analyze Simulation")
        if st.session_state.get("md_simulation"):
            md_sim = st.session_state.md_simulation
            analysis_type = st.selectbox("Analysis Type", ["MSD", "Diffusion Coefficient"])

            if analysis_type == "MSD":
                if st.button("Calculate MSD"):
                    msd_res = md_sim.calculate_msd()
                    fig = md_sim.plot_msd(msd_res, with_fit=True)
                    st.plotly_chart(fig)
            elif analysis_type == "Diffusion Coefficient":
                if st.button("Calculate D"):
                    msd_res = md_sim.calculate_msd()
                    D = md_sim.calculate_diffusion_coefficient(msd_res)
                    st.metric("Diffusion Coefficient", f"{D:.4f}")

    with md_tabs[2]:
        st.header("Compare with SPT")
        if st.session_state.get("md_simulation") and st.session_state.get("tracks_data") is not None:
            comp_type = st.selectbox("Comparison", ["Diffusion Coefficient", "MSD Curves"])

            if comp_type == "Diffusion Coefficient":
                if st.button("Compare D"):
                    md_D = st.session_state.md_simulation.calculate_diffusion_coefficient()
                    # Assume SPT analysis done
                    spt_D = 0.5 # Placeholder
                    st.metric("MD D", f"{md_D:.4f}")
                    st.metric("SPT D", f"{spt_D:.4f}")
            elif comp_type == "MSD Curves":
                if st.button("Compare MSD"):
                    md_msd = st.session_state.md_simulation.calculate_msd()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=md_msd['lag_time'], y=md_msd['msd'], name="MD"))
                    st.plotly_chart(fig)
        else:
            st.warning("Load both MD simulation and SPT data.")
