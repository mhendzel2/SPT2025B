import streamlit as st
import numpy as np
import pandas as pd
from diffusion_simulator import DiffusionSimulator
import plotly.express as px

# Optional external engine adapter
try:
    from external_simulators.nuclear_diffusion_si_adapter import (
        run_simulation as run_nd_si,
        is_available as nd_si_available,
    )
except Exception:
    run_nd_si = None  # type: ignore
    def nd_si_available(*args, **kwargs):  # type: ignore
        return False

def show_simulation_page():
    """
    Displays the simulation page in the Streamlit app.
    """
    st.title("Particle Diffusion Simulation")

    st.write("""
    This page allows you to run particle diffusion simulations within defined boundaries,
    using channel masks generated from your images.
    """)

    # Check if masks are available
    if 'available_masks' not in st.session_state or not st.session_state.available_masks:
        st.warning("No masks available. Please generate masks in the 'Image Processing' tab first.")
        return

    # 1. Mask Selection
    st.header("1. Select a Mask")
    mask_names = list(st.session_state.available_masks.keys())
    selected_mask_name = st.selectbox("Choose a mask for simulation boundaries:", mask_names)

    if selected_mask_name:
        mask_data = st.session_state.available_masks[selected_mask_name]
        st.info(f"Selected mask '{selected_mask_name}' with shape {mask_data.shape} and {len(np.unique(mask_data))} unique values.")

    # 2. Simulation Parameters
    st.header("2. Configure Simulation Parameters")
    engine = st.selectbox(
        "Simulation engine",
        ["Built-in DiffusionSimulator", "Nuclear Diffusion SI (external)"]
    )
    col1, col2 = st.columns(2)
    with col1:
        particle_diameter = st.slider("Particle Diameter (nm)", 1.0, 100.0, 10.0, 0.1)
        mobility = st.slider("Mobility (step size)", 0.1, 10.0, 1.0, 0.1)
    with col2:
        num_steps = st.slider("Number of Steps", 100, 5000, 1000, 100)
        num_particles = st.slider("Number of Particles", 1, 100, 10, 1)

    repo_path = None
    D_val = None
    if engine == "Nuclear Diffusion SI (external)":
        with st.expander("External engine options"):
            repo_path = st.text_input("Local repo path (optional)", value="") or None
            D_val = st.number_input("Diffusion coefficient D (a.u.)", min_value=0.0, value=1.0)

    # 3. Run Simulation
    st.header("3. Run Simulation")
    if st.button("Start Simulation"):
        if selected_mask_name:
            with st.spinner("Running simulation..."):
                try:
                    if engine == "Nuclear Diffusion SI (external)":
                        if run_nd_si is None:
                            st.error("External engine not available in this build.")
                            return
                        params = {
                            "D": float(D_val) if D_val is not None else 1.0,
                            "steps": int(num_steps),
                            "n_particles": int(num_particles),
                            "seed": 0,
                        }
                        tracks_df = run_nd_si(mask_data.astype(np.uint8), params, repo_path=repo_path, allow_builtin_fallback=True)
                        st.session_state.simulation_engine = "external"
                        st.session_state.external_tracks = tracks_df
                        st.success("External simulation finished successfully!")
                    else:
                        # Initialize the simulator
                        simulator = DiffusionSimulator()

                        # Load the mask as a boundary
                        simulator.boundary_map = (mask_data == 0).astype(np.uint8)
                        simulator.region_map = mask_data.astype(np.uint8)

                        # Run the simulation
                        simulation_results = simulator.run_multi_particle_simulation(
                            particle_diameters=[particle_diameter],
                            mobility=mobility,
                            num_steps=num_steps,
                            num_particles_per_size=num_particles
                        )

                        st.session_state.simulation_engine = "builtin"
                        st.session_state.simulation_results = simulation_results
                        st.session_state.simulator = simulator

                        st.success("Simulation finished successfully!")

                except Exception as e:
                    st.error(f"An error occurred during the simulation: {e}")
        else:
            st.error("Please select a mask first.")

    # 4. Display Results
    if st.session_state.get('simulation_engine') == 'builtin' and 'simulation_results' in st.session_state and 'simulator' in st.session_state:
        st.header("4. Simulation Results (Built-in)")
        simulator = st.session_state.simulator
        results_df = st.session_state.simulation_results

        st.dataframe(results_df)

        # Plotting
        st.subheader("Trajectory Plot")
        if len(simulator.all_trajectories) > 0:
            traj_idx_to_plot = st.slider("Select Trajectory to View", 0, len(simulator.all_trajectories) - 1, 0)
            fig_traj = simulator.plot_trajectory(trajectory_idx=traj_idx_to_plot, mode='3d')
            st.plotly_chart(fig_traj, use_container_width=True)
        else:
            st.info("No trajectories were generated. This can happen if particles cannot find a valid starting position.")

        st.subheader("MSD Analysis")
        try:
            fig_msd = simulator.plot_msd_analysis()
            st.plotly_chart(fig_msd, use_container_width=True)
        except ValueError as e:
            st.warning(f"Could not generate MSD plot: {e}")

    if st.session_state.get('simulation_engine') == 'external' and 'external_tracks' in st.session_state:
        st.header("4. Simulation Results (External)")
        tracks_df = st.session_state.external_tracks
        st.dataframe(tracks_df)

        # Basic trajectory visualization per track_id
        if not tracks_df.empty:
            track_ids = sorted(tracks_df['track_id'].unique().tolist())
            sel_id = st.selectbox("Select track to visualize", track_ids)
            tdf = tracks_df[tracks_df['track_id'] == sel_id].sort_values('frame')
            if {'x','y','z'}.issubset(tdf.columns):
                fig = px.line_3d(tdf, x='x', y='y', z='z', color_discrete_sequence=['#1f77b4'])
            else:
                fig = px.line(tdf, x='x', y='y')
            st.plotly_chart(fig, use_container_width=True)
