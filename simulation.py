import streamlit as st
import numpy as np
import pandas as pd
from diffusion_simulator import DiffusionSimulator

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
    col1, col2 = st.columns(2)
    with col1:
        particle_diameter = st.slider("Particle Diameter (nm)", 1.0, 100.0, 10.0, 0.1)
        mobility = st.slider("Mobility (step size)", 0.1, 10.0, 1.0, 0.1)
    with col2:
        num_steps = st.slider("Number of Steps", 100, 5000, 1000, 100)
        num_particles = st.slider("Number of Particles", 1, 100, 10, 1)

    # 3. Run Simulation
    st.header("3. Run Simulation")
    if st.button("Start Simulation"):
        if selected_mask_name:
            with st.spinner("Running simulation..."):
                try:
                    # Initialize the simulator
                    simulator = DiffusionSimulator()

                    # Load the mask as a boundary
                    # The mask from session state is directly the numpy array
                    simulator.boundary_map = (mask_data == 0).astype(np.uint8)
                    simulator.region_map = mask_data.astype(np.uint8)

                    # Run the simulation
                    simulation_results = simulator.run_multi_particle_simulation(
                        particle_diameters=[particle_diameter],
                        mobility=mobility,
                        num_steps=num_steps,
                        num_particles_per_size=num_particles
                    )

                    st.session_state.simulation_results = simulation_results
                    st.session_state.simulator = simulator

                    st.success("Simulation finished successfully!")

                except Exception as e:
                    st.error(f"An error occurred during the simulation: {e}")
        else:
            st.error("Please select a mask first.")

    # 4. Display Results
    if 'simulation_results' in st.session_state and 'simulator' in st.session_state:
        st.header("4. Simulation Results")
        simulator = st.session_state.simulator
        results_df = st.session_state.simulation_results

        st.dataframe(results_df)

        # Plotting
        st.subheader("Trajectory Plot")
        if len(simulator.all_trajectories) > 0:
            # Slider to select which trajectory to plot
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
