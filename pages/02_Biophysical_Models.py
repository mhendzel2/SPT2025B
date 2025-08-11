import streamlit as st
import pandas as pd
from analysis_manager import AnalysisManager

st.set_page_config(page_title="Biophysical Models", layout="wide")
st.title("Biophysical Models")

am = AnalysisManager()

with st.form("biofits"):
    col1, col2, col3 = st.columns(3)
    with col1:
        run_rouse = st.checkbox("Run Rouse", True)
        rouse_fit_alpha = st.checkbox("Fit α (Rouse)", False)
        n_beads = st.number_input("n_beads", 10, 5000, 100, 10)
        friction = st.number_input("Friction coeff (N·s/m)", value=1e-8, format="%.2e")
    with col2:
        run_zimm = st.checkbox("Run Zimm", False)
        viscosity = st.number_input("Viscosity (Pa·s)", value=0.001, format="%.3f")
        hydrorad = st.number_input("Hydrodynamic radius (m)", value=5e-9, format="%.2e")
    with col3:
        run_rept = st.checkbox("Run Reptation", False)
        tube_diam = st.number_input("Tube diameter (m)", value=100e-9, format="%.2e")
        contour_len = st.number_input("Contour length (m)", value=1000e-9, format="%.2e")
    st.markdown("**Global**")
    temperature = st.number_input("Temperature (K)", value=300.0)
    max_lag = st.number_input("MSD max lag", 5, 200, 20)
    min_len = st.number_input("Min track length", 3, 200, 5)
    submitted = st.form_submit_button("Run fits", type="primary")

if submitted:
    params = dict(
        run_rouse=run_rouse, rouse_fit_alpha=rouse_fit_alpha, n_beads=int(n_beads),
        friction_coefficient=float(friction),
        run_zimm=run_zimm, solvent_viscosity_Pa_s=float(viscosity), hydrodynamic_radius_m=float(hydrorad),
        run_reptation=run_rept, tube_diameter_m=float(tube_diam), contour_length_m=float(contour_len),
        temperature_K=float(temperature), max_lag=int(max_lag), min_track_length=int(min_len)
    )
    with st.spinner("Fitting models..."):
        res = am.run_biophysical_analysis(params)
    if not res.get("success"):
        st.error(res.get("error", "Failed"))
    else:
        for key in ("rouse", "zimm", "reptation"):
            block = res.get(key)
            if not block:
                continue
            st.subheader(key.capitalize())
            if not block.get("success"):
                st.warning(block.get("error", "Fit failed"))
                continue
            par = block.get("parameters", {})
            if par:
                st.write(pd.DataFrame([par]).T.rename(columns={0: "value"}))
            fig = block.get("visualization")
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            st.caption(f"R² = {block.get('r_squared', float('nan')):.3f}")
