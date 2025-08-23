import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis_manager import AnalysisManager
from advanced_biophysical_metrics import AdvancedMetricsAnalyzer, MetricConfig
from visualization import plot_advanced_metric_diagnostics

def show_advanced_biophysical_metrics():
    st.subheader("Advanced Biophysical Metrics")

    am = AnalysisManager()

    with st.form("adv_biomets"):
        c1, c2, c3 = st.columns(3)
        with c1:
            max_lag = st.number_input("Max lag (frames)", 2, 500, 20)
            min_len = st.number_input("Min track length", 3, 500, 5)
            log_lag = st.checkbox("Log-spaced lags", True)
        with c2:
            n_bins = st.number_input("van Hove bins", 20, 200, 60, 5)
            n_boot = st.number_input("Bootstrap samples (0=off)", 0, 5000, 500, 50)
            seed = st.number_input("Random seed", 0, 10000, 1234, 1)
        with c3:
            px = st.number_input("Pixel size (μm/px)", value=float(am.state.get_pixel_size()))
            dt = st.number_input("Frame interval (s/frame)", value=float(am.state.get_frame_interval()))
        submitted = st.form_submit_button("Run", type="primary")

    if submitted:
        params = dict(
            max_lag=int(max_lag),
            min_track_length=int(min_len),
            log_lag=bool(log_lag),
            n_hist_bins=int(n_bins),
            n_bootstrap=int(n_boot),
            seed=int(seed),
            pixel_size=float(px),
            frame_interval=float(dt)
        )
        with st.spinner("Computing advanced metrics..."):
            res = am.run_advanced_biophysical_metrics(params)
        if not res.get('success'):
            st.error(res.get('error', 'Failed'))
            st.stop()

        st.subheader("Summary")
        st.write(res['summary'])

        # Create the composite diagnostics plot
        st.subheader("Advanced Metric Diagnostics")
        lags = res['summary']['lags']
        if lags:
            # Prepare data for the plot
            erg_df = res.get('ergodicity', pd.DataFrame())
            ngp_df = res.get('ngp', pd.DataFrame())

            # Get Van Hove data for a short and long lag
            cfg = MetricConfig(**params)
            analyzer = AdvancedMetricsAnalyzer(am.state.get_raw_tracks(), cfg)

            short_lag_idx = lags[0]
            long_lag_idx = lags[min(len(lags)-1, 5)] # Use 5th lag or last if fewer

            vh_short = analyzer.van_hove(short_lag_idx)
            vh_long = analyzer.van_hove(long_lag_idx)

            short_lag_time_val = short_lag_idx * params.get('frame_interval', 1.0)
            long_lag_time_val = long_lag_idx * params.get('frame_interval', 1.0)

            diag_fig = plot_advanced_metric_diagnostics(
                ergodicity_df=erg_df,
                ngp_df=ngp_df,
                van_hove_short_lag=vh_short,
                van_hove_long_lag=vh_long,
                short_lag_time=short_lag_time_val,
                long_lag_time=long_lag_time_val
            )
            st.plotly_chart(diag_fig, use_container_width=True)

            # Keep expanders for detailed tables
            with st.expander("Ergodicity Table"):
                st.dataframe(erg_df, use_container_width=True, hide_index=True)
            with st.expander("NGP Table"):
                st.dataframe(ngp_df, use_container_width=True, hide_index=True)
        else:
            st.warning("Not enough data to generate diagnostic plots.")


        # VACF
        vacf = res['vacf']
        if isinstance(vacf, pd.DataFrame) and not vacf.empty:
            st.subheader("Velocity Autocorrelation (VACF)")
            f = go.Figure()
            f.add_trace(go.Scatter(x=vacf['tau_s'], y=vacf['VACF'], mode='lines+markers', name='VACF'))
            f.update_layout(xaxis_title="τ (s)", yaxis_title="VACF")
            st.plotly_chart(f, use_container_width=True)
            with st.expander("VACF table"):
                st.dataframe(vacf, use_container_width=True, hide_index=True)

        # TAMSD / EAMSD & Hurst
        tamsd = res['tamsd']; eamsd = res['eamsd']
        if not tamsd.empty and not eamsd.empty:
            st.subheader("TAMSD vs EAMSD")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=eamsd['tau_s'], y=eamsd['eamsd'], mode='lines+markers', name='EAMSD'))
            q = tamsd.groupby('tau_s')['tamsd'].quantile([0.25,0.5,0.75]).unstack()
            fig.add_trace(go.Scatter(x=q.index, y=q[0.5], mode='lines', name='TAMSD median'))
            fig.add_trace(go.Scatter(x=q.index, y=q[0.25], mode='lines', name='TAMSD Q1', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=q.index, y=q[0.75], mode='lines', name='TAMSD Q3', line=dict(dash='dot'), fill='tonexty'))
            fig.update_layout(xaxis_title="τ (s)", yaxis_title="MSD (μm²)")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Hurst exponents (per track)"):
                st.dataframe(res['hurst'], use_container_width=True, hide_index=True)

def show_biophysical_models():
    st.subheader("Biophysical Models")

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
