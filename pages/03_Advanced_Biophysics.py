import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from analysis_manager import AnalysisManager
from advanced_biophysical_metrics import AdvancedMetricsAnalyzer, MetricConfig

st.set_page_config(page_title="Advanced Biophysical Metrics", layout="wide")
st.title("Advanced Biophysical Metrics")

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

    # NGP
    ngp = res['ngp']
    if isinstance(ngp, pd.DataFrame) and not ngp.empty:
        st.subheader("Non-Gaussian Parameter")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ngp['tau_s'], y=ngp['NGP_1D'], mode='lines+markers', name='NGP 1D (Δx)'))
        fig.add_trace(go.Scatter(x=ngp['tau_s'], y=ngp['NGP_2D'], mode='lines+markers', name='NGP 2D (Δr)'))
        fig.update_layout(xaxis_title="τ (s)", yaxis_title="NGP", hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("NGP table"):
            st.dataframe(ngp, use_container_width=True, hide_index=True)

    # Ergodicity metrics
    erg = res['ergodicity']
    if isinstance(erg, pd.DataFrame) and not erg.empty:
        st.subheader("Ergodicity Metrics")
        c1, c2 = st.columns(2)
        with c1:
            f1 = go.Figure()
            f1.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_ratio'], mode='lines+markers', name='EB ratio'))
            if {'EB_ratio_CI_low','EB_ratio_CI_high'}.issubset(erg.columns):
                f1.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_ratio_CI_low'], mode='lines', name='CI low', line=dict(dash='dot')))
                f1.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_ratio_CI_high'], mode='lines', name='CI high', line=dict(dash='dot'), fill='tonexty'))
            f1.update_layout(xaxis_title="τ (s)", yaxis_title="⟨TAMSD⟩/EAMSD")
            st.plotly_chart(f1, use_container_width=True)
        with c2:
            f2 = go.Figure()
            f2.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_parameter'], mode='lines+markers', name='EB parameter'))
            if {'EB_param_CI_low','EB_param_CI_high'}.issubset(erg.columns):
                f2.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_param_CI_low'], mode='lines', name='CI low', line=dict(dash='dot')))
                f2.add_trace(go.Scatter(x=erg['tau_s'], y=erg['EB_param_CI_high'], mode='lines', name='CI high', line=dict(dash='dot'), fill='tonexty'))
            f2.update_layout(xaxis_title="τ (s)", yaxis_title="EB parameter")
            st.plotly_chart(f2, use_container_width=True)
        with st.expander("Ergodicity table"):
            st.dataframe(erg, use_container_width=True, hide_index=True)

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

    # van Hove
    st.subheader("van Hove Self-part Distributions")
    lags = res['summary']['lags']
    if lags:
        chosen_lag = st.select_slider("Lag (frames)", options=lags, value=lags[0])
        clip = st.slider("Clip to σ (0 = none)", 0.0, 6.0, 0.0, 0.5)
        cfg = MetricConfig(pixel_size=float(px), frame_interval=float(dt),
                           min_track_length=int(min_len), max_lag=max(lags),
                           log_lag=False, n_hist_bins=int(n_bins), seed=int(seed),
                           n_bootstrap=int(n_boot))
        analyzer = AdvancedMetricsAnalyzer(am.state.get_raw_tracks(), cfg)
        vh = analyzer.van_hove(chosen_lag, bins=int(n_bins), clip_sigma=None if clip == 0 else clip)
        if vh['dx_centers'].size:
            c1, c2 = st.columns(2)
            with c1:
                fx = go.Figure()
                fx.add_trace(go.Bar(x=vh['dx_centers'], y=vh['dx_density'], name='p(Δx)'))
                fx.update_layout(xaxis_title="Δx (μm)", yaxis_title="density")
                st.plotly_chart(fx, use_container_width=True)
            with c2:
                fr = go.Figure()
                fr.add_trace(go.Bar(x=vh['r_centers'], y=vh['r_density'], name='p(Δr)'))
                fr.update_layout(xaxis_title="Δr (μm)", yaxis_title="density")
                st.plotly_chart(fr, use_container_width=True)

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
