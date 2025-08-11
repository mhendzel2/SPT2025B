import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from state_manager import get_state_manager
from data_loader_unify import assemble_tracks_from_project

# Lazy imports of segmentation wrappers
from biophysical_models import run_bocpd_segmentation, run_hmm_segmentation

st.set_page_config(page_title="Phase Comparison (BOCPD vs HMM)", layout="wide")
st.title("Phase Comparison: BOCPD vs Bayesian HMM")

state = get_state_manager()

# Data guard
if state.get_raw_tracks().empty:
    project = st.session_state.get('current_project', None)
    if project is not None:
        state.set_raw_tracks(assemble_tracks_from_project(project))
if state.get_raw_tracks().empty:
    st.warning("No track data loaded. Please load data first.")
    st.stop()

tracks_df = state.get_raw_tracks()
px = state.get_pixel_size()
dt = state.get_frame_interval()

track_ids = sorted(tracks_df['track_id'].unique().tolist())
track_id = st.selectbox("Track", track_ids)

c1, c2, c3, c4 = st.columns(4)
with c1:
    lag_b = st.number_input("BOCPD lag (frames)", 1, 30, 1)
with c2:
    tau = st.number_input("BOCPD τ (expected run length)", 5.0, 2000.0, 50.0, step=5.0)
with c3:
    lag_h = st.number_input("HMM lag (frames)", 1, 30, 1)
with c4:
    K = st.number_input("HMM states", 2, 10, 3)

if st.button("Run comparison", type="primary"):
    single = tracks_df[tracks_df['track_id'] == track_id].copy()
    if single.empty:
        st.error("Selected track not found.")
        st.stop()

    # BOCPD
    bocpd_cfg = dict(pixel_size=px, frame_interval=dt, lag_frames=int(lag_b),
                     hazard_tau=float(tau), min_segment_len=3, rmax=500)
    bocpd_res = run_bocpd_segmentation(single, bocpd_cfg)
    if not bocpd_res.get('success'):
        st.error(bocpd_res.get('error', 'BOCPD failed'))
        st.stop()
    seg_b = bocpd_res['tracks'].get(track_id, {}).get('segments', pd.DataFrame())

    # HMM
    hmm_cfg = dict(pixel_size=px, frame_interval=dt, lag_frames=int(lag_h),
                   n_states=int(K), sticky_kappa=10.0, max_iter=75)
    hmm_res = run_hmm_segmentation(single, hmm_cfg)
    if not hmm_res.get('success'):
        st.error(hmm_res.get('error', 'HMM failed'))
        st.stop()
    seg_h = hmm_res['tracks'].get(track_id, {}).get('segments', pd.DataFrame())
    states_h = hmm_res['tracks'].get(track_id, {}).get('states', pd.DataFrame())

    # Plot
    fig = go.Figure()
    for i, row in seg_b.reset_index(drop=True).iterrows():
        t0 = row['start_frame'] * dt
        t1 = row['end_frame'] * dt
        yv = row.get('D_mean', row.get('D', np.nan))
        fig.add_trace(go.Scatter(
            x=[t0, t1], y=[yv, yv], mode='lines',
            line=dict(width=5), name='BOCPD D', showlegend=(i == 0)
        ))
    palette = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#B6E880','#FF6692']
    for j, row in seg_h.reset_index(drop=True).iterrows():
        start = row['start_frame']
        end = row['end_frame']
        tmid = 0.5 * (start + end) * dt
        width = (end - start + 1) * dt
        dv = row.get('D_state', row.get('D', np.nan))
        stid = int(row.get('state', j))
        fig.add_trace(go.Bar(
            x=[tmid], y=[dv], width=[width],
            name=f"HMM state {stid}",
            marker_color=palette[stid % len(palette)],
            opacity=0.45, showlegend=(j == 0)
        ))
    fig.update_layout(
        barmode='overlay',
        xaxis_title="Time (s)",
        yaxis_title="Diffusion coefficient (μm²/s)",
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True,
                    key=f"cmp-{int(track_id)}-{int(lag_b)}-{int(lag_h)}-{K}")

    colA, colB = st.columns(2)
    with colA:
        st.subheader("BOCPD segments")
        st.dataframe(seg_b, hide_index=True, use_container_width=True)
        st.download_button("Download BOCPD CSV",
                           seg_b.to_csv(index=False).encode(),
                           file_name=f"track_{track_id}_bocpd.csv",
                           mime="text/csv",
                           key=f"dl-b-{track_id}")
    with colB:
        st.subheader("HMM segments")
        st.dataframe(seg_h, hide_index=True, use_container_width=True)
        st.download_button("Download HMM CSV",
                           seg_h.to_csv(index=False).encode(),
                           file_name=f"track_{track_id}_hmm.csv",
                           mime="text/csv",
                           key=f"dl-h-{track_id}")
    if isinstance(states_h, pd.DataFrame) and not states_h.empty:
        st.subheader("HMM state summary")
        st.dataframe(states_h, hide_index=True, use_container_width=True)
