import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from synthetic_tracks import Phase, make_dataset, boundaries_from_truth, boundaries_from_segments, boundary_f1

# Fallback wrappers if AnalysisManager not present
try:
    from analysis_manager import AnalysisManager
    am = AnalysisManager()
    run_bocpd = lambda cfg: am.run_bocpd_segmentation(cfg)
    run_hmm = lambda cfg: am.run_hmm_segmentation(cfg)
except ImportError:
    # Directly use wrapper functions if available
    try:
        from biophysical_models import run_bocpd_segmentation, run_hmm_segmentation
        run_bocpd = lambda cfg: run_bocpd_segmentation(st.session_state['tracks_df'], cfg)
        run_hmm = lambda cfg: run_hmm_segmentation(st.session_state['tracks_df'], cfg)
    except Exception:
        run_bocpd = lambda cfg: {'success': False, 'error': 'No segmentation backend'}
        run_hmm = lambda cfg: {'success': False, 'error': 'No segmentation backend'}

st.set_page_config(page_title="Synthetic Lab", layout="wide")
st.title("Synthetic Data Lab – Segmentation Regression")

with st.expander("Synthetic dataset parameters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        dt = st.number_input("Frame interval dt (s)", value=0.1, min_value=0.001, format="%.3f")
        px = st.number_input("Pixel size (μm/px)", value=0.1, min_value=0.0001, format="%.4f")
        sigma_loc = st.number_input("Localization noise σ (μm)", value=0.02, min_value=0.0, format="%.3f")
    with c2:
        n1 = st.number_input("Phase1 frames", 10, 5000, 120)
        D1 = st.number_input("Phase1 D (μm²/s)", 0.0001, 10.0, 0.02)
        v1 = st.number_input("Phase1 vx (μm/s)", -2.0, 2.0, 0.0)
        n_tracks = st.number_input("Tracks", 1, 10, 1)
    with c3:
        n2 = st.number_input("Phase2 frames", 10, 5000, 120)
        D2 = st.number_input("Phase2 D (μm²/s)", 0.0001, 10.0, 0.06)
        v2 = st.number_input("Phase2 vx (μm/s)", -2.0, 2.0, 0.0)
        seed = st.number_input("Random seed", 0, 1_000_000, 7)
    c4, c5 = st.columns(2)
    with c4:
        n3 = st.number_input("Phase3 frames", 10, 5000, 120)
        D3 = st.number_input("Phase3 D (μm²/s)", 0.0001, 10.0, 0.02)
    with c5:
        v3 = st.number_input("Phase3 vx (μm/s)", -2.0, 2.0, 0.2)

if st.button("Generate synthetic dataset", type="primary"):
    phases = [
        Phase(n=int(n1), D=float(D1), vx=float(v1)),
        Phase(n=int(n2), D=float(D2), vx=float(v2)),
        Phase(n=int(n3), D=float(D3), vx=float(v3))
    ]
    tracks, truths = make_dataset(
        n_tracks=int(n_tracks), phases=phases, dt=float(dt),
        pixel_size=float(px), sigma_loc=float(sigma_loc), seed=int(seed)
    )
    st.session_state['tracks_df'] = tracks
    st.session_state['synthetic_truth'] = truths
    st.success(f"Created {int(n_tracks)} track(s), total frames {len(tracks)}.")

tracks_df = st.session_state.get('tracks_df', pd.DataFrame())
if tracks_df.empty:
    st.info("Generate dataset to continue.")
    st.stop()

st.subheader("Segmentation decoders")
c1, c2, c3 = st.columns(3)
with c1:
    lag_b = st.number_input("BOCPD lag (frames)", 1, 50, 1)
    tau = st.number_input("BOCPD expected run length τ", 5.0, 5000.0, 60.0, step=5.0)
with c2:
    min_seg = st.number_input("BOCPD min segment length", 1, 500, 8)
    rmax = st.number_input("BOCPD rmax", 10, 5000, 400, step=10)
with c3:
    K = st.number_input("HMM states", 2, 12, 3)
    sticky = st.number_input("HMM stickiness κ", 0.0, 200.0, 10.0)
    hmm_iter = st.number_input("HMM max iter", 10, 1000, 75, step=5)

if st.button("Run decoders", type="primary"):
    with st.spinner("Running BOCPD/HMM..."):
        # Provide dataframe directly if needed by fallback
        cfg_b = dict(pixel_size=float(px), frame_interval=float(dt), lag_frames=int(lag_b),
                     hazard_tau=float(tau), min_segment_len=int(min_seg), rmax=int(rmax))
        cfg_h = dict(pixel_size=float(px), frame_interval=float(dt), lag_frames=1,
                     n_states=int(K), sticky_kappa=float(sticky), max_iter=int(hmm_iter))
        # Fallback path expects tracks in session
        st.session_state['bocpd_res'] = run_bocpd(cfg_b)
        st.session_state['hmm_res'] = run_hmm(cfg_h)
    if not st.session_state['bocpd_res'].get('success'):
        st.error(st.session_state['bocpd_res'].get('error', 'BOCPD failed'))
    if not st.session_state['hmm_res'].get('success'):
        st.error(st.session_state['hmm_res'].get('error', 'HMM failed'))

res_b = st.session_state.get('bocpd_res')
res_h = st.session_state.get('hmm_res')
truths = st.session_state.get('synthetic_truth', {})

if res_b and res_h and truths and res_b.get('success') and res_h.get('success'):
    st.subheader("Per-track results")
    tids = sorted(tracks_df['track_id'].unique())
    tid = st.selectbox("Track ID", tids)
    truth = truths.get(tid, pd.DataFrame())
    seg_b = res_b.get('tracks', {}).get(tid, {}).get('segments', pd.DataFrame())
    seg_h = res_h.get('tracks', {}).get(tid, {}).get('segments', pd.DataFrame())

    tb = boundaries_from_truth(truth)
    pb = boundaries_from_segments(seg_b)
    ph = boundaries_from_segments(seg_h)
    mb = boundary_f1(tb, pb, tol=3)
    mh = boundary_f1(tb, ph, tol=3)

    cA, cB = st.columns(2)
    with cA:
        st.write(f"BOCPD boundary F1 = {mb['f1']:.3f} (P={mb['precision']:.2f}, R={mb['recall']:.2f})")
        st.dataframe(seg_b, use_container_width=True)
    with cB:
        st.write(f"HMM boundary F1 = {mh['f1']:.3f} (P={mh['precision']:.2f}, R={mh['recall']:.2f})")
        st.dataframe(seg_h, use_container_width=True)

    fig = go.Figure()
    # Truth D
    for _, row in truth.iterrows():
        fig.add_trace(go.Scatter(
            x=[row.start_frame * dt, row.end_frame * dt],
            y=[row.D, row.D],
            mode='lines',
            line=dict(width=2, dash='dot'),
            name='Truth D',
            showlegend=False
        ))
    # BOCPD mean D (if columns exist)
    if not seg_b.empty and 'D_mean' in seg_b.columns:
        for i, (_, row) in enumerate(seg_b.iterrows()):
            fig.add_trace(go.Scatter(
                x=[row.start_frame * dt, row.end_frame * dt],
                y=[row.D_mean, row.D_mean],
                mode='lines',
                line=dict(width=5),
                name='BOCPD D',
                showlegend=(i == 0)
            ))
    # HMM states bars (if columns)
    if not seg_h.empty and 'D_state' in seg_h.columns and 'state' in seg_h.columns:
        palette = ['#636EFA','#EF553B','#00CC96','#AB63FA','#FFA15A','#19D3F3','#B6E880','#FF6692']
        for j, (_, row) in enumerate(seg_h.iterrows()):
            mid = 0.5 * (row.start_frame + row.end_frame) * dt
            width = (row.end_frame - row.start_frame + 1) * dt
            fig.add_trace(go.Bar(
                x=[mid],
                y=[row.D_state],
                width=[width],
                marker_color=palette[row.state % len(palette)],
                opacity=0.5,
                name=f"HMM state {row.state}",
                showlegend=(j == 0)
            ))
    fig.update_layout(
        xaxis_title="Time (s)",
        yaxis_title="D (μm²/s)",
        barmode='overlay',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True, key=f"synthetic-track-{tid}")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download track CSV",
            tracks_df[tracks_df['track_id'] == tid].to_csv(index=False).encode(),
            file_name=f"synthetic_track_{tid}.csv",
            mime="text/csv",
            key=f"dl-track-{tid}"
        )
    with c2:
        if not seg_b.empty:
            st.download_button(
                "Download BOCPD segments",
                seg_b.to_csv(index=False).encode(),
                file_name=f"synthetic_track_{tid}_bocpd.csv",
                mime="text/csv",
                key=f"dl-bocpd-{tid}"
            )
    with c3:
        if not seg_h.empty:
            st.download_button(
                "Download HMM segments",
                seg_h.to_csv(index=False).encode(),
                file_name=f"synthetic_track_{tid}_hmm.csv",
                mime="text/csv",
                key=f"dl-hmm-{tid}"
            )
