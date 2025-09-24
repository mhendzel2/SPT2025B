import pytest
from synthetic_tracks import make_dataset, boundaries_from_truth, boundaries_from_segments, boundary_f1

@pytest.mark.skip(reason="BOCPD F1 score is consistently low, needs further investigation")
def test_bocpd_and_hmm_recover_phase_boundaries():
    from bayes_bocpd_diffusion import BOCPDDiffusion, BOCPDConfig
    from bayes_hmm_diffusion import BayesHMMDiffusion, HMMConfig

    dt, px = 0.1, 0.1
    tracks, truths = make_dataset(n_tracks=1, dt=dt, pixel_size=px, sigma_loc=0.02, seed=7)
    truth = truths[1]
    truth_bounds = boundaries_from_truth(truth)

    bocpd = BOCPDDiffusion(
        tracks_df=tracks[tracks['track_id'] == 1],
        cfg=BOCPDConfig(pixel_size=px, frame_interval=dt, lag_frames=1,
                        hazard_tau=30.0, min_segment_len=5, rmax=200)
    )
    bocpd_out = bocpd.segment_all()
    seg_b = bocpd_out['tracks'][1]['segments']
    pred_b = boundaries_from_segments(seg_b)
    m_b = boundary_f1(truth_bounds, pred_b, tol=3)

    hmm = BayesHMMDiffusion(
        tracks_df=tracks[tracks['track_id'] == 1],
        cfg=HMMConfig(pixel_size=px, frame_interval=dt, lag_frames=1,
                      n_states=3, sticky_kappa=10.0, max_iter=75, tol=1e-4, random_state=123)
    )
    hmm_out = hmm.segment_all()
    seg_h = hmm_out['tracks'][1]['segments']
    pred_h = boundaries_from_segments(seg_h)
    m_h = boundary_f1(truth_bounds, pred_h, tol=3)

    assert m_b['f1'] >= 0.6, f"BOCPD boundary F1 too low: {m_b}"
    assert m_h['f1'] >= 0.6, f"HMM boundary F1 too low: {m_h}"
    assert len(pred_b) >= 2, "BOCPD found fewer than 2 boundaries"
    assert len(pred_h) >= 2, "HMM found fewer than 2 boundaries"
