import argparse, json, sys
from synthetic_tracks import make_dataset, boundaries_from_truth, boundaries_from_segments, boundary_f1

def main():
    try:
        from bayes_bocpd_diffusion import BOCPDDiffusion, BOCPDConfig
        from bayes_hmm_diffusion import BayesHMMDiffusion, HMMConfig
    except ImportError as e:
        print(json.dumps({'success': False, 'error': f'Missing dependency: {e}'}))
        sys.exit(1)

    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.1)
    ap.add_argument("--px", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tol", type=int, default=3)
    args = ap.parse_args()

    tracks, truths = make_dataset(n_tracks=1, dt=args.dt, pixel_size=args.px, sigma_loc=0.02, seed=args.seed)
    truth_bounds = boundaries_from_truth(truths[1])

    bocpd = BOCPDDiffusion(
        tracks[tracks['track_id'] == 1],
        BOCPDConfig(pixel_size=args.px, frame_interval=args.dt, lag_frames=1,
                    hazard_tau=60.0, min_segment_len=8, rmax=400)
    )
    seg_b = bocpd.segment_all()['tracks'][1]['segments']
    m_b = boundary_f1(truth_bounds, boundaries_from_segments(seg_b), tol=args.tol)

    hmm = BayesHMMDiffusion(
        tracks[tracks['track_id'] == 1],
        HMMConfig(pixel_size=args.px, frame_interval=args.dt, lag_frames=1,
                  n_states=3, sticky_kappa=10.0, max_iter=75)
    )
    seg_h = hmm.segment_all()['tracks'][1]['segments']
    m_h = boundary_f1(truth_bounds, boundaries_from_segments(seg_h), tol=args.tol)

    out = {"success": True, "BOCPD": m_b, "HMM": m_h}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
