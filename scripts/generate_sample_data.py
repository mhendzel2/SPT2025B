#!/usr/bin/env python3
"""Generate simple synthetic SPT sample data and export to CSV."""

from __future__ import annotations

import argparse

import pandas as pd

from synthetic_track_generator import (
    generate_brownian_motion,
    generate_confined_motion,
    generate_directed_motion,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate synthetic SPT sample data")
    parser.add_argument("--output", default="sample_data.csv", help="Output CSV path")
    parser.add_argument("--n-steps", type=int, default=200, help="Frames per track")
    parser.add_argument("--n-tracks", type=int, default=10, help="Tracks per motion class")
    parser.add_argument("--dt", type=float, default=0.1, help="Frame interval in seconds")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    brownian = generate_brownian_motion(n_steps=args.n_steps, D=0.05, dt=args.dt, n_tracks=args.n_tracks)
    confined = generate_confined_motion(n_steps=args.n_steps, D=0.03, dt=args.dt, L=1.5, n_tracks=args.n_tracks)
    directed = generate_directed_motion(n_steps=args.n_steps, D=0.01, dt=args.dt, v=0.5, n_tracks=args.n_tracks)

    dataset = pd.concat([brownian, confined, directed], ignore_index=True)
    dataset.to_csv(args.output, index=False)
    print(f"wrote {len(dataset)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
