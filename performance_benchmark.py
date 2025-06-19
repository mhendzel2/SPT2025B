#!/usr/bin/env python3
"""
Performance benchmark script for MSD calculation optimization.
Compares original nested loop implementation with vectorized version.
"""

import numpy as np
import pandas as pd
import time
from typing import Dict, List, Any

def calculate_msd_original(tracks_df: pd.DataFrame, max_lag: int = 20, pixel_size: float = 1.0, 
                          frame_interval: float = 1.0, min_track_length: int = 5) -> pd.DataFrame:
    """Original implementation with nested loops for comparison."""
    if tracks_df.empty:
        raise ValueError("tracks_df cannot be empty")
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    grouped = tracks_df.groupby('track_id')
    msd_results = {'track_id': [], 'lag_time': [], 'msd': [], 'n_points': []}
    
    for track_id, track_data in grouped:
        if len(track_data) < min_track_length:
            continue
            
        track_data = track_data.sort_values('frame')
        x = track_data['x'].values.astype(float) * pixel_size
        y = track_data['y'].values.astype(float) * pixel_size
        frames = track_data['frame'].values
        
        for lag in range(1, min(max_lag + 1, len(track_data))):
            sd_list = []
            lag_time_list = []
            
            for i in range(len(track_data) - lag):
                dx = x[i + lag] - x[i]
                dy = y[i + lag] - y[i]
                sd = dx**2 + dy**2
                dt = (frames[i + lag] - frames[i]) * frame_interval
                
                sd_list.append(sd)
                lag_time_list.append(dt)
            
            if sd_list:
                msd_results['track_id'].append(track_id)
                mean_lag_time = np.mean(lag_time_list)
                msd_results['lag_time'].append(mean_lag_time)
                msd_results['msd'].append(np.mean(sd_list))
                msd_results['n_points'].append(len(sd_list))
    
    return pd.DataFrame(msd_results)

def calculate_msd_optimized(tracks_df: pd.DataFrame, max_lag: int = 20, pixel_size: float = 1.0, 
                           frame_interval: float = 1.0, min_track_length: int = 5) -> pd.DataFrame:
    """Optimized implementation with vectorized operations."""
    if tracks_df.empty:
        raise ValueError("tracks_df cannot be empty")
    
    required_columns = ['track_id', 'frame', 'x', 'y']
    missing_columns = [col for col in required_columns if col not in tracks_df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    grouped = tracks_df.groupby('track_id')
    msd_results = {'track_id': [], 'lag_time': [], 'msd': [], 'n_points': []}
    
    for track_id, track_data in grouped:
        if len(track_data) < min_track_length:
            continue
            
        track_data = track_data.sort_values('frame')
        x = track_data['x'].values.astype(float) * pixel_size
        y = track_data['y'].values.astype(float) * pixel_size
        frames = track_data['frame'].values
        
        for lag in range(1, min(max_lag + 1, len(track_data))):
            n_points = len(track_data) - lag
            if n_points <= 0:
                continue
                
            dx = x[lag:] - x[:-lag]
            dy = y[lag:] - y[:-lag]
            sd = dx**2 + dy**2
            dt = (frames[lag:] - frames[:-lag]) * frame_interval
            
            if len(sd) > 0:
                msd_results['track_id'].append(track_id)
                msd_results['lag_time'].append(np.mean(dt))
                msd_results['msd'].append(np.mean(sd))
                msd_results['n_points'].append(len(sd))
    
    return pd.DataFrame(msd_results)

def generate_test_data(n_tracks: int = 50, n_points_per_track: int = 200) -> pd.DataFrame:
    """Generate synthetic particle tracking data for benchmarking."""
    np.random.seed(42)
    tracks_data = []
    
    for track_id in range(n_tracks):
        x_start, y_start = np.random.uniform(0, 100, 2)
        x_positions = np.cumsum(np.random.normal(0, 1, n_points_per_track)) + x_start
        y_positions = np.cumsum(np.random.normal(0, 1, n_points_per_track)) + y_start
        
        for i in range(n_points_per_track):
            tracks_data.append({
                'track_id': track_id,
                'frame': i,
                'x': x_positions[i],
                'y': y_positions[i]
            })
    
    return pd.DataFrame(tracks_data)

def run_benchmark():
    """Run performance benchmark comparing original vs optimized implementations."""
    print("SPT2025B MSD Calculation Performance Benchmark")
    print("=" * 50)
    
    test_configs = [
        (10, 100),   # Small: 10 tracks, 100 points each
        (50, 200),   # Medium: 50 tracks, 200 points each  
        (100, 500),  # Large: 100 tracks, 500 points each
    ]
    
    for n_tracks, n_points in test_configs:
        print(f"\nTesting with {n_tracks} tracks, {n_points} points per track")
        print("-" * 40)
        
        test_data = generate_test_data(n_tracks, n_points)
        print(f"Generated {len(test_data)} total data points")
        
        start_time = time.time()
        result_original = calculate_msd_original(test_data, max_lag=20)
        original_time = time.time() - start_time
        
        start_time = time.time()
        result_optimized = calculate_msd_optimized(test_data, max_lag=20)
        optimized_time = time.time() - start_time
        
        results_match = len(result_original) == len(result_optimized)
        if results_match and len(result_original) > 0:
            msd_diff = np.abs(result_original['msd'].values.astype(float) - result_optimized['msd'].values.astype(float))
            results_match = np.all(msd_diff < 1e-10)
        
        speedup = original_time / optimized_time if optimized_time > 0 else float('inf')
        
        print(f"Original implementation: {original_time:.4f} seconds")
        print(f"Optimized implementation: {optimized_time:.4f} seconds")
        print(f"Speedup: {speedup:.1f}x")
        print(f"Results match: {'✓' if results_match else '✗'}")
        print(f"MSD calculations: {len(result_original)}")

if __name__ == "__main__":
    run_benchmark()
