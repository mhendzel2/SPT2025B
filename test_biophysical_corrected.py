"""
Test script for corrected biophysical models.
"""

import numpy as np
import pandas as pd
from biophysical_models_corrected import (
    RouseModel, ConfinedDiffusionModel, AnomalousDiffusionModel, 
    FBMModel, WLCModel, ActiveTransportModel, run_full_analysis
)
from report_builder import ReportBuilder

def test_rouse_model():
    print("Testing Rouse Model...")
    # Generate synthetic data: MSD = 2*d*D*t + Gamma*t^0.5
    t = np.linspace(0.1, 10, 20)
    d = 2
    D_macro = 0.1
    Gamma = 0.5
    msd_true = 2 * d * D_macro * t + Gamma * t**0.5
    # Add noise
    msd_noisy = msd_true + np.random.normal(0, 0.05, size=len(t))
    
    model = RouseModel(dimension=2)
    result = model.fit(t, msd_noisy)
    
    print(f"True Params: D_macro={D_macro}, Gamma={Gamma}")
    print(f"Fitted Params: {result['parameters']}")
    assert result['success']
    assert abs(result['parameters']['D_macro'] - D_macro) < 0.1
    assert abs(result['parameters']['Gamma'] - Gamma) < 0.2

def test_confined_model():
    print("\nTesting Confined Model...")
    # Generate synthetic data
    t = np.linspace(0.1, 10, 20)
    L = 2.0
    D = 0.5
    # Approx formula for generation
    msd_true = L**2 * (1 - np.exp(-4*D*t/L**2))
    msd_noisy = msd_true + np.random.normal(0, 0.05, size=len(t))
    
    model = ConfinedDiffusionModel(dimension=2)
    result = model.fit(t, msd_noisy)
    
    print(f"True Params: L={L}, D={D}")
    print(f"Fitted Params: {result['parameters']}")
    assert result['success']
    # Note: The fitted model uses the series expansion, so it might differ slightly from the simple generation formula
    assert abs(result['parameters']['L_conf'] - L) < 0.2

def test_full_pipeline():
    print("\nTesting Full Pipeline...")
    # Create a dummy tracks dataframe
    tracks = []
    for i in range(5):
        # Create a track
        t = np.arange(0, 50) * 0.1
        x = np.cumsum(np.random.normal(0, np.sqrt(2*0.1*0.1), size=50))
        y = np.cumsum(np.random.normal(0, np.sqrt(2*0.1*0.1), size=50))
        df = pd.DataFrame({'track_id': i, 'frame': np.arange(50), 'x': x/0.1, 'y': y/0.1}) # pixels
        tracks.append(df)
        
    tracks_df = pd.concat(tracks)
    
    results = run_full_analysis(tracks_df, pixel_size=0.1, frame_interval=0.1)
    print(f"Best Model: {results['best_model']}")
    
    # Generate report
    builder = ReportBuilder(output_dir="test_reports")
    report_path = builder.generate_html_report(results)
    print(f"Report generated at: {report_path}")

if __name__ == "__main__":
    test_rouse_model()
    test_confined_model()
    test_full_pipeline()
