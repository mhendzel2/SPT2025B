
import pandas as pd
import numpy as np
from analysis import analyze_motion
from biophysical_models import EnergyLandscapeMapper

def test_vacf():
    print("Testing VACF calculation...")
    # Create a synthetic track: constant velocity in x, zero in y
    # v_x = 1.0, v_y = 0.0
    # x(t) = t, y(t) = 0
    frames = np.arange(0, 20)
    x = frames * 1.0
    y = np.zeros_like(frames)
    
    tracks_df = pd.DataFrame({
        'track_id': [1] * len(frames),
        'frame': frames,
        'x': x,
        'y': y
    })
    
    # Analyze motion
    results = analyze_motion(tracks_df, analyze_velocity_autocorr=True, pixel_size=1.0, frame_interval=1.0)
    track_result = results['track_results'].iloc[0]
    
    # Check if keys exist
    if 'velocity_autocorr' not in track_result:
        print("FAIL: 'velocity_autocorr' not found in results.")
        return
    if 'velocity_autocorr_raw' not in track_result:
        print("FAIL: 'velocity_autocorr_raw' not found in results.")
        return
        
    vacf = track_result['velocity_autocorr']
    vacf_raw = track_result['velocity_autocorr_raw']
    
    print(f"VACF (normalized): {vacf[:5]}")
    print(f"VACF (raw): {vacf_raw[:5]}")
    
    # For constant velocity, VACF should be 1.0 (normalized) and v^2 (raw)
    # v_x = 1, v_y = 0 -> v^2 = 1
    
    if np.isclose(vacf[1], 1.0, atol=0.1): # lag 1
        print("PASS: Normalized VACF is approximately 1.0 for constant velocity.")
    else:
        print(f"FAIL: Normalized VACF {vacf[1]} != 1.0")
        
    if np.isclose(vacf_raw[1], 1.0, atol=0.1): # lag 1
        print("PASS: Raw VACF is approximately 1.0 for constant velocity.")
    else:
        print(f"FAIL: Raw VACF {vacf_raw[1]} != 1.0")

def test_energy_landscape_errors():
    print("\nTesting EnergyLandscapeMapper error handling...")
    # Create dummy data
    tracks_df = pd.DataFrame({
        'track_id': [1, 1, 1],
        'frame': [0, 1, 2],
        'x': [0, 1, 2],
        'y': [0, 0, 0]
    })
    
    mapper = EnergyLandscapeMapper(tracks_df)
    
    # Test drift method
    try:
        mapper.map_energy_landscape(method='drift')
        print("FAIL: 'drift' method did not raise NotImplementedError.")
    except NotImplementedError as e:
        print(f"PASS: 'drift' method raised NotImplementedError: {e}")
    except Exception as e:
        print(f"FAIL: 'drift' method raised unexpected exception: {type(e).__name__}: {e}")

    # Test kramers method
    try:
        mapper.map_energy_landscape(method='kramers')
        print("FAIL: 'kramers' method did not raise NotImplementedError.")
    except NotImplementedError as e:
        print(f"PASS: 'kramers' method raised NotImplementedError: {e}")
    except Exception as e:
        print(f"FAIL: 'kramers' method raised unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_vacf()
    test_energy_landscape_errors()
