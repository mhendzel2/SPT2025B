"""
Test script for enhanced diffusion simulator integration.
"""

import numpy as np
import pandas as pd
from diffusion_simulator import DiffusionSimulator

def test_enhanced_diffusion_simulator():
    """Test enhanced diffusion simulator functionality."""
    simulator = DiffusionSimulator()
    
    assert simulator.pixel_size_nm == 1.0
    print("✓ 1nm scaling initialized correctly")
    
    simulator.set_optical_slice_thickness(250.0)
    assert simulator.optical_slice_thickness_nm == 250.0
    print("✓ Optical slice thickness setting works")
    
    simulator.set_optical_slice_thickness(175.0)
    assert simulator.optical_slice_thickness_nm == 150.0
    print("✓ Optical slice thickness rounds to valid values")
    
    crowding_map = simulator.generate_crowding_map((50, 50, 10), 30)
    assert crowding_map.shape == (50, 50, 10)
    assert np.sum(crowding_map) > 0
    print("✓ Crowding map generation works")
    
    gel_map = simulator.generate_gel_map((50, 50, 10), 10)
    assert gel_map.shape == (50, 50, 10)
    print("✓ Gel map generation works")
    
    simulator.region_map = np.zeros((50, 50, 10), dtype=np.uint8)
    simulator.region_map[10:40, 10:40, :] = 1
    simulator.region_map[20:30, 20:30, :] = 2
    
    simulator.set_region_properties(1, "Cytoplasm", 0.8)
    simulator.set_region_properties(2, "Nucleus", 1.2)
    
    assert simulator.region_names[1] == "Cytoplasm"
    assert simulator.partition_coefficients[1] == 0.8
    assert simulator.region_names[2] == "Nucleus"
    assert simulator.partition_coefficients[2] == 1.2
    print("✓ Region properties setting works")
    
    simulator.add_liquid_liquid_boundary(1, 2, 0.3, 0.7)
    assert simulator.partition_coefficients["1_to_2"] == 0.3
    assert simulator.partition_coefficients["2_to_1"] == 0.7
    print("✓ Liquid-liquid boundary setting works")
    
    simulator.boundary_map = None
    final_disp = simulator.run_single_simulation(10.0, 0.5, 100)
    assert not np.isnan(final_disp)
    assert final_disp > 0
    print("✓ Single particle simulation with regions works")
    
    tracks_df = simulator.convert_to_tracks_format()
    required_columns = ['track_id', 'frame', 'x', 'y', 'z', 'Quality']
    assert all(col in tracks_df.columns for col in required_columns)
    print("✓ Track format conversion works")
    
    results_df = simulator.run_multi_particle_simulation([5, 10, 20], 0.5, 100, 2)
    assert len(results_df) == 6
    assert 'particle_diameter' in results_df.columns
    assert 'final_displacement_sq' in results_df.columns
    print("✓ Multi-particle simulation works")
    
    msd_result = simulator.calculate_msd()
    assert 'lag_time' in msd_result
    assert 'msd' in msd_result
    assert len(msd_result['lag_time']) > 0
    print("✓ MSD calculation with proper scaling works")
    
    region_stats = simulator.get_region_occupancy_stats()
    assert isinstance(region_stats, dict)
    print("✓ Region occupancy statistics work")
    
    fig = simulator.plot_trajectory(mode='3d')
    assert fig is not None
    print("✓ Trajectory plotting works")
    
    print("All enhanced diffusion simulator tests passed!")

if __name__ == "__main__":
    test_enhanced_diffusion_simulator()
