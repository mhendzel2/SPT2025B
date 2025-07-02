"""
Diffusion simulator with boundary constraints for SPT Analysis.
Integrates with nuclear segmentation masks and MD simulation framework.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, List, Tuple, Any, Optional, Union
import plotly.graph_objects as go
import plotly.express as px

class DiffusionSimulator:
    """
    Diffusion simulator with boundary constraints using nuclear masks.
    Supports 1nm scaling, optical slice imaging, and multiple regions with partition coefficients.
    """
    
    def __init__(self):
        self.boundary_map: Optional[np.ndarray] = None
        self.region_map: Optional[np.ndarray] = None
        self.partition_coefficients: Dict[int, float] = {}
        self.region_names: Dict[int, str] = {}
        self.simulation_results: Optional[pd.DataFrame] = None
        self.trajectory: Optional[np.ndarray] = None
        self.simulation_params: Dict[str, Any] = {}
        self.pixel_size_nm: float = 1.0  # 1nm scaling as requested
        self.optical_slice_thickness_nm: float = 200.0  # Default 200nm thickness
    
    def set_optical_slice_thickness(self, thickness_nm: float) -> None:
        """Set optical slice thickness in nanometers (100-700nm in 50nm steps)."""
        valid_thicknesses = list(range(100, 751, 50))
        if thickness_nm not in valid_thicknesses:
            thickness_nm = min(valid_thicknesses, key=lambda x: abs(x - thickness_nm))
        self.optical_slice_thickness_nm = float(thickness_nm)
    
    def load_nuclear_mask_as_boundary(self, mask_name: str, simulate_optical_slice: bool = False) -> bool:
        """Load nuclear segmentation mask as boundary constraint with optional optical slice simulation."""
        available_masks = st.session_state.get('available_masks', {})
        if mask_name not in available_masks:
            return False
        
        mask = available_masks[mask_name]
        
        if simulate_optical_slice:
            z_depth = int(self.optical_slice_thickness_nm / self.pixel_size_nm)
        else:
            z_depth = max(10, int(self.optical_slice_thickness_nm / self.pixel_size_nm))
        
        if mask.ndim == 2:
            self.boundary_map = np.zeros((mask.shape[0], mask.shape[1], z_depth), dtype=np.uint8)
            self.region_map = np.zeros((mask.shape[0], mask.shape[1], z_depth), dtype=np.uint8)
            
            for z in range(z_depth):
                self.boundary_map[:, :, z] = (mask == 0).astype(np.uint8)
                self.region_map[:, :, z] = mask.astype(np.uint8)
        else:
            self.boundary_map = (mask == 0).astype(np.uint8)
            self.region_map = mask.astype(np.uint8)
        
        unique_regions = np.unique(self.region_map)
        for region_id in unique_regions:
            if region_id not in self.partition_coefficients:
                self.partition_coefficients[region_id] = 1.0
            if region_id not in self.region_names:
                self.region_names[region_id] = f"Region_{region_id}"
        
        return True
    
    def set_region_properties(self, region_id: int, name: str, partition_coefficient: float) -> None:
        """Set properties for a specific region including partition coefficient."""
        self.region_names[region_id] = name
        self.partition_coefficients[region_id] = partition_coefficient
    
    def add_liquid_liquid_boundary(self, region1_id: int, region2_id: int, 
                                 partition_coeff_1_to_2: float, partition_coeff_2_to_1: float) -> None:
        """Add liquid-liquid phase boundary with bidirectional partition coefficients."""
        self.partition_coefficients[f"{region1_id}_to_{region2_id}"] = partition_coeff_1_to_2
        self.partition_coefficients[f"{region2_id}_to_{region1_id}"] = partition_coeff_2_to_1
    
    def generate_crowding_map(self, dimensions: Tuple[int, int, int], crowding_percentage: float) -> np.ndarray:
        """Generate random crowding barriers."""
        if not (0 < crowding_percentage < 100):
            return np.zeros(dimensions, dtype=np.uint8)
        
        total_voxels = np.prod(dimensions)
        num_barriers = int(total_voxels * (crowding_percentage / 100.0))
        
        boundary_map = np.zeros(total_voxels, dtype=np.uint8)
        boundary_map[:num_barriers] = 1
        np.random.shuffle(boundary_map)
        
        return boundary_map.reshape(dimensions)
    
    def generate_gel_map(self, dimensions: Tuple[int, int, int], pore_size: int) -> np.ndarray:
        """Generate hydrogel grid barriers."""
        boundary_map = np.zeros(dimensions, dtype=np.uint8)
        pore_size = max(1, pore_size)
        
        for i in range(0, dimensions[0], pore_size):
            boundary_map[i:i+1, :, :] = 1
        for j in range(0, dimensions[1], pore_size):
            boundary_map[:, j:j+1, :] = 1
        for k in range(0, dimensions[2], pore_size):
            boundary_map[:, :, k:k+1] = 1
            
        return boundary_map
    
    def run_single_simulation(self, particle_diameter: float, mobility: float, 
                            num_steps: int, boundary_map: Optional[np.ndarray] = None) -> float:
        """Run single particle diffusion simulation with partition coefficient support."""
        if boundary_map is None:
            boundary_map = self.boundary_map
        
        if boundary_map is None:
            container_dims = (100, 100, 100)
        else:
            container_dims = boundary_map.shape
        
        start_pos = np.random.rand(3) * container_dims
        if boundary_map is not None:
            max_tries, tries = 1000, 0
            while (boundary_map[int(start_pos[0]) % container_dims[0],
                             int(start_pos[1]) % container_dims[1],
                             int(start_pos[2]) % container_dims[2]] == 1):
                start_pos = np.random.rand(3) * container_dims
                tries += 1
                if tries > max_tries:
                    return np.nan
        
        current_pos = np.copy(start_pos)
        trajectory = [current_pos.copy()]
        current_region = None
        
        if self.region_map is not None:
            px, py, pz = int(start_pos[0]) % container_dims[0], int(start_pos[1]) % container_dims[1], int(start_pos[2]) % container_dims[2]
            if (0 <= px < self.region_map.shape[0] and 
                0 <= py < self.region_map.shape[1] and 
                0 <= pz < self.region_map.shape[2]):
                current_region = self.region_map[px, py, pz]
        
        for _ in range(num_steps):
            random_step = (np.random.rand(3) - 0.5) * 2
            random_step = random_step / np.linalg.norm(random_step) * mobility
            proposed_pos = current_pos + random_step
            
            px_int, py_int, pz_int = int(proposed_pos[0]), int(proposed_pos[1]), int(proposed_pos[2])
            if not (0 <= px_int < container_dims[0] and 
                   0 <= py_int < container_dims[1] and 
                   0 <= pz_int < container_dims[2]):
                continue
            
            if boundary_map is not None and boundary_map[px_int, py_int, pz_int] == 1:
                continue
            
            if self.region_map is not None:
                if (0 <= px_int < self.region_map.shape[0] and 
                    0 <= py_int < self.region_map.shape[1] and 
                    0 <= pz_int < self.region_map.shape[2]):
                    
                    new_region = self.region_map[px_int, py_int, pz_int]
                    
                    if current_region is not None and new_region != current_region:
                        partition_key = f"{current_region}_to_{new_region}"
                        if partition_key in self.partition_coefficients:
                            partition_prob = self.partition_coefficients[partition_key]
                            if np.random.rand() > partition_prob:
                                continue
                        elif new_region in self.partition_coefficients:
                            partition_prob = self.partition_coefficients[new_region]
                            if np.random.rand() > partition_prob:
                                continue
                    
                    current_region = new_region
            
            current_pos = proposed_pos
            trajectory.append(current_pos.copy())
        
        self.trajectory = np.array(trajectory)
        
        return np.sum((current_pos - start_pos)**2)
    
    def run_multi_particle_simulation(self, particle_diameters: List[float], 
                                    mobility: float, num_steps: int, 
                                    num_particles_per_size: int = 10) -> pd.DataFrame:
        """Run simulation for multiple particle sizes."""
        results = []
        all_trajectories = []
        
        for diameter in particle_diameters:
            for particle_idx in range(num_particles_per_size):
                final_displacement = self.run_single_simulation(diameter, mobility, num_steps)
                
                results.append({
                    'particle_diameter': diameter,
                    'particle_id': f"{diameter}nm_{particle_idx}",
                    'final_displacement_sq': final_displacement,
                    'diffusion_coefficient': final_displacement / (6 * num_steps) if not np.isnan(final_displacement) else np.nan
                })
                
                if self.trajectory is not None:
                    all_trajectories.append(self.trajectory.copy())
        
        self.simulation_results = pd.DataFrame(results)
        return self.simulation_results
    
    def convert_to_tracks_format(self, max_particles: int = 10) -> pd.DataFrame:
        """Convert simulation trajectory to SPT tracks format."""
        if self.trajectory is None:
            raise ValueError("No trajectory data available")
        
        data = []
        for frame_idx, pos in enumerate(self.trajectory):
            data.append({
                'track_id': 0,
                'frame': frame_idx,
                'x': pos[0],
                'y': pos[1],
                'z': pos[2],
                'Quality': 1.0
            })
        
        return pd.DataFrame(data)
    
    def calculate_msd(self, max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Calculate MSD from simulation trajectory with proper scaling."""
        if self.trajectory is None:
            raise ValueError("No trajectory data available")
        
        n_frames = len(self.trajectory)
        if max_lag is None:
            max_lag = n_frames // 2
        else:
            max_lag = min(max_lag, n_frames - 1)
        
        lag_times = range(1, max_lag + 1)
        msd_values = []
        
        for lag in lag_times:
            displacements = self.trajectory[lag:] - self.trajectory[:-lag]
            squared_displacements = np.sum(displacements**2, axis=1) * (self.pixel_size_nm ** 2)
            msd_values.append(np.mean(squared_displacements))
        
        return {
            'lag_time': np.array(lag_times),
            'msd': np.array(msd_values)
        }
    
    def get_region_occupancy_stats(self) -> Dict[str, Any]:
        """Calculate time spent in each region during simulation."""
        if self.trajectory is None or self.region_map is None:
            return {}
        
        region_times = {}
        total_steps = len(self.trajectory)
        
        for i, pos in enumerate(self.trajectory):
            px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
            if (0 <= px < self.region_map.shape[0] and 
                0 <= py < self.region_map.shape[1] and 
                0 <= pz < self.region_map.shape[2]):
                
                region_id = self.region_map[px, py, pz]
                region_name = self.region_names.get(region_id, f"Region_{region_id}")
                
                if region_name not in region_times:
                    region_times[region_name] = 0
                region_times[region_name] += 1
        
        for region_name in region_times:
            region_times[region_name] = region_times[region_name] / total_steps
        
        return region_times
    
    def plot_trajectory(self, mode: str = '3d') -> go.Figure:
        """Plot simulation trajectory."""
        if self.trajectory is None:
            raise ValueError("No trajectory data available")
        
        if mode == '3d':
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=self.trajectory[:, 0],
                y=self.trajectory[:, 1],
                z=self.trajectory[:, 2],
                mode='lines+markers',
                marker=dict(size=3),
                line=dict(width=2),
                name='Simulated Particle'
            ))
            fig.update_layout(
                title="Simulated Particle Trajectory",
                scene=dict(
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    zaxis_title="Z Position"
                )
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=self.trajectory[:, 0],
                y=self.trajectory[:, 1],
                mode='lines+markers',
                marker=dict(size=5),
                line=dict(width=2),
                name='Simulated Particle'
            ))
            fig.update_layout(
                title="Simulated Particle Trajectory (XY Plane)",
                xaxis_title="X Position",
                yaxis_title="Y Position"
            )
        
        return fig
