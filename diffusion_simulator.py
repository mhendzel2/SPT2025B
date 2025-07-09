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
        self.all_trajectories: List[np.ndarray] = []
    
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
    
    def _find_valid_starting_position(self, container_dims: Tuple[int, int, int], 
                                    boundary_map: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Find a valid starting position that's not in a boundary."""
        max_tries = 1000
        for _ in range(max_tries):
            start_pos = np.random.rand(3) * container_dims
            
            if boundary_map is None:
                return start_pos
            
            px, py, pz = int(start_pos[0]) % container_dims[0], int(start_pos[1]) % container_dims[1], int(start_pos[2]) % container_dims[2]
            
            if boundary_map[px, py, pz] == 0:
                return start_pos
        
        return None
    
    def _is_valid_position(self, pos: np.ndarray, container_dims: Tuple[int, int, int], 
                          boundary_map: Optional[np.ndarray] = None) -> bool:
        """Check if position is valid (within bounds and not in boundary)."""
        px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
        
        if not (0 <= px < container_dims[0] and 0 <= py < container_dims[1] and 0 <= pz < container_dims[2]):
            return False
        
        if boundary_map is not None and boundary_map[px, py, pz] == 1:
            return False
        
        return True
    
    def _check_partition_crossing(self, current_region: int, new_region: int) -> bool:
        """Check if particle can cross from current region to new region based on partition coefficients."""
        if current_region == new_region:
            return True
        
        # Check for specific region-to-region partition coefficient
        partition_key = f"{current_region}_to_{new_region}"
        if partition_key in self.partition_coefficients:
            partition_prob = self.partition_coefficients[partition_key]
            return np.random.rand() < partition_prob
        
        # Check for general region partition coefficient
        if new_region in self.partition_coefficients:
            partition_prob = self.partition_coefficients[new_region]
            return np.random.rand() < partition_prob
        
        return True
    
    def run_single_simulation(self, particle_diameter: float, mobility: float, 
                            num_steps: int, boundary_map: Optional[np.ndarray] = None) -> float:
        """Run single particle diffusion simulation with partition coefficient support."""
        if boundary_map is None:
            boundary_map = self.boundary_map
        
        if boundary_map is None:
            container_dims = (100, 100, 100)
        else:
            container_dims = boundary_map.shape
        
        # Find valid starting position
        start_pos = self._find_valid_starting_position(container_dims, boundary_map)
        if start_pos is None:
            return np.nan
        
        current_pos = np.copy(start_pos)
        trajectory = [current_pos.copy()]
        current_region = None
        
        # Determine starting region
        if self.region_map is not None:
            px, py, pz = int(start_pos[0]) % container_dims[0], int(start_pos[1]) % container_dims[1], int(start_pos[2]) % container_dims[2]
            if (0 <= px < self.region_map.shape[0] and 
                0 <= py < self.region_map.shape[1] and 
                0 <= pz < self.region_map.shape[2]):
                current_region = self.region_map[px, py, pz]
        
        # Simulation loop
        for step in range(num_steps):
            # Generate random step with size dependent on mobility
            random_direction = np.random.normal(0, 1, 3)
            random_direction = random_direction / np.linalg.norm(random_direction)
            step_size = np.random.exponential(mobility)
            random_step = random_direction * step_size
            
            proposed_pos = current_pos + random_step
            
            # Check if proposed position is valid
            if not self._is_valid_position(proposed_pos, container_dims, boundary_map):
                continue
            
            # Check region crossing if region map exists
            if self.region_map is not None:
                px, py, pz = int(proposed_pos[0]), int(proposed_pos[1]), int(proposed_pos[2])
                if (0 <= px < self.region_map.shape[0] and 
                    0 <= py < self.region_map.shape[1] and 
                    0 <= pz < self.region_map.shape[2]):
                    
                    new_region = self.region_map[px, py, pz]
                    
                    if current_region is not None and new_region != current_region:
                        if not self._check_partition_crossing(current_region, new_region):
                            continue
                    
                    current_region = new_region
            
            # Accept the move
            current_pos = proposed_pos
            trajectory.append(current_pos.copy())
        
        self.trajectory = np.array(trajectory)
        
        # Calculate final displacement squared
        final_displacement_sq = np.sum((current_pos - start_pos)**2)
        return final_displacement_sq
    
    def run_multi_particle_simulation(self, particle_diameters: List[float], 
                                    mobility: float, num_steps: int, 
                                    num_particles_per_size: int = 10) -> pd.DataFrame:
        """Run simulation for multiple particle sizes."""
        results = []
        self.all_trajectories = []
        
        for diameter in particle_diameters:
            for particle_idx in range(num_particles_per_size):
                final_displacement_sq = self.run_single_simulation(diameter, mobility, num_steps)
                
                # Calculate diffusion coefficient using Einstein relation
                # D = <r²>/(6*t) for 3D diffusion
                time_interval = num_steps * 1.0  # Assuming unit time steps
                diffusion_coefficient = final_displacement_sq / (6 * time_interval) if not np.isnan(final_displacement_sq) else np.nan
                
                results.append({
                    'particle_diameter': diameter,
                    'particle_id': f"{diameter}nm_{particle_idx}",
                    'final_displacement_sq': final_displacement_sq,
                    'diffusion_coefficient': diffusion_coefficient,
                    'mobility': mobility,
                    'num_steps': num_steps
                })
                
                if self.trajectory is not None:
                    self.all_trajectories.append(self.trajectory.copy())
        
        self.simulation_results = pd.DataFrame(results)
        
        # Store simulation parameters
        self.simulation_params = {
            'particle_diameters': particle_diameters,
            'mobility': mobility,
            'num_steps': num_steps,
            'num_particles_per_size': num_particles_per_size,
            'pixel_size_nm': self.pixel_size_nm,
            'optical_slice_thickness_nm': self.optical_slice_thickness_nm
        }
        
        return self.simulation_results
    
    def convert_to_tracks_format(self, trajectory_idx: int = 0) -> pd.DataFrame:
        """Convert simulation trajectory to SPT tracks format."""
        if not self.all_trajectories or trajectory_idx >= len(self.all_trajectories):
            if self.trajectory is None:
                raise ValueError("No trajectory data available")
            trajectory = self.trajectory
        else:
            trajectory = self.all_trajectories[trajectory_idx]
        
        data = []
        for frame_idx, pos in enumerate(trajectory):
            data.append({
                'track_id': trajectory_idx,
                'frame': frame_idx,
                'x': pos[0] * self.pixel_size_nm,  # Convert to nm
                'y': pos[1] * self.pixel_size_nm,
                'z': pos[2] * self.pixel_size_nm,
                'Quality': 1.0,
                'particle_id': trajectory_idx
            })
        
        return pd.DataFrame(data)
    
    def convert_all_to_tracks_format(self) -> pd.DataFrame:
        """Convert all simulation trajectories to SPT tracks format."""
        if not self.all_trajectories:
            raise ValueError("No trajectory data available")
        
        all_data = []
        for traj_idx, trajectory in enumerate(self.all_trajectories):
            for frame_idx, pos in enumerate(trajectory):
                all_data.append({
                    'track_id': traj_idx,
                    'frame': frame_idx,
                    'x': pos[0] * self.pixel_size_nm,
                    'y': pos[1] * self.pixel_size_nm,
                    'z': pos[2] * self.pixel_size_nm,
                    'Quality': 1.0,
                    'particle_id': traj_idx
                })
        
        return pd.DataFrame(all_data)
    
    def calculate_msd(self, trajectory_idx: Optional[int] = None, max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Calculate MSD from simulation trajectory with proper scaling."""
        if trajectory_idx is not None and trajectory_idx < len(self.all_trajectories):
            trajectory = self.all_trajectories[trajectory_idx]
        elif self.trajectory is not None:
            trajectory = self.trajectory
        else:
            raise ValueError("No trajectory data available")
        
        n_frames = len(trajectory)
        if max_lag is None:
            max_lag = n_frames // 2
        else:
            max_lag = min(max_lag, n_frames - 1)
        
        lag_times = np.arange(1, max_lag + 1)
        msd_values = []
        
        for lag in lag_times:
            displacements = trajectory[lag:] - trajectory[:-lag]
            squared_displacements = np.sum(displacements**2, axis=1) * (self.pixel_size_nm ** 2)
            msd_values.append(np.mean(squared_displacements))
        
        return {
            'lag_time': lag_times,
            'msd': np.array(msd_values)
        }
    
    def calculate_ensemble_msd(self, max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Calculate ensemble MSD from all trajectories."""
        if not self.all_trajectories:
            raise ValueError("No trajectory data available")
        
        # Find minimum trajectory length
        min_length = min(len(traj) for traj in self.all_trajectories)
        
        if max_lag is None:
            max_lag = min_length // 2
        else:
            max_lag = min(max_lag, min_length - 1)
        
        lag_times = np.arange(1, max_lag + 1)
        ensemble_msd = []
        
        for lag in lag_times:
            all_msd_values = []
            for trajectory in self.all_trajectories:
                if len(trajectory) > lag:
                    displacements = trajectory[lag:] - trajectory[:-lag]
                    squared_displacements = np.sum(displacements**2, axis=1) * (self.pixel_size_nm ** 2)
                    all_msd_values.extend(squared_displacements)
            
            ensemble_msd.append(np.mean(all_msd_values))
        
        return {
            'lag_time': lag_times,
            'msd': np.array(ensemble_msd)
        }
    
    def get_region_occupancy_stats(self, trajectory_idx: Optional[int] = None) -> Dict[str, Any]:
        """Calculate time spent in each region during simulation."""
        if trajectory_idx is not None and trajectory_idx < len(self.all_trajectories):
            trajectory = self.all_trajectories[trajectory_idx]
        elif self.trajectory is not None:
            trajectory = self.trajectory
        else:
            return {}
        
        if self.region_map is None:
            return {}
        
        region_times = {}
        total_steps = len(trajectory)
        
        for i, pos in enumerate(trajectory):
            px, py, pz = int(pos[0]), int(pos[1]), int(pos[2])
            if (0 <= px < self.region_map.shape[0] and 
                0 <= py < self.region_map.shape[1] and 
                0 <= pz < self.region_map.shape[2]):
                
                region_id = self.region_map[px, py, pz]
                region_name = self.region_names.get(region_id, f"Region_{region_id}")
                
                if region_name not in region_times:
                    region_times[region_name] = 0
                region_times[region_name] += 1
        
        # Convert to percentages
        for region_name in region_times:
            region_times[region_name] = region_times[region_name] / total_steps * 100
        
        return region_times
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the simulation."""
        if self.simulation_results is None:
            return {}
        
        summary = {
            'total_particles': len(self.simulation_results),
            'particle_diameters': self.simulation_results['particle_diameter'].unique().tolist(),
            'mean_diffusion_coefficient': self.simulation_results['diffusion_coefficient'].mean(),
            'std_diffusion_coefficient': self.simulation_results['diffusion_coefficient'].std(),
            'successful_simulations': self.simulation_results['diffusion_coefficient'].notna().sum(),
            'failed_simulations': self.simulation_results['diffusion_coefficient'].isna().sum(),
            'simulation_parameters': self.simulation_params
        }
        
        return summary
    
    def plot_trajectory(self, trajectory_idx: Optional[int] = None, mode: str = '3d') -> go.Figure:
        """Plot simulation trajectory."""
        if trajectory_idx is not None and trajectory_idx < len(self.all_trajectories):
            trajectory = self.all_trajectories[trajectory_idx]
        elif self.trajectory is not None:
            trajectory = self.trajectory
        else:
            raise ValueError("No trajectory data available")
        
        # Convert to nm for plotting
        trajectory_nm = trajectory * self.pixel_size_nm
        
        if mode == '3d':
            fig = go.Figure()
            
            # Plot trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory_nm[:, 0],
                y=trajectory_nm[:, 1],
                z=trajectory_nm[:, 2],
                mode='lines+markers',
                marker=dict(size=3, color='blue'),
                line=dict(width=2, color='blue'),
                name='Particle Trajectory'
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter3d(
                x=[trajectory_nm[0, 0]],
                y=[trajectory_nm[0, 1]],
                z=[trajectory_nm[0, 2]],
                mode='markers',
                marker=dict(size=8, color='green', symbol='circle'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter3d(
                x=[trajectory_nm[-1, 0]],
                y=[trajectory_nm[-1, 1]],
                z=[trajectory_nm[-1, 2]],
                mode='markers',
                marker=dict(size=8, color='red', symbol='circle'),
                name='End'
            ))
            
            fig.update_layout(
                title="Simulated Particle Trajectory (3D)",
                scene=dict(
                    xaxis_title="X Position (nm)",
                    yaxis_title="Y Position (nm)",
                    zaxis_title="Z Position (nm)"
                ),
                width=800,
                height=600
            )
        else:
            fig = go.Figure()
            
            # Plot trajectory
            fig.add_trace(go.Scatter(
                x=trajectory_nm[:, 0],
                y=trajectory_nm[:, 1],
                mode='lines+markers',
                marker=dict(size=3, color='blue'),
                line=dict(width=2, color='blue'),
                name='Particle Trajectory'
            ))
            
            # Mark start and end points
            fig.add_trace(go.Scatter(
                x=[trajectory_nm[0, 0]],
                y=[trajectory_nm[0, 1]],
                mode='markers',
                marker=dict(size=10, color='green', symbol='circle'),
                name='Start'
            ))
            
            fig.add_trace(go.Scatter(
                x=[trajectory_nm[-1, 0]],
                y=[trajectory_nm[-1, 1]],
                mode='markers',
                marker=dict(size=10, color='red', symbol='circle'),
                name='End'
            ))
            
            fig.update_layout(
                title="Simulated Particle Trajectory (XY Plane)",
                xaxis_title="X Position (nm)",
                yaxis_title="Y Position (nm)",
                width=800,
                height=600
            )
        
        return fig
    
    def plot_msd_analysis(self, trajectory_idx: Optional[int] = None) -> go.Figure:
        """Plot MSD analysis with fitting."""
        try:
            if trajectory_idx is not None:
                msd_data = self.calculate_msd(trajectory_idx)
            else:
                msd_data = self.calculate_ensemble_msd()
            
            fig = go.Figure()
            
            # Plot MSD data
            fig.add_trace(go.Scatter(
                x=msd_data['lag_time'],
                y=msd_data['msd'],
                mode='markers+lines',
                name='MSD Data',
                marker=dict(size=6, color='blue')
            ))
            
            # Fit linear regression for diffusion coefficient
            if len(msd_data['lag_time']) > 1:
                slope, intercept = np.polyfit(msd_data['lag_time'], msd_data['msd'], 1)
                fitted_line = slope * msd_data['lag_time'] + intercept
                
                fig.add_trace(go.Scatter(
                    x=msd_data['lag_time'],
                    y=fitted_line,
                    mode='lines',
                    name=f'Linear Fit (D = {slope/6:.3e} nm²/step)',
                    line=dict(dash='dash', color='red')
                ))
            
            fig.update_layout(
                title="Mean Square Displacement Analysis",
                xaxis_title="Lag Time (steps)",
                yaxis_title="MSD (nm²)",
                width=800,
                height=600
            )
            
            return fig
            
        except Exception as e:
            # Return empty figure if MSD calculation fails
            fig = go.Figure()
            fig.update_layout(
                title=f"MSD Analysis Error: {str(e)}",
                xaxis_title="Lag Time (steps)",
                yaxis_title="MSD (nm²)"
            )
            return fig
    
    def plot_diffusion_coefficients(self) -> go.Figure:
        """Plot diffusion coefficients by particle size."""
        if self.simulation_results is None:
            raise ValueError("No simulation results available")
        
        fig = go.Figure()
        
        # Box plot of diffusion coefficients by particle diameter
        for diameter in self.simulation_results['particle_diameter'].unique():
            data = self.simulation_results[
                self.simulation_results['particle_diameter'] == diameter
            ]['diffusion_coefficient'].dropna()
            
            fig.add_trace(go.Box(
                y=data,
                name=f'{diameter} nm',
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title="Diffusion Coefficients by Particle Size",
            xaxis_title="Particle Diameter",
            yaxis_title="Diffusion Coefficient (nm²/step)",
            width=800,
            height=600
        )
        
        return fig
    
    def plot_region_occupancy(self, trajectory_idx: Optional[int] = None) -> go.Figure:
        """Plot region occupancy statistics."""
        region_stats = self.get_region_occupancy_stats(trajectory_idx)
        
        if not region_stats:
            fig = go.Figure()
            fig.update_layout(title="No region data available")
            return fig
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=list(region_stats.keys()),
            y=list(region_stats.values()),
            name='Region Occupancy'
        ))
        
        fig.update_layout(
            title="Region Occupancy Statistics",
            xaxis_title="Region",
            yaxis_title="Time Spent (%)",
            width=800,
            height=600
        )
        
        return fig
