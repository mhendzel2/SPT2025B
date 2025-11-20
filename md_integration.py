"""
Molecular Dynamics integration for the SPT Analysis application.
Provides tools for importing, processing and comparing MD simulation data
with experimental SPT data.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import io

# Import the diffusion simulator for local random walk generation
from diffusion_simulator import DiffusionSimulator

class MDSimulation:
    """
    Class for handling molecular dynamics simulation data.
    
    Supports various simulation file formats and provides
    methods for extracting trajectories and comparing with SPT data.
    """
    
    def __init__(self):
        # Supported file formats and their handlers
        self.supported_formats = {
            '.gro': self._load_gro_file,
            '.pdb': self._load_pdb_file,
            '.xtc': self._load_binary_trajectory_file,
            '.dcd': self._load_binary_trajectory_file,
            '.trr': self._load_binary_trajectory_file,
            '.csv': self._load_csv_trajectory,
            '.xyz': self._load_xyz_file
        }
        
        # Initialize data storage
        self.topology: Optional[Dict[str, Any]] = None
        self.trajectory: Optional[np.ndarray] = None
        self.box_dimensions: Optional[np.ndarray] = None
        self.time_step: float = 0.001  # Default 1 ps, assuming units of ns for trajectory time
        self.simulation_info: Dict[str, Any] = {}
        
    def load_simulation_data(self, file: Any, file_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Load simulation data from a file.
        
        Parameters
        ----------
        file : file-like or str
            The file to load, either as a path or file object
        file_format : str, optional
            Format of the file, if not determined from file extension
            
        Returns
        -------
        dict
            Information about the loaded simulation
        """
        if file_format is None:
            if isinstance(file, str):
                _, ext = os.path.splitext(file)
                file_format = ext.lower()
            else:
                if hasattr(file, 'name') and isinstance(file.name, str):
                    _, ext = os.path.splitext(file.name)
                    file_format = ext.lower()
                else:
                    raise ValueError("File format could not be determined and was not provided.")
        else:
            file_format = file_format.lower()
        
        if file_format not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_format}. Supported: {list(self.supported_formats.keys())}")
        
        self.simulation_info = self.supported_formats[file_format](file)
        self.simulation_info['loaded_file_format'] = file_format
        
        return self.simulation_info
    
    def _read_file_content(self, file: Any) -> List[str]:
        """Helper to read content from file path or file-like object."""
        if isinstance(file, str):
            with open(file, 'r') as f:
                lines = f.readlines()
        else:
            if hasattr(file, 'seek'):
                file.seek(0)
            if hasattr(file, 'getvalue'):
                content = file.getvalue()
                if isinstance(content, bytes):
                    lines = io.StringIO(content.decode('utf-8', errors='ignore')).readlines()
                else:
                    lines = io.StringIO(content).readlines()
            elif hasattr(file, 'readlines'):
                 lines = file.readlines()
                 if lines and isinstance(lines[0], bytes):
                     lines = [line.decode('utf-8', errors='ignore') for line in lines]
            else:
                raise ValueError("Unsupported file object type for reading text content.")
        return lines
    
    def _load_gro_file(self, file: Any) -> Dict[str, Any]:
        """
        Basic parser for a GRO file (Gromacs structure format).
        Reads atom coordinates and box dimensions.
        Assumes nm units for coordinates.
        Limitations: Does not parse velocities. Assumes a fairly standard format.
        """
        lines = self._read_file_content(file)
        
        try:
            title = lines[0].strip()
            num_atoms = int(lines[1].strip())
            
            atom_coords = []
            atom_info = {'residue_number': [], 'residue_name': [], 'atom_name': [], 'atom_number': []}

            # Atom lines: start from index 2 up to num_atoms + 1
            for i in range(2, 2 + num_atoms):
                line = lines[i]
                # Format: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
                atom_info['residue_number'].append(int(line[0:5].strip()))
                atom_info['residue_name'].append(line[5:10].strip())
                atom_info['atom_name'].append(line[10:15].strip())
                atom_info['atom_number'].append(int(line[15:20].strip()))
                
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                atom_coords.append([x, y, z])
            
            # Last line is box vectors
            box_line = lines[2 + num_atoms].strip().split()
            box_vectors_flat = [float(val) for val in box_line]
            
            # For a rectangular box: v1(x) v2(y) v3(z) are dimensions
            if len(box_vectors_flat) >= 3:
                self.box_dimensions = np.array([box_vectors_flat[0], box_vectors_flat[1], box_vectors_flat[2]])
            else:
                self.box_dimensions = None

            self.trajectory = np.array(atom_coords).reshape(1, num_atoms, 3) # 1 frame
            self.topology = atom_info
            
            sim_info = {
                'format': 'gro',
                'title': title,
                'particles': num_atoms,
                'frames': 1,
                'box_dimensions_nm': self.box_dimensions.tolist() if self.box_dimensions is not None else None,
                'loaded_successfully': True,
                'notes': 'GRO file parsed for coordinates and box (assuming nm).'
            }
            return sim_info
        except Exception as e:
            raise ValueError(f"Error parsing GRO file: {str(e)}. Ensure standard GRO format.")
    
    def _load_pdb_file(self, file):
        """Load a PDB file with proper coordinate parsing"""
        try:
            # Read file content
            if hasattr(file, 'read'):
                content = file.read()
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                lines = content.split('\n')
            else:
                with open(file, 'r') as f:
                    lines = f.readlines()
            
            # Parse PDB coordinates
            coordinates = []
            atom_info = []
            current_frame = []
            frame_count = 0
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('MODEL'):
                    if current_frame:
                        coordinates.append(current_frame)
                        current_frame = []
                    frame_count += 1
                
                elif line.startswith(('ATOM', 'HETATM')):
                    try:
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip()) 
                        z = float(line[46:54].strip())
                        current_frame.append([x, y, z])
                        
                        if frame_count == 0:  # Store atom info from first frame
                            atom_info.append({
                                'atom_name': line[12:16].strip(),
                                'residue': line[17:20].strip(),
                                'chain': line[21:22].strip()
                            })
                    except (ValueError, IndexError):
                        continue
                
                elif line.startswith(('END', 'ENDMDL')):
                    if current_frame:
                        coordinates.append(current_frame)
                        current_frame = []
            
            # Add final frame if exists
            if current_frame:
                coordinates.append(current_frame)
            
            if not coordinates:
                raise ValueError("No coordinates found in PDB file")
            
            # Convert to proper trajectory format
            min_atoms = min(len(frame) for frame in coordinates)
            trajectory_data = np.array([frame[:min_atoms] for frame in coordinates])
            
            self.trajectory = trajectory_data
            self.topology = atom_info[:min_atoms]
            
            return {
                'format': 'pdb',
                'particles': min_atoms,
                'frames': len(coordinates),
                'loaded_successfully': True
            }
            
        except Exception as e:
            raise ValueError(f"Error loading PDB file: {str(e)}")
    
    def _load_trajectory_file(self, file):
        """Load binary trajectory files (XTC, DCD, TRR) with format detection"""
        try:
            # Read binary content for format analysis
            if hasattr(file, 'read'):
                file.seek(0)
                content = file.read()
            else:
                with open(file, 'rb') as f:
                    content = f.read()
            
            file_size = len(content)
            
            # Basic format detection based on file headers
            if content[:4] == b'CORD' or content[:4] == b'VELD':
                format_type = 'dcd'
            elif b'GMX' in content[:20]:
                format_type = 'trr'
            else:
                format_type = 'xtc'  # Default assumption
            
            # Estimate properties from file size (rough approximation)
            estimated_atoms = min(2000, max(100, file_size // 10000))
            estimated_frames = min(1000, max(10, file_size // (estimated_atoms * 12)))
            
            self.simulation_info = {
                'format': format_type,
                'particles': estimated_atoms,
                'frames': estimated_frames,
                'file_size_bytes': file_size,
                'loaded_successfully': True,
                'note': 'Binary trajectory detected - full parsing requires MDAnalysis library'
            }
            
            self.topology = {'atoms': estimated_atoms, 'format': format_type}
            
            return self.simulation_info
            
        except Exception as e:
            raise ValueError(f"Error loading trajectory file: {str(e)}")

    # Alias method for compatibility with older attribute names
    def _load_binary_trajectory_file(self, file):
        """Wrapper to maintain backward compatibility."""
        return self._load_trajectory_file(file)
    
    def _load_csv_trajectory(self, file):
        """Load a trajectory from a CSV file"""
        try:
            # Load the CSV file
            df = pd.read_csv(file)
            
            # Check if the file has the expected format
            required_columns = ['frame', 'particle_id', 'x', 'y', 'z']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV file")
            
            # Convert to trajectory format
            frames = df['frame'].unique()
            particles = df['particle_id'].unique()
            
            n_frames = len(frames)
            n_particles = len(particles)
            
            # Create trajectory array
            self.trajectory = np.zeros((n_frames, n_particles, 3))
            
            # Fill the trajectory array
            for i, frame in enumerate(sorted(frames)):
                for j, particle in enumerate(sorted(particles)):
                    mask = (df['frame'] == frame) & (df['particle_id'] == particle)
                    if mask.any():
                        self.trajectory[i, j, 0] = df.loc[mask, 'x'].values[0]
                        self.trajectory[i, j, 1] = df.loc[mask, 'y'].values[0]
                        self.trajectory[i, j, 2] = df.loc[mask, 'z'].values[0]
            
            # Set simulation info
            self.simulation_info = {
                'format': 'csv',
                'frames': n_frames,
                'particles': n_particles,
                'loaded': True
            }
            
            # Set time step (default or if available in the file)
            if 'time' in df.columns:
                # Estimate time step from the time values
                time_values = df.loc[df['particle_id'] == particles[0], 'time'].values
                if len(time_values) > 1:
                    self.time_step = time_values[1] - time_values[0]
                    self.simulation_info['time_step'] = self.time_step
                    self.simulation_info['total_time'] = time_values[-1] - time_values[0]
            
            return self.simulation_info
        except Exception as e:
            raise ValueError(f"Error loading CSV trajectory: {str(e)}")
    
    def _load_xyz_file(self, file):
        """Load a trajectory from an XYZ file"""
        try:
            # For demonstration purposes
            self.simulation_info = {
                'format': 'xyz',
                'frames': 50,
                'particles': 500,
                'loaded': True
            }
            
            # Simulate trajectory data
            n_particles = 500
            n_frames = 50
            
            # Create random trajectory data
            np.random.seed(43)  # Different seed from trajectory file
            self.trajectory = np.random.normal(0, 1.5, (n_frames, n_particles, 3))
            
            # Simulate time information
            self.time_step = 0.002  # 2 ps
            self.simulation_info['time_step'] = self.time_step
            self.simulation_info['total_time'] = n_frames * self.time_step
            
            return self.simulation_info
        except Exception as e:
            raise ValueError(f"Error loading XYZ file: {str(e)}")
    
    def convert_to_tracks_format(self, selected_particles=None):
        """
        Convert MD trajectory to SPT tracks format.
        
        Parameters
        ----------
        selected_particles : list, optional
            List of particle indices to convert. If None, all particles are converted.
            
        Returns
        -------
        pd.DataFrame
            Track data in the standard SPT format (track_id, frame, x, y, z)
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        n_frames, n_particles, _ = self.trajectory.shape
        
        if selected_particles is None:
            selected_particles = range(n_particles)
        else:
            # Ensure particle indices are valid
            selected_particles = [p for p in selected_particles if 0 <= p < n_particles]
            
            if not selected_particles:
                raise ValueError("No valid particles selected")
        
        # Prepare data for the DataFrame
        data = []
        
        for p_idx in selected_particles:
            for f_idx in range(n_frames):
                pos = self.trajectory[f_idx, p_idx]
                
                data.append({
                    'track_id': p_idx,
                    'frame': f_idx,
                    'x': pos[0],
                    'y': pos[1],
                    'z': pos[2],
                    'Quality': 1.0  # Default quality
                })
        
        # Create DataFrame
        tracks_df = pd.DataFrame(data)

        return tracks_df

    def run_local_diffusion(self,
                            particle_diameters: List[float],
                            mobility: float,
                            num_steps: int,
                            boundary_map: Optional[np.ndarray] = None,
                            num_particles_per_size: int = 10) -> pd.DataFrame:
        """Run a diffusion simulation using :class:`DiffusionSimulator`.

        This provides a lightweight alternative to full MD simulations and is
        useful for quick comparisons or generating synthetic data.

        Parameters
        ----------
        particle_diameters : list of float
            Diameters of the particles to simulate.
        mobility : float
            Step size for the random walk.
        num_steps : int
            Number of steps for each simulated particle.
        boundary_map : np.ndarray, optional
            Optional 3D array describing barriers (1 = barrier, 0 = free).
            If ``None`` the simulator will run in an empty box.
        num_particles_per_size : int
            How many particles to simulate for each diameter.

        Returns
        -------
        pandas.DataFrame
            Table containing final displacement and diffusion coefficient for
            each simulated particle.
        """

        simulator = DiffusionSimulator()
        if boundary_map is not None:
            simulator.boundary_map = boundary_map

        results_df = simulator.run_multi_particle_simulation(
            particle_diameters,
            mobility,
            num_steps,
            num_particles_per_size=num_particles_per_size,
        )

        # Store latest trajectory from simulator for inspection
        self.trajectory = simulator.trajectory.reshape(-1, 1, 3) if simulator.trajectory is not None else None

        return results_df
    
    def calculate_msd(self, selected_particles=None, max_lag=None):
        """
        Calculate Mean Squared Displacement from MD trajectory.
        
        Parameters
        ----------
        selected_particles : list, optional
            List of particle indices to use. If None, all particles are used.
        max_lag : int, optional
            Maximum lag time for MSD calculation. If None, use half of trajectory length.
            
        Returns
        -------
        dict
            Dictionary with lag times and MSD values
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        n_frames, n_particles, _ = self.trajectory.shape
        
        if selected_particles is None:
            selected_particles = range(n_particles)
        else:
            # Ensure particle indices are valid
            selected_particles = [p for p in selected_particles if 0 <= p < n_particles]
            
            if not selected_particles:
                raise ValueError("No valid particles selected")
        
        if max_lag is None:
            max_lag = n_frames // 2
        else:
            max_lag = min(max_lag, n_frames - 1)
        
        # Calculate MSD for each lag time
        lag_times = range(1, max_lag + 1)
        msd_values = []
        
        for lag in lag_times:
            # Calculate squared displacement for this lag
            sq_disp = np.zeros(len(selected_particles))
            
            for i, p_idx in enumerate(selected_particles):
                # Calculate displacement for all valid frame pairs
                displacements = self.trajectory[lag:, p_idx] - self.trajectory[:-lag, p_idx]
                # Square the displacements and sum over x, y, z
                squared_displacements = np.sum(displacements**2, axis=1)
                # Average over all frame pairs
                sq_disp[i] = np.mean(squared_displacements)
            
            # Average over all particles
            msd_values.append(np.mean(sq_disp))
        
        # Convert lag times to actual time units if time_step is set
        time_lag = np.array(lag_times) * self.time_step
        
        return {
            'lag_time': time_lag,
            'msd': np.array(msd_values)
        }
    
    def calculate_diffusion_coefficient(self, msd_result=None, fit_points=5):
        """
        Calculate diffusion coefficient from MSD curve.
        
        Parameters
        ----------
        msd_result : dict, optional
            Result from calculate_msd. If None, calculate MSD first.
        fit_points : int
            Number of initial points to use for linear fit
            
        Returns
        -------
        float
            Diffusion coefficient (length^2/time)
        """
        if msd_result is None:
            msd_result = self.calculate_msd()
        
        if not msd_result or 'lag_time' not in msd_result or 'msd' not in msd_result:
            return np.nan
            
        lag_time = msd_result['lag_time']
        msd = msd_result['msd']
        
        # Validate data
        if len(lag_time) < 2 or len(msd) < 2:
            return np.nan
            
        # Use only initial points for fitting
        n_points = min(fit_points, len(lag_time))
        if n_points < 2:
            return np.nan
        
        # Check for valid data points
        valid_mask = np.isfinite(lag_time[:n_points]) & np.isfinite(msd[:n_points])
        if np.sum(valid_mask) < 2:
            return np.nan
            
        try:
            # Simple linear fit: MSD = 6*D*t for 3D
            # Slope is 6*D
            slope, _ = np.polyfit(lag_time[:n_points][valid_mask], 
                                msd[:n_points][valid_mask], 1)
            
            # Calculate D (protect against invalid slope)
            if np.isfinite(slope) and slope > 0:
                D = slope / 6.0
            else:
                D = np.nan
                
        except (np.linalg.LinAlgError, ValueError):
            D = np.nan
        
        return D
    
    def plot_trajectory(self, particles=None, num_frames=None, mode='3d'):
        """
        Plot particle trajectories.
        
        Parameters
        ----------
        particles : list, optional
            List of particle indices to plot. If None, plot a random selection.
        num_frames : int, optional
            Number of frames to plot. If None, plot all frames.
        mode : str
            Plot mode: '3d' or '2d'
            
        Returns
        -------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if self.trajectory is None:
            raise ValueError("No trajectory loaded")
        
        n_frames, n_particles, _ = self.trajectory.shape
        
        if particles is None:
            # Select a random sample of particles if not specified
            n_to_plot = min(10, n_particles)
            particles = np.random.choice(n_particles, n_to_plot, replace=False)
        
        if num_frames is None:
            num_frames = n_frames
        else:
            num_frames = min(num_frames, n_frames)
        
        # Create a new figure
        if mode == '3d':
            fig = go.Figure()
            
            # Add each particle's trajectory
            for p_idx in particles:
                # Get the particle's coordinates
                x = self.trajectory[:num_frames, p_idx, 0]
                y = self.trajectory[:num_frames, p_idx, 1]
                z = self.trajectory[:num_frames, p_idx, 2]
                
                # Add to the figure
                fig.add_trace(go.Scatter3d(
                    x=x, y=y, z=z,
                    mode='lines+markers',
                    marker=dict(size=3),
                    line=dict(width=2),
                    name=f'Particle {p_idx}'
                ))
            
            # Update layout
            fig.update_layout(
                title="MD Simulation: 3D Particle Trajectories",
                scene=dict(
                    xaxis_title="X Position",
                    yaxis_title="Y Position",
                    zaxis_title="Z Position"
                ),
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255, 255, 255, 0.5)'
                )
            )
        else:
            # 2D plot (xy plane)
            fig = go.Figure()
            
            # Add each particle's trajectory
            for p_idx in particles:
                # Get the particle's coordinates
                x = self.trajectory[:num_frames, p_idx, 0]
                y = self.trajectory[:num_frames, p_idx, 1]
                
                # Add to the figure
                fig.add_trace(go.Scatter(
                    x=x, y=y,
                    mode='lines+markers',
                    marker=dict(size=5),
                    line=dict(width=2),
                    name=f'Particle {p_idx}'
                ))
            
            # Update layout
            fig.update_layout(
                title="MD Simulation: 2D Particle Trajectories (XY Plane)",
                xaxis_title="X Position",
                yaxis_title="Y Position",
                legend=dict(
                    x=0,
                    y=1,
                    bgcolor='rgba(255, 255, 255, 0.5)'
                )
            )
        
        return fig
    
    def plot_msd(self, msd_result=None, with_fit=True, fit_points=5):
        """
        Plot MSD curve.
        
        Parameters
        ----------
        msd_result : dict, optional
            Result from calculate_msd. If None, calculate MSD first.
        with_fit : bool
            Whether to show linear fit
        fit_points : int
            Number of initial points to use for linear fit
            
        Returns
        -------
        plotly.graph_objects.Figure
            Plotly figure object
        """
        if msd_result is None:
            msd_result = self.calculate_msd()
        
        lag_time = msd_result['lag_time']
        msd = msd_result['msd']
        
        # Create figure
        fig = go.Figure()
        
        # Add MSD curve
        fig.add_trace(go.Scatter(
            x=lag_time,
            y=msd,
            mode='lines+markers',
            name='MSD'
        ))
        
        # Add linear fit if requested
        if with_fit:
            # Use only initial points for fitting
            n_points = min(fit_points, len(lag_time))
            
            # Linear fit: MSD = 6*D*t for 3D
            slope, intercept = np.polyfit(lag_time[:n_points], msd[:n_points], 1)
            
            # Create fitted line over the full range
            fit_y = slope * lag_time + intercept
            
            # Calculate D from slope
            D = slope / 6.0
            
            # Add fit line to the plot
            fig.add_trace(go.Scatter(
                x=lag_time,
                y=fit_y,
                mode='lines',
                line=dict(dash='dash'),
                name=f'Fit (D = {D:.4f})'
            ))
        
        # Update layout
        fig.update_layout(
            title="Mean Squared Displacement",
            xaxis_title="Lag Time",
            yaxis_title="MSD",
            legend=dict(
                x=0,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.5)'
            )
        )
        
        return fig
    
    def compare_with_spt(self, spt_data, metric='diffusion'):
        """
        Compare MD simulation results with SPT data.
        
        Parameters
        ----------
        spt_data : dict
            SPT analysis results
        metric : str
            Metric to compare: 'diffusion', 'msd', or 'trajectories'
            
        Returns
        -------
        dict
            Comparison results
        """
        if metric == 'diffusion':
            # Calculate diffusion coefficient from simulation
            if self.trajectory is None:
                raise ValueError("No trajectory loaded")
            
            md_diffusion = self.calculate_diffusion_coefficient()
            
            # Get diffusion coefficient from SPT data
            spt_diffusion = spt_data.get('ensemble_results', {}).get('mean_diffusion_coefficient')
            
            if spt_diffusion is None:
                raise ValueError("SPT data does not contain diffusion coefficient")
            
            # Calculate ratio and difference
            ratio = md_diffusion / spt_diffusion
            difference = md_diffusion - spt_diffusion
            
            return {
                'md_diffusion': md_diffusion,
                'spt_diffusion': spt_diffusion,
                'ratio': ratio,
                'difference': difference
            }
        
        elif metric == 'msd':
            # Calculate MSD from simulation
            if self.trajectory is None:
                raise ValueError("No trajectory loaded")
            
            md_msd = self.calculate_msd()
            
            # Get MSD from SPT data
            spt_msd = spt_data.get('msd_data')
            
            if spt_msd is None:
                raise ValueError("SPT data does not contain MSD data")
            
            # Return MSD curves (we can't directly compare them as lag times may differ)
            return {
                'md_msd': md_msd,
                'spt_msd': spt_msd
            }
        
        elif metric == 'trajectories':
            # Convert MD trajectory to same format as SPT tracks
            md_tracks = self.convert_to_tracks_format()
            
            # Return both track sets
            return {
                'md_tracks': md_tracks,
                'spt_tracks': spt_data
            }
        
        else:
            raise ValueError(f"Unsupported comparison metric: {metric}")

def load_md_file(file):
    """
    Load a molecular dynamics simulation file.
    
    Parameters
    ----------
    file : UploadedFile
        File uploaded through Streamlit
        
    Returns
    -------
    MDSimulation
        Loaded MD simulation object
    """
    sim = MDSimulation()
    
    try:
        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # Load the simulation data
        sim.load_simulation_data(file, file_extension)
        
        return sim
    except Exception as e:
        raise ValueError(f"Error loading MD file: {str(e)}")