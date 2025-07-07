Okay, I can provide basic implementations for some of the placeholder file loading methods in your `md_integration.py` script.

**Important Considerations Before Proceeding:**

* **Complexity of MD Formats:** Molecular dynamics file formats (especially binary ones like XTC, DCD, TRR, but also text ones like PDB and GRO) can have many variations and complexities. The parsers provided below are **highly simplified** and make assumptions. They are intended for illustrative purposes and may not work correctly or robustly with all files of these types.
* **Recommendation for Production Use:** For any serious or production-level work, it is **strongly recommended** to use established, well-tested libraries such as:
    * **MDAnalysis:** Supports a wide array of topology and trajectory formats.
    * **mdtraj:** Another excellent library for trajectory analysis.
    * **Biopython:** Specifically `Bio.PDB` for PDB file parsing.
    These libraries handle the intricacies of the formats, provide error checking, and offer much more functionality.
* **Binary Formats (XTC, DCD, TRR):** Directly parsing these binary trajectory formats without dedicated libraries is extremely complex due to compression schemes and specific record structures. The placeholder for `_load_trajectory_file` will be updated to reflect this and recommend using specialized libraries, as providing a from-scratch parser is not feasible here.
* **File Object Handling:** The provided code assumes the `file` argument is a Streamlit `UploadedFile` object or a similar file-like object that can be read. For text files, we'll decode its content.

Below is the modified `md_integration.py` with basic parser implementations for `.xyz` and `.gro`, and a slightly more detailed PDB coordinate extractor. The CSV loader is retained from the previous version.

```python
"""
Molecular Dynamics integration for the SPT Analysis application.
Provides tools for importing, processing and comparing MD simulation data
with experimental SPT data.
"""

import pandas as pd
import numpy as np
# import streamlit as st # Not used directly in this module
# import matplotlib.pyplot as plt # Not used directly in this module
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Any, Optional, Union
import os
import io # Added for StringIO

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
            '.xtc': self._load_binary_trajectory_file, # Updated to reflect placeholder status
            '.dcd': self._load_binary_trajectory_file, # Updated
            '.trr': self._load_binary_trajectory_file, # Updated
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
            Format of the file, if not determined from file extension (e.g., '.gro')
            
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
        if isinstance(file, str): # If it's a path
            with open(file, 'r') as f:
                lines = f.readlines()
        else: # Assuming a file-like object (e.g., UploadedFile)
            # Make sure to reset stream position if it has been read before
            if hasattr(file, 'seek'):
                file.seek(0)
            if hasattr(file, 'getvalue'): # For BytesIO or objects from st.file_uploader
                content = file.getvalue()
                if isinstance(content, bytes):
                    lines = io.StringIO(content.decode('utf-8', errors='ignore')).readlines()
                else: # Assumed already decoded string
                    lines = io.StringIO(content).readlines()
            elif hasattr(file, 'readlines'): # For standard text file objects opened with 'r'
                 lines = file.readlines()
                 # If lines are bytes, decode them
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
                atom_info['atom_name'].append(line[10:15].strip()) # Corrected from 10:15 based on common format inspection (often 5s)
                atom_info['atom_number'].append(int(line[15:20].strip())) # Corrected from 15:20
                
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                atom_coords.append([x, y, z])
            
            # Last line is box vectors
            box_line = lines[2 + num_atoms].strip().split()
            box_vectors_flat = [float(val) for val in box_line]
            
            # For a rectangular box: v1(x) v2(y) v3(z) are dimensions
            # Other components (off-diagonal for triclinic) are ignored in this basic parser.
            if len(box_vectors_flat) >= 3:
                self.box_dimensions = np.array([box_vectors_flat[0], box_vectors_flat[1], box_vectors_flat[2]])
            else: # Fallback if box line is malformed or not rectangular
                self.box_dimensions = None # Or some default/warning


            self.trajectory = np.array(atom_coords).reshape(1, num_atoms, 3) # 1 frame
            self.topology = atom_info # Store parsed atom names, res_names etc.
            
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

    def _load_pdb_file(self, file: Any) -> Dict[str, Any]:
        """
        Basic parser for a PDB file (Protein Data Bank format).
        Reads ATOM/HETATM records for coordinates. Supports multiple MODELs as frames.
        Assumes Angstrom units for coordinates, converts to nm.
        Limitations: Does not parse full topology, connectivity, or complex PDB features.
        """
        lines = self._read_file_content(file)
        
        frames_coords: List[List[List[float]]] = []
        current_frame_coords: List[List[float]] = []
        atom_info_first_model: Dict[str, List] = {'atom_name': [], 'residue_name': [], 'residue_number': [], 'chain_id': []}
        model_count = 0
        box_dims_A: Optional[np.ndarray] = None # Box dimensions in Angstrom

        try:
            for line in lines:
                record_type = line[0:6].strip()
                
                if record_type == "MODEL":
                    model_count += 1
                    if current_frame_coords: # Starting a new model, save previous one
                        frames_coords.append(current_frame_coords)
                        current_frame_coords = []
                elif record_type in ("ATOM", "HETATM"):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    current_frame_coords.append([x / 10.0, y / 10.0, z / 10.0]) # Convert Angstrom to nm
                    
                    if model_count <= 1 or not frames_coords: # Store atom info only from first model for topology
                        atom_info_first_model['atom_name'].append(line[12:16].strip())
                        atom_info_first_model['residue_name'].append(line[17:20].strip())
                        atom_info_first_model['chain_id'].append(line[21:22].strip())
                        atom_info_first_model['residue_number'].append(int(line[22:26].strip()))
                elif record_type == "ENDMDL":
                    if current_frame_coords: # End of current model
                        frames_coords.append(current_frame_coords)
                        current_frame_coords = []
                elif record_type == "CRYST1" and box_dims_A is None: # Only take first CRYST1
                    try:
                        a = float(line[6:15].strip())
                        b = float(line[15:24].strip())
                        c = float(line[24:33].strip())
                        # alpha = float(line[33:40].strip()) # Ignored for simple rectangular box
                        # beta = float(line[40:47].strip())  # Ignored
                        # gamma = float(line[47:54].strip()) # Ignored
                        box_dims_A = np.array([a, b, c])
                    except ValueError:
                        pass # Ignore malformed CRYST1

            if not frames_coords and current_frame_coords: # Single "frame" PDB without MODEL records
                frames_coords.append(current_frame_coords)
            
            if not frames_coords:
                raise ValueError("No ATOM/HETATM records found or no models processed.")

            # Check consistency of atom counts across frames
            num_atoms = len(frames_coords[0])
            for frame_data in frames_coords[1:]:
                if len(frame_data) != num_atoms:
                    raise ValueError("Inconsistent number of atoms across PDB MODELs.")

            self.trajectory = np.array(frames_coords) # (n_frames, n_atoms, 3)
            self.topology = atom_info_first_model
            self.box_dimensions = box_dims_A / 10.0 if box_dims_A is not None else None # Convert to nm
            
            sim_info = {
                'format': 'pdb',
                'particles': num_atoms,
                'frames': len(frames_coords),
                'box_dimensions_nm': self.box_dimensions.tolist() if self.box_dimensions is not None else None,
                'loaded_successfully': True,
                'notes': 'PDB file parsed for ATOM/HETATM coordinates (converted to nm). Supports multiple MODELs.'
            }
            return sim_info
        except Exception as e:
            raise ValueError(f"Error parsing PDB file: {str(e)}")

    def _load_binary_trajectory_file(self, file: Any) -> Dict[str, Any]:
        """
        Placeholder for loading binary trajectory files (XTC, DCD, TRR).
        A real implementation requires specialized libraries like MDAnalysis or mdtraj.
        """
        sim_info = {
            'format': 'binary_trajectory (XTC/DCD/TRR)',
            'loaded_successfully': False,
            'notes': 'This is a placeholder. Loading XTC, DCD, or TRR files requires libraries like MDAnalysis or mdtraj. These formats are complex and binary.'
        }
        # To make the app somewhat runnable with these types, could return simulated data:
        # np.random.seed(42)
        # self.trajectory = np.random.normal(0, 2, (100, 1000, 3))
        # sim_info['frames'] = 100
        # sim_info['particles'] = 1000
        # sim_info['loaded_successfully'] = True 
        # sim_info['notes'] = 'Placeholder: Simulated trajectory for XTC/DCD/TRR.'
        # self.time_step = 0.002
        # sim_info['time_step_ps'] = 2.0
        # sim_info['total_duration_ps'] = 100 * 2.0
        return sim_info
    
    def _load_csv_trajectory(self, file: Any) -> Dict[str, Any]:
        """Load a trajectory from a CSV file."""
        try:
            # If file is UploadedFile, read its content
            if not isinstance(file, (str, pd.DataFrame, io.StringIO, io.BytesIO)):
                if hasattr(file, 'getvalue'): # Streamlit UploadedFile
                    csv_content = file.getvalue()
                    if isinstance(csv_content, bytes):
                        csv_stream = io.StringIO(csv_content.decode('utf-8', errors='ignore'))
                    else: # Already a string
                        csv_stream = io.StringIO(csv_content)
                    df = pd.read_csv(csv_stream)
                else:
                    raise ValueError("Unsupported file type for CSV loading, expected path or file-like object.")
            else: # If already a path or suitable stream
                df = pd.read_csv(file)

            required_columns = ['frame', 'particle_id', 'x', 'y', 'z']
            for col in required_columns:
                if col not in df.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV file.")
            
            unique_frames = sorted(df['frame'].unique())
            unique_particle_ids = sorted(df['particle_id'].unique())
            
            n_frames = len(unique_frames)
            n_particles = len(unique_particle_ids)
            
            frame_to_idx = {frame_val: i for i, frame_val in enumerate(unique_frames)}
            particle_id_to_idx = {p_id: i for i, p_id in enumerate(unique_particle_ids)}

            self.trajectory = np.full((n_frames, n_particles, 3), np.nan)
            
            for _, row in df.iterrows():
                frame_idx = frame_to_idx.get(row['frame'])
                particle_idx = particle_id_to_idx.get(row['particle_id'])
                
                if frame_idx is not None and particle_idx is not None:
                    self.trajectory[frame_idx, particle_idx, 0] = row['x']
                    self.trajectory[frame_idx, particle_idx, 1] = row['y']
                    self.trajectory[frame_idx, particle_idx, 2] = row['z']
            
            sim_info = {
                'format': 'csv', 'frames': n_frames, 'particles': n_particles,
                'loaded_successfully': True, 'notes': 'Trajectory loaded from CSV.'
            }
            
            if 'time' in df.columns and n_particles > 0:
                first_particle_id = unique_particle_ids[0]
                time_values = df.loc[df['particle_id'] == first_particle_id, 'time'].sort_values().values
                if len(time_values) > 1:
                    dt_values = np.diff(time_values)
                    if len(dt_values) > 0:
                        estimated_dt = np.mean(dt_values)
                        if estimated_dt > 1e-9:
                            self.time_step = estimated_dt 
                            sim_info['time_step_in_file_units'] = self.time_step
                            sim_info['total_duration_in_file_units'] = time_values[-1] - time_values[0]
            else:
                sim_info['notes'] += f" Using default time_step: {self.time_step} (assumed units)."

            return sim_info
        except Exception as e:
            raise ValueError(f"Error loading CSV trajectory: {str(e)}")
    
    def _load_xyz_file(self, file: Any) -> Dict[str, Any]:
        """
        Basic parser for a multi-frame XYZ file.
        Assumes consistent number of atoms per frame.
        Units are assumed to be Angstroms and are converted to nm.
        """
        lines = self._read_file_content(file)
        
        frames_coords: List[List[List[float]]] = []
        atom_symbols_first_frame: List[str] = []
        
        i = 0
        num_atoms_expected = -1
        
        try:
            while i < len(lines):
                if not lines[i].strip(): # Skip empty lines between frames if any
                    i += 1
                    continue

                num_atoms_this_frame = int(lines[i].strip())
                if num_atoms_expected == -1:
                    num_atoms_expected = num_atoms_this_frame
                elif num_atoms_this_frame != num_atoms_expected:
                    raise ValueError(f"Inconsistent number of atoms in XYZ file frames. Expected {num_atoms_expected}, found {num_atoms_this_frame}.")

                # Skip comment line
                i += 2 
                
                current_frame_coords: List[List[float]] = []
                for _ in range(num_atoms_this_frame):
                    if i >= len(lines):
                        raise ValueError("XYZ file ended prematurely while reading atom coordinates.")
                    parts = lines[i].strip().split()
                    if len(parts) < 4:
                        raise ValueError(f"Malformed atom line in XYZ file: '{lines[i].strip()}'")
                    
                    if len(frames_coords) == 0: # Only store symbols from the first frame
                        atom_symbols_first_frame.append(parts[0])
                    
                    # Coordinates are typically in Angstroms, convert to nm
                    x = float(parts[1]) / 10.0
                    y = float(parts[2]) / 10.0
                    z = float(parts[3]) / 10.0
                    current_frame_coords.append([x, y, z])
                    i += 1
                
                if len(current_frame_coords) == num_atoms_expected:
                    frames_coords.append(current_frame_coords)
                else: # Should not happen if previous checks pass
                    raise ValueError("Error processing frame, atom count mismatch.")

            if not frames_coords:
                raise ValueError("No frames processed from XYZ file.")

            self.trajectory = np.array(frames_coords)
            if self.topology is None: # Initialize topology if not set by other loaders
                self.topology = {'atom_symbols': atom_symbols_first_frame}
            else: # Append/update if exists
                self.topology['atom_symbols'] = atom_symbols_first_frame


            # Default time_step for XYZ, user might need to set this based on their simulation
            self.time_step = getattr(self, 'time_step', 0.001) # Use existing or default (e.g. 1ps)

            sim_info = {
                'format': 'xyz',
                'particles': num_atoms_expected,
                'frames': len(frames_coords),
                'loaded_successfully': True,
                'notes': f'XYZ file parsed ({len(frames_coords)} frames, {num_atoms_expected} atoms/frame). Coords converted to nm. Assumed time_step: {self.time_step}.',
                'time_step_assumed': self.time_step, # Indicate time step is assumed
                'total_duration_assumed': len(frames_coords) * self.time_step
            }
            return sim_info
        except Exception as e:
            raise ValueError(f"Error parsing XYZ file: {str(e)}")

    def convert_to_tracks_format(self, selected_particles: Optional[List[int]] = None) -> pd.DataFrame:
        """Converts MD trajectory to SPT tracks format."""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded to convert.")
        
        n_frames, n_total_particles, _ = self.trajectory.shape
        
        particles_to_convert_indices: List[int]
        if selected_particles is None:
            particles_to_convert_indices = list(range(n_total_particles))
        else:
            particles_to_convert_indices = [p for p in selected_particles if 0 <= p < n_total_particles]
            if not particles_to_convert_indices:
                return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y', 'z', 'Quality'])
        
        data_list = []
        for p_original_idx in particles_to_convert_indices:
            for f_idx in range(n_frames):
                pos = self.trajectory[f_idx, p_original_idx]
                if not np.any(np.isnan(pos)):
                    data_list.append({
                        'track_id': p_original_idx,
                        'frame': f_idx,
                        'x': pos[0], 'y': pos[1], 'z': pos[2],
                        'Quality': 1.0 
                    })
        
        return pd.DataFrame(data_list)

    def calculate_msd(self, selected_particles: Optional[List[int]] = None, max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Calculates Mean Squared Displacement."""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded to calculate MSD.")
        
        n_frames, n_total_particles, _ = self.trajectory.shape
        
        particles_for_msd_indices: List[int]
        if selected_particles is None:
            particles_for_msd_indices = list(range(n_total_particles))
        else:
            particles_for_msd_indices = [p for p in selected_particles if 0 <= p < n_total_particles]
        
        if not particles_for_msd_indices:
            return {'lag_time_steps': np.array([]), 'lag_time_actual': np.array([]), 'msd': np.array([])}

        if max_lag is None:
            max_lag = n_frames // 2
        else:
            max_lag = min(max_lag, n_frames - 1)
        
        if max_lag < 1:
            return {'lag_time_steps': np.array([]), 'lag_time_actual': np.array([]), 'msd': np.array([])}

        lag_steps = np.arange(1, max_lag + 1)
        msd_values_all_lags = np.zeros(len(lag_steps))
        
        for lag_idx, current_lag_step in enumerate(lag_steps):
            all_sq_disp_for_lag: List[float] = []
            for p_original_idx in particles_for_msd_indices:
                particle_traj = self.trajectory[:, p_original_idx, :]
                displacements = particle_traj[current_lag_step:] - particle_traj[:-current_lag_step]
                valid_displacements = displacements[~np.isnan(displacements).any(axis=1)]
                if valid_displacements.shape[0] > 0:
                    squared_displacements = np.sum(valid_displacements**2, axis=1)
                    all_sq_disp_for_lag.extend(squared_displacements)
            
            if all_sq_disp_for_lag:
                msd_values_all_lags[lag_idx] = np.mean(all_sq_disp_for_lag)
            else:
                msd_values_all_lags[lag_idx] = np.nan
        
        actual_lag_times = lag_steps * self.time_step
        
        return {
            'lag_time_steps': lag_steps, 
            'lag_time_actual': actual_lag_times, 
            'msd': msd_values_all_lags
        }

    def calculate_diffusion_coefficient(self, msd_result: Optional[Dict[str, np.ndarray]] = None, fit_points: int = 5) -> float:
        """Calculates diffusion coefficient from MSD curve."""
        if self.trajectory is None: # Need trajectory to determine dimensionality
            # This case should ideally be caught by MSD calculation first
            return np.nan 

        if msd_result is None:
            msd_result = self.calculate_msd()
        
        actual_lag_times = msd_result.get('lag_time_actual', np.array([]))
        msd_values = msd_result.get('msd', np.array([]))
        
        if len(actual_lag_times) == 0 or len(msd_values) == 0 or len(actual_lag_times) != len(msd_values):
            return np.nan

        num_available_points = len(actual_lag_times)
        n_points_to_fit = min(fit_points, num_available_points)
        
        if n_points_to_fit < 2: return np.nan
            
        fit_lags = actual_lag_times[:n_points_to_fit]
        fit_msds = msd_values[:n_points_to_fit]
        
        valid_fit_indices = ~np.isnan(fit_lags) & ~np.isnan(fit_msds)
        fit_lags_valid = fit_lags[valid_fit_indices]
        fit_msds_valid = fit_msds[valid_fit_indices]

        if len(fit_lags_valid) < 2: return np.nan

        try:
            slope, _ = np.polyfit(fit_lags_valid, fit_msds_valid, 1)
            dimensionality = self.trajectory.shape[2] 
            D = slope / (2.0 * dimensionality)
            return D
        except (np.linalg.LinAlgError, ValueError):
            return np.nan

    def plot_trajectory(self, particles: Optional[List[int]] = None, num_frames: Optional[int] = None, mode: str = '3d') -> go.Figure:
        """Plots particle trajectories."""
        if self.trajectory is None:
            return go.Figure().update_layout(title_text="No trajectory data available to plot.")
        
        n_total_frames, n_total_particles, _ = self.trajectory.shape
        
        particles_to_plot: List[int]
        if particles is None:
            n_to_plot = min(10, n_total_particles)
            particles_to_plot = list(np.random.choice(n_total_particles, n_to_plot, replace=False)) if n_total_particles > 0 else []
        else:
            particles_to_plot = [p for p in particles if 0 <= p < n_total_particles]

        frames_to_plot = num_frames if num_frames is not None else n_total_frames
        frames_to_plot = min(frames_to_plot, n_total_frames)

        fig = go.Figure()
        if not particles_to_plot:
             fig.update_layout(title_text=f"MD Simulation: No Particles Selected/Available for {mode.upper()} Plot")
             return fig

        plot_title_suffix = "3D Particle Trajectories" if mode == '3d' else "2D Particle Trajectories (XY Plane)"
        fig.update_layout(title=f"MD Simulation: {plot_title_suffix}")

        for p_idx in particles_to_plot:
            traj_data = self.trajectory[:frames_to_plot, p_idx, :]
            valid_pts = ~np.isnan(traj_data).any(axis=1) # Filter out NaN coordinates for plotting
            x = traj_data[valid_pts, 0]
            y = traj_data[valid_pts, 1]
            
            if mode == '3d':
                z = traj_data[valid_pts, 2]
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines+markers', marker=dict(size=3), name=f'Particle {p_idx}'))
                fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
            else: # 2D
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', marker=dict(size=5), name=f'Particle {p_idx}'))
                fig.update_layout(xaxis_title="X", yaxis_title="Y")
        return fig

    def plot_msd(self, msd_result: Optional[Dict[str, np.ndarray]] = None, with_fit: bool = True, fit_points: int = 5) -> go.Figure:
        """Plots MSD curve."""
        if msd_result is None:
            msd_result = self.calculate_msd()

        actual_lag_times = msd_result.get('lag_time_actual', np.array([]))
        msd_values = msd_result.get('msd', np.array([]))
        
        fig = go.Figure()
        plot_title = "Mean Squared Displacement"

        if len(actual_lag_times) > 0 and len(msd_values) > 0:
            valid_msd_indices = ~np.isnan(actual_lag_times) & ~np.isnan(msd_values)
            plot_lags = actual_lag_times[valid_msd_indices]
            plot_msds = msd_values[valid_msd_indices]

            if len(plot_lags) > 0:
                fig.add_trace(go.Scatter(x=plot_lags, y=plot_msds, mode='lines+markers', name='MSD'))
                if with_fit:
                    D_fit = self.calculate_diffusion_coefficient(msd_result={'lag_time_actual': plot_lags, 'msd': plot_msds}, fit_points=fit_points)
                    if not np.isnan(D_fit):
                        n_pts_fit = min(fit_points, len(plot_lags))
                        if n_pts_fit >= 2:
                            slope, intercept = np.polyfit(plot_lags[:n_pts_fit], plot_msds[:n_pts_fit], 1)
                            fit_line_y_poly = slope * plot_lags + intercept
                            fig.add_trace(go.Scatter(x=plot_lags, y=fit_line_y_poly, mode='lines', line=dict(dash='dash'), name=f'Fit (D = {D_fit:.4g})'))
                            plot_title += f" (Est. D: {D_fit:.4g})"
            else: # No valid MSD points after NaN filtering
                plot_title += " (No valid data points)"
        else:
            plot_title += " (No data available)"
        
        fig.update_layout(
            title=plot_title, 
            xaxis_title=f"Lag Time (using time_step: {self.time_step})", 
            yaxis_title="MSD"
        )
        return fig
    
    def compare_with_spt(self, spt_data: Dict[str, Any], metric: str = 'diffusion') -> Dict[str, Any]:
        """Compares MD simulation results with SPT data."""
        md_val: Any = np.nan
        spt_val: Any = np.nan
        comparison_results: Dict[str, Any] = {}

        if metric == 'diffusion':
            md_val = self.calculate_diffusion_coefficient()
            spt_val = spt_data.get('ensemble_results', {}).get('mean_diffusion_coefficient')
            comparison_results = {'md_diffusion': md_val, 'spt_diffusion': spt_val}
            if spt_val is not None and not np.isnan(spt_val) and not np.isnan(md_val):
                comparison_results['ratio'] = md_val / spt_val if spt_val != 0 else np.inf
                comparison_results['difference'] = md_val - spt_val
            else:
                comparison_results['ratio'] = np.nan
                comparison_results['difference'] = np.nan
            return comparison_results

        elif metric == 'msd':
            md_msd_data = self.calculate_msd()
            # spt_msd_data is often a DataFrame from analysis.calculate_msd
            spt_msd_df = spt_data.get('msd_data') 
            if isinstance(spt_msd_df, pd.DataFrame) and not spt_msd_df.empty:
                # Aggregate SPT MSD for comparison (e.g., mean MSD per lag_time)
                if 'lag_time' in spt_msd_df.columns and 'msd' in spt_msd_df.columns:
                     spt_msd_aggregated = spt_msd_df.groupby('lag_time')['msd'].mean().reset_index()
                     spt_msd_dict = {'lag_time_actual': spt_msd_aggregated['lag_time'].values, 'msd': spt_msd_aggregated['msd'].values}
                else: # Fallback if columns differ
                     spt_msd_dict = None
            else:
                spt_msd_dict = None
            return {'md_msd': md_msd_data, 'spt_msd_aggregated': spt_msd_dict}
        
        elif metric == 'trajectories':
            md_tracks_df = self.convert_to_tracks_format()
            spt_tracks_val = spt_data.get('tracks_data', spt_data if isinstance(spt_data, pd.DataFrame) else None)
            return {'md_tracks': md_tracks_df, 'spt_tracks': spt_tracks_val}
        
        else:
            raise ValueError(f"Unsupported comparison metric: {metric}")

def load_md_file(uploaded_file: Any) -> MDSimulation:
    """
    Load a molecular dynamics simulation file from a Streamlit UploadedFile object.
    """
    sim = MDSimulation()
    if uploaded_file is None:
        raise ValueError("No file provided for loading.")

    try:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        
        sim.load_simulation_data(uploaded_file, file_format=file_extension)
        if sim.simulation_info: # Check if populated
            sim.simulation_info['original_filename'] = file_name
        else: # Should not happen if load_simulation_data throws error on failure
            sim.simulation_info = {'original_filename': file_name, 'loaded_successfully': False, 'notes': 'Loading failed unexpectedly.'}

        return sim
    except Exception as e:
        filename_msg = f"'{uploaded_file.name}'" if hasattr(uploaded_file, 'name') else "'unknown file'"
        raise ValueError(f"Error loading MD file {filename_msg}: {str(e)}")

```