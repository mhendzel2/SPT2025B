"""
Molecular Dynamics integration for the SPT Analysis application.
Provides tools for importing, processing and comparing MD simulation data
with experimental SPT data.
"""

from typing import Dict, List, Tuple, Any, Optional
import os
import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go


class MDSimulation:
    """
    Class for handling molecular dynamics simulation data.
    Supports common text formats and provides trajectory processing utilities.
    """

    def __init__(self):
        # Supported file formats and their handlers
        self.supported_formats = {
            '.gro': self._load_gro_file,
            '.pdb': self._load_pdb_file,
            '.xtc': self._load_binary_trajectory_file,  # placeholder
            '.dcd': self._load_binary_trajectory_file,  # placeholder
            '.trr': self._load_binary_trajectory_file,  # placeholder
            '.csv': self._load_csv_trajectory,
            '.xyz': self._load_xyz_file,
        }
        # Data storage
        self.topology: Optional[Dict[str, Any]] = None
        self.trajectory: Optional[np.ndarray] = None  # shape: (frames, particles, 3)
        self.box_dimensions: Optional[np.ndarray] = None  # nm
        self.time_step: float = 0.001  # default time step (arbitrary units)
        self.simulation_info: Dict[str, Any] = {}

    # -------- Public API --------

    def load_simulation_data(self, file: Any, file_format: Optional[str] = None) -> Dict[str, Any]:
        """
        Load simulation data from a file path or a file-like object.

        Returns a dict with metadata and load status.
        """
        fmt = self._detect_format(file, file_format)
        if fmt not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {fmt}. Supported: {list(self.supported_formats.keys())}")

        info = self.supported_formats[fmt](file)
        info['loaded_file_format'] = fmt
        self.simulation_info = info
        return info

    def convert_to_tracks_format(self, selected_particles: Optional[List[int]] = None) -> pd.DataFrame:
        """Convert loaded trajectory to SPT tracks df: columns [track_id, frame, x, y, z, Quality]."""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded to convert.")
        n_frames, n_particles, _ = self.trajectory.shape

        if selected_particles is None:
            selected = list(range(n_particles))
        else:
            selected = [p for p in selected_particles if 0 <= p < n_particles]
            if not selected:
                return pd.DataFrame(columns=['track_id', 'frame', 'x', 'y', 'z', 'Quality'])

        rows: List[Dict[str, Any]] = []
        for pid in selected:
            coords = self.trajectory[:, pid, :]
            valid = ~np.isnan(coords).any(axis=1)
            for f in np.where(valid)[0]:
                x, y, z = coords[f]
                rows.append({'track_id': pid, 'frame': int(f), 'x': float(x), 'y': float(y), 'z': float(z), 'Quality': 1.0})
        return pd.DataFrame(rows)

    def calculate_msd(self, selected_particles: Optional[List[int]] = None, max_lag: Optional[int] = None) -> Dict[str, np.ndarray]:
        """Compute ensemble MSD across selected particles."""
        if self.trajectory is None:
            raise ValueError("No trajectory loaded to calculate MSD.")
        n_frames, n_particles, _ = self.trajectory.shape

        if selected_particles is None:
            selected = list(range(n_particles))
        else:
            selected = [p for p in selected_particles if 0 <= p < n_particles]
        if not selected:
            return {'lag_time_steps': np.array([]), 'lag_time_actual': np.array([]), 'msd': np.array([])}

        max_lag = min(max_lag or (n_frames // 2), n_frames - 1)
        if max_lag < 1:
            return {'lag_time_steps': np.array([]), 'lag_time_actual': np.array([]), 'msd': np.array([])}

        lags = np.arange(1, max_lag + 1, dtype=int)
        msd_vals = np.zeros_like(lags, dtype=float)

        for i, lag in enumerate(lags):
            disp2_all: List[float] = []
            for pid in selected:
                traj = self.trajectory[:, pid, :]
                valid = ~np.isnan(traj).any(axis=1)
                if valid.sum() <= lag:
                    continue
                # Only use contiguous valid segments
                x = traj[valid]
                if len(x) <= lag:
                    continue
                d = x[lag:] - x[:-lag]
                if len(d) > 0:
                    disp2_all.extend(np.sum(d * d, axis=1))
            msd_vals[i] = np.mean(disp2_all) if disp2_all else np.nan

        return {
            'lag_time_steps': lags,
            'lag_time_actual': lags * self.time_step,
            'msd': msd_vals
        }

    def calculate_diffusion_coefficient(self, msd_result: Optional[Dict[str, np.ndarray]] = None, fit_points: int = 5) -> float:
        """Estimate D from initial linear region of MSD: MSD = 2*d*D*t."""
        if self.trajectory is None:
            return np.nan
        if msd_result is None:
            msd_result = self.calculate_msd()
        t = msd_result.get('lag_time_actual', np.array([]))
        y = msd_result.get('msd', np.array([]))
        if t.size < 2 or y.size < 2:
            return np.nan
        k = int(min(max(fit_points, 2), len(t)))
        tv = t[:k]
        yv = y[:k]
        mask = np.isfinite(tv) & np.isfinite(yv)
        tv = tv[mask]
        yv = yv[mask]
        if tv.size < 2:
            return np.nan
        try:
            slope, intercept = np.polyfit(tv, yv, 1)
            d_dim = self.trajectory.shape[2]  # 3 for 3D
            D = slope / (2.0 * d_dim)
            return float(D)
        except Exception:
            return np.nan

    def plot_trajectory(self, particles: Optional[List[int]] = None, num_frames: Optional[int] = None, mode: str = '3d') -> go.Figure:
        """Create a Plotly figure for trajectories (2D or 3D)."""
        fig = go.Figure()
        if self.trajectory is None:
            return fig.update_layout(title="No trajectory data available.")
        n_frames, n_particles, _ = self.trajectory.shape

        if particles is None:
            n_show = min(10, n_particles)
            rng = np.random.default_rng(42)
            particles = list(rng.choice(n_particles, n_show, replace=False)) if n_particles > 0 else []
        frames_to_plot = min(num_frames or n_frames, n_frames)

        if not particles:
            return fig.update_layout(title="No particles to plot.")
        title = "3D Particle Trajectories" if mode == '3d' else "2D Particle Trajectories (XY)"
        fig.update_layout(title=title)

        for pid in particles:
            traj = self.trajectory[:frames_to_plot, pid, :]
            valid = ~np.isnan(traj).any(axis=1)
            x = traj[valid, 0]
            y = traj[valid, 1]
            if mode == '3d':
                z = traj[valid, 2]
                fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode='lines', name=f'P{pid}'))
                fig.update_layout(scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"))
            else:
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=f'P{pid}'))
                fig.update_layout(xaxis_title="X", yaxis_title="Y")
        return fig

    def plot_msd(self, msd_result: Optional[Dict[str, np.ndarray]] = None, with_fit: bool = True, fit_points: int = 5) -> go.Figure:
        """Create a Plotly figure for MSD."""
        if msd_result is None:
            msd_result = self.calculate_msd()
        t = msd_result.get('lag_time_actual', np.array([]))
        y = msd_result.get('msd', np.array([]))
        fig = go.Figure()
        if t.size and y.size:
            fig.add_trace(go.Scatter(x=t, y=y, mode='lines+markers', name='MSD'))
            if with_fit and t.size >= 2:
                k = int(min(max(fit_points, 2), len(t)))
                slope, intercept = np.polyfit(t[:k], y[:k], 1)
                fig.add_trace(go.Scatter(x=t, y=slope * t + intercept, mode='lines', name='Linear fit', line=dict(dash='dash')))
        fig.update_layout(title="Mean Squared Displacement", xaxis_title=f"Lag Time (Δt, step={self.time_step})", yaxis_title="MSD")
        return fig

    def compare_with_spt(self, spt_data: Dict[str, Any], metric: str = 'diffusion') -> Dict[str, Any]:
        """
        Compare MD results with SPT results.
        metric: 'diffusion' | 'msd' | 'trajectories'
        """
        if metric == 'diffusion':
            md_D = self.calculate_diffusion_coefficient()
            spt_D = spt_data.get('ensemble_results', {}).get('mean_diffusion_coefficient')
            out = {'md_diffusion': md_D, 'spt_diffusion': spt_D}
            if spt_D is not None and np.isfinite(md_D):
                out['ratio'] = md_D / spt_D if spt_D else np.inf
                out['difference'] = md_D - spt_D
            else:
                out['ratio'] = np.nan
                out['difference'] = np.nan
            return out

        if metric == 'msd':
            md_msd = self.calculate_msd()
            spt_msd_df = spt_data.get('msd_data')
            spt_msd = None
            if isinstance(spt_msd_df, pd.DataFrame) and not spt_msd_df.empty:
                if 'lag_time' in spt_msd_df.columns and 'msd' in spt_msd_df.columns:
                    g = spt_msd_df.groupby('lag_time')['msd'].mean().reset_index()
                    spt_msd = {'lag_time_actual': g['lag_time'].values, 'msd': g['msd'].values}
            return {'md_msd': md_msd, 'spt_msd_aggregated': spt_msd}

        if metric == 'trajectories':
            md_tracks = self.convert_to_tracks_format()
            spt_tracks = spt_data.get('tracks_data', spt_data if isinstance(spt_data, pd.DataFrame) else None)
            return {'md_tracks': md_tracks, 'spt_tracks': spt_tracks}

        raise ValueError(f"Unsupported comparison metric: {metric}")

    # -------- Internal loaders/helpers --------

    def _detect_format(self, file: Any, file_format: Optional[str]) -> str:
        if file_format:
            return file_format.lower()
        if isinstance(file, str):
            return os.path.splitext(file)[1].lower()
        if hasattr(file, 'name') and isinstance(file.name, str):
            return os.path.splitext(file.name)[1].lower()
        raise ValueError("Cannot determine file format; please provide file_format explicitly.")

    def _read_file_content(self, file: Any) -> List[str]:
        """Read text content lines from path or file-like object."""
        if isinstance(file, str):
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                return f.readlines()
        # file-like
        if hasattr(file, 'seek'):
            file.seek(0)
        if hasattr(file, 'getvalue'):
            content = file.getvalue()
            if isinstance(content, bytes):
                return io.StringIO(content.decode('utf-8', errors='ignore')).readlines()
            return io.StringIO(content).readlines()
        if hasattr(file, 'read'):
            lines = file.readlines()
            if lines and isinstance(lines[0], bytes):
                lines = [ln.decode('utf-8', errors='ignore') for ln in lines]
            return lines
        raise ValueError("Unsupported file object type for reading text content.")

    def _load_gro_file(self, file: Any) -> Dict[str, Any]:
        """Parse minimal GRO file (nm units)."""
        lines = self._read_file_content(file)
        try:
            title = lines[0].strip()
            num_atoms = int(lines[1].strip())
            atom_coords: List[List[float]] = []
            atom_info = {'residue_number': [], 'residue_name': [], 'atom_name': [], 'atom_number': []}
            for i in range(2, 2 + num_atoms):
                line = lines[i]
                atom_info['residue_number'].append(int(line[0:5].strip()))
                atom_info['residue_name'].append(line[5:10].strip())
                atom_info['atom_name'].append(line[10:15].strip())
                atom_info['atom_number'].append(int(line[15:20].strip()))
                x = float(line[20:28].strip())
                y = float(line[28:36].strip())
                z = float(line[36:44].strip())
                atom_coords.append([x, y, z])
            # Box line
            box_vals = [float(v) for v in lines[2 + num_atoms].strip().split()]
            self.box_dimensions = np.array(box_vals[:3]) if len(box_vals) >= 3 else None
            self.trajectory = np.array(atom_coords, dtype=float).reshape(1, num_atoms, 3)
            self.topology = atom_info
            return {
                'format': 'gro',
                'title': title,
                'particles': num_atoms,
                'frames': 1,
                'box_dimensions_nm': self.box_dimensions.tolist() if self.box_dimensions is not None else None,
                'loaded_successfully': True,
                'notes': 'GRO parsed (coords, box).',
            }
        except Exception as e:
            raise ValueError(f"Error parsing GRO file: {e}")

    def _load_pdb_file(self, file: Any) -> Dict[str, Any]:
        """Parse basic PDB (Angstrom -> nm), supports multiple MODEL frames."""
        lines = self._read_file_content(file)
        frames: List[List[List[float]]] = []
        current: List[List[float]] = []
        topo: Dict[str, List[Any]] = {'atom_name': [], 'residue_name': [], 'residue_number': [], 'chain_id': []}
        boxA: Optional[np.ndarray] = None
        try:
            in_model = False
            for line in lines:
                rec = line[0:6].strip()
                if rec == 'MODEL':
                    if current:
                        frames.append(current)
                        current = []
                    in_model = True
                elif rec in ('ATOM', 'HETATM'):
                    x = float(line[30:38].strip()) / 10.0
                    y = float(line[38:46].strip()) / 10.0
                    z = float(line[46:54].strip()) / 10.0
                    current.append([x, y, z])
                    if len(frames) == 0 and not in_model:
                        topo['atom_name'].append(line[12:16].strip())
                        topo['residue_name'].append(line[17:20].strip())
                        topo['chain_id'].append(line[21:22].strip())
                        topo['residue_number'].append(int(line[22:26].strip()))
                elif rec == 'ENDMDL':
                    if current:
                        frames.append(current)
                        current = []
                        in_model = False
                elif rec == 'CRYST1' and boxA is None:
                    try:
                        a = float(line[6:15].strip())
                        b = float(line[15:24].strip())
                        c = float(line[24:33].strip())
                        boxA = np.array([a, b, c])
                    except Exception:
                        pass
            if not frames and current:
                frames.append(current)
            if not frames:
                raise ValueError("No coordinates found in PDB.")
            n_atoms = len(frames[0])
            for f in frames:
                if len(f) != n_atoms:
                    raise ValueError("Inconsistent atom count across models.")
            self.trajectory = np.array(frames, dtype=float)
            self.topology = topo
            self.box_dimensions = boxA / 10.0 if boxA is not None else None
            return {
                'format': 'pdb',
                'particles': n_atoms,
                'frames': len(frames),
                'box_dimensions_nm': self.box_dimensions.tolist() if self.box_dimensions is not None else None,
                'loaded_successfully': True,
                'notes': 'PDB parsed (coords converted to nm).',
            }
        except Exception as e:
            raise ValueError(f"Error parsing PDB file: {e}")

    def _load_csv_trajectory(self, file: Any) -> Dict[str, Any]:
        """Load trajectory from CSV expecting columns: frame, particle_id, x, y, z."""
        try:
            if isinstance(file, (io.StringIO, io.BytesIO, str)):
                df = pd.read_csv(file)
            else:
                # Streamlit UploadedFile
                if hasattr(file, 'getvalue'):
                    buf = file.getvalue()
                    if isinstance(buf, bytes):
                        df = pd.read_csv(io.StringIO(buf.decode('utf-8', errors='ignore')))
                    else:
                        df = pd.read_csv(io.StringIO(buf))
                else:
                    df = pd.read_csv(file)
            required = ['frame', 'particle_id', 'x', 'y', 'z']
            for c in required:
                if c not in df.columns:
                    raise ValueError(f"Required column '{c}' not found in CSV.")
            frames = np.sort(df['frame'].unique())
            pids = np.sort(df['particle_id'].unique())
            f_map = {v: i for i, v in enumerate(frames)}
            p_map = {v: i for i, v in enumerate(pids)}
            traj = np.full((len(frames), len(pids), 3), np.nan, dtype=float)
            for _, r in df.iterrows():
                fi = f_map.get(r['frame'])
                pi = p_map.get(r['particle_id'])
                traj[fi, pi, 0] = float(r['x'])
                traj[fi, pi, 1] = float(r['y'])
                traj[fi, pi, 2] = float(r['z'])
            self.trajectory = traj
            info = {
                'format': 'csv',
                'frames': len(frames),
                'particles': len(pids),
                'loaded_successfully': True,
                'notes': 'CSV trajectory loaded.',
            }
            if 'time' in df.columns:
                t = np.sort(df.loc[df['particle_id'] == pids[0], 'time'].values)
                if t.size > 1:
                    dt = np.mean(np.diff(t))
                    if np.isfinite(dt) and dt > 0:
                        self.time_step = float(dt)
                        info['time_step_in_file_units'] = self.time_step
                        info['total_duration_in_file_units'] = float(t[-1] - t[0])
            else:
                info['notes'] += " Using default time_step."
            return info
        except Exception as e:
            raise ValueError(f"Error loading CSV trajectory: {e}")

    def _load_xyz_file(self, file: Any) -> Dict[str, Any]:
        """Parse simple multi-frame XYZ (Angstrom -> nm)."""
        lines = self._read_file_content(file)
        frames: List[List[List[float]]] = []
        atom_symbols: List[str] = []
        i = 0
        expected = -1
        try:
            while i < len(lines):
                if not lines[i].strip():
                    i += 1
                    continue
                n = int(lines[i].strip())
                if expected == -1:
                    expected = n
                elif n != expected:
                    raise ValueError("Inconsistent atom count in XYZ frames.")
                i += 1  # comment
                i += 1
                fcoords: List[List[float]] = []
                for _ in range(n):
                    parts = lines[i].strip().split()
                    if len(parts) < 4:
                        raise ValueError("Malformed XYZ atom line.")
                    if not frames:
                        atom_symbols.append(parts[0])
                    x = float(parts[1]) / 10.0
                    y = float(parts[2]) / 10.0
                    z = float(parts[3]) / 10.0
                    fcoords.append([x, y, z])
                    i += 1
                frames.append(fcoords)
            if not frames:
                raise ValueError("No frames found in XYZ.")
            self.trajectory = np.array(frames, dtype=float)
            self.topology = {'atom_symbols': atom_symbols}
            return {
                'format': 'xyz',
                'particles': expected,
                'frames': len(frames),
                'loaded_successfully': True,
                'notes': 'XYZ parsed (coords converted to nm).',
            }
        except Exception as e:
            raise ValueError(f"Error parsing XYZ file: {e}")

    def _load_binary_trajectory_file(self, file: Any) -> Dict[str, Any]:
        """Placeholder for XTC/DCD/TRR; requires mdtraj/MDAnalysis to implement."""
        return {
            'format': 'binary_trajectory',
            'loaded_successfully': False,
            'notes': 'Binary trajectory formats require external libraries (mdtraj/MDAnalysis).'
        }


def load_md_file(uploaded_file: Any) -> MDSimulation:
    """Helper to load a molecular dynamics file (e.g., from Streamlit uploader)."""
    sim = MDSimulation()
    if uploaded_file is None:
        raise ValueError("No file provided for loading.")
    try:
        name = getattr(uploaded_file, 'name', 'unknown')
        ext = os.path.splitext(name)[1].lower()
        sim.load_simulation_data(uploaded_file, file_format=ext)
        sim.simulation_info['original_filename'] = name
        return sim
    except Exception as e:
        raise ValueError(f"Error loading MD file '{getattr(uploaded_file, 'name', 'unknown')}': {e}")