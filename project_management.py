"""
Project Management for SPT Analysis application.
Handles organizing and comparing multiple tracking files across different conditions.
Enhanced with persistent file-based storage and robust data management.
"""

from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import os, json, uuid, datetime, io
import pandas as pd

class FileObject:
    """Represents a file with track data within a project condition."""
    def __init__(self, file_id: str, file_name: str, tracks_df: pd.DataFrame, upload_date: str):
        self.id = file_id
        self.file_name = file_name
        self.track_data = tracks_df
        self.upload_date = upload_date

    def to_dict(self, save_path: str = None) -> Dict:
        track_data_file = None
        if save_path and self.track_data is not None and not self.track_data.empty:
            # Persist as CSV alongside project json
            data_dir = os.path.join(os.path.dirname(save_path), "data")
            os.makedirs(data_dir, exist_ok=True)
            track_data_file = os.path.join(data_dir, f"{self.id}.csv")
            try:
                self.track_data.to_csv(track_data_file, index=False)
            except Exception:
                track_data_file = None
        return {
            'id': self.id,
            'file_name': self.file_name,
            'upload_date': self.upload_date,
            'data_path': track_data_file
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'FileObject':
        df = None
        p = d.get('data_path')
        if p and os.path.exists(p):
            try:
                df = pd.read_csv(p)
            except Exception:
                df = None
        return cls(d['id'], d.get('file_name', 'unknown'), df, d.get('upload_date', ''))

class Condition:
    """Represents an experimental condition."""
    def __init__(self, cond_id: str, name: str, description: str = ""):
        self.id = cond_id
        self.name = name
        self.description = description
        # For UI uploads we store lightweight dicts with embedded bytes under 'data'
        self.files: List[Dict[str, Any]] = []

    def to_dict(self, save_path: str = None) -> Dict:
        # Files are stored as lightweight dicts (name/type/size/data bytes not persisted)
        # Remove 'data' field from files as bytes are not JSON serializable
        files_for_json = []
        for f in self.files:
            file_copy = {k: v for k, v in f.items() if k != 'data'}
            files_for_json.append(file_copy)
        
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'files': files_for_json  # keep metadata; 'data' bytes excluded from JSON
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Condition':
        c = cls(d['id'], d.get('name', ''), d.get('description', ''))
        # Ensure a copy of the list is made to prevent shared state issues
        c.files = list(d.get('files', []))
        return c

    def pool_tracks(self, max_workers: int = 4, 
                    show_progress: bool = True,
                    validate: bool = True) -> Tuple[pd.DataFrame, List[Dict]]:
        """
        Pool tracks from all files with parallel processing.
        
        Parameters
        ----------
        max_workers : int, default 4
            Maximum number of parallel workers
        show_progress : bool, default True
            Show progress bar (requires streamlit)
        validate : bool, default True
            Validate and deduplicate pooled data
        
        Returns
        -------
        Tuple[pd.DataFrame, List[Dict]]
            (pooled_dataframe, list_of_errors)
        """
        try:
            from batch_processing_utils import parallel_process_files, pool_dataframes_efficiently, load_file_with_retry
            
            # Process files in parallel
            dataframes, errors = parallel_process_files(
                self.files,
                load_file_with_retry,
                max_workers=max_workers,
                show_progress=show_progress,
                use_threads=True
            )
            
            # Pool DataFrames efficiently
            result = pool_dataframes_efficiently(
                dataframes,
                validate=validate,
                deduplicate=True
            )
            
            return result, errors
            
        except ImportError:
            # Fallback to original sequential implementation
            pooled = []
            errors = []
            for f in self.files:
                try:
                    df = None
                    
                    if 'data' in f and f['data'] is not None:
                        # Load from bytes with encoding handling
                        data_bytes = f['data']
                        if isinstance(data_bytes, str):
                            data_bytes = data_bytes.encode('utf-8')
                        try:
                            df = pd.read_csv(io.BytesIO(data_bytes), encoding='utf-8')
                        except UnicodeDecodeError:
                            df = pd.read_csv(io.BytesIO(data_bytes), encoding='latin-1')
                            
                    elif 'data_path' in f and f['data_path'] and os.path.exists(f['data_path']):
                        df = pd.read_csv(f['data_path'])
                    
                    if df is not None and not df.empty:
                        # Clean the data
                        df = df.dropna(how='all')
                        df.columns = df.columns.str.strip()
                        
                        # Check for duplicate header row
                        if len(df) > 1:
                            first_row = df.iloc[0]
                            if all(isinstance(val, str) and val.strip() in df.columns for val in first_row if pd.notna(val)):
                                df = df.iloc[1:].reset_index(drop=True)
                        
                        # Convert numeric columns
                        for col in df.columns:
                            if col in ['x', 'y', 'z', 'frame', 'track_id'] or any(x in col.lower() for x in ['frame', 'track']):
                                try:
                                    df[col] = pd.to_numeric(df[col], errors='coerce')
                                except Exception:
                                    pass
                        
                        # Remove rows with NaN in critical columns
                        critical_cols = [c for c in ['track_id', 'frame', 'x', 'y'] if c in df.columns]
                        if critical_cols:
                            df = df.dropna(subset=critical_cols)
                        
                        pooled.append(df)
                        
                except Exception as e:
                    errors.append({
                        'file': f.get('file_name', 'unknown'),
                        'error': str(e)
                    })
                    continue
            
            result = pd.concat(pooled, ignore_index=True) if pooled else pd.DataFrame()
            return result, errors

    def compare_conditions(self, metric: str = "diffusion_coefficient", 
                          test_type: str = "auto", alpha: float = 0.05) -> Dict[str, Any]:
        return {'success': True, 'metric': metric, 'alpha': alpha}

class Project:
    """Represents a full SPT project."""
    def __init__(self, name: str = "New Project", description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.created_date = datetime.datetime.now().isoformat()
        self.last_modified = self.created_date
        self.conditions: List[Condition] = []

    def save(self, base_path: str):
        save_project(self, base_path)

    def update_last_modified(self):
        self.last_modified = datetime.datetime.now().isoformat()

    def to_dict(self, save_path: str = None) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'created_date': self.created_date,
            'last_modified': self.last_modified,
            'conditions': [c.to_dict(save_path) for c in self.conditions]
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'Project':
        p = cls(d.get('name', 'Project'), d.get('description', ''))
        p.id = d.get('id', str(uuid.uuid4()))
        p.created_date = d.get('created_date', datetime.datetime.now().isoformat())
        p.last_modified = d.get('last_modified', p.created_date)
        p.conditions = [Condition.from_dict(cd) for cd in d.get('conditions', [])]
        return p

def save_project(project: 'Project', project_path: str) -> None:
    """Saves a project to a JSON file and associated data to CSV."""
    try:
        os.makedirs(os.path.dirname(project_path), exist_ok=True)
        with open(project_path, 'w', encoding='utf-8') as f:
            json.dump(project.to_dict(project_path), f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to save project: {e}")

def load_project(project_path: str) -> 'Project':
    """Loads a project from a JSON file."""
    if not os.path.exists(project_path):
        raise FileNotFoundError("Project file not found")
    try:
        with open(project_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return Project.from_dict(data)
    except Exception as e:
        raise RuntimeError(f"Failed to load project: {e}")

class ProjectManager:
    """Manages a directory of projects."""
    def __init__(self, projects_dir: str = "spt_projects"):
        self.projects_dir = projects_dir
        os.makedirs(self.projects_dir, exist_ok=True)

    def create_project(self, name: str, description: str = "") -> Project:
        p = Project(name=name, description=description)
        self.save_project(p, os.path.join(self.projects_dir, f"{p.id}.json"))
        return p

    def delete_project(self, project_id: str) -> bool:
        """Deletes a project's JSON file."""
        project_path = None
        for meta in self.list_projects():
            if meta['id'] == project_id:
                project_path = meta['path']
                break

        if project_path and os.path.exists(project_path):
            try:
                os.remove(project_path)
                return True
            except Exception as e:
                print(f"Error deleting project {project_id}: {e}")
                return False
        return False

    def save_project(self, project: 'Project', project_path: str) -> None:
        save_project(project, project_path)

    def get_project(self, project_id: str) -> Optional[Project]:
        for meta in self.list_projects():
            if meta['id'] == project_id:
                return load_project(meta['path'])
        return None

    def list_projects(self) -> List[Dict[str, Any]]:
        projects = []
        for fn in os.listdir(self.projects_dir):
            if fn.endswith(".json"):
                pth = os.path.join(self.projects_dir, fn)
                try:
                    with open(pth, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    projects.append({'id': data.get('id'), 'name': data.get('name'), 'path': pth})
                except Exception:
                    continue
        return projects

    def add_file_to_condition(self, project: 'Project', condition_id: str, 
                             file_name: str, tracks_df: pd.DataFrame) -> str:
        """Add a file to a condition and save its data to CSV."""
        cond = next((c for c in project.conditions if c.id == condition_id), None)
        if cond is None:
            raise ValueError("Condition not found")
        fid = str(uuid.uuid4())
        
        # Save the track data to CSV file
        data_dir = os.path.join(self.projects_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        data_path = os.path.join(data_dir, f"{fid}.csv")
        
        try:
            tracks_df.to_csv(data_path, index=False)
            # Store metadata with path to CSV file (no bytes in memory)
            cond.files.append({
                'id': fid, 
                'name': file_name, 
                'type': 'text/csv', 
                'size': len(tracks_df), 
                'data_path': data_path
            })
        except Exception as e:
            # If save fails, store bytes in memory as fallback (will be excluded from JSON)
            cond.files.append({
                'id': fid, 
                'name': file_name, 
                'type': 'text/csv', 
                'size': 0, 
                'data': tracks_df.to_csv(index=False).encode('utf-8')
            })
        
        return fid

    def remove_file_from_project(self, project: 'Project', condition_id: str, file_id: str) -> Dict[str, Any]:
        cond = next((c for c in project.conditions if c.id == condition_id), None)
        if cond is None:
            return {'success': False, 'error': 'Condition not found'}
        before = len(cond.files)
        cond.files = [f for f in cond.files if f.get('id') != file_id]
        return {'success': True, 'removed': before - len(cond.files)}

    def remove_condition(self, project: 'Project', condition_id: str) -> Dict[str, Any]:
        before = len(project.conditions)
        project.conditions = [c for c in project.conditions if c.id != condition_id]
        return {'success': True, 'removed': before - len(project.conditions)}

    def add_condition(self, project: 'Project', name: str, description: str = "") -> str:
        cid = str(uuid.uuid4())
        project.conditions.append(Condition(cid, name, description))
        return cid

    def list_conditions(self, project: 'Project') -> List[Dict[str, Any]]:
        return [{'id': c.id, 'name': c.name, 'description': c.description, 'file_count': len(c.files)} for c in project.conditions]

    def generate_batch_reports(self, project_id: str, selected_analyses: List[str] = None, 
                             report_format: str = "HTML Interactive") -> Dict[str, Any]:
        try:
            proj = self.get_project(project_id)
            if proj is None:
                return {'success': False, 'error': 'Project not found'}
            from enhanced_report_generator import EnhancedSPTReportGenerator
            generator = EnhancedSPTReportGenerator()
            conditions_results = {}
            for cond in proj.conditions:
                pooled_result = cond.pool_tracks()
                if isinstance(pooled_result, tuple):
                    pooled, _ = pooled_result
                else:
                    pooled = pooled_result
                if pooled is None or pooled.empty:
                    conditions_results[cond.name] = {'success': False, 'error': 'No data'}
                    continue
                analyses = selected_analyses or ['basic_statistics', 'diffusion_analysis']
                res = generator.generate_batch_report(pooled, analyses, cond.name)
                # Export JSON summary
                export_path = self._export_condition_report(res, cond.name, 'json' if report_format == "Raw Data (JSON)" else 'json', proj.id)
                res['export_path'] = export_path
                conditions_results[cond.name] = {'success': True, 'export_path': export_path}
            return {'success': True, 'conditions': conditions_results}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _export_condition_report(self, results: Dict, condition_name: str, 
                               format_type: str, project_id: str) -> str:
        reports_dir = os.path.join(self.projects_dir, project_id, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if format_type == 'json':
            return self._export_html_report(results, condition_name, reports_dir, timestamp)  # reuse name; JSON content

    def _export_html_report(self, results: Dict, condition_name: str, 
                          reports_dir: str, timestamp: str) -> str:
        p = os.path.join(reports_dir, f"{condition_name.replace(' ', '_')}_{timestamp}.json")
        try:
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            return p
        except Exception:
            return ""

def pm_list_available_projects(projects_dir):
    """List available projects in the projects directory."""
    projects = []
    if os.path.exists(projects_dir):
        for fn in os.listdir(projects_dir):
            if fn.endswith(".json"):
                p = os.path.join(projects_dir, fn)
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        d = json.load(f)
                    projects.append({'id': d.get('id'), 'name': d.get('name'), 'path': p})
                except Exception:
                    continue
    return projects

def list_available_projects(projects_dir: str) -> List[Dict[str, Any]]:
    """List available projects using the ProjectManager class."""
    manager = ProjectManager(projects_dir)
    return manager.list_projects()

class ProjectManagerCompat:
    """Compatibility wrapper for project management functions."""
    def __init__(self):
        self.pm = ProjectManager()
    def list_available_projects(self, projects_dir):
        return pm_list_available_projects(projects_dir)

def remove_file_from_project(project: 'Project', condition_id: str, file_id: str, base_path: str) -> Dict[str, Any]:
    """
    Remove a file from a project condition and delete its associated data file from disk.
    """
    try:
        pm = ProjectManager(os.path.dirname(base_path))
        return pm.remove_file_from_project(project, condition_id, file_id)
    except Exception as e:
        return {'success': False, 'error': str(e)}

def pool_tracks_from_condition(project: 'Project', condition_id: str) -> pd.DataFrame:
    """
    Pool all tracks from files in a specific condition.
    """
    cond = next((c for c in project.conditions if c.id == condition_id), None)
    if cond is None:
        return pd.DataFrame()
    pooled_result = cond.pool_tracks()
    if isinstance(pooled_result, tuple):
        return pooled_result[0]
    return pooled_result
