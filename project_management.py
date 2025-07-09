"""
Project Management for SPT Analysis application.
Handles organizing and comparing multiple tracking files across different conditions.
Enhanced with persistent file-based storage and robust data management.
"""

import os
import json
import datetime
import uuid
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy import stats

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
            track_data_file = f"{self.id}_tracks.csv"
            track_data_path = os.path.join(save_path, track_data_file)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(track_data_path), exist_ok=True)
            
            # Save track data to CSV
            try:
                self.track_data.to_csv(track_data_path, index=False)
            except Exception as e:
                print(f"Warning: Could not save track data for {self.file_name}: {e}")
                track_data_file = None
        
        return {
            "id": self.id,
            "file_name": self.file_name,
            "track_data_file": track_data_file,
            "upload_date": self.upload_date,
        }

    @classmethod
    def from_dict(cls, data: Dict, project_path: str = None):
        tracks_df = pd.DataFrame()
        if data.get('track_data_file') and project_path:
            track_data_path = os.path.join(os.path.dirname(project_path), data['track_data_file'])
            if os.path.exists(track_data_path):
                try:
                    tracks_df = pd.read_csv(track_data_path)
                except Exception as e:
                    print(f"Warning: Could not load track data from {track_data_path}: {e}")
                    tracks_df = pd.DataFrame()
        return cls(data['id'], data['file_name'], tracks_df, data['upload_date'])

class Condition:
    """Represents an experimental condition."""
    def __init__(self, cond_id: str, name: str, description: str = ""):
        self.id = cond_id
        self.name = name
        self.description = description
        self.files: Dict[str, FileObject] = {}

    def to_dict(self, save_path: str = None) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "files": {fid: f.to_dict(save_path) for fid, f in self.files.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict, project_path: str = None):
        condition = cls(data.get("id"), data.get("name", ""), data.get("description", ""))
        for fid, fdata in data.get("files", {}).items():
            condition.files[fid] = FileObject.from_dict(fdata, project_path)
        return condition

    def pool_tracks(self) -> pd.DataFrame:
        """Pool all tracks from all files in this condition."""
        all_tracks = []
        for file_obj in self.files.values():
            if file_obj.track_data is not None and not file_obj.track_data.empty:
                tracks_copy = file_obj.track_data.copy()
                tracks_copy['file_id'] = file_obj.id
                tracks_copy['file_name'] = file_obj.file_name
                all_tracks.append(tracks_copy)
        
        if all_tracks:
            pooled = pd.concat(all_tracks, ignore_index=True)
            return pooled
        else:
            return pd.DataFrame()

    def compare_conditions(self, metric: str = "diffusion_coefficient", 
                          test_type: str = "auto", alpha: float = 0.05) -> Dict[str, Any]:
        """Compare this condition against itself or provide summary statistics."""
        pooled_data = self.pool_tracks()
        
        if pooled_data.empty:
            return {
                'success': False,
                'error': 'No data available for comparison',
                'condition_name': self.name
            }
        
        # Calculate basic statistics
        if metric in pooled_data.columns:
            values = pooled_data[metric].dropna()
            
            if len(values) == 0:
                return {
                    'success': False,
                    'error': f'No valid values found for metric {metric}',
                    'condition_name': self.name
                }
            
            statistics = {
                'mean': float(values.mean()),
                'median': float(values.median()),
                'std': float(values.std()),
                'min': float(values.min()),
                'max': float(values.max()),
                'n_values': len(values),
                'condition_name': self.name
            }
            
            return {
                'success': True,
                'statistics': statistics,
                'metric': metric
            }
        else:
            return {
                'success': False,
                'error': f'Metric {metric} not found in data',
                'condition_name': self.name,
                'available_columns': list(pooled_data.columns)
            }

class Project:
    """Represents a full SPT project, now with a save method."""
    def __init__(self, name: str = "New Project", description: str = ""):
        self.id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.creation_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_modified = self.creation_date
        self.conditions: Dict[str, Condition] = {}

    def save(self, base_path: str):
        """Saves the project to its designated file path."""
        project_dir = os.path.join(base_path, self.id)
        os.makedirs(project_dir, exist_ok=True)
        project_file_path = os.path.join(project_dir, "project.json")
        save_project(self, project_file_path)

    def update_last_modified(self):
        self.last_modified = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def to_dict(self, save_path: str = None) -> Dict:
        return {
            "id": self.id, 
            "name": self.name, 
            "description": self.description,
            "creation_date": self.creation_date, 
            "last_modified": self.last_modified,
            "conditions": {cid: c.to_dict(save_path) for cid, c in self.conditions.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict, project_path: str = None) -> 'Project':
        project = cls(data.get("name", "Unknown"), data.get("description", ""))
        project.id = data.get("id", str(uuid.uuid4()))
        project.creation_date = data.get("creation_date", project.creation_date)
        project.last_modified = data.get("last_modified", project.last_modified)
        for cid, cdata in data.get("conditions", {}).items():
            project.conditions[cid] = Condition.from_dict(cdata, project_path)
        return project

def save_project(project: 'Project', project_path: str) -> None:
    """Saves a project to a JSON file and associated data to CSV."""
    try:
        project_dir = os.path.dirname(project_path)
        os.makedirs(project_dir, exist_ok=True)
        
        # Pass the directory where the csv should be saved
        project_data = project.to_dict(save_path=project_dir)
        
        with open(project_path, 'w') as f:
            json.dump(project_data, f, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save project to {project_path}: {str(e)}")

def load_project(project_path: str) -> 'Project':
    """Loads a project from a JSON file."""
    if not os.path.exists(project_path):
        raise FileNotFoundError(f"Project file not found: {project_path}")
    
    try:
        with open(project_path, 'r') as f:
            project_data = json.load(f)
        return Project.from_dict(project_data, project_path)
    except Exception as e:
        raise ValueError(f"Failed to load project from {project_path}: {str(e)}")

class ProjectManager:
    """
    Manages a directory of projects, ensuring data is saved to and loaded from files.
    """
    def __init__(self, projects_dir: str = "spt_projects"):
        self.projects_dir = projects_dir
        if not os.path.exists(self.projects_dir):
            os.makedirs(self.projects_dir)

    def create_project(self, name: str, description: str = "") -> Project:
        """Creates a new project and saves it immediately."""
        project = Project(name=name, description=description)
        project.save(self.projects_dir)
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Loads a project from the projects directory."""
        project_path = os.path.join(self.projects_dir, project_id, "project.json")
        if os.path.exists(project_path):
            try:
                return load_project(project_path)
            except Exception as e:
                print(f"Error loading project {project_id}: {e}")
                return None
        return None

    def list_projects(self) -> List[Dict[str, Any]]:
        """Lists all available projects by reading their JSON files."""
        projects = []
        if not os.path.exists(self.projects_dir):
            return projects
            
        for project_id in os.listdir(self.projects_dir):
            project_path = os.path.join(self.projects_dir, project_id, "project.json")
            if os.path.exists(project_path):
                try:
                    with open(project_path, 'r') as f:
                        data = json.load(f)
                        projects.append({
                            'id': data.get('id', project_id),
                            'name': data.get('name', 'Untitled'),
                            'description': data.get('description', ''),
                            'creation_date': data.get('creation_date', ''),
                            'last_modified': data.get('last_modified', '1970-01-01 00:00:00')
                        })
                except Exception as e:
                    print(f"Error reading project {project_id}: {e}")
                    continue
        return sorted(projects, key=lambda p: p['last_modified'], reverse=True)

    def add_file_to_condition(self, project: 'Project', condition_id: str, 
                             file_name: str, tracks_df: pd.DataFrame) -> str:
        """Add a file to a project condition."""
        if condition_id not in project.conditions:
            raise ValueError(f"Condition {condition_id} not found")
        
        file_id = str(uuid.uuid4())
        upload_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        file_obj = FileObject(file_id, file_name, tracks_df, upload_date)
        project.conditions[condition_id].files[file_id] = file_obj
        
        project.update_last_modified()
        project.save(self.projects_dir)
        
        return file_id

    def remove_file_from_project(self, project: 'Project', condition_id: str, file_id: str) -> Dict[str, Any]:
        """
        Remove a file from a project condition and delete its associated data file from disk.
        """
        try:
            if condition_id not in project.conditions:
                return {"success": False, "error": f"Condition {condition_id} not found"}
            
            condition = project.conditions[condition_id]
            if file_id not in condition.files:
                return {"success": False, "error": f"File {file_id} not found in condition"}

            file_obj = condition.files[file_id]
            file_name = file_obj.file_name
            
            # Construct the full path to the data file and delete it
            project_dir = os.path.join(self.projects_dir, project.id)
            file_dict = file_obj.to_dict()
            track_data_filename = file_dict.get('track_data_file')
            
            if track_data_filename:
                track_data_path = os.path.join(project_dir, track_data_filename)
                if os.path.exists(track_data_path):
                    os.remove(track_data_path)

            del condition.files[file_id]
            project.update_last_modified()
            project.save(self.projects_dir)
            
            return {"success": True, "message": f"Removed file '{file_name}' from condition '{condition.name}'"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def remove_condition(self, project: 'Project', condition_id: str) -> Dict[str, Any]:
        """Remove a condition and all its associated files from disk."""
        try:
            if condition_id not in project.conditions:
                return {"success": False, "error": f"Condition {condition_id} not found"}
            
            condition = project.conditions[condition_id]
            condition_name = condition.name
            
            # Remove all files in the condition
            file_ids = list(condition.files.keys())
            for file_id in file_ids:
                result = self.remove_file_from_project(project, condition_id, file_id)
                if not result["success"]:
                    return result
            
            # Remove the condition itself
            del project.conditions[condition_id]
            project.update_last_modified()
            project.save(self.projects_dir)
            
            return {"success": True, "message": f"Removed condition '{condition_name}' and all associated files"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}

    def add_condition(self, project: 'Project', name: str, description: str = "") -> str:
        """Add a condition to a project."""
        try:
            condition_id = str(uuid.uuid4())
            condition = Condition(condition_id, name, description)
            project.conditions[condition_id] = condition
            project.update_last_modified()
            project.save(self.projects_dir)
            return condition_id
        except Exception as e:
            raise ValueError(f"Failed to add condition: {str(e)}")

    def list_conditions(self, project: 'Project') -> List[Dict[str, Any]]:
        """List all conditions in a project."""
        try:
            conditions = []
            for condition_id, condition in project.conditions.items():
                conditions.append({
                    'id': condition.id,
                    'name': condition.name,
                    'description': condition.description,
                    'files_count': len(condition.files)
                })
            return conditions
        except Exception:
            return []

    def generate_batch_reports(self, project_id: str, selected_analyses: List[str] = None, 
                             report_format: str = "HTML Interactive") -> Dict[str, Any]:
        """Generate automated reports for all conditions in a project."""
        project = self.get_project(project_id)
        if not project:
            return {'success': False, 'error': f'Project {project_id} not found'}
        
        from enhanced_report_generator import EnhancedSPTReportGenerator
        from analysis_manager import AnalysisManager
        
        generator = EnhancedSPTReportGenerator()
        analysis_manager = AnalysisManager()
        
        batch_results = {
            'project_name': project.name,
            'conditions': {},
            'summary': {},
            'success': True
        }
        
        if not selected_analyses:
            selected_analyses = ['basic_statistics', 'diffusion_analysis', 'motion_classification']
        
        for condition_id, condition in project.conditions.items():
            try:
                pooled_tracks = condition.pool_tracks()
                if pooled_tracks.empty:
                    batch_results['conditions'][condition.name] = {
                        'success': False, 'error': 'No track data available'
                    }
                    continue
                
                condition_results = generator.generate_batch_report(
                    pooled_tracks, selected_analyses, condition.name
                )
                
                if report_format in ["HTML Interactive", "PDF Report"]:
                    export_path = self._export_condition_report(
                        condition_results, condition.name, report_format, project.id
                    )
                    condition_results['export_path'] = export_path
                
                batch_results['conditions'][condition.name] = condition_results
                
            except Exception as e:
                batch_results['conditions'][condition.name] = {
                    'success': False, 'error': str(e)
                }
        
        batch_results['summary'] = self._generate_project_summary(batch_results['conditions'])
        
        return batch_results
    
    def _export_condition_report(self, results: Dict, condition_name: str, 
                               format_type: str, project_id: str) -> str:
        """Export condition report to HTML or PDF."""
        import os
        from datetime import datetime
        
        reports_dir = os.path.join(self.projects_dir, project_id, "reports")
        os.makedirs(reports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format_type == "HTML Interactive":
            return self._export_html_report(results, condition_name, reports_dir, timestamp)
        elif format_type == "PDF Report":
            return self._export_pdf_report(results, condition_name, reports_dir, timestamp)
        
        return ""
    
    def _export_html_report(self, results: Dict, condition_name: str, 
                          reports_dir: str, timestamp: str) -> str:
        """Export results as interactive HTML report."""
        import plotly.io as pio
        
        filename = f"{condition_name}_report_{timestamp}.html"
        filepath = os.path.join(reports_dir, filename)
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SPT Analysis Report - {condition_name}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
        .figure {{ margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Single Particle Tracking Analysis Report</h1>
        <h2>Condition: {condition_name}</h2>
        <p>Generated: {timestamp}</p>
    </div>
"""
        
        for analysis_key, result in results.get('analysis_results', {}).items():
            if result.get('success', True):
                html_content += f'<div class="section"><h3>{analysis_key.replace("_", " ").title()}</h3>'
                
                if 'summary' in result:
                    html_content += '<h4>Summary Statistics:</h4><ul>'
                    for key, value in result['summary'].items():
                        html_content += f'<li><strong>{key}:</strong> {value}</li>'
                    html_content += '</ul>'
                
                html_content += '</div>'
        
        for fig_key, figure in results.get('figures', {}).items():
            if figure:
                fig_html = pio.to_html(figure, include_plotlyjs=False, div_id=f"fig_{fig_key}")
                html_content += f'<div class="figure">{fig_html}</div>'
        
        html_content += '</body></html>'
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filepath
    
    def _export_pdf_report(self, results: Dict, condition_name: str, 
                         reports_dir: str, timestamp: str) -> str:
        """Export results as PDF report using matplotlib."""
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
        
        filename = f"{condition_name}_report_{timestamp}.pdf"
        filepath = os.path.join(reports_dir, filename)
        
        with PdfPages(filepath) as pdf:
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.text(0.5, 0.8, 'Single Particle Tracking Analysis Report', 
                    ha='center', va='center', fontsize=20, weight='bold')
            ax.text(0.5, 0.7, f'Condition: {condition_name}', 
                    ha='center', va='center', fontsize=16)
            ax.text(0.5, 0.6, f'Generated: {timestamp}', 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            for analysis_key, result in results.get('analysis_results', {}).items():
                if result.get('success', True) and 'summary' in result:
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.text(0.5, 0.9, analysis_key.replace('_', ' ').title(), 
                            ha='center', va='center', fontsize=16, weight='bold')
                    
                    y_pos = 0.8
                    for key, value in result['summary'].items():
                        ax.text(0.1, y_pos, f'{key}: {value}', 
                                ha='left', va='center', fontsize=12)
                        y_pos -= 0.05
                    
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        return filepath
    
    def _generate_project_summary(self, conditions_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics across all conditions."""
        summary = {
            'total_conditions': len(conditions_results),
            'successful_conditions': 0,
            'failed_conditions': 0,
            'total_analyses': 0
        }
        
        for condition_name, result in conditions_results.items():
            if result.get('success', True):
                summary['successful_conditions'] += 1
                if 'analysis_results' in result:
                    summary['total_analyses'] += len(result['analysis_results'])
            else:
                summary['failed_conditions'] += 1
        
        return summary

def pm_list_available_projects(projects_dir):
    """List available projects in the projects directory."""
    import os
    projects = []
    if os.path.exists(projects_dir):
        for item in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, item)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, "project_config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            import json
                            config = json.load(f)
                            projects.append({
                                'name': config.get('name', item),
                                'path': project_path,
                                'description': config.get('description', ''),
                                'created': config.get('created', ''),
                                'last_modified': config.get('last_modified', '')
                            })
                    except Exception:
                        pass
    return projects

def list_available_projects(projects_dir: str) -> List[Dict[str, Any]]:
    """List available projects using the ProjectManager class."""
    manager = ProjectManager(projects_dir)
    return manager.list_projects()

class ProjectManagerCompat:
    """Compatibility wrapper for project management functions."""
    
    def __init__(self):
        self.projects_dir = "./projects"
    
    def list_available_projects(self, projects_dir):
        """List available projects."""
        return pm_list_available_projects(projects_dir)

def remove_file_from_project(project: 'Project', condition_id: str, file_id: str, base_path: str) -> Dict[str, Any]:
    """
    Remove a file from a project condition and delete its associated data file from disk.
    
    Parameters
    ----------
    project : Project
        The project to remove file from
    condition_id : str
        ID of the condition containing the file
    file_id : str
        ID of the file to remove
    base_path : str
        The base directory where projects are stored, needed to locate the CSV file.
    """
    try:
        if condition_id not in project.conditions:
            return {"success": False, "error": f"Condition {condition_id} not found"}
        
        condition = project.conditions[condition_id]
        if file_id not in condition.files:
            return {"success": False, "error": f"File {file_id} not found in condition"}

        # Get file object before deleting its record
        file_obj = condition.files[file_id]
        file_name = file_obj.file_name
        
        # Delete the associated CSV file
        file_dict = file_obj.to_dict()
        track_data_filename = file_dict.get('track_data_file')
        
        if track_data_filename:
            # Construct the full path to the data file
            project_dir = os.path.join(base_path, project.id)
            track_data_path = os.path.join(project_dir, track_data_filename)
            if os.path.exists(track_data_path):
                os.remove(track_data_path)

        # Now, remove the file record from the project
        del condition.files[file_id]
        project.update_last_modified()
        project.save(base_path)
        
        return {
            "success": True,
            "message": f"Successfully removed file '{file_name}' and its data from condition '{condition.name}'"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def pool_tracks_from_condition(project: 'Project', condition_id: str) -> pd.DataFrame:
    """
    Pool all tracks from files in a specific condition using vectorized operations.
    Enhanced with conflict resolution for track IDs across multiple files.
    """
    if condition_id not in project.conditions:
        return pd.DataFrame()
    
    return project.conditions[condition_id].pool_tracks()
                    
                    ax.axis('off')
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
        
        return filepath
    
    def _generate_project_summary(self, conditions_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics across all conditions."""
        summary = {
            'total_conditions': len(conditions_results),
            'successful_conditions': 0,
            'failed_conditions': 0,
            'total_analyses': 0
        }
        
        for condition_name, result in conditions_results.items():
            if result.get('success', True):
                summary['successful_conditions'] += 1
                if 'analysis_results' in result:
                    summary['total_analyses'] += len(result['analysis_results'])
            else:
                summary['failed_conditions'] += 1
        
        return summary

def pm_list_available_projects(projects_dir):
    """List available projects in the projects directory."""
    import os
    projects = []
    if os.path.exists(projects_dir):
        for item in os.listdir(projects_dir):
            project_path = os.path.join(projects_dir, item)
            if os.path.isdir(project_path):
                config_path = os.path.join(project_path, "project_config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            import json
                            config = json.load(f)
                            projects.append({
                                'name': config.get('name', item),
                                'path': project_path,
                                'description': config.get('description', ''),
                                'created': config.get('created', ''),
                                'last_modified': config.get('last_modified', '')
                            })
                    except Exception:
                        pass
    return projects

def list_available_projects(projects_dir: str) -> List[Dict[str, Any]]:
    """List available projects using the ProjectManager class."""
    manager = ProjectManager(projects_dir)
    return manager.list_projects()

class ProjectManagerCompat:
    """Compatibility wrapper for project management functions."""
    
    def __init__(self):
        self.projects_dir = "./projects"
    
    def list_available_projects(self, projects_dir):
        """List available projects."""
        return pm_list_available_projects(projects_dir)

def remove_file_from_project(project: 'Project', condition_id: str, file_id: str, base_path: str) -> Dict[str, Any]:
    """
    Remove a file from a project condition and delete its associated data file from disk.
    
    Parameters
    ----------
    project : Project
        The project to remove file from
    condition_id : str
        ID of the condition containing the file
    file_id : str
        ID of the file to remove
    base_path : str
        The base directory where projects are stored, needed to locate the CSV file.
    """
    try:
        if condition_id not in project.conditions:
            return {"success": False, "error": f"Condition {condition_id} not found"}
        
        condition = project.conditions[condition_id]
        if file_id not in condition.files:
            return {"success": False, "error": f"File {file_id} not found in condition"}

        # Get file object before deleting its record
        file_obj = condition.files[file_id]
        file_name = file_obj.file_name
        
        # Delete the associated CSV file
        file_dict = file_obj.to_dict()
        track_data_filename = file_dict.get('track_data_file')
        
        if track_data_filename:
            # Construct the full path to the data file
            project_dir = os.path.join(base_path, project.id)
            track_data_path = os.path.join(project_dir, track_data_filename)
            if os.path.exists(track_data_path):
                os.remove(track_data_path)

        # Now, remove the file record from the project
        del condition.files[file_id]
        project.update_last_modified()
        project.save(base_path)
        
        return {
            "success": True,
            "message": f"Successfully removed file '{file_name}' and its data from condition '{condition.name}'"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def pool_tracks_from_condition(project: 'Project', condition_id: str) -> pd.DataFrame:
    """
    Pool all tracks from files in a specific condition using vectorized operations.
    Enhanced with conflict resolution for track IDs across multiple files.
    """
    if condition_id not in project.conditions:
        return pd.DataFrame()
    
    return project.conditions[condition_id].pool_tracks()
