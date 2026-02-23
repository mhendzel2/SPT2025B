#!/usr/bin/env python3
"""
Test script to create a sample project and test batch processing functionality.
"""
import sys
import os
import pandas as pd
import numpy as np
import datetime
sys.path.append('.')

def create_sample_tracking_data():
    """Create sample tracking data for testing."""
    np.random.seed(42)
    
    tracks = []
    for track_id in range(1, 6):  # 5 tracks
        n_points = np.random.randint(20, 50)
        
        x_start = np.random.uniform(0, 100)
        y_start = np.random.uniform(0, 100)
        
        x_positions = [x_start]
        y_positions = [y_start]
        
        for i in range(1, n_points):
            x_positions.append(x_positions[-1] + np.random.normal(0, 2))
            y_positions.append(y_positions[-1] + np.random.normal(0, 2))
        
        for frame, (x, y) in enumerate(zip(x_positions, y_positions)):
            tracks.append({
                'track_id': track_id,
                'frame': frame,
                'x': x,
                'y': y,
                'z': 0.0,
                'quality': np.random.uniform(0.8, 1.0)
            })
    
    return pd.DataFrame(tracks)

def test_project_creation_and_batch_processing():
    """Test creating a project and running batch processing."""
    try:
        from project_management import ProjectManager, Project, Condition, FileObject
        
        pm = ProjectManager()
        
        project = Project(
            name="Test Batch Processing Project",
            description="Sample project for testing batch processing functionality"
        )
        
        sample_data = create_sample_tracking_data()
        
        condition1_data = sample_data[sample_data['track_id'] <= 2].copy()
        condition2_data = sample_data[sample_data['track_id'] > 2].copy()
        
        condition1 = Condition(
            cond_id="control_001",
            name="Control",
            description="Control condition"
        )
        
        condition2 = Condition(
            cond_id="treatment_001", 
            name="Treatment",
            description="Treatment condition"
        )
        
        file1 = FileObject(
            file_id="control_file_001",
            file_name="control_tracks.csv",
            tracks_df=condition1_data,
            upload_date=str(datetime.datetime.now())
        )
        
        file2 = FileObject(
            file_id="treatment_file_001",
            file_name="treatment_tracks.csv", 
            tracks_df=condition2_data,
            upload_date=str(datetime.datetime.now())
        )
        
        condition1.files[file1.id] = file1
        condition2.files[file2.id] = file2
        
        project.conditions[condition1.id] = condition1
        project.conditions[condition2.id] = condition2
        
        project.save(pm.projects_dir)
        print(f"✅ Created test project with ID: {project.id}")
        
        selected_analyses = ['basic_statistics', 'diffusion_analysis']
        results = pm.generate_batch_reports(
            project.id, selected_analyses, "HTML Interactive"
        )
        
        if results['success']:
            print("✅ Batch processing completed successfully!")
            for condition_name, condition_result in results['conditions'].items():
                if condition_result.get('success', True):
                    print(f"  ✅ {condition_name}: Report generated")
                    if 'export_path' in condition_result:
                        print(f"    📄 Exported to: {condition_result['export_path']}")
                else:
                    print(f"  ❌ {condition_name}: {condition_result.get('error', 'Unknown error')}")
        else:
            print("❌ Batch processing failed")
            
        return results
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing project creation and batch processing...")
    results = test_project_creation_and_batch_processing()
    
    if results and results.get('success'):
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print("❌ Tests failed")
        sys.exit(1)
