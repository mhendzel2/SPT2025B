#!/usr/bin/env python3
"""
Test script to verify PDF export functionality.
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
    for track_id in range(1, 4):  # 3 tracks
        n_points = np.random.randint(15, 25)
        
        x_start = np.random.uniform(0, 50)
        y_start = np.random.uniform(0, 50)
        
        x_positions = [x_start]
        y_positions = [y_start]
        
        for i in range(1, n_points):
            x_positions.append(x_positions[-1] + np.random.normal(0, 1))
            y_positions.append(y_positions[-1] + np.random.normal(0, 1))
        
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

def test_pdf_export():
    """Test PDF export functionality."""
    try:
        from project_management import ProjectManager, Project, Condition, FileObject
        
        pm = ProjectManager()
        
        project = Project(
            name="PDF Export Test Project",
            description="Testing PDF export functionality"
        )
        
        sample_data = create_sample_tracking_data()
        
        condition = Condition(
            cond_id="pdf_test_001",
            name="PDF Test Condition",
            description="Test condition for PDF export"
        )
        
        file_obj = FileObject(
            file_id="pdf_test_file_001",
            file_name="pdf_test_tracks.csv",
            tracks_df=sample_data,
            upload_date=str(datetime.datetime.now())
        )
        
        condition.files[file_obj.id] = file_obj
        project.conditions[condition.id] = condition
        
        project.save(pm.projects_dir)
        print(f"✅ Created PDF test project with ID: {project.id}")
        
        selected_analyses = ['basic_statistics']
        results = pm.generate_batch_reports(
            project.id, selected_analyses, "PDF Report"
        )
        
        if results['success']:
            print("✅ PDF batch processing completed successfully!")
            for condition_name, condition_result in results['conditions'].items():
                if condition_result.get('success', True):
                    print(f"  ✅ {condition_name}: PDF report generated")
                    if 'export_path' in condition_result:
                        export_path = condition_result['export_path']
                        print(f"    📄 Exported to: {export_path}")
                        
                        if os.path.exists(export_path):
                            file_size = os.path.getsize(export_path)
                            print(f"    📊 PDF file size: {file_size} bytes")
                            if file_size > 0:
                                print("    ✅ PDF file created successfully with content")
                            else:
                                print("    ❌ PDF file is empty")
                        else:
                            print("    ❌ PDF file not found")
                else:
                    print(f"  ❌ {condition_name}: {condition_result.get('error', 'Unknown error')}")
        else:
            print("❌ PDF batch processing failed")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
        return results
        
    except Exception as e:
        print(f"❌ PDF export test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_html_content():
    """Check the content of generated HTML files."""
    try:
        projects_dir = "./spt_projects"
        if not os.path.exists(projects_dir):
            print("❌ No projects directory found")
            return False
            
        project_dirs = [d for d in os.listdir(projects_dir) if os.path.isdir(os.path.join(projects_dir, d))]
        if not project_dirs:
            print("❌ No project directories found")
            return False
            
        latest_project = max(project_dirs, key=lambda d: os.path.getctime(os.path.join(projects_dir, d)))
        reports_dir = os.path.join(projects_dir, latest_project, "reports")
        
        if not os.path.exists(reports_dir):
            print("❌ No reports directory found")
            return False
            
        html_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
        if not html_files:
            print("❌ No HTML files found")
            return False
            
        print(f"✅ Found {len(html_files)} HTML report files")
        
        html_file = os.path.join(reports_dir, html_files[0])
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        checks = [
            ('HTML structure', '<html>' in content and '</html>' in content),
            ('Title present', 'Single Particle Tracking Analysis Report' in content),
            ('Condition name', any(cond in content for cond in ['Control', 'Treatment'])),
            ('CSS styling', '<style>' in content),
            ('Plotly integration', 'plotly' in content.lower())
        ]
        
        print("📋 HTML Content Validation:")
        all_passed = True
        for check_name, passed in checks:
            status = "✅" if passed else "❌"
            print(f"  {status} {check_name}")
            if not passed:
                all_passed = False
                
        return all_passed
        
    except Exception as e:
        print(f"❌ HTML content check failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing PDF export and HTML content validation...")
    
    html_ok = check_html_content()
    
    pdf_results = test_pdf_export()
    
    if html_ok and pdf_results and pdf_results.get('success'):
        print("🎉 All export functionality tests passed!")
        sys.exit(0)
    else:
        print("❌ Some export tests failed")
        sys.exit(1)
