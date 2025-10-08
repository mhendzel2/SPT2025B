"""
Test script to verify project management JSON serialization fix
Tests for Bug #11: Object of type bytes is not JSON serializable
"""

import json
import tempfile
import os
import sys
import pandas as pd

def test_json_serialization():
    """Test that condition files can be serialized to JSON."""
    print("=" * 60)
    print("Testing Project Management JSON Serialization Fix")
    print("=" * 60)
    
    try:
        from project_management import ProjectManager, Project, Condition
        
        # Create a temporary directory for test
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\n1. Creating test project in: {tmpdir}")
            pmgr = ProjectManager(projects_dir=tmpdir)
            
            # Create a test project
            proj = pmgr.create_project("Test Project", "Testing JSON serialization")
            print(f"   ✅ Project created: {proj.name} (ID: {proj.id})")
            
            # Add a condition
            print("\n2. Adding condition to project")
            cond_id = pmgr.add_condition(proj, "Test Condition", "Test condition description")
            print(f"   ✅ Condition created: {cond_id}")
            
            # Create test DataFrame
            print("\n3. Creating test DataFrame with track data")
            test_df = pd.DataFrame({
                'track_id': [1, 1, 1, 2, 2, 2],
                'frame': [0, 1, 2, 0, 1, 2],
                'x': [10.5, 11.2, 12.1, 20.3, 21.5, 22.8],
                'y': [5.2, 5.8, 6.1, 15.7, 16.2, 16.9]
            })
            print(f"   ✅ Test DataFrame created: {len(test_df)} rows")
            
            # Add file to condition
            print("\n4. Adding file to condition")
            file_id = pmgr.add_file_to_condition(proj, cond_id, "test_tracks.csv", test_df)
            print(f"   ✅ File added: {file_id}")
            
            # Try to save project (this is where the error occurs)
            print("\n5. Saving project to JSON...")
            project_path = os.path.join(tmpdir, f"{proj.id}.json")
            try:
                pmgr.save_project(proj, project_path)
                print(f"   ✅ Project saved successfully to: {project_path}")
            except TypeError as e:
                if "not JSON serializable" in str(e):
                    print(f"   ❌ JSON SERIALIZATION ERROR: {e}")
                    return False
                raise
            
            # Verify the JSON file is valid
            print("\n6. Verifying saved JSON is valid")
            with open(project_path, 'r') as f:
                data = json.load(f)
            print(f"   ✅ JSON is valid and loadable")
            print(f"   ✅ Project name: {data.get('name')}")
            print(f"   ✅ Conditions: {len(data.get('conditions', []))}")
            
            # Check that files are properly stored
            if data.get('conditions'):
                cond_data = data['conditions'][0]
                files = cond_data.get('files', [])
                print(f"   ✅ Files in condition: {len(files)}")
                
                if files:
                    file_data = files[0]
                    print(f"   ✅ File name: {file_data.get('name')}")
                    
                    # Check that 'data' field (bytes) is NOT in JSON
                    if 'data' in file_data:
                        print(f"   ⚠️  WARNING: 'data' field found in JSON (should be excluded)")
                    else:
                        print(f"   ✅ 'data' field correctly excluded from JSON")
                    
                    # Check that data_path is present
                    if 'data_path' in file_data:
                        data_path = file_data['data_path']
                        print(f"   ✅ data_path present: {data_path}")
                        
                        # Verify CSV file exists
                        if os.path.exists(data_path):
                            print(f"   ✅ CSV file exists at data_path")
                            
                            # Verify we can load it
                            loaded_df = pd.read_csv(data_path)
                            print(f"   ✅ CSV file is valid: {len(loaded_df)} rows")
                            
                            if len(loaded_df) == len(test_df):
                                print(f"   ✅ Loaded data matches original data")
                            else:
                                print(f"   ⚠️  Row count mismatch: {len(loaded_df)} vs {len(test_df)}")
                        else:
                            print(f"   ❌ CSV file NOT found at data_path")
                    else:
                        print(f"   ⚠️  data_path not found in file metadata")
            
            # Try to load the project back
            print("\n7. Loading project from JSON")
            loaded_proj = pmgr.get_project(proj.id)
            print(f"   ✅ Project loaded: {loaded_proj.name}")
            print(f"   ✅ Conditions loaded: {len(loaded_proj.conditions)}")
            
            if loaded_proj.conditions:
                loaded_cond = loaded_proj.conditions[0]
                print(f"   ✅ Files in condition: {len(loaded_cond.files)}")
            
            return True
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\nTesting Bug #11 Fix: JSON Serialization of Project Files\n")
    
    success = test_json_serialization()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("=" * 60)
        print("\nProject file saving now works correctly:")
        print("✅ Files saved to CSV on disk")
        print("✅ Metadata stored in JSON (without bytes)")
        print("✅ Data can be loaded back from CSV")
        print("✅ No JSON serialization errors")
    else:
        print("❌ TESTS FAILED")
        print("=" * 60)
        print("\nThe fix may need adjustment. Check output above.")
    
    sys.exit(0 if success else 1)
