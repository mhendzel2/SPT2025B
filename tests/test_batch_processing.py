#!/usr/bin/env python3
"""
Test script for batch processing functionality.
"""
import sys
import os
sys.path.append('.')

def test_batch_processing_imports():
    """Test that all batch processing modules can be imported."""
    try:
        from project_management import ProjectManager
        from enhanced_project_management import show_batch_processing_interface
        from enhanced_report_generator import EnhancedSPTReportGenerator
        print("✅ All batch processing imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_project_manager_batch_methods():
    """Test that ProjectManager has batch processing methods."""
    try:
        from project_management import ProjectManager
        pm = ProjectManager()
        
        assert hasattr(pm, 'generate_batch_reports'), "generate_batch_reports method missing"
        assert hasattr(pm, '_export_html_report'), "_export_html_report method missing"
        assert hasattr(pm, '_export_pdf_report'), "_export_pdf_report method missing"
        
        print("✅ ProjectManager batch processing methods available")
        return True
    except Exception as e:
        print(f"❌ ProjectManager test failed: {e}")
        return False

def test_enhanced_report_generator():
    """Test that EnhancedSPTReportGenerator has batch report method."""
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        generator = EnhancedSPTReportGenerator()
        
        assert hasattr(generator, 'generate_batch_report'), "generate_batch_report method missing"
        
        print("✅ EnhancedSPTReportGenerator batch processing method available")
        return True
    except Exception as e:
        print(f"❌ EnhancedSPTReportGenerator test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing batch processing functionality...")
    
    tests = [
        test_batch_processing_imports,
        test_project_manager_batch_methods,
        test_enhanced_report_generator
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n{passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All batch processing tests passed!")
        sys.exit(0)
    else:
        print("❌ Some tests failed")
        sys.exit(1)
