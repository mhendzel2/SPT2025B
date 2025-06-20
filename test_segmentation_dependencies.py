"""
Test script to verify segmentation dependencies are properly installed and imported.
"""

def test_segmentation_imports():
    """Test that segmentation dependencies can be imported correctly."""
    results = {}
    
    try:
        from segment_anything import sam_model_registry, SamPredictor
        results['segment_anything'] = True
        print('✓ SAM dependencies available')
    except ImportError as e:
        results['segment_anything'] = False
        print(f'✗ SAM import failed: {e}')
    
    try:
        from cellpose import models
        CellposeModel = models.CellposeModel
        results['cellpose'] = True
        print('✓ Cellpose dependencies available')
    except ImportError as e:
        results['cellpose'] = False
        print(f'✗ Cellpose import failed: {e}')
    
    try:
        from channel_manager import channel_manager
        results['channel_manager'] = True
        print('✓ Channel manager available')
        
        is_valid, msg = channel_manager.validate_name('Test Channel')
        print(f'✓ Channel validation works: {is_valid}, {msg}')
        
        long_name = 'This is a very long channel name that exceeds limit'
        is_valid, msg = channel_manager.validate_name(long_name)
        if not is_valid and '24 characters' in msg:
            print('✓ 24-character limit validation works')
        else:
            print('✗ 24-character limit validation failed')
            
    except Exception as e:
        results['channel_manager'] = False
        print(f'✗ Channel manager failed: {e}')
    
    return results

if __name__ == "__main__":
    print("Testing segmentation dependencies and channel naming system...")
    results = test_segmentation_imports()
    
    success_count = sum(results.values())
    total_count = len(results)
    
    print(f"\nTest Results: {success_count}/{total_count} components working")
    
    if success_count == total_count:
        print("✅ All segmentation dependencies and channel naming system working correctly!")
    else:
        print("❌ Some components failed - check error messages above")
