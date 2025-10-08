"""
Test script to verify session state initialization fixes
Tests for Bug #8 (image_data) and Bug #9 (mask_images)
"""

def test_session_state_initialization():
    """Test that all required session state variables are initialized."""
    import streamlit as st
    from utils import initialize_session_state
    
    print("=" * 60)
    print("Testing Session State Initialization")
    print("=" * 60)
    
    # Initialize session state
    initialize_session_state()
    
    # Test image_data
    print("\n1. Testing image_data initialization:")
    if 'image_data' in st.session_state:
        print("   ✅ 'image_data' key exists in session state")
        print(f"   ✅ Initial value: {st.session_state.image_data}")
    else:
        print("   ❌ 'image_data' key NOT found in session state")
        return False
    
    # Test mask_images
    print("\n2. Testing mask_images initialization:")
    if 'mask_images' in st.session_state:
        print("   ✅ 'mask_images' key exists in session state")
        print(f"   ✅ Initial value: {st.session_state.mask_images}")
    else:
        print("   ❌ 'mask_images' key NOT found in session state")
        return False
    
    # Test safe access patterns
    print("\n3. Testing safe access patterns:")
    try:
        # Pattern from app.py line 3757
        if 'image_data' not in st.session_state or st.session_state.image_data is None:
            print("   ✅ Safe image_data check works (currently None)")
        
        # Pattern from app.py line 1796
        if 'mask_images' not in st.session_state or st.session_state.mask_images is None:
            print("   ✅ Safe mask_images check works (currently None)")
        
        return True
    except KeyError as e:
        print(f"   ❌ KeyError during safe access: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        return False

def test_other_initializations():
    """Test other critical session state variables."""
    import streamlit as st
    
    print("\n4. Testing other session state variables:")
    
    required_vars = [
        'tracks_data',
        'track_statistics',
        'analysis_results',
        'recent_analyses',
        'pixel_size',
        'frame_interval',
        'active_page',
        'available_masks',
        'mask_metadata'
    ]
    
    all_present = True
    for var in required_vars:
        if var in st.session_state:
            print(f"   ✅ '{var}' initialized")
        else:
            print(f"   ❌ '{var}' NOT initialized")
            all_present = False
    
    return all_present

if __name__ == "__main__":
    try:
        result1 = test_session_state_initialization()
        result2 = test_other_initializations()
        
        print("\n" + "=" * 60)
        if result1 and result2:
            print("🎉 ALL TESTS PASSED!")
            print("=" * 60)
            print("\nSession state properly initialized.")
            print("Navigation fixes for both Tracking and Image Processing")
            print("pages should now work correctly.")
        else:
            print("⚠️ SOME TESTS FAILED")
            print("=" * 60)
            print("\nPlease review the output above for details.")
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ TEST ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
