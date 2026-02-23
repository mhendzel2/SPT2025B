"""
Test script for the reptation model implementation.
"""

import sys
import pandas as pd
import numpy as np

def test_reptation_model():
    """Test Reptation model fitting."""
    print("\n" + "="*70)
    print("TESTING REPTATION MODEL")
    print("="*70)
    
    try:
        from biophysical_models import PolymerPhysicsModel
        
        # Generate synthetic subdiffusive data (α ~ 0.3 for reptation)
        n_points = 50
        lag_times = np.linspace(0.1, 5, n_points)
        alpha_reptation = 0.35  # Early reptation regime
        K = 0.05
        msd = K * (lag_times ** alpha_reptation)
        # Add some noise
        msd = msd * (1 + np.random.randn(n_points) * 0.05)
        
        msd_df = pd.DataFrame({
            'lag_time': lag_times,
            'msd': msd
        })
        
        print("\n1. Initializing Reptation model...")
        model = PolymerPhysicsModel(
            msd_data=msd_df,
            pixel_size=0.1,
            frame_interval=0.1
        )
        print(f"   ✓ Model initialized with {len(msd_df)} MSD points")
        
        print("\n2. Fitting Reptation model...")
        result = model.fit_reptation_model(
            temperature=300.0,
            tube_diameter=100e-9,  # 100 nm
            contour_length=1000e-9  # 1 μm
        )
        
        if not result.get('success'):
            print(f"   ❌ Reptation fit failed: {result.get('error')}")
            return False
        
        print(f"   ✓ Reptation model fitted successfully")
        
        params = result.get('parameters', {})
        print(f"\n   Results:")
        print(f"   - Alpha: {params.get('alpha', 'N/A'):.3f}")
        print(f"   - K_reptation: {params.get('K_reptation', 'N/A'):.6f}")
        print(f"   - Regime: {params.get('regime', 'N/A')}")
        print(f"   - Regime Phase: {params.get('regime_phase', 'N/A')}")
        print(f"   - Tube Diameter (estimated): {params.get('tube_diameter_estimated', 0)*1e9:.1f} nm")
        if params.get('reptation_time'):
            print(f"   - Reptation Time: {params.get('reptation_time'):.3e} s")
        
        # Check interpretation
        if 'interpretation' in result:
            print(f"\n   Interpretation: {result['interpretation']}")
        
        # Verify fitted curve exists
        if 'fitted_curve' in result:
            fit = result['fitted_curve']
            print(f"\n   ✓ Fitted curve generated with {len(fit['lag_time'])} points")
        else:
            print(f"\n   ⚠ Warning: No fitted curve in results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_zimm_model():
    """Test Zimm model fitting."""
    print("\n" + "="*70)
    print("TESTING ZIMM MODEL")
    print("="*70)
    
    try:
        from biophysical_models import PolymerPhysicsModel
        
        # Generate synthetic Zimm data (α = 2/3)
        n_points = 50
        lag_times = np.linspace(0.1, 5, n_points)
        alpha_zimm = 2.0/3.0
        K = 0.1
        msd = K * (lag_times ** alpha_zimm)
        # Add some noise
        msd = msd * (1 + np.random.randn(n_points) * 0.05)
        
        msd_df = pd.DataFrame({
            'lag_time': lag_times,
            'msd': msd
        })
        
        print("\n1. Initializing Zimm model...")
        model = PolymerPhysicsModel(
            msd_data=msd_df,
            pixel_size=0.1,
            frame_interval=0.1
        )
        print(f"   ✓ Model initialized with {len(msd_df)} MSD points")
        
        print("\n2. Fitting Zimm model...")
        result = model.fit_zimm_model(
            fit_alpha=True,
            solvent_viscosity=0.001,  # Water at 25°C
            hydrodynamic_radius=5e-9,  # 5 nm
            temperature=300.0
        )
        
        if not result.get('success'):
            print(f"   ❌ Zimm fit failed: {result.get('error')}")
            return False
        
        print(f"   ✓ Zimm model fitted successfully")
        
        params = result.get('parameters', {})
        print(f"\n   Results:")
        print(f"   - Alpha: {params.get('alpha', 'N/A'):.3f}")
        print(f"   - K_zimm: {params.get('K_zimm', 'N/A'):.6f}")
        if 'D_zimm_theory' in params:
            print(f"   - D (Zimm Theory): {params['D_zimm_theory']:.3e} m²/s")
        
        # Verify fitted curve exists
        if 'fitted_curve' in result:
            fit = result['fitted_curve']
            print(f"\n   ✓ Fitted curve generated with {len(fit['lag_time'])} points")
        else:
            print(f"\n   ⚠ Warning: No fitted curve in results")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔬 Testing Polymer Physics Models (Reptation & Zimm)" + "\n")
    
    # Run tests
    test1_passed = test_reptation_model()
    test2_passed = test_zimm_model()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Reptation Model:  {'✓ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Zimm Model:       {'✓ PASSED' if test2_passed else '❌ FAILED'}")
    
    if all([test1_passed, test2_passed]):
        print("\n✅ All polymer physics model tests PASSED!")
        sys.exit(0)
    else:
        print("\n❌ Some tests FAILED. Check the output above.")
        sys.exit(1)
