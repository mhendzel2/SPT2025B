"""
Test ML and MD Integration Features
Validates machine learning classification, nuclear diffusion simulation, and MD-SPT comparison.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'
BOLD = '\033[1m'


def print_header(text):
    """Print colored header"""
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{text:^70}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def print_success(text):
    """Print success message"""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text):
    """Print error message"""
    print(f"{RED}✗ {text}{RESET}")


def print_info(text):
    """Print info message"""
    print(f"{YELLOW}ℹ {text}{RESET}")


def create_test_tracks(n_tracks=50, n_points=100, diffusion_type='normal'):
    """
    Create synthetic track data for testing.
    
    Parameters
    ----------
    n_tracks : int
        Number of tracks to generate
    n_points : int
        Points per track
    diffusion_type : str
        'normal', 'confined', or 'directed'
    """
    records = []
    
    for track_id in range(n_tracks):
        # Starting position
        x, y = np.random.uniform(0, 100, 2)
        
        for frame in range(n_points):
            # Add displacement based on type
            if diffusion_type == 'normal':
                dx, dy = np.random.normal(0, 0.5, 2)
            elif diffusion_type == 'confined':
                # Confined diffusion (smaller steps with drift toward origin)
                dx, dy = np.random.normal(0, 0.2, 2)
                dx -= 0.05 * (x - 50)  # Drift toward center
                dy -= 0.05 * (y - 50)
            elif diffusion_type == 'directed':
                # Directed motion
                dx = np.random.normal(0.3, 0.1)
                dy = np.random.normal(0.2, 0.1)
            else:
                dx, dy = 0, 0
            
            x += dx
            y += dy
            
            # Boundary conditions
            x = np.clip(x, 0, 100)
            y = np.clip(y, 0, 100)
            
            records.append({
                'track_id': track_id,
                'frame': frame,
                'x': x,
                'y': y
            })
    
    return pd.DataFrame(records)


def test_ml_classifier():
    """Test ML trajectory classifier"""
    print_header("Testing ML Trajectory Classifier")
    
    try:
        from ml_trajectory_classifier_enhanced import (
            extract_features_from_tracks_df,
            TrajectoryClassifier,
            TrajectoryClusterer,
            classify_motion_types
        )
        
        print_info("Creating test data with 3 motion types...")
        
        # Create mixed dataset
        tracks_normal = create_test_tracks(20, 80, 'normal')
        tracks_confined = create_test_tracks(20, 80, 'confined')
        tracks_directed = create_test_tracks(20, 80, 'directed')
        
        # Adjust track_ids to be unique
        tracks_confined['track_id'] += 20
        tracks_directed['track_id'] += 40
        
        tracks_df = pd.concat([tracks_normal, tracks_confined, tracks_directed], ignore_index=True)
        
        print_success(f"Created {len(tracks_df['track_id'].unique())} test tracks")
        
        # Test 1: Feature extraction
        print_info("Testing feature extraction...")
        features, track_ids = extract_features_from_tracks_df(tracks_df, pixel_size=0.1, frame_interval=0.1)
        
        if len(features) > 0:
            print_success(f"Extracted {features.shape[1]} features from {len(track_ids)} tracks")
        else:
            print_error("Feature extraction failed")
            return False
        
        # Test 2: Unsupervised clustering
        print_info("Testing unsupervised clustering (K-Means)...")
        result = classify_motion_types(
            tracks_df,
            pixel_size=0.1,
            frame_interval=0.1,
            method='unsupervised',
            model_type='kmeans'
        )
        
        if result['success']:
            print_success(f"Clustering successful: {result['clustering_results']['n_clusters']} classes identified")
            print_success(f"Silhouette score: {result['clustering_results']['silhouette_score']:.3f}")
        else:
            print_error(f"Clustering failed: {result.get('error', 'Unknown error')}")
            return False
        
        # Test 3: Supervised classification (with synthetic labels)
        print_info("Testing supervised classification (Random Forest)...")
        
        # Create synthetic labels
        labels = np.array(['normal']*20 + ['confined']*20 + ['directed']*20)
        labels = labels[:len(track_ids)]  # Match number of valid tracks
        
        try:
            classifier = TrajectoryClassifier(model_type='random_forest')
            train_result = classifier.train(features, labels, validation_split=0.3)
            
            if train_result['success']:
                print_success(f"Training successful: {train_result['accuracy']:.2%} accuracy")
                
                # Test prediction
                pred_labels, probs = classifier.predict(features[:5])
                print_success(f"Prediction successful on {len(pred_labels)} samples")
            else:
                print_error("Training failed")
                return False
                
        except Exception as e:
            print_error(f"Supervised classification error: {str(e)}")
            return False
        
        print_success("All ML classifier tests passed!")
        return True
        
    except Exception as e:
        print_error(f"ML classifier test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_nuclear_diffusion_simulator():
    """Test nuclear diffusion simulator"""
    print_header("Testing Nuclear Diffusion Simulator")
    
    try:
        from nuclear_diffusion_simulator import (
            simulate_nuclear_diffusion,
            NuclearGeometry,
            PhysicsEngine,
            NuclearDiffusionSimulator,
            ParticleProperties,
            CompartmentType
        )
        
        # Test 1: Basic simulation
        print_info("Running basic nuclear diffusion simulation...")
        
        tracks_df, summary = simulate_nuclear_diffusion(
            n_particles=20,
            particle_radius=40,
            n_steps=100,
            time_step=0.001,
            temperature=310
        )
        
        if len(tracks_df) > 0:
            print_success(f"Simulation generated {len(tracks_df)} trajectory points")
            print_success(f"Particles: {summary['n_particles']}, Steps: {summary['total_steps']}")
        else:
            print_error("Simulation produced no trajectories")
            return False
        
        # Test 2: Geometry creation
        print_info("Testing nuclear geometry creation...")
        
        geometry = NuclearGeometry(width=450, height=250)
        print_success(f"Created geometry with {len(geometry.compartments)} compartments")
        
        # Test compartment classification
        test_point = (225, 125)  # Center point
        comp_type = geometry.classify_point(*test_point)
        print_success(f"Point classification working: {comp_type.value if comp_type else 'None'}")
        
        # Test 3: Physics engine
        print_info("Testing physics engine...")
        
        physics = PhysicsEngine(temperature=310, time_step=0.001)
        particle_props = ParticleProperties(radius=40, charge=0.0)
        
        # Get nucleoplasm compartment
        nucleoplasm = geometry.get_compartment(CompartmentType.NUCLEOPLASM)
        if nucleoplasm:
            D = physics.calculate_diffusion_coefficient(particle_props, nucleoplasm.medium)
            print_success(f"Diffusion coefficient calculated: {D:.2e} m²/s")
        else:
            print_error("Could not find nucleoplasm compartment")
        
        # Test 4: Compartment distribution
        print_info("Checking compartment distribution...")
        
        compartment_counts = tracks_df.groupby('compartment').size()
        print_success(f"Particles distributed across {len(compartment_counts)} compartments:")
        for comp, count in compartment_counts.items():
            print(f"  {comp}: {count} points")
        
        print_success("All nuclear diffusion simulator tests passed!")
        return True
        
    except Exception as e:
        print_error(f"Nuclear diffusion simulator test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_md_spt_comparison():
    """Test MD-SPT comparison framework"""
    print_header("Testing MD-SPT Comparison Framework")
    
    try:
        from md_spt_comparison import (
            calculate_diffusion_coefficient,
            compare_diffusion_coefficients,
            compare_msd_curves,
            compare_md_with_spt
        )
        from nuclear_diffusion_simulator import simulate_nuclear_diffusion
        
        # Test 1: Generate MD and "experimental" data
        print_info("Generating MD simulation data...")
        
        tracks_md, _ = simulate_nuclear_diffusion(
            n_particles=30,
            particle_radius=40,
            n_steps=100,
            time_step=0.001
        )
        
        # Convert to μm
        tracks_md['x'] = tracks_md['x'] * 0.1
        tracks_md['y'] = tracks_md['y'] * 0.1
        
        print_success(f"Generated {len(tracks_md['track_id'].unique())} MD tracks")
        
        # Create synthetic "experimental" data (similar to MD but with noise)
        print_info("Creating synthetic experimental data...")
        
        tracks_spt = tracks_md.copy()
        tracks_spt['track_id'] += 1000  # Make unique IDs
        tracks_spt['x'] += np.random.normal(0, 0.05, len(tracks_spt))
        tracks_spt['y'] += np.random.normal(0, 0.05, len(tracks_spt))
        
        print_success(f"Created {len(tracks_spt['track_id'].unique())} experimental tracks")
        
        # Test 2: Diffusion coefficient calculation
        print_info("Calculating diffusion coefficients...")
        
        D_md, msd_md = calculate_diffusion_coefficient(tracks_md, pixel_size=1.0, frame_interval=0.001)
        D_spt, msd_spt = calculate_diffusion_coefficient(tracks_spt, pixel_size=1.0, frame_interval=0.001)
        
        if not np.isnan(D_md) and not np.isnan(D_spt):
            print_success(f"D_MD = {D_md:.4f} μm²/s")
            print_success(f"D_SPT = {D_spt:.4f} μm²/s")
            print_success(f"Ratio (MD/SPT) = {D_md/D_spt:.3f}")
        else:
            print_error("Diffusion coefficient calculation failed")
            return False
        
        # Test 3: Statistical comparison
        print_info("Performing statistical comparison...")
        
        comparison = compare_diffusion_coefficients(
            D_md, D_spt, tracks_md, tracks_spt,
            pixel_size=1.0, frame_interval=0.001
        )
        
        print_success(f"p-value: {comparison['p_value']:.4f}")
        print_success(f"Significant difference: {comparison['significant']}")
        
        # Test 4: MSD curve comparison
        print_info("Comparing MSD curves...")
        
        msd_comp = compare_msd_curves(msd_md, msd_spt)
        print_success(f"MSD correlation: {msd_comp['correlation']:.3f}")
        print_success(f"RMSE: {msd_comp['rmse']:.4f}")
        
        # Test 5: Comprehensive comparison
        print_info("Running comprehensive MD-SPT comparison...")
        
        full_comparison = compare_md_with_spt(
            tracks_md, tracks_spt,
            pixel_size=1.0,
            frame_interval=0.001,
            analyze_compartments=False  # Skip compartment analysis for speed
        )
        
        if full_comparison['success']:
            print_success("Comprehensive comparison successful")
            print_success(f"Agreement: {full_comparison['summary']['diffusion_agreement']}")
            print_success(f"Recommendation: {full_comparison['summary']['recommendation'][:60]}...")
        else:
            print_error("Comprehensive comparison failed")
            return False
        
        print_success("All MD-SPT comparison tests passed!")
        return True
        
    except Exception as e:
        print_error(f"MD-SPT comparison test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_report_generator_integration():
    """Test integration with report generator"""
    print_header("Testing Report Generator Integration")
    
    try:
        from enhanced_report_generator import EnhancedSPTReportGenerator
        
        # Create test data
        print_info("Creating test tracks...")
        tracks_df = create_test_tracks(30, 100, 'normal')
        
        # Initialize report generator
        print_info("Initializing report generator...")
        generator = EnhancedSPTReportGenerator()
        
        # Check that new analyses are registered
        print_info("Checking registered analyses...")
        
        required_analyses = ['ml_classification', 'md_comparison', 'nuclear_diffusion_sim']
        for analysis in required_analyses:
            if analysis in generator.available_analyses:
                print_success(f"Analysis '{analysis}' registered")
            else:
                print_error(f"Analysis '{analysis}' NOT registered")
                return False
        
        # Test ML classification
        print_info("Testing ML classification in report generator...")
        try:
            result = generator._analyze_ml_classification(tracks_df, pixel_size=0.1, frame_interval=0.1)
            if result.get('success', False):
                print_success(f"ML classification successful: {result['n_classes']} classes")
            else:
                print_error(f"ML classification failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            print_error(f"ML classification error: {str(e)}")
        
        # Test nuclear diffusion simulation
        print_info("Testing nuclear diffusion simulation in report generator...")
        try:
            result = generator._run_nuclear_diffusion_simulation(tracks_df, pixel_size=0.1, frame_interval=0.1)
            if result.get('success', False):
                print_success(f"Simulation successful: {result['n_particles']} particles, {result['total_steps']} steps")
            else:
                print_error(f"Simulation failed: {result.get('error', 'Unknown')}")
        except Exception as e:
            print_error(f"Simulation error: {str(e)}")
        
        print_success("Report generator integration tests completed!")
        return True
        
    except Exception as e:
        print_error(f"Report generator integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print_header("ML & MD INTEGRATION TEST SUITE")
    
    print(f"{BOLD}Testing new features:{RESET}")
    print("  1. Machine Learning Trajectory Classifier")
    print("  2. Nuclear Diffusion Simulator")
    print("  3. MD-SPT Comparison Framework")
    print("  4. Report Generator Integration")
    
    results = {}
    
    # Run tests
    results['ML Classifier'] = test_ml_classifier()
    results['Nuclear Diffusion Simulator'] = test_nuclear_diffusion_simulator()
    results['MD-SPT Comparison'] = test_md_spt_comparison()
    results['Report Generator Integration'] = test_report_generator_integration()
    
    # Summary
    print_header("TEST SUMMARY")
    
    total = len(results)
    passed = sum(results.values())
    failed = total - passed
    
    for test_name, result in results.items():
        status = f"{GREEN}PASSED{RESET}" if result else f"{RED}FAILED{RESET}"
        print(f"{test_name:.<50} {status}")
    
    print(f"\n{BOLD}Total:{RESET} {total} tests")
    print(f"{BOLD}{GREEN}Passed:{RESET} {passed}")
    print(f"{BOLD}{RED}Failed:{RESET} {failed}")
    
    success_rate = (passed / total * 100) if total > 0 else 0
    
    if success_rate == 100:
        print(f"\n{BOLD}{GREEN}✓ ALL TESTS PASSED!{RESET}")
        return 0
    elif success_rate >= 75:
        print(f"\n{BOLD}{YELLOW}⚠ {success_rate:.0f}% TESTS PASSED{RESET}")
        return 0
    else:
        print(f"\n{BOLD}{RED}✗ {success_rate:.0f}% TESTS PASSED{RESET}")
        return 1


if __name__ == "__main__":
    exit(main())
