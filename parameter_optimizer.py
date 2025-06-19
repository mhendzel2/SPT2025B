"""
Parameter Optimization Tool for Particle Detection
Automatically tests different detection parameters against ground truth data
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io, feature, filters, measure, morphology
from skimage.filters import threshold_local
import pywt
from scipy.spatial.distance import cdist
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
from tracking import detect_particles, detect_particles_enhanced
from concurrent.futures import ProcessPoolExecutor # Change this import
import multiprocessing as mp

class ParticleDetectionOptimizer:
    def __init__(self, image_path, ground_truth_csv, num_sigma=10, min_distance=2.0):
        """Initialize enhanced optimizer with multi-frame testing and parallel processing"""
        self.image_path = image_path
        self.csv_path = ground_truth_csv
        self.image_stack = None
        self.results = []
        self.ground_truth = None
        self.num_sigma = num_sigma
        self.min_distance = min_distance
        
    def load_ground_truth(self):
        """Load ground truth coordinates from CSV or XML file"""
        try:
            if hasattr(self.csv_path, 'read'):
                # It's a Streamlit uploaded file - need to reset pointer
                self.csv_path.seek(0)
                
                # Check file extension
                file_extension = self.csv_path.name.lower().split('.')[-1]
                
                if file_extension == 'csv':
                    self.ground_truth = pd.read_csv(self.csv_path)
                    
                    # Check for coordinate columns (both lowercase and uppercase)
                    if 'x' in self.ground_truth.columns and 'y' in self.ground_truth.columns:
                        # Already correct format
                        pass
                    elif 'X' in self.ground_truth.columns and 'Y' in self.ground_truth.columns:
                        # Rename to lowercase for consistency
                        self.ground_truth = self.ground_truth.rename(columns={'X': 'x', 'Y': 'y'})
                    else:
                        raise ValueError("Ground truth CSV must contain 'x'/'X' and 'y'/'Y' columns")
                
                elif file_extension == 'xml':
                    # Load TrackMate XML file
                    from data_loader import load_tracks_file
                    tracks_df = load_tracks_file(self.csv_path)
                    
                    # Use only frame 0 for ground truth (first frame spots)
                    if 'frame' in tracks_df.columns:
                        self.ground_truth = tracks_df[tracks_df['frame'] == 0][['x', 'y']].copy()
                        # Add Frame column (set to 0 for compatibility)
                        self.ground_truth['Frame'] = 0
                    else:
                        # If no frame column, use all spots
                        self.ground_truth = tracks_df[['x', 'y']].copy()
                        self.ground_truth['Frame'] = 0
                else:
                    raise ValueError("Unsupported file format. Use CSV or XML files.")
            else:
                # It's a file path
                self.ground_truth = pd.read_csv(self.csv_path)
                
                # Check for coordinate columns (both lowercase and uppercase)
                if 'x' in self.ground_truth.columns and 'y' in self.ground_truth.columns:
                    # Already correct format
                    pass
                elif 'X' in self.ground_truth.columns and 'Y' in self.ground_truth.columns:
                    # Rename to lowercase for consistency
                    self.ground_truth = self.ground_truth.rename(columns={'X': 'x', 'Y': 'y'})
                else:
                    raise ValueError("Ground truth CSV must contain 'x'/'X' and 'y'/'Y' columns")
            
            return True
        except Exception as e:
            st.error(f"Error loading ground truth: {str(e)}")
            return False

    def load_image(self):
        """Load TIFF image stack"""
        try:
            if hasattr(self.image_path, 'read'):
                # It's a Streamlit uploaded file - need to reset pointer
                self.image_path.seek(0)
                self.image_stack = io.imread(self.image_path)
            else:
                # It's a file path
                self.image_stack = io.imread(self.image_path)
            return True
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return False
    
    def get_ground_truth_for_frame(self, frame_idx):
        """Get ground truth particles for a specific frame"""
        frame_gt = self.ground_truth[self.ground_truth['Frame'] == frame_idx]
        if len(frame_gt) > 0:
            return frame_gt[['x', 'y']].values
        return np.array([])
    
    def calculate_detection_metrics(self, detected_particles, ground_truth_particles, tolerance=2.0):
        """Calculate detection performance metrics"""
        if len(detected_particles) == 0:
            return {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': len(ground_truth_particles),
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        if len(ground_truth_particles) == 0:
            return {
                'true_positives': 0,
                'false_positives': len(detected_particles),
                'false_negatives': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            }
        
        # Calculate distances between detected and ground truth particles
        distances = cdist(detected_particles, ground_truth_particles)
        
        # Find matches within tolerance
        true_positives = 0
        matched_gt = set()
        matched_det = set()
        
        for i in range(len(detected_particles)):
            min_dist_idx = np.argmin(distances[i])
            min_dist = distances[i][min_dist_idx]
            
            if min_dist <= tolerance and min_dist_idx not in matched_gt:
                true_positives += 1
                matched_gt.add(min_dist_idx)
                matched_det.add(i)
        
        false_positives = len(detected_particles) - true_positives
        false_negatives = len(ground_truth_particles) - true_positives
        
        precision = true_positives / len(detected_particles) if len(detected_particles) > 0 else 0
        recall = true_positives / len(ground_truth_particles) if len(ground_truth_particles) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
    
    def detect_particles_log(self, image, **kwargs):
        """Use tracking module implementation."""
        return detect_particles(image, method="LoG", **kwargs)
    def detect_particles_dog(self, image, **kwargs):
        """Use tracking module implementation."""
        return detect_particles(image, method="DoG", **kwargs)
    def detect_particles_wavelet(self, image, **kwargs):
        """Use tracking module implementation."""
        return detect_particles_enhanced(image, method="Wavelet", **kwargs)
    def detect_particles_intensity(self, image, threshold_mode, threshold_factor, percentile_value, std_multiplier, particle_size, morphology_cleanup, erosion_size, dilation_size):
        """Intensity-based particle detection"""
        try:
            if threshold_mode == "Auto (Otsu)":
                thresh = filters.threshold_otsu(image)
            elif threshold_mode == "Manual":
                thresh = threshold_factor * np.mean(image)
            elif threshold_mode == "Percentile":
                thresh = np.percentile(image, percentile_value)
            elif threshold_mode == "Mean + Std":
                thresh = np.mean(image) + std_multiplier * np.std(image)
            
            mask = image > thresh
            
            if morphology_cleanup:
                if erosion_size > 0:
                    mask = morphology.erosion(mask, morphology.disk(erosion_size))
                if dilation_size > 0:
                    mask = morphology.dilation(mask, morphology.disk(dilation_size))
            
            labeled_mask = measure.label(mask)
            props = measure.regionprops(labeled_mask)
            
            min_area = int(particle_size * particle_size * 0.5)
            particles = []
            for prop in props:
                if prop.area >= min_area:
                    particles.append([prop.centroid[1], prop.centroid[0]])  # x, y
            
            return np.array(particles) if particles else np.array([])
        except:
            return np.array([])
    
    def test_log_params(self, params):
        """Test single LoG parameter combination - for parallel processing"""
        min_sigma, max_sigma, threshold, overlap, test_frames = params
        
        if max_sigma <= min_sigma:
            return None
            
        avg_f1 = 0
        for frame_idx in test_frames:
            if frame_idx < len(self.image_stack):
                image = self.image_stack[frame_idx]
                gt_particles = self.get_ground_truth_for_frame(frame_idx)
                
                # Use configurable num_sigma from parameters or default
                num_sigma = getattr(self, 'num_sigma', 10)
                detected = self.detect_particles_log(image, min_sigma, max_sigma, num_sigma, threshold, overlap)
                metrics = self.calculate_detection_metrics(detected, gt_particles)
                avg_f1 += metrics['f1_score']
        
        avg_f1 /= len(test_frames)
        
        # Calculate particle size from sigma range
        estimated_particle_size = (min_sigma + max_sigma) / 2 * 2.0  # Convert sigma to diameter
        
        return {
            'method': 'LoG',
            'particle_size': estimated_particle_size,
            'threshold_factor': threshold,
            'min_distance': getattr(self, 'min_distance', 2.0),
            'min_sigma': min_sigma,
            'max_sigma': max_sigma,
            'num_sigma': num_sigma,
            'overlap_threshold': overlap,
            'f1_score': avg_f1
        }

    def optimize_log_parameters(self, test_frames=[0, 1, 2]):
        """Optimize LoG detection parameters with parallel processing"""
        st.write("🔍 Optimizing LoG parameters with parallel processing...")
        
        # Reduced parameter ranges for faster testing
        min_sigma_range = np.arange(0.6, 1.8, 0.3)
        max_sigma_range = np.arange(1.2, 2.5, 0.3)
        threshold_range = np.arange(0.02, 0.08, 0.02)
        overlap_range = np.arange(0.3, 0.7, 0.2)
        
        # Create parameter combinations
        param_combinations = []
        for min_sigma in min_sigma_range:
            for max_sigma in max_sigma_range:
                for threshold in threshold_range:
                    for overlap in overlap_range:
                        param_combinations.append((min_sigma, max_sigma, threshold, overlap, test_frames))
        
        st.write(f"Testing {len(param_combinations)} parameter combinations...")
        progress_bar = st.progress(0)
        
        # Use parallel processing
        results = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.test_log_params, params) for params in param_combinations]
            
            for i, future in enumerate(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(param_combinations))
        
        # Find best result
        if results:
            return max(results, key=lambda x: x['f1_score'])
        return None
    
    def optimize_dog_parameters(self, test_frames=list(range(10))):
        """Optimize DoG detection parameters"""
        st.write("🔍 Optimizing DoG parameters...")
        
        particle_size = 1.25
        # Better ranges for 1.25px particles
        sigma1_range = np.arange(0.8, 1.8, 0.3)
        sigma2_range = np.arange(1.5, 3.0, 0.3)
        threshold_range = np.arange(85, 99, 3)  # Percentile-based thresholds
        
        best_score = 0
        best_params = None
        
        progress_bar = st.progress(0)
        total_tests = len(sigma1_range) * len(sigma2_range) * len(threshold_range)
        current_test = 0
        
        for sigma1 in sigma1_range:
            for sigma2 in sigma2_range:
                if sigma2 <= sigma1 + 0.3:  # Ensure meaningful difference
                    continue
                for threshold in threshold_range:
                    avg_f1 = 0
                    valid_frames = 0
                    
                    for frame_idx in test_frames:
                        if frame_idx < len(self.image_stack):
                            image = self.image_stack[frame_idx]
                            gt_particles = self.get_ground_truth_for_frame(frame_idx)
                            
                            if len(gt_particles) > 0:  # Only test frames with ground truth
                                detected = self.detect_particles_dog(image, sigma1, sigma2, threshold, "Manual", particle_size)
                                metrics = self.calculate_detection_metrics(detected, gt_particles)
                                avg_f1 += metrics['f1_score']
                                valid_frames += 1
                    
                    if valid_frames > 0:
                        avg_f1 /= valid_frames
                        
                        if avg_f1 > best_score:
                            best_score = avg_f1
                            best_params = {
                                'method': 'DoG',
                                'sigma1': sigma1,
                                'sigma2': sigma2,
                                'threshold': threshold,
                                'particle_size': particle_size,
                                'f1_score': avg_f1
                            }
                    
                    current_test += 1
                    progress_bar.progress(current_test / total_tests)
        
        return best_params
    
    def optimize_wavelet_parameters(self, test_frames=list(range(10))):
        """Optimize Wavelet detection parameters"""
        st.write("🔍 Optimizing Wavelet parameters...")
        
        particle_size = 1.25
        wavelet_types = ['db4', 'haar']  # Reduced for faster testing
        levels_range = [2, 3]
        enhancement_range = [1.5, 2.0, 2.5]
        threshold_range = [1.5, 2.0, 2.5]
        
        best_score = 0
        best_params = None
        
        progress_bar = st.progress(0)
        total_tests = len(wavelet_types) * len(levels_range) * len(enhancement_range) * len(threshold_range)
        current_test = 0
        
        for wavelet_type in wavelet_types:
            for levels in levels_range:
                for enhancement in enhancement_range:
                    for threshold in threshold_range:
                        avg_f1 = 0
                        valid_frames = 0
                        
                        for frame_idx in test_frames:
                            if frame_idx < len(self.image_stack):
                                image = self.image_stack[frame_idx]
                                gt_particles = self.get_ground_truth_for_frame(frame_idx)
                                
                                if len(gt_particles) > 0:  # Only test frames with ground truth
                                    detected = self.detect_particles_wavelet(image, wavelet_type, levels, enhancement, 0.1, threshold, particle_size)
                                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                                    avg_f1 += metrics['f1_score']
                                    valid_frames += 1
                        
                        if valid_frames > 0:
                            avg_f1 /= valid_frames
                            
                            if avg_f1 > best_score:
                                best_score = avg_f1
                                best_params = {
                                    'method': 'Wavelet',
                                    'wavelet_type': wavelet_type,
                                    'levels': levels,
                                    'enhancement': enhancement,
                                    'threshold_factor': threshold,
                                    'particle_size': particle_size,
                                    'f1_score': avg_f1
                                }
                        
                        current_test += 1
                        progress_bar.progress(current_test / total_tests)
        
        return best_params
    
    def optimize_intensity_parameters(self, test_frames=[0, 1, 2]):
        """Optimize Intensity detection parameters"""
        st.write("🔍 Optimizing Intensity parameters...")
        
        particle_size = 2.0
        percentile_range = np.arange(85, 98, 2)
        std_multiplier_range = np.arange(1.0, 3.0, 0.5)
        
        best_score = 0
        best_params = None
        
        progress_bar = st.progress(0)
        total_tests = len(percentile_range) + len(std_multiplier_range)
        current_test = 0
        
        # Test percentile method
        for percentile in percentile_range:
            avg_f1 = 0
            for frame_idx in test_frames:
                if frame_idx < len(self.image_stack):
                    image = self.image_stack[frame_idx]
                    gt_particles = self.get_ground_truth_for_frame(frame_idx)
                    
                    detected = self.detect_particles_intensity(
                        image, "Percentile", 1.0, percentile, 2.0, particle_size, True, 1, 1
                    )
                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                    avg_f1 += metrics['f1_score']
            
            avg_f1 /= len(test_frames)
            
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = {
                    'method': 'Intensity',
                    'threshold_mode': 'Percentile',
                    'percentile': percentile,
                    'particle_size': particle_size,
                    'f1_score': avg_f1
                }
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # Test Mean + Std method
        for std_mult in std_multiplier_range:
            avg_f1 = 0
            for frame_idx in test_frames:
                if frame_idx < len(self.image_stack):
                    image = self.image_stack[frame_idx]
                    gt_particles = self.get_ground_truth_for_frame(frame_idx)
                    
                    detected = self.detect_particles_intensity(
                        image, "Mean + Std", 1.0, 90, std_mult, particle_size, True, 1, 1
                    )
                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                    avg_f1 += metrics['f1_score']
            
            avg_f1 /= len(test_frames)
            
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = {
                    'method': 'Intensity',
                    'threshold_mode': 'Mean + Std',
                    'std_multiplier': std_mult,
                    'particle_size': particle_size,
                    'f1_score': avg_f1
                }
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        return best_params
    
    def get_ground_truth_for_frame(self, frame_idx):
        """Get ground truth particles for a specific frame"""
        if self.ground_truth is None:
            return np.array([])
        
        frame_gt = self.ground_truth[self.ground_truth['frame'] == frame_idx]
        if len(frame_gt) > 0:
            return frame_gt[['x', 'y']].values
        return np.array([])

    def test_log_params_parallel(self, params_combo):
        """Test a single LoG parameter combination across multiple frames"""
        min_sigma, max_sigma, threshold, test_frames = params_combo
        f1_scores = []
        
        for frame_idx in test_frames:
            if frame_idx < len(self.image_stack):
                if len(self.image_stack.shape) == 3:  # Multi-frame stack
                    image = self.image_stack[frame_idx]
                else:  # Single frame
                    image = self.image_stack
                gt_particles = self.get_ground_truth_for_frame(frame_idx)
                
                if len(gt_particles) > 0:
                    detected = self.detect_particles_log(image, min_sigma, max_sigma, 20, threshold, 0.3)
                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                    f1_scores.append(metrics['f1_score'])
        
        if f1_scores:
            return {
                'method': 'LoG',
                'min_sigma': min_sigma,
                'max_sigma': max_sigma,
                'threshold': threshold,
                'f1_score': np.mean(f1_scores),
                'f1_std': np.std(f1_scores)
            }
        return None

    def optimize_log_robust_parallel(self, test_frames):
        """Enhanced parallel LoG optimization across multiple frames"""
        st.write("🔍 Optimizing LoG parameters with parallel processing...")
        
        min_sigma_range = np.linspace(0.5, 1.5, 6)
        max_sigma_range = np.linspace(1.5, 3.0, 6)
        threshold_range = np.linspace(0.01, 0.1, 6)
        
        param_combinations = []
        for min_sigma in min_sigma_range:
            for max_sigma in max_sigma_range:
                if max_sigma > min_sigma + 0.5:
                    for threshold in threshold_range:
                        param_combinations.append((min_sigma, max_sigma, threshold, test_frames))
        
        if not param_combinations:
            st.warning("⚠️ LoG: No valid parameter combinations to test.")
            return None

        st.write(f"Testing {len(param_combinations)} LoG parameter combinations in parallel...")
        
        progress_bar = st.progress(0)
        results = []
        
        max_workers = min(mp.cpu_count(), 8)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.test_log_params_parallel, combo) for combo in param_combinations]
            
            for i, future in enumerate(futures):
                result = future.result()
                if result:
                    results.append(result)
                progress_bar.progress((i + 1) / len(param_combinations))
        
        if results:
            best_params = max(results, key=lambda x: x['f1_score'])
            st.write(f"✅ LoG: Tested {len(results)} valid combinations.")
            return best_params
        st.warning("⚠️ LoG: No parameter combinations yielded valid results.")
        return None

    def run_comprehensive_optimization(self, test_frames=list(range(10))):
        """Run enhanced comprehensive optimization across multiple frames"""
        if not self.load_ground_truth():
            return None
        if not self.load_image():
            return None
        
        st.write(f"🚀 **Enhanced Multi-Frame Parameter Optimization**")
        
        # Filter test_frames to include only those that are valid and have ground truth
        valid_frames_for_testing = []
        for frame_idx in test_frames:
            if len(self.image_stack.shape) == 3:  # Multi-frame stack
                max_frame = len(self.image_stack) - 1
            else:  # Single frame
                max_frame = 0
                
            if 0 <= frame_idx <= max_frame:
                if len(self.get_ground_truth_for_frame(frame_idx)) > 0:
                    valid_frames_for_testing.append(frame_idx)
        
        if not valid_frames_for_testing:
            st.error(f"❌ No frames with ground truth data found. Cannot optimize.")
            return []
        
        st.write(f"📊 Optimizing using {len(valid_frames_for_testing)} frames with ground truth data")
        
        all_results = []
        
        # Test enhanced LoG with parallel processing
        try:
            log_result = self.optimize_log_robust_parallel(valid_frames_for_testing)
            if log_result and log_result.get('f1_score', -1) > 0:
                all_results.append(log_result)
                st.success(f"✅ Enhanced LoG: Best F1={log_result['f1_score']:.3f} ±{log_result.get('f1_std', 0):.3f}")
            else:
                st.warning("⚠️ Enhanced LoG: No effective parameters found.")
        except Exception as e:
            st.error(f"❌ Enhanced LoG optimization error: {e}")
        
        # Test standard methods for comparison
        try:
            st.write("🔍 Testing standard DoG detection...")
            dog_result = self.optimize_dog_parameters(valid_frames_for_testing)
            if dog_result:
                all_results.append(dog_result)
                st.success(f"✅ DoG optimization complete - F1 Score: {dog_result['f1_score']:.3f}")
        except Exception as e:
            st.warning(f"⚠️ DoG optimization encountered an issue: {e}")
        
        # Test intensity detection
        try:
            st.write("🔍 Testing Intensity detection...")
            intensity_result = self.optimize_intensity_parameters_fast(valid_frames_for_testing)
            if intensity_result:
                all_results.append(intensity_result)
                st.success(f"✅ Intensity optimization complete - F1 Score: {intensity_result['f1_score']:.3f}")
        except Exception as e:
            st.warning(f"⚠️ Intensity optimization encountered an issue: {e}")
        
        # Show enhanced results comparison
        if all_results:
            all_results.sort(key=lambda x: x['f1_score'], reverse=True)
            best_method_overall = all_results[0]
            st.write(f"🏆 **Overall Best Method:** {best_method_overall['method']} with F1 Score: {best_method_overall['f1_score']:.3f}")
            
            # Create enhanced comparison plot
            if len(all_results) > 0:
                methods = [r['method'] for r in all_results]
                scores = [r['f1_score'] for r in all_results]
                stds = [r.get('f1_std', 0) for r in all_results]
                
                plot_df = pd.DataFrame({'Method': methods, 'F1 Score': scores, 'F1 Std': stds})

                fig = px.bar(plot_df, x='Method', y='F1 Score', error_y='F1 Std',
                             title="Enhanced Detection Method Comparison (Mean F1 Score with Std Dev)",
                             labels={'F1 Score': 'Mean F1 Score (+/- Std Dev)'})
                fig.update_layout(xaxis_title="Method", yaxis_title="Mean F1 Score")
                st.plotly_chart(fig)
        else:
            st.error("❌ No detection methods yielded successful results!")
        
        return all_results

    def run_optimization(self, test_frames=[0, 1, 2]):
        """Legacy run optimization function for backward compatibility"""
        return self.run_comprehensive_optimization(test_frames)
    
    def optimize_intensity_parameters_fast(self, test_frames=[0, 1, 2]):
        """Fast intensity optimization with fewer parameters"""
        st.write("🔍 Quick intensity optimization...")
        
        particle_size = 2.0
        percentile_range = [85, 90, 95]
        std_multiplier_range = [1.5, 2.0, 2.5]
        
        best_score = 0
        best_params = None
        
        progress_bar = st.progress(0)
        total_tests = len(percentile_range) + len(std_multiplier_range)
        current_test = 0
        
        # Test percentile method
        for percentile in percentile_range:
            avg_f1 = 0
            for frame_idx in test_frames:
                if frame_idx < len(self.image_stack):
                    image = self.image_stack[frame_idx]
                    gt_particles = self.get_ground_truth_for_frame(frame_idx)
                    
                    detected = self.detect_particles_intensity(
                        image, "Percentile", 1.0, percentile, 2.0, particle_size, True, 1, 1
                    )
                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                    avg_f1 += metrics['f1_score']
            
            avg_f1 /= len(test_frames)
            
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = {
                    'method': 'Intensity',
                    'threshold_mode': 'Percentile',
                    'percentile': percentile,
                    'particle_size': particle_size,
                    'f1_score': avg_f1
                }
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        # Test Mean + Std method
        for std_mult in std_multiplier_range:
            avg_f1 = 0
            for frame_idx in test_frames:
                if frame_idx < len(self.image_stack):
                    image = self.image_stack[frame_idx]
                    gt_particles = self.get_ground_truth_for_frame(frame_idx)
                    
                    detected = self.detect_particles_intensity(
                        image, "Mean + Std", 1.0, 90, std_mult, particle_size, True, 1, 1
                    )
                    metrics = self.calculate_detection_metrics(detected, gt_particles)
                    avg_f1 += metrics['f1_score']
            
            avg_f1 /= len(test_frames)
            
            if avg_f1 > best_score:
                best_score = avg_f1
                best_params = {
                    'method': 'Intensity',
                    'threshold_mode': 'Mean + Std',
                    'std_multiplier': std_mult,
                    'particle_size': particle_size,
                    'f1_score': avg_f1
                }
            
            current_test += 1
            progress_bar.progress(current_test / total_tests)
        
        return best_params
    
    def create_results_visualization(self, results):
        """Create visualization of optimization results"""
        if not results:
            st.error("No optimization results to display")
            return
        
        # Create results DataFrame
        df_results = pd.DataFrame(results)
        
        # Sort by F1 score
        df_results = df_results.sort_values('f1_score', ascending=False)
        
        # Display results table
        st.subheader("🏆 Optimization Results")
        st.dataframe(df_results.style.highlight_max(subset=['f1_score']))
        
        # Create bar chart
        fig = px.bar(
            df_results, 
            x='method', 
            y='f1_score',
            title='Detection Performance by Method',
            labels={'f1_score': 'F1 Score', 'method': 'Detection Method'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig)
        
        # Show best parameters
        best_result = df_results.iloc[0]
        st.subheader("🎯 Best Detection Method")
        st.success(f"**{best_result['method']}** achieved the highest F1 score: {best_result['f1_score']:.3f}")
        
        # Display optimal parameters
        st.write("**Optimal Parameters:**")
        for key, value in best_result.items():
            if key not in ['method', 'f1_score']:
                st.write(f"- **{key}**: {value}")
        
        return best_result
# Use ProcessPoolExecutor instead of ThreadPoolExecutor
max_workers = min(mp.cpu_count(), 8)
with ProcessPoolExecutor(max_workers=max_workers) as executor: # Change this line
    futures = [executor.submit(self.test_log_params_parallel, combo) for combo in param_combinations]
    
    for i, future in enumerate(futures):
        result = future.result()
        if result:
            results.append(result)
        progress_bar.progress((i + 1) / len(param_combinations))
def create_optimizer_interface():
    """Create Streamlit interface for parameter optimization"""
    st.header("🔬 Particle Detection Parameter Optimizer")
    st.write("Automatically find the best detection settings for your particles!")
    
    # File paths
    image_path = "attached_assets/Call1_beads.tif"
    csv_path = "attached_assets/Cropped_spots.csv"
    
    if st.button("🚀 Start Parameter Optimization", type="primary"):
        optimizer = ParticleDetectionOptimizer(image_path, csv_path)
        
        with st.spinner("Running optimization tests..."):
            results = optimizer.run_optimization()
        
        if results:
            best_params = optimizer.create_results_visualization(results)
            
            # Save results for later use
            st.session_state['optimal_params'] = best_params
            
            st.success("✅ Optimization complete! You can now use these optimal parameters in the main detection interface.")
        else:
            st.error("❌ Optimization failed. Please check your image and CSV files.")

if __name__ == "__main__":
    create_optimizer_interface()