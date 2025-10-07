"""
Parallel Processing Support for 2025 SPT Features

Adds multiprocessing capabilities to all analysis modules:
- biased_inference batch analysis
- acquisition_advisor validation
- equilibrium_validator checks
- ddm_analyzer image processing
- ihmm_blur_analysis state segmentation
- microsecond_sampling MSD calculation

Uses ProcessPoolExecutor for CPU-bound tasks with automatic
core count detection and progress reporting.

Author: SPT2025B Team
Date: October 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Any, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from tqdm.auto import tqdm
import warnings


def get_optimal_workers(n_tasks: int = None) -> int:
    """
    Determine optimal number of worker processes.
    
    Parameters
    ----------
    n_tasks : int, optional
        Number of tasks to process. If provided, limits workers to n_tasks.
    
    Returns
    -------
    int
        Optimal number of workers (usually n_cpus - 1, max n_tasks)
    """
    n_cpus = mp.cpu_count()
    
    # Leave one core for system
    n_workers = max(1, n_cpus - 1)
    
    # Don't spawn more workers than tasks
    if n_tasks is not None:
        n_workers = min(n_workers, n_tasks)
    
    return n_workers


def parallel_batch_analyze(tracks_df: pd.DataFrame,
                          analysis_function: Callable,
                          n_workers: Optional[int] = None,
                          show_progress: bool = True,
                          **kwargs) -> pd.DataFrame:
    """
    Generic parallel batch analysis for track-level functions.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracking data with track_id column
    analysis_function : Callable
        Function to apply to each track. Should accept track coordinates
        and kwargs, return dict with results.
    n_workers : int, optional
        Number of parallel workers. Default: auto-detect
    show_progress : bool
        Show progress bar
    **kwargs
        Additional arguments passed to analysis_function
    
    Returns
    -------
    pd.DataFrame
        Results with one row per track
    """
    # Group by track_id
    track_groups = list(tracks_df.groupby('track_id'))
    n_tracks = len(track_groups)
    
    if n_tracks == 0:
        return pd.DataFrame()
    
    # Determine workers
    if n_workers is None:
        n_workers = get_optimal_workers(n_tracks)
    
    # Serial execution for small datasets or single worker
    if n_tracks < 10 or n_workers == 1:
        results = []
        for track_id, group in (tqdm(track_groups, desc="Analyzing tracks") 
                               if show_progress else track_groups):
            result = _process_single_track(track_id, group, analysis_function, kwargs)
            results.append(result)
        return pd.DataFrame(results)
    
    # Parallel execution
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(_process_single_track, track_id, group, analysis_function, kwargs): track_id
            for track_id, group in track_groups
        }
        
        # Collect results with progress bar
        if show_progress:
            pbar = tqdm(total=n_tracks, desc="Analyzing tracks (parallel)")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                track_id = futures[future]
                warnings.warn(f"Track {track_id} failed: {e}")
                results.append({'track_id': track_id, 'success': False, 'error': str(e)})
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    return pd.DataFrame(results)


def _process_single_track(track_id: Any, group: pd.DataFrame,
                         analysis_function: Callable, kwargs: Dict) -> Dict:
    """
    Process single track (helper for parallelization).
    
    Parameters
    ----------
    track_id : Any
        Track identifier
    group : pd.DataFrame
        Track data
    analysis_function : Callable
        Analysis function
    kwargs : Dict
        Additional arguments
    
    Returns
    -------
    Dict
        Result with track_id
    """
    # Sort by frame
    group = group.sort_values('frame')
    
    # Extract coordinates
    has_z = 'z' in group.columns
    if has_z:
        coords = group[['x', 'y', 'z']].values
    else:
        coords = group[['x', 'y']].values
    
    # Apply pixel size if provided
    if 'pixel_size' in kwargs or 'pixel_size_um' in kwargs:
        pixel_size = kwargs.get('pixel_size', kwargs.get('pixel_size_um', 1.0))
        coords = coords * pixel_size
    
    # Run analysis
    try:
        result = analysis_function(coords, **kwargs)
        result['track_id'] = track_id
        result['N_steps'] = len(coords) - 1
        return result
    except Exception as e:
        return {
            'track_id': track_id,
            'success': False,
            'error': str(e),
            'N_steps': len(coords) - 1
        }


def parallel_ddm_analysis(image_stacks: List[np.ndarray],
                         ddm_analyzer,
                         n_workers: Optional[int] = None,
                         show_progress: bool = True,
                         **kwargs) -> List[Dict]:
    """
    Parallel DDM analysis for multiple image stacks.
    
    Parameters
    ----------
    image_stacks : list of np.ndarray
        List of 3D image arrays (n_frames, height, width)
    ddm_analyzer : DDMAnalyzer
        DDM analyzer instance
    n_workers : int, optional
        Number of workers
    show_progress : bool
        Show progress bar
    **kwargs
        Arguments for compute_image_structure_function
    
    Returns
    -------
    list of Dict
        DDM results for each stack
    """
    n_stacks = len(image_stacks)
    
    if n_workers is None:
        n_workers = get_optimal_workers(n_stacks)
    
    if n_stacks < 3 or n_workers == 1:
        # Serial
        results = []
        for stack in (tqdm(image_stacks, desc="DDM analysis") if show_progress else image_stacks):
            result = ddm_analyzer.compute_image_structure_function(stack, **kwargs)
            results.append(result)
        return results
    
    # Parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(ddm_analyzer.compute_image_structure_function, stack, **kwargs): i
            for i, stack in enumerate(image_stacks)
        }
        
        if show_progress:
            pbar = tqdm(total=n_stacks, desc="DDM analysis (parallel)")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                warnings.warn(f"Stack {idx} failed: {e}")
                results.append({'success': False, 'error': str(e), 'stack_index': idx})
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    return results


def parallel_ihmm_segmentation(tracks_list: List[np.ndarray],
                               ihmm_analyzer,
                               n_workers: Optional[int] = None,
                               show_progress: bool = True,
                               **kwargs) -> List[Dict]:
    """
    Parallel iHMM state segmentation for multiple tracks.
    
    Parameters
    ----------
    tracks_list : list of np.ndarray
        List of track coordinates
    ihmm_analyzer : iHMMBlurAnalyzer
        iHMM analyzer instance
    n_workers : int, optional
        Number of workers
    show_progress : bool
        Show progress bar
    **kwargs
        Arguments for fit method
    
    Returns
    -------
    list of Dict
        iHMM results for each track
    """
    n_tracks = len(tracks_list)
    
    if n_workers is None:
        n_workers = get_optimal_workers(n_tracks)
    
    if n_tracks < 10 or n_workers == 1:
        # Serial
        results = []
        for track in (tqdm(tracks_list, desc="iHMM segmentation") if show_progress else tracks_list):
            result = ihmm_analyzer.fit(track, **kwargs)
            results.append(result)
        return results
    
    # Parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(ihmm_analyzer.fit, track, **kwargs): i
            for i, track in enumerate(tracks_list)
        }
        
        if show_progress:
            pbar = tqdm(total=n_tracks, desc="iHMM segmentation (parallel)")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                warnings.warn(f"Track {idx} failed: {e}")
                results.append({'success': False, 'error': str(e), 'track_index': idx})
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    return results


def parallel_equilibrium_validation(rheology_results_list: List[Dict],
                                   equilibrium_validator,
                                   n_workers: Optional[int] = None,
                                   show_progress: bool = True) -> List[Dict]:
    """
    Parallel equilibrium validation for multiple samples.
    
    Parameters
    ----------
    rheology_results_list : list of Dict
        List of rheology results to validate
    equilibrium_validator : EquilibriumValidator
        Validator instance
    n_workers : int, optional
        Number of workers
    show_progress : bool
        Show progress bar
    
    Returns
    -------
    list of Dict
        Validation results for each sample
    """
    n_samples = len(rheology_results_list)
    
    if n_workers is None:
        n_workers = get_optimal_workers(n_samples)
    
    if n_samples < 5 or n_workers == 1:
        # Serial
        results = []
        for rheology in (tqdm(rheology_results_list, desc="Equilibrium validation") 
                        if show_progress else rheology_results_list):
            result = _validate_single_sample(equilibrium_validator, rheology)
            results.append(result)
        return results
    
    # Parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_validate_single_sample, equilibrium_validator, rheology): i
            for i, rheology in enumerate(rheology_results_list)
        }
        
        if show_progress:
            pbar = tqdm(total=n_samples, desc="Equilibrium validation (parallel)")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                idx = futures[future]
                warnings.warn(f"Sample {idx} failed: {e}")
                results.append({'success': False, 'error': str(e), 'sample_index': idx})
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    return results


def _validate_single_sample(validator, rheology_result: Dict) -> Dict:
    """Helper for single equilibrium validation."""
    # Extract VACF if available
    if 'vacf' in rheology_result:
        vacf_result = validator.check_vacf_symmetry(rheology_result['vacf'])
    else:
        vacf_result = {'success': False, 'error': 'No VACF data'}
    
    # Extract 1P/2P data if available
    if 'one_point' in rheology_result and 'two_point' in rheology_result:
        agreement_result = validator.check_1p_2p_agreement(
            rheology_result['one_point'],
            rheology_result['two_point']
        )
    else:
        agreement_result = {'success': False, 'error': 'No 1P/2P data'}
    
    # Generate composite validity report
    validity = validator.generate_validity_report(
        rheology_result.get('tracks_df'),
        rheology_result
    )
    
    return {
        'success': True,
        'vacf_check': vacf_result,
        'agreement_check': agreement_result,
        'overall_validity': validity
    }


# Convenience functions for each module

def parallel_biased_inference_batch(tracks_df: pd.DataFrame,
                                    corrector,
                                    pixel_size_um: float,
                                    dt: float,
                                    localization_error_um: float,
                                    exposure_time: float = 0.0,
                                    method: str = 'auto',
                                    n_workers: Optional[int] = None,
                                    show_progress: bool = True) -> pd.DataFrame:
    """
    Parallel biased inference batch analysis.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracking data
    corrector : BiasedInferenceCorrector
        Corrector instance
    pixel_size_um : float
        Pixel size
    dt : float
        Frame interval
    localization_error_um : float
        Localization precision
    exposure_time : float
        Exposure time
    method : str
        'auto', 'CVE', 'MLE', or 'MSD'
    n_workers : int, optional
        Number of workers
    show_progress : bool
        Show progress bar
    
    Returns
    -------
    pd.DataFrame
        Results per track
    """
    from functools import partial
    
    # Create analysis function with fixed parameters
    def analyze_track_wrapper(coords, dt, localization_error, exposure_time, method):
        dimensions = coords.shape[1]
        return corrector.analyze_track(
            coords, dt, localization_error, exposure_time, method, dimensions
        )
    
    analysis_func = partial(
        analyze_track_wrapper,
        dt=dt,
        localization_error=localization_error_um,
        exposure_time=exposure_time,
        method=method
    )
    
    return parallel_batch_analyze(
        tracks_df,
        analysis_func,
        n_workers=n_workers,
        show_progress=show_progress,
        pixel_size=pixel_size_um
    )


def parallel_microsecond_batch(tracks_df: pd.DataFrame,
                              handler,
                              pixel_size: float = 1.0,
                              n_workers: Optional[int] = None,
                              show_progress: bool = True,
                              **kwargs) -> List[Dict]:
    """
    Parallel microsecond sampling batch analysis.
    
    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracking data
    handler : IrregularSamplingHandler
        Handler instance
    pixel_size : float
        Pixel size
    n_workers : int, optional
        Number of workers
    show_progress : bool
        Show progress bar
    **kwargs
        Additional arguments for calculate_msd_irregular
    
    Returns
    -------
    list of Dict
        MSD results per track
    """
    track_groups = list(tracks_df.groupby('track_id'))
    n_tracks = len(track_groups)
    
    if n_workers is None:
        n_workers = get_optimal_workers(n_tracks)
    
    if n_tracks < 10 or n_workers == 1:
        # Serial
        results = []
        for track_id, group in (tqdm(track_groups, desc="Microsecond MSD") 
                               if show_progress else track_groups):
            result = handler.calculate_msd_irregular(group, pixel_size, **kwargs)
            result['track_id'] = track_id
            results.append(result)
        return results
    
    # Parallel
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(handler.calculate_msd_irregular, group, pixel_size, **kwargs): track_id
            for track_id, group in track_groups
        }
        
        if show_progress:
            pbar = tqdm(total=n_tracks, desc="Microsecond MSD (parallel)")
        
        for future in as_completed(futures):
            try:
                result = future.result()
                track_id = futures[future]
                result['track_id'] = track_id
                results.append(result)
            except Exception as e:
                track_id = futures[future]
                warnings.warn(f"Track {track_id} failed: {e}")
                results.append({'track_id': track_id, 'success': False, 'error': str(e)})
            
            if show_progress:
                pbar.update(1)
        
        if show_progress:
            pbar.close()
    
    return results


if __name__ == "__main__":
    print(f"Parallel processing module loaded")
    print(f"CPU count: {mp.cpu_count()}")
    print(f"Optimal workers: {get_optimal_workers()}")
