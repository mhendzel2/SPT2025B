"""
Batch Processing Utilities for SPT2025B.
Provides parallel processing, progress tracking, and error handling for batch operations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from datetime import datetime
import logging
from pathlib import Path
import io

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False


class BatchProcessingProgress:
    """Track and report batch processing progress."""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        """
        Initialize progress tracker.
        
        Parameters
        ----------
        total_items : int
            Total number of items to process
        description : str, default "Processing"
            Description of the operation
        """
        self.total_items = total_items
        self.completed_items = 0
        self.failed_items = 0
        self.description = description
        self.start_time = datetime.now()
        self.errors = []
        
        # Create UI elements if in Streamlit context
        if STREAMLIT_AVAILABLE:
            try:
                self.progress_bar = st.progress(0.0)
                self.status_text = st.empty()
                self.has_ui = True
            except Exception:
                self.progress_bar = None
                self.status_text = None
                self.has_ui = False
        else:
            self.progress_bar = None
            self.status_text = None
            self.has_ui = False
    
    def update(self, increment: int = 1, status: str = None, error: str = None):
        """
        Update progress.
        
        Parameters
        ----------
        increment : int, default 1
            Number of items completed in this update
        status : str, optional
            Custom status message
        error : str, optional
            Error message if item failed
        """
        self.completed_items += increment
        
        if error:
            self.failed_items += 1
            self.errors.append(error)
            logging.error(f"Batch processing error: {error}")
        
        if self.has_ui:
            progress = self.completed_items / self.total_items
            self.progress_bar.progress(min(progress, 1.0))
            
            elapsed = (datetime.now() - self.start_time).total_seconds()
            if self.completed_items > 0:
                eta = elapsed / self.completed_items * (self.total_items - self.completed_items)
                eta_str = f"{int(eta)}s" if eta < 60 else f"{int(eta/60)}m {int(eta%60)}s"
            else:
                eta_str = "calculating..."
            
            status_msg = status or f"{self.description} {self.completed_items}/{self.total_items}"
            if self.failed_items > 0:
                status_msg += f" ({self.failed_items} failed)"
            status_msg += f" - ETA: {eta_str}"
            
            self.status_text.text(status_msg)
        else:
            # Console output if no UI
            if self.completed_items % max(1, self.total_items // 10) == 0:
                progress = (self.completed_items / self.total_items) * 100
                print(f"{self.description}: {progress:.1f}% ({self.completed_items}/{self.total_items})")
    
    def complete(self):
        """Mark processing as complete and clean up UI."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if self.has_ui:
            self.progress_bar.empty()
            
            if self.failed_items > 0:
                st.warning(f"✅ Completed with {self.failed_items} errors in {elapsed:.1f}s")
                with st.expander("Show errors"):
                    for error in self.errors:
                        st.error(error)
            else:
                self.status_text.success(f"✅ Completed {self.total_items} items in {elapsed:.1f}s")
        else:
            # Console output
            if self.failed_items > 0:
                print(f"Completed with {self.failed_items} errors in {elapsed:.1f}s")
                for error in self.errors[:5]:  # Show first 5 errors
                    print(f"  Error: {error}")
            else:
                print(f"Completed {self.total_items} items in {elapsed:.1f}s")


def parallel_process_files(
    files: List[Dict[str, Any]],
    process_func: Callable[[Dict], Tuple[Optional[pd.DataFrame], Optional[str]]],
    max_workers: int = 4,
    show_progress: bool = True,
    use_threads: bool = True
) -> Tuple[List[pd.DataFrame], List[Dict[str, str]]]:
    """
    Process multiple files in parallel with progress tracking.
    
    Parameters
    ----------
    files : List[Dict]
        List of file information dictionaries
    process_func : Callable
        Function to process each file, should return (dataframe, error_message)
    max_workers : int, default 4
        Maximum number of parallel workers
    show_progress : bool, default True
        Show progress bar
    use_threads : bool, default True
        Use threads (True) or processes (False)
    
    Returns
    -------
    Tuple[List[pd.DataFrame], List[Dict]]
        (list_of_dataframes, list_of_errors)
    """
    results = []
    errors = []
    
    if show_progress:
        progress = BatchProcessingProgress(len(files), "Loading files")
    
    # Choose executor type
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor
    
    # Use parallel processing
    with executor_class(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_func, f): f 
            for f in files
        }
        
        # Process completed tasks
        for future in as_completed(future_to_file):
            file_info = future_to_file[future]
            
            try:
                df, error = future.result()
                
                if df is not None:
                    results.append(df)
                if error is not None:
                    errors.append({
                        'file': file_info.get('file_name', 'unknown'),
                        'error': error
                    })
                
                if show_progress:
                    progress.update(error=error)
            except Exception as e:
                error_msg = f"Unexpected error processing {file_info.get('file_name', 'unknown')}: {str(e)}"
                errors.append({
                    'file': file_info.get('file_name', 'unknown'),
                    'error': error_msg
                })
                if show_progress:
                    progress.update(error=error_msg)
    
    if show_progress:
        progress.complete()
    
    return results, errors


def load_file_with_retry(
    file_info: Dict[str, Any],
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Load a file with retry logic and robust data cleaning.
    
    Parameters
    ----------
    file_info : Dict
        File information dictionary
    max_retries : int, default 3
        Maximum number of retry attempts
    retry_delay : float, default 1.0
        Delay between retries in seconds
    
    Returns
    -------
    Tuple[Optional[pd.DataFrame], Optional[str]]
        (dataframe, error_message)
    """
    import time
    
    for attempt in range(max_retries):
        try:
            df = None
            
            if 'data' in file_info and file_info['data'] is not None:
                # Load from bytes
                data_bytes = file_info['data']
                if isinstance(data_bytes, str):
                    data_bytes = data_bytes.encode('utf-8')
                
                # Try reading with standard CSV parser
                try:
                    df = pd.read_csv(io.BytesIO(data_bytes), encoding='utf-8')
                except UnicodeDecodeError:
                    # Try alternative encoding
                    df = pd.read_csv(io.BytesIO(data_bytes), encoding='latin-1')
                    
            elif 'data_path' in file_info and file_info['data_path']:
                if Path(file_info['data_path']).exists():
                    df = pd.read_csv(file_info['data_path'])
            
            if df is None:
                return None, "No valid data source found"
            
            # Clean the data
            if not df.empty:
                # Remove any completely empty rows
                df = df.dropna(how='all')
                
                # Standardize column names (strip whitespace, lowercase)
                df.columns = df.columns.str.strip()
                
                # Check if first row might be a duplicate header
                if len(df) > 1:
                    # If all values in first row are strings matching column names
                    first_row = df.iloc[0]
                    if all(isinstance(val, str) and val.strip() in df.columns for val in first_row if pd.notna(val)):
                        logging.info(f"Removing duplicate header row from {file_info.get('name', 'file')}")
                        df = df.iloc[1:].reset_index(drop=True)
                
                # Convert numeric columns
                numeric_cols = ['x', 'y', 'z', 'frame', 'track_id', 'Frame', 'Track', 'TrackID']
                for col in df.columns:
                    if col in numeric_cols or any(x in col.lower() for x in ['frame', 'track', 'pos']):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except Exception:
                            pass
                
                # Remove rows with NaN in critical columns
                critical_cols = [c for c in ['track_id', 'frame', 'x', 'y'] if c in df.columns]
                if critical_cols:
                    df = df.dropna(subset=critical_cols)
            
            return df, None
            
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed, retrying: {str(e)}")
                time.sleep(retry_delay)
            else:
                error_msg = f"Failed after {max_retries} attempts: {str(e)}"
                logging.error(error_msg)
                return None, error_msg
    
    return None, "Max retries exceeded"


def pool_dataframes_efficiently(
    dataframes: List[pd.DataFrame],
    validate: bool = True,
    deduplicate: bool = True
) -> pd.DataFrame:
    """
    Efficiently pool multiple DataFrames with validation and column standardization.
    
    Parameters
    ----------
    dataframes : List[pd.DataFrame]
        List of DataFrames to pool
    validate : bool, default True
        Validate schema consistency
    deduplicate : bool, default True
        Remove duplicate rows
    
    Returns
    -------
    pd.DataFrame
        Pooled DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    # Filter out empty dataframes
    dataframes = [df for df in dataframes if not df.empty]
    
    if not dataframes:
        return pd.DataFrame()
    
    # Standardize column names across all dataframes
    standardized_dfs = []
    for df in dataframes:
        df = df.copy()
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        # Standardize common column name variations
        column_mapping = {
            'Track': 'track_id',
            'TrackID': 'track_id',
            'track_ID': 'track_id',
            'Frame': 'frame',
            'FRAME': 'frame',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'Position X': 'x',
            'Position Y': 'y',
            'Position Z': 'z',
        }
        
        # Apply mapping if columns exist
        rename_dict = {}
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                rename_dict[old_name] = new_name
        
        if rename_dict:
            df = df.rename(columns=rename_dict)
            logging.info(f"Standardized column names: {rename_dict}")
        
        standardized_dfs.append(df)
    
    dataframes = standardized_dfs
    
    # Validate schema consistency
    if validate and len(dataframes) > 1:
        first_cols = set(dataframes[0].columns)
        inconsistent = False
        for i, df in enumerate(dataframes[1:], 1):
            if set(df.columns) != first_cols:
                logging.warning(f"DataFrame {i} has inconsistent columns: {set(df.columns).symmetric_difference(first_cols)}")
                inconsistent = True
        
        if inconsistent:
            # Align columns across all dataframes
            all_cols = sorted(set.union(*[set(df.columns) for df in dataframes]))
            dataframes = [df.reindex(columns=all_cols) for df in dataframes]
            logging.info(f"Aligned all DataFrames to common schema with {len(all_cols)} columns")
    
    # Concatenate efficiently
    result = pd.concat(dataframes, ignore_index=True, copy=False)
    
    # Remove duplicates if requested
    if deduplicate and 'track_id' in result.columns and 'frame' in result.columns:
        before_count = len(result)
        result = result.drop_duplicates(subset=['track_id', 'frame'], keep='first')
        after_count = len(result)
        if before_count > after_count:
            logging.info(f"Removed {before_count - after_count} duplicate rows")
    
    return result


def batch_analyze_tracks(
    tracks_list: List[pd.DataFrame],
    analysis_func: Callable[[pd.DataFrame, Dict], Dict],
    analysis_params: Dict[str, Any],
    max_workers: int = 4,
    show_progress: bool = True
) -> List[Dict[str, Any]]:
    """
    Perform analysis on multiple track DataFrames in parallel.
    
    Parameters
    ----------
    tracks_list : List[pd.DataFrame]
        List of track DataFrames
    analysis_func : Callable
        Analysis function to apply to each DataFrame
    analysis_params : Dict
        Parameters for the analysis function
    max_workers : int, default 4
        Maximum number of parallel workers
    show_progress : bool, default True
        Show progress bar
    
    Returns
    -------
    List[Dict]
        List of analysis results
    """
    results = []
    
    if show_progress:
        progress = BatchProcessingProgress(len(tracks_list), "Analyzing tracks")
    
    def process_single(tracks_df: pd.DataFrame) -> Dict:
        """Process a single track DataFrame."""
        try:
            return analysis_func(tracks_df, **analysis_params)
        except Exception as e:
            logging.error(f"Analysis failed: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # Use thread pool for analysis
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single, tracks) for tracks in tracks_list]
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                
                if show_progress:
                    error = None if result.get('success', False) else result.get('error')
                    progress.update(error=error)
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                results.append({'success': False, 'error': error_msg})
                if show_progress:
                    progress.update(error=error_msg)
    
    if show_progress:
        progress.complete()
    
    return results
