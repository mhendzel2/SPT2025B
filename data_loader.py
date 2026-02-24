"""
Data loading utilities for the SPT Analysis application.
Handles loading various file formats for images and track data.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import os
import json
import re
from typing import List, Dict, Tuple, Any, Optional, Union
from PIL import Image
from utils import format_track_data, validate_tracks_dataframe
from special_file_handlers import load_trackmate_file, load_cropped_cell3_spots, load_ms2_spots_file, load_imaris_file, load_trackmate_xml_file
from mvd2_handler import load_mvd2_file
from volocity_handler import load_volocity_file
from state_manager import StateManager
from logging_config import get_logger
from security_utils import SecureFileHandler

# Initialize logger
logger = get_logger(__name__)


def validate_column_mapping(df, x_col, y_col, frame_col, track_id_col):
    """Validate that detected columns make sense for tracking data."""
    if not all(col in df.columns for col in [x_col, y_col, frame_col, track_id_col]):
        return False
    
    # Check if x,y columns are numeric
    if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
        return False
    
    # Check if frame column is numeric (if provided)
    if frame_col and not pd.api.types.is_numeric_dtype(df[frame_col]):
        return False
    
    # Check if track_id column exists (if provided)
    if track_id_col and not pd.api.types.is_numeric_dtype(df[track_id_col]):
        return False
    
    return True


def clean_tracks(df: pd.DataFrame, warn_user: bool = True) -> pd.DataFrame:
    """
    Clean track data by removing invalid values (NaN, Inf) and warning the user.
    
    Performs comprehensive validation:
    - Checks for NaN and Inf values in critical columns (x, y, z, frame, track_id)
    - Removes rows with invalid values
    - Warns user about data quality issues
    - Ensures all numeric columns are valid
    
    Parameters
    ----------
    df : pd.DataFrame
        Track data with standard columns (track_id, frame, x, y, optional z)
    warn_user : bool
        Whether to display Streamlit warnings about invalid data
        
    Returns
    -------
    pd.DataFrame
        Cleaned track data with invalid rows removed
    """
    if df.empty:
        return df
    
    initial_rows = len(df)
    
    # Define critical columns that must be valid
    critical_cols = ['track_id', 'frame', 'x', 'y']
    if 'z' in df.columns:
        critical_cols.append('z')
    
    # Filter to columns that exist
    existing_critical_cols = [col for col in critical_cols if col in df.columns]
    
    if not existing_critical_cols:
        logger.warning("No critical columns found for validation")
        return df
    
    # Check for NaN values
    nan_mask = df[existing_critical_cols].isna().any(axis=1)
    nan_count = nan_mask.sum()
    
    # Check for Inf values
    inf_mask = np.isinf(df[existing_critical_cols].select_dtypes(include=[np.number])).any(axis=1)
    inf_count = inf_mask.sum()
    
    # Combined invalid mask
    invalid_mask = nan_mask | inf_mask
    invalid_count = invalid_mask.sum()
    
    if invalid_count > 0:
        # Log details about invalid data
        logger.warning(f"Found {invalid_count} invalid rows: {nan_count} with NaN, {inf_count} with Inf")
        
        if warn_user:
            # Detailed breakdown by column
            invalid_details = []
            for col in existing_critical_cols:
                col_nan = df[col].isna().sum()
                col_inf = np.isinf(df[col]).sum() if pd.api.types.is_numeric_dtype(df[col]) else 0
                if col_nan > 0 or col_inf > 0:
                    invalid_details.append(f"  • {col}: {col_nan} NaN, {col_inf} Inf")
            
            warning_msg = f"⚠️ **Data Quality Warning**: Found {invalid_count} invalid rows ({invalid_count/initial_rows*100:.1f}%)\n\n"
            warning_msg += "\n".join(invalid_details)
            warning_msg += f"\n\nThese rows will be removed. {initial_rows - invalid_count} valid rows remain."
            
            st.warning(warning_msg)
        
        # Remove invalid rows
        df_clean = df[~invalid_mask].copy()
        
        # Verify cleaning worked
        remaining_invalid = df_clean[existing_critical_cols].isna().any().any() or \
                           np.isinf(df_clean[existing_critical_cols].select_dtypes(include=[np.number])).any().any()
        
        if remaining_invalid:
            logger.error("Clean operation failed - invalid values still present")
            if warn_user:
                st.error("❌ Critical error: Unable to fully clean data. Please check input file.")
        else:
            logger.info(f"Successfully cleaned {invalid_count} invalid rows")
        
        return df_clean
    
    else:
        logger.info("Data validation passed - no invalid values found")
        return df


def load_image_file(file) -> List[np.ndarray]:
    """
    Load an image file into a NumPy array.
    
    Parameters
    ----------
    file : UploadedFile
        File uploaded through Streamlit
        
    Returns
    -------
    list
        List of image frames as NumPy arrays
    """
    file_extension = os.path.splitext(file.name)[1].lower()
    
    # For standard image formats
    if file_extension in ['.png', '.jpg', '.jpeg']:
        try:
            image = Image.open(file)
            return [np.array(image)]
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    # For TIFF files (which may be multi-page)
    elif file_extension in ['.tif', '.tiff']:
        try:
            frames = []
            with Image.open(file) as img:
                # Check if the image is a multi-frame TIFF
                try:
                    # Get the number of frames
                    for i in range(10000):  # Assuming no more than 10000 frames
                        img.seek(i)
                        frame_array = np.array(img)
                        frames.append(frame_array)
                except EOFError:
                    # End of file reached
                    pass
            
            if not frames:
                # If no frames were read, try a simpler approach
                image = Image.open(file)
                frames = [np.array(image)]
            
            # If we have multiple frames, determine if it's a timelapse or multichannel image
            if len(frames) > 1:
                # Check if all frames have the same dimensions
                first_shape = frames[0].shape[:2]  # height, width
                
                # Heuristic: If there are many frames (>10), it's likely a timelapse
                # If there are few frames (<=10), it might be multichannel data
                TIMELAPSE_THRESHOLD = 10
                
                if all(frame.shape[:2] == first_shape for frame in frames):
                    if len(frames) > TIMELAPSE_THRESHOLD:
                        # Many frames with same dimensions = timelapse series
                        st.info(f"Detected {len(frames)} frames - treating as timelapse series")
                        return frames
                    else:
                        # Few frames - could be multichannel, but check frame shape
                        if len(frames[0].shape) == 2:  # Grayscale frames
                            # For <=10 grayscale frames, ask user or default to timelapse
                            # Default to timelapse for safety (common use case)
                            st.info(f"Detected {len(frames)} grayscale frames - treating as timelapse series")
                            st.info("💡 Tip: If this is a multichannel image, please convert it to a proper multichannel TIFF format.")
                            return frames
                        elif len(frames[0].shape) == 3:  # Already RGB frames
                            # Each frame is already multichannel, return as separate frames
                            st.info(f"Detected {len(frames)} RGB frames - treating as time series")
                            return frames
                else:
                    st.info(f"Detected {len(frames)} frames with different dimensions - treating as time series")
                    return frames
            
            return frames
        except Exception as e:
            raise ValueError(f"Error loading TIFF image: {str(e)}")
    
    # For MVD2 files (Olympus spinning disk microscopy)
    elif file_extension in ['.mvd2', '.mvd']:
        try:
            frames, metadata = load_mvd2_file(file)
            return frames
        except Exception as e:
            raise ValueError(f"Error loading MVD2 file: {str(e)}")
    
    # For Volocity UIC files (Perkin Elmer spinning disk microscopy)
    elif file_extension in ['.uic', '.vol']:
        try:
            frames, metadata = load_volocity_file(file)
            return frames
        except Exception as e:
            raise ValueError(f"Error loading Volocity file: {str(e)}")
    
    # Unsupported format
    else:
        raise ValueError(f"Unsupported image format: {file_extension}")

def _standardize_trackmate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect TrackMate-style exported columns and map to internal schema.
    Expected incoming columns (example):
        TRACK_ID, POSITION_X, POSITION_Y, POSITION_Z, FRAME, POSITION_T (optional)
    Output standardized columns:
        track_id, x, y, z (if present), frame, t (if POSITION_T present)
    Non-existing targets are skipped gracefully.
    """
    if not isinstance(df, pd.DataFrame):
        return df

    # Heuristic: presence of several signature columns
    signature = {"TRACK_ID", "POSITION_X", "POSITION_Y", "FRAME"}
    if not signature.issubset(set(df.columns)):
        return df  # Not TrackMate style; leave untouched

    mappings = {
        "TRACK_ID": "track_id",
        "POSITION_X": "x",
        "POSITION_Y": "y",
        "POSITION_Z": "z",
        "FRAME": "frame",
        "POSITION_T": "t",
    }
    rename_map = {src: dst for src, dst in mappings.items() if src in df.columns}
    df = df.rename(columns=rename_map)

    # Ensure integer track_id / frame where possible
    for col in ("track_id", "frame"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


def _safe_to_numeric_series(df: pd.DataFrame, columns) -> None:
    """
    Convert listed columns to numeric if they exist and are 1-D Series.
    Skips silently if column missing or not convertible.
    """
    for col in columns:
        if col in df.columns:
            ser = df[col]
            # Only convert if it's a Series (not already numeric or not an object like list-of-lists)
            if isinstance(ser, pd.Series):
                df[col] = pd.to_numeric(ser, errors="coerce")


def _remove_redundant_header_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove accidental repeated header rows inside the data.
    Heuristics:
      - A row is flagged if, for key columns, the cell value (case-insensitive)
        equals the column name or a known header alias (e.g. 'Track ID').
      - Or if a high fraction of non-null string cells match column headers.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    col_lower = {c.lower(): c for c in df.columns}
    header_aliases = {
        "track id": {"track id", "track_id", "TRACK_ID"},
        "frame": {"frame", "FRAME"},
        "x": {"x", "position_x", "POSITION_X"},
        "y": {"y", "position_y", "POSITION_Y"},
        "z": {"z", "position_z", "POSITION_Z"},
        "t": {"t", "position_t", "POSITION_T"},
        "quality": {"quality", "QUALITY"},
    }
    # Map actual present columns to alias sets (only those that exist)
    present_alias_sets = []
    for canonical, aliases in header_aliases.items():
        # include if at least one alias corresponds to an existing column name
        if any(a in df.columns for a in aliases):
            present_alias_sets.append(aliases)

    header_name_set = set(a for s in present_alias_sets for a in s)
    header_name_set |= set(df.columns)
    header_name_set_lower = {h.lower() for h in header_name_set}

    rows_to_drop = []
    for idx, row in df.iterrows():
        values = row.values
        # Count header-like matches
        header_like = 0
        string_cells = 0
        key_column_hits = 0

        for col, val in row.items():
            if isinstance(val, str):
                string_cells += 1
                v = val.strip().lower()
                if v in header_name_set_lower:
                    header_like += 1
                    # If this value equals (case-insensitive) the column name or alias for a key column
                    for alias_set in present_alias_sets:
                        if v in {a.lower() for a in alias_set}:
                            key_column_hits += 1

        if string_cells == 0:
            continue

        fraction_header_like = header_like / max(1, string_cells)

        # Heuristic conditions:
        # 1. All key columns present in this row appear as header tokens
        # 2. Or >= 70% of its string cells look like header tokens and at least 3 such cells
        if (key_column_hits >= max(2, len(present_alias_sets) // 2)) or (
            header_like >= 3 and fraction_header_like >= 0.7
        ):
            rows_to_drop.append(idx)

    if rows_to_drop:
        df = df.drop(rows_to_drop)
        df = df.reset_index(drop=True)
        # Optional: simple debug print (replace with logging if available)
        try:
            print(f"Removed {len(rows_to_drop)} repeated header row(s): indices {rows_to_drop}")
        except Exception:
            pass
    return df

def _remove_units_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unit-annotation rows that sometimes follow header rows in Excel/CSV exports.
    Heuristics:
      - Count string cells in a row; count cells containing unit-like tokens
        (e.g., 'μm', 'um', 'micron', 'nm', 'pixel', 'sec', 'ms', 'quality').
      - Drop the row if there are at least 2 unit-like hits AND they comprise >= 50% of string cells.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    unit_patterns = [
        r"\bμm\b", r"\bum\b", r"micron", r"\bnm\b", r"pixel", r"\bsec\b", r"\bms\b",
        r"\bseconds?\b", r"\bmicrons?\b", r"quality", r"\bframes?\b", r"\bHz\b",
        r"\[[^\]]*\]",  # [um], [nm], etc.
    ]
    unit_regex = re.compile("|".join(unit_patterns), re.IGNORECASE)

    rows_to_drop = []
    for idx, row in df.iterrows():
        string_cells = 0
        unit_hits = 0
        for val in row.values:
            if isinstance(val, str):
                string_cells += 1
                if unit_regex.search(val.strip()):
                    unit_hits += 1
        if string_cells > 0 and unit_hits >= 2 and unit_hits / string_cells >= 0.5:
            rows_to_drop.append(idx)

    if rows_to_drop:
        df = df.drop(rows_to_drop).reset_index(drop=True)
        try:
            print(f"Removed {len(rows_to_drop)} unit annotation row(s): indices {rows_to_drop}")
        except Exception:
            pass
    return df

def load_tracks_file(
    file,
    pixel_size: Optional[float] = None,
    frame_interval: Optional[float] = None,
) -> pd.DataFrame:
    """
    Load track data from various file formats with security validation.
    
    Parameters
    ----------
    file : UploadedFile
        File uploaded through Streamlit
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the track data
    """
    try:
        # Prefer uploaded/file-like objects over path handling.
        has_file_like_api = hasattr(file, 'getvalue') or hasattr(file, 'read')

        # Support pathlib/str inputs used in tests and CLI contexts.
        if (not has_file_like_api) and isinstance(file, (str, os.PathLike)):
            file_path = os.fspath(file)
            with open(file_path, 'rb') as f_in:
                file_bytes = f_in.read()
            file_obj = io.BytesIO(file_bytes)
            file_obj.name = os.path.basename(file_path)
            file_obj.size = len(file_bytes)
            file = file_obj

        # Normalize size for in-memory uploads/test doubles (e.g., MagicMock).
        if hasattr(file, 'getvalue'):
            try:
                payload = file.getvalue()
                payload_size = len(payload) if payload is not None else 0
                current_size = getattr(file, 'size', None)
                if not isinstance(current_size, (int, float, np.integer, np.floating)):
                    file.size = int(payload_size)
            except Exception:
                pass

        # Validate filename
        safe_filename = SecureFileHandler.validate_filename(file.name)
        logger.info(f"Loading track file: {safe_filename}")
        
        # Validate file size
        SecureFileHandler.validate_file_size(file)
        
        file_extension = os.path.splitext(file.name)[1].lower()
        logger.debug(f"File extension: {file_extension}, Size: {file.size / 1024:.1f} KB")
        
    except ValueError as e:
        logger.error(f"File validation failed: {e}")
        st.error(f"File validation failed: {e}")
        return pd.DataFrame()
    
    # For Excel files
    if file_extension in ['.xlsx', '.xls']:
        try:
            # Read Excel file with openpyxl engine
            try:
                df = pd.read_excel(file, engine='openpyxl')
            except ImportError:
                st.error("Excel support requires the 'openpyxl' package. Please install it to load Excel files.")
                return pd.DataFrame()
            
            st.info(f"Loaded Excel file with {len(df)} rows and columns: {list(df.columns)}")
            
            # Clean typical artifacts: repeated header rows and units rows inside the data
            df = _remove_redundant_header_rows(df)
            df = _remove_units_rows(df)
            
            # Auto-detect column mapping for Excel files with enhanced detection
            column_map = {}
            potential_mappings = {}
            
            # Look for standard tracking columns with flexible naming
            for col in df.columns:
                col_str = str(col).strip()
                col_lower = col_str.lower()
                
                # Track ID detection (more comprehensive)
                if (col_str in ['track_id', 'Track_ID', 'TrackID', 'track ID', 'ID'] or
                    'track' in col_lower and 'id' in col_lower or
                    col_lower in ['particle_id', 'spot_id'] or
                    any(keyword in col_lower for keyword in ['track', 'particle', 'spot']) and any(keyword in col_lower for keyword in ['id', 'number', 'index'])):
                    potential_mappings['track_id'] = col
                    column_map[col] = 'track_id'
                    
                # Frame detection (more comprehensive)
                elif (col_str in ['Frame', 'frame', 'Time', 'time', 'T', 't'] or
                      col_lower in ['frame', 'time', 't', 'timepoint', 'frames']):
                    potential_mappings['frame'] = col
                    column_map[col] = 'frame'
                    
                # X coordinate detection
                elif (col_str in ['X', 'x', 'x_position', 'X_position'] or
                      col_lower in ['x', 'x_position', 'pos_x', 'position_x', 'x_coord']):
                    potential_mappings['x'] = col
                    column_map[col] = 'x'
                    
                # Y coordinate detection  
                elif (col_str in ['Y', 'y', 'y_position', 'Y_position'] or
                      col_lower in ['y', 'y_position', 'pos_y', 'position_y', 'y_coord']):
                    potential_mappings['y'] = col
                    column_map[col] = 'y'
                    
                # Z coordinate detection
                elif (col_str in ['Z', 'z', 'z_position', 'Z_position'] or
                      col_lower in ['z', 'z_position', 'pos_z', 'position_z', 'z_coord']):
                    potential_mappings['z'] = col
                    column_map[col] = 'z'
            
            st.info(f"Detected column mappings: {potential_mappings}")
            
            # If we don't have automatic mappings, try to infer from data structure
            required_cols = ['track_id', 'frame', 'x', 'y']
            if not all(req_col in potential_mappings for req_col in required_cols):
                st.warning("Could not automatically detect all required columns. Attempting manual inference...")
                
                # Look at numeric columns and their ranges to infer likely mappings
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                st.info(f"Numeric columns found: {numeric_cols}")
                
                for col in numeric_cols:
                    if col not in column_map:
                        unique_count = df[col].nunique()
                        col_range = df[col].max() - df[col].min()
                        
                        # Heuristic: track_id usually has moderate unique count
                        if 'track_id' not in potential_mappings and unique_count > 1 and unique_count < len(df) * 0.5:
                            potential_mappings['track_id'] = col
                            column_map[col] = 'track_id'
                            st.info(f"Inferred '{col}' as track_id (unique values: {unique_count})")
                            
                        # Heuristic: frame usually starts from 0 or 1 and increments
                        elif 'frame' not in potential_mappings and df[col].min() >= 0 and unique_count > 1:
                            potential_mappings['frame'] = col
                            column_map[col] = 'frame'
                            st.info(f"Inferred '{col}' as frame (range: {df[col].min()} - {df[col].max()})")
                            
                        # Heuristic: x,y coordinates usually have larger ranges
                        elif 'x' not in potential_mappings and col_range > 10:
                            potential_mappings['x'] = col
                            column_map[col] = 'x'
                            st.info(f"Inferred '{col}' as x coordinate (range: {col_range:.2f})")
                            
                        elif 'y' not in potential_mappings and col_range > 10:
                            potential_mappings['y'] = col
                            column_map[col] = 'y'
                            st.info(f"Inferred '{col}' as y coordinate (range: {col_range:.2f})")
            
            # Apply column mapping
            tracks_df = df.rename(columns=column_map)

            # If renaming created duplicate standardized names (e.g., two columns mapped to 'x'),
            # keep the first occurrence to ensure 1-D selections below.
            if tracks_df.columns.duplicated().any():
                dup_names = tracks_df.columns[tracks_df.columns.duplicated()].unique().tolist()
                st.error(f"⚠️ Critical: Duplicate columns detected: {dup_names}")
                st.info(f"Keeping first occurrence of each duplicate column. Please review your data file.")
                
                # Log for debugging
                import logging
                logging.warning(f"Dropped duplicate columns in file: {dup_names}")
                
                tracks_df = tracks_df.loc[:, ~tracks_df.columns.duplicated(keep='first')]
            
            # Check if we have required columns
            missing_cols = [col for col in required_cols if col not in tracks_df.columns]
            
            if missing_cols:
                st.error(f"Excel file is missing required columns: {missing_cols}")
                st.info("Available columns: " + ", ".join(df.columns.tolist()))
                st.info("Detected mappings: " + str(potential_mappings))
                
                # Show data sample to help user understand structure
                st.info("First few rows of your data:")
                st.dataframe(df.head())
                return pd.DataFrame()
            
            # Convert to numeric where appropriate with validation
            for col in ['track_id', 'frame', 'x', 'y', 'z']:
                if col in tracks_df.columns:
                    # Guard against duplicate columns resulting in a DataFrame selection
                    col_obj = tracks_df[col]
                    if isinstance(col_obj, pd.DataFrame):
                        # Pick the first occurrence
                        col_obj = col_obj.iloc[:, 0]
                    
                    # Track conversion issues
                    original_count = len(col_obj)
                    converted = pd.to_numeric(col_obj, errors='coerce')
                    nan_count = converted.isna().sum()
                    
                    # Report conversion issues
                    if nan_count > 0:
                        nan_pct = (nan_count / original_count) * 100
                        if nan_pct > 10:
                            st.error(f"⚠️ Column '{col}': {nan_count} ({nan_pct:.1f}%) invalid values converted to NaN")
                        else:
                            st.warning(f"Column '{col}': {nan_count} invalid values converted to NaN")
                    
                    tracks_df[col] = converted
            
            # Remove rows with NaN values in essential columns
            tracks_df = tracks_df.dropna(subset=['track_id', 'frame', 'x', 'y'])
            
            if tracks_df.empty:
                st.error("No valid track data found in Excel file after processing")
                return pd.DataFrame()
            
            # Generate frame numbers if missing or invalid
            if 'frame' not in tracks_df.columns or tracks_df['frame'].isna().all():
                st.warning("No valid frame information found. Generating sequential frame numbers.")
                # Group by track_id and assign sequential frame numbers
                tracks_df['frame'] = tracks_df.groupby('track_id').cumcount()
            
            st.success(f"Successfully loaded {len(tracks_df)} data points from {tracks_df['track_id'].nunique()} tracks")
            standardized_df = format_track_data(tracks_df)
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            
            # Clean invalid values (NaN, Inf)
            standardized_df = clean_tracks(standardized_df, warn_user=True)
            
            sm = StateManager.get_instance()
            sm.set_tracks(standardized_df, filename=file.name)
            
            return standardized_df
            
        except Exception as e:
            st.error(f"Error loading Excel file: {str(e)}")
            import traceback
            st.error(f"Detailed error: {traceback.format_exc()}")
            return pd.DataFrame()
    
    # For CSV files
    elif file_extension == '.csv':
        try:
            # Read the first few lines to check format and delimiter
            file_stream = io.StringIO(file.getvalue().decode("utf-8"))
            header_lines = [file_stream.readline() for _ in range(5)]
            sample = "".join(header_lines)
            file_stream.seek(0)
            
            # Determine delimiter
            if ";" in sample:
                sep = ";"
            elif "," in sample:
                sep = ","
            elif "\t" in sample:
                sep = "\t"
            else:
                sep = ","  # Default to comma
            
            # Determine file format
            # Special handling for MS2_spots format
            if "ms2_spots" in file.name.lower():
                file_stream.seek(0)
                tracks_df = load_ms2_spots_file(file_stream, sep=sep)
                
            # Special handling for Cropped_cell3_spots format
            elif "cell3_spots" in file.name.lower():
                file_stream.seek(0)
                tracks_df = load_cropped_cell3_spots(file_stream, sep=sep)
                
            # Special handling for TrackMate format
            elif any(("TRACK_ID" in line or "POSITION_X" in line or "Track ID" in line) for line in header_lines):
                file_stream.seek(0)
                tracks_df = load_trackmate_file(file_stream, sep=sep)
                
            # Standard CSV format with simple structure
            else:
                file_stream.seek(0)
                tracks_df = pd.read_csv(file_stream, sep=sep)
                
                # Enhanced column mapping for various formats including intensity channels
                column_map = {}
                for col in tracks_df.columns:
                    col_clean = col.strip()
                    col_lower = col_clean.lower()
                    
                    # Track ID mappings
                    if col_clean == 'track_id':
                        column_map[col] = 'track_id'
                    elif 'track' in col_lower and 'id' in col_lower:
                        column_map[col] = 'track_id'
                    # Frame mappings  
                    elif col_clean == 'Frame':
                        column_map[col] = 'frame'
                    elif col_clean == 'frame':
                        column_map[col] = 'frame'
                    elif col_clean == 'T':
                        column_map[col] = 'frame'
                    # Coordinate mappings
                    elif col_clean == 'X':
                        column_map[col] = 'x'
                    elif col_clean == 'x':
                        column_map[col] = 'x'
                    elif col_clean == 'Y':
                        column_map[col] = 'y'
                    elif col_clean == 'y':
                        column_map[col] = 'y'
                    elif col_clean == 'Z':
                        column_map[col] = 'z'
                    elif col_clean == 'z':
                        column_map[col] = 'z'
                    # Other useful columns
                    elif col_clean == 'Quality':
                        column_map[col] = 'quality'
                    elif 'spot' in col_lower and 'id' in col_lower:
                        column_map[col] = 'spot_id'
                
                # Keep intensity and other analysis columns without mapping
                # This preserves all the channel information for analysis
                intensity_keywords = ['intensity', 'mean', 'median', 'min', 'max', 'total', 'sum', 'std', 
                                    'contrast', 'snr', 'signal', 'noise', 'ch1', 'ch2', 'ch3']
                
                # Apply the mapping only to basic tracking columns
                tracks_df = tracks_df.rename(columns=column_map)
                
                # Ensure we have required columns
                required_cols = ['track_id', 'frame', 'x', 'y']
                missing_cols = [col for col in required_cols if col not in tracks_df.columns]
                
                if missing_cols:
                    # Try to find unmapped columns that could be our missing ones
                    unmapped_cols = [col for col in tracks_df.columns if col not in column_map.values()]
                    numeric_cols = tracks_df.select_dtypes(include=[np.number]).columns.tolist()
                    
                    # Auto-assign numeric columns if we have enough
                    if len(numeric_cols) >= len(required_cols):
                        auto_map = {}
                        remaining_required = missing_cols.copy()
                        
                        for col in numeric_cols:
                            if 'track' in col.lower() and 'track_id' in remaining_required:
                                auto_map[col] = 'track_id'
                                remaining_required.remove('track_id')
                            elif 'frame' in col.lower() and 'frame' in remaining_required:
                                auto_map[col] = 'frame'
                                remaining_required.remove('frame')
                            elif col.lower() == 'x' and 'x' in remaining_required:
                                auto_map[col] = 'x'
                                remaining_required.remove('x')
                            elif col.lower() == 'y' and 'y' in remaining_required:
                                auto_map[col] = 'y'
                                remaining_required.remove('y')
                        
                        tracks_df = tracks_df.rename(columns=auto_map)
            
            # Enhanced handling for Cropped_spots files - use standardized column names
            if ('Cropped_spots' in file.name or 'cropped_spots' in file.name.lower()) and \
               all(col in tracks_df.columns for col in ['track_id', 'frame', 'x', 'y']):
                
                # Use standardized lowercase column names that were created by the column mapping above
                data_dict = {
                    'track_id': pd.to_numeric(tracks_df['track_id'], errors='coerce'),
                    'frame': pd.to_numeric(tracks_df['frame'], errors='coerce'),  # Use lowercase 'frame'
                    'x': pd.to_numeric(tracks_df['x'], errors='coerce'),          # Use lowercase 'x'
                    'y': pd.to_numeric(tracks_df['y'], errors='coerce')           # Use lowercase 'y'
                }
                
                # Check for optional columns using lowercase names
                if 'z' in tracks_df.columns:
                    data_dict['z'] = pd.to_numeric(tracks_df['z'], errors='coerce')
                else:
                    data_dict['z'] = pd.Series([0] * len(tracks_df), dtype=float)
                    
                if 'quality' in tracks_df.columns:
                    data_dict['quality'] = pd.to_numeric(tracks_df['quality'], errors='coerce')
                else:
                    data_dict['quality'] = pd.Series([1] * len(tracks_df), dtype=float)
                
                result_df = pd.DataFrame(data_dict)
                
                # Remove rows with NaN values in essential columns
                result_df = result_df.dropna(subset=['track_id', 'frame', 'x', 'y'])
                if not result_df.empty:
                    result_df['track_id'] = result_df['track_id'].astype(int)
                    result_df['frame'] = result_df['frame'].astype(int)
                
                return result_df
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            
            # Clean invalid values (NaN, Inf)
            standardized_df = clean_tracks(standardized_df, warn_user=True)

            return standardized_df
            
        except pd.errors.ParserError as e:
            st.error("❌ CSV parsing failed - check file format")
            st.info("💡 Suggestions:")
            st.info("- Verify the delimiter (comma, semicolon, tab)")
            st.info("- Check for malformed rows")
            st.info("- Ensure consistent number of columns")
            raise ValueError(f"CSV parsing error: {str(e)}")
        except UnicodeDecodeError as e:
            st.error("❌ File encoding issue")
            st.info("💡 Try saving your file as UTF-8 encoded CSV")
            raise ValueError(f"Encoding error: {str(e)}")
        except ValueError as e:
            # Re-raise validation errors
            raise
        except Exception as e:
            import traceback
            st.error(f"❌ Unexpected error loading CSV file")
            st.error(f"Error details: {str(e)}")
            if st.checkbox("Show detailed traceback"):
                st.code(traceback.format_exc())
            raise
    
    # For Excel files
    elif file_extension in ['.xlsx', '.xls']:
        try:
            tracks_df = pd.read_excel(file)
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)

            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            
            return standardized_df
            
        except Exception as e:
            raise ValueError(f"Error loading Excel file: {str(e)}")
    
    # For HDF5 files
    elif file_extension == '.h5':
        try:
            import h5py
            
            # Create a temporary file to save the uploaded content
            with open("temp.h5", "wb") as f:
                f.write(file.getvalue())
            
            # Open the HDF5 file
            with h5py.File("temp.h5", "r") as h5file:
                # Try to find datasets containing track data
                track_datasets = []
                
                def find_track_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset) and (
                        "track" in name.lower() or 
                        "trajectory" in name.lower() or
                        "particle" in name.lower()
                    ):
                        track_datasets.append(name)
                
                h5file.visititems(find_track_datasets)
                
                if not track_datasets:
                    # If no specific track datasets were found, use top-level datasets
                    track_datasets = [name for name, obj in h5file.items() 
                                      if isinstance(obj, h5py.Dataset)]
                
                # Load the first dataset found
                if track_datasets:
                    data = h5file[track_datasets[0]][()]
                    
                    # Convert to DataFrame if it's a structured array
                    if hasattr(data, 'dtype') and data.dtype.names is not None:
                        tracks_df = pd.DataFrame({name: data[name] for name in data.dtype.names})
                    else:
                        # Try to interpret the data as a simple array
                        tracks_df = pd.DataFrame(data)
                        
                    # Standardize the track data format
                    standardized_df = format_track_data(tracks_df)
                    
                    # Validate the standardized dataframe
                    is_valid, message = validate_tracks_dataframe(standardized_df)
                    if not is_valid:
                        raise ValueError(f"Track data validation failed: {message}")

                    return standardized_df
                else:
                    raise ValueError("No suitable datasets found in the HDF5 file")
                
        except Exception as e:
            raise ValueError(f"Error loading HDF5 file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists("temp.h5"):
                os.remove("temp.h5")
    
    # For Imaris files (.ims)
    elif file_extension == '.ims':
        try:
            tracks_df = load_imaris_file(file)
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading Imaris file: {str(e)}")
    
    # For Volocity files (.uic, .aisf, .aiix)
    elif file_extension in ['.uic', '.aisf', '.aiix']:
        try:
            tracks_df = load_volocity_file(file)
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading Volocity file: {str(e)}")
    
    # For MVD2 files (.mvd2)
    elif file_extension == '.mvd2':
        try:
            tracks_df = load_mvd2_file(file)
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading MVD2 file: {str(e)}")
    
    # For XML files (TrackMate)
    elif file_extension == '.xml':
        try:
            file_stream = io.StringIO(file.getvalue().decode("utf-8"))
            tracks_df = load_trackmate_xml_file(file_stream)
            
            # Check if we got any data
            if tracks_df is None or tracks_df.empty:
                st.warning("⚠️ No track data found in TrackMate XML file")
                st.info("💡 Possible reasons:")
                st.info("- File may not contain tracked particles")
                st.info("- Track data may be in an unsupported format")
                st.info("- Try exporting tracks from TrackMate in CSV format instead")
                raise ValueError("No track or spot data found in TrackMate XML file")
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)

            # Validate the standardized dataframe
            is_valid, message = validate_tracks_dataframe(standardized_df)
            if not is_valid:
                raise ValueError(f"Track data validation failed: {message}")
            
            return standardized_df
            
        except ValueError as e:
            # Re-raise validation errors with original message
            raise
        except Exception as e:
            st.error("❌ Error loading TrackMate XML file")
            st.info("💡 Suggestions:")
            st.info("- Verify this is a valid TrackMate XML export")
            st.info("- Check that the file contains track data")
            st.info("- Try exporting as CSV from TrackMate instead")
            if st.checkbox("Show detailed XML error"):
                st.code(str(e))
            raise ValueError(f"Error loading TrackMate XML file: {str(e)}")
    
    # For JSON files
    elif file_extension == '.json':
        try:
            # Parse JSON content
            track_data = json.loads(file.getvalue().decode("utf-8"))
            
            # Handle different JSON structures
            if isinstance(track_data, list):
                # List of tracks
                tracks_df = pd.DataFrame(track_data)
            elif isinstance(track_data, dict):
                # Dictionary format
                if "tracks" in track_data:
                    tracks_df = pd.DataFrame(track_data["tracks"])
                else:
                    # Try to flatten the dictionary
                    flattened_data = []
                    for track_id, track_info in track_data.items():
                        if isinstance(track_info, list):
                            for point in track_info:
                                if isinstance(point, dict):
                                    point['track_id'] = track_id
                                    flattened_data.append(point)
                                else:
                                    # Handle case where points are arrays
                                    flattened_data.append({
                                        'track_id': track_id,
                                        'x': point[0] if len(point) > 0 else None,
                                        'y': point[1] if len(point) > 1 else None,
                                        'frame': point[2] if len(point) > 2 else None
                                    })
                        elif isinstance(track_info, dict):
                            for frame, coords in track_info.items():
                                point = {
                                    'track_id': track_id,
                                    'frame': frame
                                }
                                if isinstance(coords, list):
                                    point['x'] = coords[0] if len(coords) > 0 else None
                                    point['y'] = coords[1] if len(coords) > 1 else None
                                elif isinstance(coords, dict):
                                    point.update(coords)
                                flattened_data.append(point)
                    
                    tracks_df = pd.DataFrame(flattened_data)
            else:
                raise ValueError("Unrecognized JSON structure")
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
            # Clean invalid values (NaN, Inf)
            standardized_df = clean_tracks(standardized_df, warn_user=True)
            
            return standardized_df
            
        except Exception as e:
            raise ValueError(f"Error loading JSON file: {str(e)}")
    
    # For Imaris IMS files
    elif file_extension == '.ims':
        try:
            # Create a temporary file to save the uploaded content
            with open("temp.ims", "wb") as f:
                f.write(file.getvalue())
            
            # Load the Imaris file using our specialized handler
            imaris_data = load_imaris_file("temp.ims")
            
            # Use tracks data if available, otherwise use spots data
            if imaris_data['tracks_df'] is not None:
                tracks_df = imaris_data['tracks_df']
            elif imaris_data['spots_df'] is not None:
                tracks_df = imaris_data['spots_df']
            else:
                raise ValueError("No track or spot data found in the Imaris file")
            
            # Store metadata in session state if needed
            if 'imaris_metadata' not in st.session_state:
                st.session_state.imaris_metadata = {}
            
            # Store metadata using the filename as key
            st.session_state.imaris_metadata[file.name] = {
                'metadata': imaris_data['metadata'],
                'image_info': imaris_data['image_data'],
                'thumbnail': imaris_data['thumbnail']
            }
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
            # Clean invalid values (NaN, Inf)
            standardized_df = clean_tracks(standardized_df, warn_user=True)
            
            return standardized_df
            
        except Exception as e:
            raise ValueError(f"Error loading Imaris file: {str(e)}")
        finally:
            # Clean up temporary file
            if os.path.exists("temp.ims"):
                os.remove("temp.ims")
    
    # Unsupported format
    else:
        raise ValueError(f"Unsupported track data format: {file_extension}")


def detect_file_format(file_or_path: Union[str, os.PathLike, Any]) -> str:
    """
    Backward-compatible file format detector.

    Returns lowercase extension without leading dot (e.g., 'csv', 'xlsx').
    """
    if isinstance(file_or_path, (str, os.PathLike)):
        name = os.path.basename(os.fspath(file_or_path))
    else:
        name = getattr(file_or_path, 'name', '')
    ext = os.path.splitext(name)[1].lower().lstrip('.')
    return ext


def load_tracking_data(
    file,
    pixel_size: Optional[float] = None,
    frame_interval: Optional[float] = None,
) -> pd.DataFrame:
    """
    Backward-compatible alias for `load_tracks_file`.
    """
    return load_tracks_file(file, pixel_size=pixel_size, frame_interval=frame_interval)
