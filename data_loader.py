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
from typing import List, Dict, Tuple, Any, Optional, Union
from PIL import Image
from utils import format_track_data
from special_file_handlers import load_trackmate_file, load_cropped_cell3_spots, load_ms2_spots_file, load_imaris_file
from mvd2_handler import load_mvd2_file
from volocity_handler import load_volocity_file


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
            
            # If we have multiple frames, check if they should be combined as channels
            if len(frames) > 1:
                # Check if all frames have the same dimensions
                first_shape = frames[0].shape[:2]  # height, width
                if all(frame.shape[:2] == first_shape for frame in frames):
                    # Try to combine frames as channels in a single image
                    if len(frames[0].shape) == 2:  # Grayscale frames
                        # Stack as channels: (height, width, channels)
                        combined = np.stack(frames, axis=2)
                        st.info(f"Detected {len(frames)} frames - combining as multichannel image")
                        return [combined]
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

def load_tracks_file(file) -> pd.DataFrame:
    """
    Load track data from various file formats.
    
    Parameters
    ----------
    file : UploadedFile
        File uploaded through Streamlit
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the track data
    """
    file_extension = os.path.splitext(file.name)[1].lower()
    
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
            
            # Convert to numeric where appropriate
            for col in ['track_id', 'frame', 'x', 'y', 'z']:
                if col in tracks_df.columns:
                    tracks_df[col] = pd.to_numeric(tracks_df[col], errors='coerce')
            
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
            return format_track_data(tracks_df)
            
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
                    data_dict['z'] = 0
                    
                if 'quality' in tracks_df.columns:
                    data_dict['quality'] = pd.to_numeric(tracks_df['quality'], errors='coerce')
                else:
                    data_dict['quality'] = 1
                
                result_df = pd.DataFrame(data_dict)
                
                # Remove rows with NaN values in essential columns
                result_df = result_df.dropna(subset=['track_id', 'frame', 'x', 'y'])
                if not result_df.empty:
                    result_df['track_id'] = result_df['track_id'].astype(int)
                    result_df['frame'] = result_df['frame'].astype(int)
                
                return result_df
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
            return standardized_df
            
        except Exception as e:
            raise ValueError(f"Error loading CSV file: {str(e)}")
    
    # For Excel files
    elif file_extension in ['.xlsx', '.xls']:
        try:
            tracks_df = pd.read_excel(file)
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
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
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading Imaris file: {str(e)}")
    
    # For Volocity files (.uic, .aisf, .aiix)
    elif file_extension in ['.uic', '.aisf', '.aiix']:
        try:
            tracks_df = load_volocity_file(file)
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading Volocity file: {str(e)}")
    
    # For MVD2 files (.mvd2)
    elif file_extension == '.mvd2':
        try:
            tracks_df = load_mvd2_file(file)
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            return standardized_df
        except Exception as e:
            raise ValueError(f"Error loading MVD2 file: {str(e)}")
    
    # For XML files (TrackMate)
    elif file_extension == '.xml':
        try:
            import xml.etree.ElementTree as ET
            
            # Parse XML content
            content = file.getvalue()
            root = ET.fromstring(content)
            
            # Initialize data containers
            track_data = []
            spots_dict = {}
            
            # Try to find spots in different XML structures
            
            # Method 1: Try standard TrackMate format
            model = root.find('.//Model')
            if model is not None:
                all_spots = model.find('.//AllSpots')
                if all_spots is not None:
                    for frame_elem in all_spots.findall('SpotsInFrame'):
                        frame = int(frame_elem.get('frame', 0))
                        
                        for spot_elem in frame_elem.findall('Spot'):
                            spot_id = spot_elem.get('ID')
                            x = float(spot_elem.get('POSITION_X', 0))
                            y = float(spot_elem.get('POSITION_Y', 0))
                            z = float(spot_elem.get('POSITION_Z', 0))
                            
                            spots_dict[spot_id] = {
                                'frame': frame,
                                'x': x,
                                'y': y,
                                'z': z
                            }
                
                # Find tracks if available
                all_tracks = model.find('.//AllTracks')
                if all_tracks is not None:
                    for track_elem in all_tracks.findall('.//Track'):
                        track_id = int(track_elem.get('TRACK_ID', -1))
                        
                        # Get spots in this track
                        spot_ids = set()
                        for edge in track_elem.findall('.//Edge'):
                            source_id = edge.get('SPOT_SOURCE_ID')
                            target_id = edge.get('SPOT_TARGET_ID')
                            spot_ids.add(source_id)
                            spot_ids.add(target_id)
                        
                        # Add track data for each spot
                        for spot_id in spot_ids:
                            if spot_id in spots_dict:
                                spot_data = spots_dict[spot_id].copy()
                                spot_data['track_id'] = track_id
                                track_data.append(spot_data)
                
                # If no tracks found, use spots as individual tracks
                if not track_data and spots_dict:
                    for spot_id, spot_data in spots_dict.items():
                        spot_data = spot_data.copy()
                        spot_data['track_id'] = 0  # Single track
                        track_data.append(spot_data)
            
            # Method 2: Try TrackMate export format with <particle> and <detection> elements
            if not track_data:
                particles = root.findall('.//particle')
                if particles:
                    for track_id, particle in enumerate(particles):
                        detections = particle.findall('.//detection')
                        for detection in detections:
                            frame = int(detection.get('t', 0))
                            x = float(detection.get('x', 0))
                            y = float(detection.get('y', 0))
                            z = float(detection.get('z', 0))
                            
                            track_data.append({
                                'track_id': track_id,
                                'frame': frame,
                                'x': x,
                                'y': y,
                                'z': z
                            })
            
            # Method 3: Try to find any spot-like elements in the XML
            if not track_data:
                # Look for any elements that might contain coordinate data
                for elem in root.iter():
                    # Check if element has coordinate attributes
                    if ('x' in elem.attrib.keys() or 'X' in elem.attrib.keys() or 
                        'POSITION_X' in elem.attrib.keys()):
                        
                        # Extract coordinates
                        x = float(elem.get('x', elem.get('X', elem.get('POSITION_X', 0))))
                        y = float(elem.get('y', elem.get('Y', elem.get('POSITION_Y', 0))))
                        z = float(elem.get('z', elem.get('Z', elem.get('POSITION_Z', 0))))
                        frame = int(elem.get('frame', elem.get('Frame', elem.get('t', 0))))
                        
                        track_data.append({
                            'track_id': 0,
                            'frame': frame,
                            'x': x,
                            'y': y,
                            'z': z
                        })
            
            if not track_data:
                raise ValueError("No track data found in TrackMate XML file")
            
            # Convert to DataFrame
            tracks_df = pd.DataFrame(track_data)
            
            # Standardize the track data format
            standardized_df = format_track_data(tracks_df)
            
            return standardized_df
            
        except ET.ParseError:
            raise ValueError("Invalid XML file format")
        except Exception as e:
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
    