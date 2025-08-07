"""
Special file handlers for specific file formats in the SPT Analysis application.
"""

import pandas as pd
import numpy as np
import io
import os
import h5py
from typing import Dict, Any, Optional, Tuple, List, Union
import re

def load_trackmate_file(file_stream, sep=",") -> pd.DataFrame:
    """
    Load a TrackMate-style CSV file with multiple header rows.
    
    Parameters
    ----------
    file_stream : io.StringIO
        File stream containing the CSV data
    sep : str
        Separator character
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed track data
    """
    # Read the first several lines to determine the structure
    file_stream.seek(0)
    header_lines = [file_stream.readline() for _ in range(10)]
    file_stream.seek(0)
    
    # Check which row contains the actual column headers
    header_row = 0
    unit_row = None
    
    # Analyze the header lines to identify the structure
    for i, line in enumerate(header_lines):
        if "ID" in line and "TRACK_ID" in line:
            header_row = i
            # The next row likely contains unit information
            if i + 1 < len(header_lines):
                unit_row = i + 1
            break
        elif "ID" in line and ("Track ID" in line or "X" in line or "POSITION_X" in line):
            header_row = i
            # The next row likely contains unit information
            if i + 1 < len(header_lines):
                unit_row = i + 1
            break
        elif "Track ID" in line and "X" in line and "Y" in line:
            header_row = i
            # The next row likely contains unit information
            if i + 1 < len(header_lines):
                unit_row = i + 1
            break
        elif "micron" in line.lower() or "sec" in line.lower() or "quality" in line.lower():
            # This is likely a unit information row
            # The previous row should be the header
            unit_row = i
            header_row = i - 1
            break
    
    # Read the CSV with the determined header row and skipping the unit row if present
    file_stream.seek(0)
    skiprows = [unit_row] if unit_row is not None else None
    
    try:
        tracks_df = pd.read_csv(file_stream, sep=sep, header=header_row, skiprows=skiprows)
    except Exception:
        # Fall back to a more flexible approach
        file_stream.seek(0)
        # Skip to the header row
        for _ in range(header_row):
            file_stream.readline()
            
        # Read the header row
        header = file_stream.readline().strip().split(sep)
        
        # Skip the unit row if present
        if unit_row is not None and unit_row == header_row + 1:
            file_stream.readline()
        
        # Create empty DataFrame with the determined headers
        tracks_df = pd.DataFrame(columns=header)
        
        row_data = []
        
        # Read and parse the data rows
        for line in file_stream:
            line = line.strip()
            if line:  # Skip empty lines
                values = line.split(sep)
                # Pad or trim values to match header length
                if len(values) < len(header):
                    values.extend([""] * (len(header) - len(values)))
                elif len(values) > len(header):
                    values = values[:len(header)]
                
                row_data.append(dict(zip(header, values)))
        
        # Create DataFrame from collected data - much faster than repeated concat
        if row_data:
            tracks_df = pd.DataFrame(row_data)
    
    # Convert numeric columns
    for col in tracks_df.columns:
        try:
            tracks_df[col] = pd.to_numeric(tracks_df[col])
        except (ValueError, TypeError):
            # Keep as string if not convertible to numeric
            pass
    
    # Map common column names to standardized format
    column_map = {
        # TrackMate standard format
        'TRACK_ID': 'track_id',
        'FRAME': 'frame',
        'POSITION_X': 'x',
        'POSITION_Y': 'y',
        'POSITION_Z': 'z',
        'POSITION_T': 'time',
        'QUALITY': 'Quality',
        
        # Cell3_spots.csv format variants
        'Track ID': 'track_id',
        'Spot ID': 'spot_id',
        'ID': 'spot_id',
        'Frame': 'frame',
        'X': 'x',
        'Y': 'y',
        'Z': 'z',
        'T': 'time',
        'Quality': 'Quality',
        
        # Additional column mappings
        'Mean ch1': 'intensity_ch1',
        'Mean intensity ch1': 'intensity_ch1',
        'MEAN_INTENSITY_CH1': 'intensity_ch1',
        'Mean ch2': 'intensity_ch2',
        'Mean intensity ch2': 'intensity_ch2',
        'MEAN_INTENSITY_CH2': 'intensity_ch2',
        'Mean ch3': 'intensity_ch3',
        'Mean intensity ch3': 'intensity_ch3',
        'MEAN_INTENSITY_CH3': 'intensity_ch3'
    }
    
    # Rename columns if they exist in our mapping
    for old_col, new_col in column_map.items():
        if old_col in tracks_df.columns:
            tracks_df.rename(columns={old_col: new_col}, inplace=True)
    
    return tracks_df

def load_cropped_cell3_spots(file_stream, sep=",") -> pd.DataFrame:
    """
    Special loader for the Cropped_cell3_spots format.
    
    Parameters
    ----------
    file_stream : io.StringIO
        File stream containing the CSV data
    sep : str
        Separator character
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed track data
    """
    # Reset file pointer
    file_stream.seek(0)
    
    # Try to read the file with header at row 2 (third row) and skip row 3 (unit information)
    try:
        tracks_df = pd.read_csv(file_stream, sep=sep, header=2, skiprows=[3])
        
        # Check if we got the right columns
        if 'X' in tracks_df.columns and 'Y' in tracks_df.columns and 'Track ID' in tracks_df.columns:
            # Map columns to standard format
            tracks_df = tracks_df.rename(columns={
                'Track ID': 'track_id',
                'Frame': 'frame',
                'X': 'x',
                'Y': 'y'
            })
            
            # Convert numeric columns
            for col in ['track_id', 'frame', 'x', 'y']:
                if col in tracks_df.columns:
                    tracks_df[col] = pd.to_numeric(tracks_df[col], errors='coerce')
            
            return tracks_df
    except Exception:
        # Fall back to the general TrackMate loader
        file_stream.seek(0)
        return load_trackmate_file(file_stream, sep)
        
    # If we're here, the first attempt failed, try with the general handler
    file_stream.seek(0)
    return load_trackmate_file(file_stream, sep)
    
def load_ms2_spots_file(file_stream, sep=",") -> pd.DataFrame:
    """
    Special loader for MS2_spots format files which have multiple header rows.
    
    Parameters
    ----------
    file_stream : io.StringIO
        File stream containing the CSV data
    sep : str
        Separator character
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the parsed track data
    """
    # Read all lines to analyze the structure
    file_stream.seek(0)
    lines = file_stream.readlines()
    
    # Find the data section
    header_row = None
    unit_row = None
    data_start_row = None
    
    # Analyze the first few lines
    for i, line in enumerate(lines[:15]):
        line = line.strip()
        # Look for a line with Track ID or TRACK_ID
        if "Track ID" in line or "TRACK_ID" in line:
            header_row = i
            # Check if next line(s) have units or abbreviations
            for j in range(i+1, min(i+4, len(lines))):
                if j < len(lines):
                    next_line = lines[j].strip()
                    if "micron" in next_line.lower() or "sec" in next_line.lower() or "(quality)" in next_line.lower():
                        unit_row = j
                        # Data starts in the next row
                        data_start_row = j + 1
                        break
            break
    
    # If we identified the structure
    if header_row is not None and data_start_row is not None:
        try:
            # First approach: try to read with pandas
            file_stream.seek(0)
            skiprows = list(range(header_row+1, data_start_row))
            df = pd.read_csv(file_stream, sep=sep, header=header_row, skiprows=skiprows)
            
            # Fix the column names (remove any leading/trailing spaces)
            df.columns = [col.strip() if isinstance(col, str) else col for col in df.columns]
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    pass
            
            # Rename columns to standard format
            column_map = {
                'Track ID': 'track_id',
                'TRACK_ID': 'track_id',
                'X': 'x', 
                'Y': 'y',
                'Z': 'z',
                'POSITION_X': 'x',
                'POSITION_Y': 'y',
                'POSITION_Z': 'z',
                'Frame': 'frame',
                'FRAME': 'frame',
                'T': 'time',
                'POSITION_T': 'time',
                'Mean ch1': 'intensity'
            }
            
            for old_col, new_col in column_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            # Return the processed dataframe
            return df
            
        except Exception as e:
            # Manual fallback approach
            file_stream.seek(0)
            
            # Get the header row for column names
            header = None
            for i, line in enumerate(lines):
                if i == header_row:
                    header = [h.strip() for h in line.strip().split(sep)]
                    break
            
            if not header:
                raise ValueError("Could not find header row in MS2 spots file")
            
            # Read data rows
            data_rows = []
            for i, line in enumerate(lines):
                if i >= data_start_row:
                    if line.strip():  # Skip empty lines
                        # Process the line
                        values = line.strip().split(sep)
                        # Make sure we have the right number of values
                        if len(values) > len(header):
                            values = values[:len(header)]
                        elif len(values) < len(header):
                            values.extend([''] * (len(header) - len(values)))
                            
                        row_dict = dict(zip(header, values))
                        data_rows.append(row_dict)
            
            # Create DataFrame
            df = pd.DataFrame(data_rows)
            
            # Convert numeric columns
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except (ValueError, TypeError):
                    pass
            
            # Rename columns to standard format
            column_map = {
                'Track ID': 'track_id',
                'TRACK_ID': 'track_id',
                'X': 'x', 
                'Y': 'y',
                'Z': 'z',
                'POSITION_X': 'x',
                'POSITION_Y': 'y',
                'POSITION_Z': 'z',
                'Frame': 'frame',
                'FRAME': 'frame',
                'T': 'time',
                'POSITION_T': 'time',
                'Mean ch1': 'intensity'
            }
            
            for old_col, new_col in column_map.items():
                if old_col in df.columns:
                    df.rename(columns={old_col: new_col}, inplace=True)
            
            return df
            
    # If we couldn't identify the structure, fall back to the general handler
    file_stream.seek(0)
    return load_trackmate_file(file_stream, sep)

def load_imaris_file(file_path: str) -> Dict[str, Any]:
    """
    Load a Bitplane Imaris IMS file and extract relevant data.
    
    IMS files are based on HDF5 format and contain multi-dimensional 
    microscopy data including tracks, spots, and images.
    
    Parameters
    ----------
    file_path : str
        Path to the IMS file
        
    Returns
    -------
    dict
        Dictionary containing extracted data from the IMS file:
        - 'tracks_df': DataFrame with track data
        - 'spots_df': DataFrame with spot data
        - 'image_data': Dictionary with image data (if available)
        - 'metadata': Dictionary with file metadata
        - 'thumbnail': Thumbnail image as NumPy array (if available)
    """
    results = {
        'tracks_df': None,
        'spots_df': None,
        'image_data': None,
        'metadata': {},
        'thumbnail': None
    }
    
    try:
        # Open the IMS file as an HDF5 file
        with h5py.File(file_path, 'r') as ims_file:
            # Extract metadata
            results['metadata'] = extract_imaris_metadata(ims_file)
            
            # Extract tracks if available
            if 'Scene8/Content/Tracks' in ims_file:
                tracks_data = extract_imaris_tracks(ims_file)
                results['tracks_df'] = tracks_data
            
            # Extract spots if available
            if 'Scene8/Content/Spots' in ims_file:
                spots_data = extract_imaris_spots(ims_file)
                results['spots_df'] = spots_data
                
                # If no tracks were found but spots have track IDs, construct tracks
                if results['tracks_df'] is None and spots_data is not None and 'track_id' in spots_data.columns:
                    results['tracks_df'] = construct_tracks_from_spots(spots_data)
            
            # Extract image data (basic info only)
            results['image_data'] = extract_imaris_image_info(ims_file)
            
            # Extract thumbnail for preview
            thumbnail = extract_imaris_thumbnail(ims_file)
            if thumbnail is not None:
                results['thumbnail'] = thumbnail
            
            # If thumbnail extraction failed, try to get a slice from the image stack
            if results['thumbnail'] is None:
                try:
                    # Get a slice from the first channel, first timepoint
                    img_stack = extract_imaris_image_stack(ims_file, channel=0, timepoint=0, max_size=512)
                    if img_stack is not None and img_stack.ndim >= 2:
                        # For 3D data, take the middle slice
                        if img_stack.ndim == 3:
                            middle_slice = img_stack.shape[0] // 2
                            results['thumbnail'] = img_stack[middle_slice]
                        else:
                            results['thumbnail'] = img_stack
                except Exception as img_err:
                    print(f"Error extracting image slice as thumbnail: {str(img_err)}")
            
        return results
    
    except Exception as e:
        raise ValueError(f"Error processing Imaris file: {str(e)}")

def extract_imaris_metadata(ims_file: h5py.File) -> Dict[str, Any]:
    """
    Extract metadata from an Imaris file.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
        
    Returns
    -------
    dict
        Dictionary containing metadata
    """
    metadata = {}
    
    # Extract basic file information
    try:
        if 'DataSetInfo' in ims_file:
            dataset_info = ims_file['DataSetInfo']
            
            # Extract common metadata fields
            for key in dataset_info.keys():
                if isinstance(dataset_info[key], h5py.Dataset):
                    try:
                        metadata[key] = dataset_info[key][()]
                        # Convert bytes to string if needed
                        if isinstance(metadata[key], bytes):
                            metadata[key] = metadata[key].decode('utf-8', errors='ignore')
                    except:
                        pass
        
        # Extract image dimensions
        if 'Scene8/Content/Image' in ims_file:
            image = ims_file['Scene8/Content/Image']
            if 'Dimensions' in image.attrs:
                metadata['Dimensions'] = list(image.attrs['Dimensions'])
            
            # Extract resolution information
            if 'ExtMin0' in image.attrs and 'ExtMax0' in image.attrs:
                x_min = image.attrs['ExtMin0']
                x_max = image.attrs['ExtMax0']
                # Convert to numeric values if they're strings
                try:
                    if isinstance(x_min, str):
                        x_min = float(x_min)
                    if isinstance(x_max, str):
                        x_max = float(x_max)
                    metadata['X_Range'] = (x_min, x_max)
                    if 'X_SIZE' in image.attrs:
                        x_size = image.attrs['X_SIZE']
                        if isinstance(x_size, str):
                            x_size = float(x_size)
                        metadata['X_Resolution'] = (x_max - x_min) / x_size
                except (ValueError, TypeError):
                    # Skip this calculation if conversion fails
                    metadata['X_Range'] = (str(x_min), str(x_max))
            
            if 'ExtMin1' in image.attrs and 'ExtMax1' in image.attrs:
                y_min = image.attrs['ExtMin1']
                y_max = image.attrs['ExtMax1']
                # Convert to numeric values if they're strings
                try:
                    if isinstance(y_min, str):
                        y_min = float(y_min)
                    if isinstance(y_max, str):
                        y_max = float(y_max)
                    metadata['Y_Range'] = (y_min, y_max)
                    if 'Y_SIZE' in image.attrs:
                        y_size = image.attrs['Y_SIZE']
                        if isinstance(y_size, str):
                            y_size = float(y_size)
                        metadata['Y_Resolution'] = (y_max - y_min) / y_size
                except (ValueError, TypeError):
                    # Skip this calculation if conversion fails
                    metadata['Y_Range'] = (str(y_min), str(y_max))
            
            if 'ExtMin2' in image.attrs and 'ExtMax2' in image.attrs:
                z_min = image.attrs['ExtMin2']
                z_max = image.attrs['ExtMax2']
                # Convert to numeric values if they're strings
                try:
                    if isinstance(z_min, str):
                        z_min = float(z_min)
                    if isinstance(z_max, str):
                        z_max = float(z_max)
                    metadata['Z_Range'] = (z_min, z_max)
                    if 'Z_SIZE' in image.attrs:
                        z_size = image.attrs['Z_SIZE']
                        if isinstance(z_size, str):
                            z_size = float(z_size)
                        metadata['Z_Resolution'] = (z_max - z_min) / z_size
                except (ValueError, TypeError):
                    # Skip this calculation if conversion fails
                    metadata['Z_Range'] = (str(z_min), str(z_max))
    
    except Exception as e:
        metadata['error'] = f"Error extracting metadata: {str(e)}"
    
    return metadata

def extract_imaris_tracks(ims_file: h5py.File) -> pd.DataFrame:
    """
    Extract track data from an Imaris file.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing track data
    """
    try:
        tracks_path = 'Scene8/Content/Tracks'
        
        if tracks_path not in ims_file:
            return None
        
        tracks_group = ims_file[tracks_path]
        
        # Initialize lists to store track data
        track_ids = []
        track_times = []
        track_positions = []
        
        # Iterate through all track groups
        for track_name in tracks_group.keys():
            if not track_name.startswith('Track'):
                continue
                
            track_group = tracks_group[track_name]
            track_id = int(track_name.replace('Track', ''))
            
            # Extract position data
            if 'Position' in track_group:
                positions = track_group['Position'][()]
                
                # Extract time data
                if 'Time' in track_group:
                    times = track_group['Time'][()]
                    
                    # For each timepoint and position
                    for i in range(len(times)):
                        track_ids.append(track_id)
                        track_times.append(times[i])
                        track_positions.append(positions[i])
        
        # Create DataFrame
        if track_ids:
            tracks_df = pd.DataFrame({
                'track_id': track_ids,
                'time': track_times,
                'x': [pos[0] for pos in track_positions],
                'y': [pos[1] for pos in track_positions],
                'z': [pos[2] if len(pos) > 2 else 0 for pos in track_positions]
            })
            
            # Sort by track_id and time
            tracks_df = tracks_df.sort_values(['track_id', 'time']).reset_index(drop=True)
            
            # Convert time to frame number if needed
            if 'time' in tracks_df.columns and 'frame' not in tracks_df.columns:
                # Estimate frame from time
                unique_times = sorted(tracks_df['time'].unique())
                time_to_frame = {t: i for i, t in enumerate(unique_times)}
                tracks_df['frame'] = tracks_df['time'].map(time_to_frame)
            
            return tracks_df
        
        return None
    
    except Exception as e:
        print(f"Error extracting tracks: {str(e)}")
        return None

def extract_imaris_spots(ims_file: h5py.File) -> pd.DataFrame:
    """
    Extract spot data from an Imaris file.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing spot data
    """
    try:
        spots_path = 'Scene8/Content/Spots'
        
        if spots_path not in ims_file:
            return None
        
        spots_group = ims_file[spots_path]
        
        # Lists to store all spot data
        all_spot_ids = []
        all_track_ids = []
        all_positions = []
        all_times = []
        all_radii = []
        all_intensities = []
        
        # Iterate through spot groups (usually organized by size/type)
        for spot_group_name in spots_group.keys():
            spot_group = spots_group[spot_group_name]
            
            # Extract position data
            if 'Position' in spot_group:
                positions = spot_group['Position'][()]
                
                # Extract time data
                if 'Time' in spot_group:
                    times = spot_group['Time'][()]
                    
                    # Extract radius data if available
                    radii = None
                    if 'Radius' in spot_group:
                        radii = spot_group['Radius'][()]
                    
                    # Extract track ID associations if available
                    track_ids = None
                    if 'TrackID' in spot_group:
                        track_ids = spot_group['TrackID'][()]
                    
                    # Extract intensity values if available
                    intensities = None
                    if 'Intensity' in spot_group:
                        intensities = spot_group['Intensity'][()]
                    
                    # For each spot
                    for i in range(len(times)):
                        all_spot_ids.append(i)
                        all_times.append(times[i])
                        all_positions.append(positions[i])
                        
                        if radii is not None and i < len(radii):
                            all_radii.append(radii[i])
                        else:
                            all_radii.append(None)
                            
                        if track_ids is not None and i < len(track_ids):
                            all_track_ids.append(track_ids[i])
                        else:
                            all_track_ids.append(None)
                            
                        if intensities is not None and i < len(intensities):
                            all_intensities.append(intensities[i])
                        else:
                            all_intensities.append(None)
        
        # Create DataFrame
        if all_spot_ids:
            data = {
                'spot_id': all_spot_ids,
                'time': all_times,
                'x': [pos[0] for pos in all_positions],
                'y': [pos[1] for pos in all_positions],
                'z': [pos[2] if len(pos) > 2 else 0 for pos in all_positions]
            }
            
            # Add track IDs if available
            if any(tid is not None for tid in all_track_ids):
                data['track_id'] = all_track_ids
            
            # Add radii if available
            if any(r is not None for r in all_radii):
                data['radius'] = all_radii
            
            # Add intensities if available
            if any(i is not None for i in all_intensities):
                data['intensity'] = all_intensities
            
            spots_df = pd.DataFrame(data)
            
            # Sort by time
            spots_df = spots_df.sort_values('time').reset_index(drop=True)
            
            # Convert time to frame number if needed
            if 'time' in spots_df.columns and 'frame' not in spots_df.columns:
                # Estimate frame from time
                unique_times = sorted(spots_df['time'].unique())
                time_to_frame = {t: i for i, t in enumerate(unique_times)}
                spots_df['frame'] = spots_df['time'].map(time_to_frame)
            
            return spots_df
        
        return None
    
    except Exception as e:
        print(f"Error extracting spots: {str(e)}")
        return None

def extract_imaris_image_info(ims_file: h5py.File) -> Dict[str, Any]:
    """
    Extract basic information about the images in an Imaris file.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
        
    Returns
    -------
    dict
        Dictionary containing image information
    """
    image_info = {
        'channels': [],
        'dimensions': None,
        'time_points': 0,
        'resolution': {}
    }
    
    try:
        if 'Scene8/Content/Image' in ims_file:
            image = ims_file['Scene8/Content/Image']
            
            # Get dimensions
            if 'Dimensions' in image.attrs:
                image_info['dimensions'] = list(image.attrs['Dimensions'])
            
            # Get channels
            channel_count = 0
            for key in image.keys():
                if key.startswith('Channel'):
                    channel_count += 1
                    
                    channel_name = f"Channel {key.replace('Channel', '')}"
                    if 'Name' in image[key].attrs:
                        channel_name = image[key].attrs['Name']
                        if isinstance(channel_name, bytes):
                            channel_name = channel_name.decode('utf-8', errors='ignore')
                    
                    image_info['channels'].append({
                        'id': key,
                        'name': channel_name
                    })
            
            # Get time points
            if 'TimeInfo' in ims_file and 'DataSetTimePoints' in ims_file['TimeInfo']:
                time_points = ims_file['TimeInfo']['DataSetTimePoints']
                image_info['time_points'] = len(time_points)
                
                # Get time interval if more than one time point
                if len(time_points) > 1:
                    try:
                        # Convert time points to numeric values if they're strings
                        t0 = time_points[0]
                        t1 = time_points[1]
                        if isinstance(t0, str):
                            t0 = float(t0)
                        if isinstance(t1, str):
                            t1 = float(t1)
                        image_info['time_interval'] = t1 - t0
                    except (ValueError, TypeError):
                        # If conversion fails, store as strings
                        image_info['time_interval'] = f"{time_points[1]} - {time_points[0]}"
            
            # Get resolution information
            for dim, axis in enumerate(['X', 'Y', 'Z']):
                if f'ExtMin{dim}' in image.attrs and f'ExtMax{dim}' in image.attrs:
                    ext_min = image.attrs[f'ExtMin{dim}']
                    ext_max = image.attrs[f'ExtMax{dim}']
                    
                    # Size in pixels
                    size_attr = f'{axis}_SIZE'
                    if size_attr in image.attrs:
                        size = image.attrs[size_attr]
                        if size > 0:
                            # Calculate resolution in physical units per pixel
                            resolution = (ext_max - ext_min) / size
                            image_info['resolution'][axis.lower()] = resolution
    
    except Exception as e:
        image_info['error'] = f"Error extracting image info: {str(e)}"
    
    return image_info

def extract_imaris_thumbnail(ims_file: h5py.File) -> Optional[np.ndarray]:
    """
    Extract a thumbnail image from an Imaris file.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
        
    Returns
    -------
    np.ndarray or None
        Thumbnail image as a NumPy array, or None if not available
    """
    try:
        # Check if thumbnail exists
        if 'Scene8/Content/Thumbnail' in ims_file:
            thumbnail = ims_file['Scene8/Content/Thumbnail']
            if isinstance(thumbnail, h5py.Dataset):
                img_data = thumbnail[()]
                
                # Process the thumbnail data based on its format
                # Typically thumbnails are stored as RGB values
                if img_data.ndim == 3 and img_data.shape[2] == 3:
                    # Standard RGB format
                    return img_data
                elif img_data.ndim == 2:
                    # Grayscale format
                    return img_data
                elif img_data.ndim == 1:
                    # 1D array, might need reshaping
                    # Try to determine image dimensions
                    if 'Scene8/Content/Thumbnail' in ims_file.attrs:
                        if 'Shape' in ims_file['Scene8/Content/Thumbnail'].attrs:
                            shape = ims_file['Scene8/Content/Thumbnail'].attrs['Shape']
                            return img_data.reshape(shape)
        
        # If no thumbnail found or couldn't process it, try to extract from main image
        if 'Scene8/Content/Image' in ims_file:
            image = ims_file['Scene8/Content/Image']
            
            # Look for the first channel and first timepoint
            for channel_key in image.keys():
                if channel_key.startswith('Channel'):
                    channel = image[channel_key]
                    
                    # Get the first timepoint
                    if 'Data' in channel and isinstance(channel['Data'], h5py.Group):
                        data_group = channel['Data']
                        timepoint_keys = [k for k in data_group.keys() if k.startswith('TimePoint')]
                        
                        if timepoint_keys:
                            # Get first timepoint
                            timepoint = data_group[timepoint_keys[0]]
                            
                            # Try to get the smallest resolution level for a thumbnail
                            resolution_levels = [k for k in timepoint.keys() if k.startswith('Resolution')]
                            
                            if resolution_levels:
                                # Use highest resolution level (smallest image)
                                resolution = timepoint[resolution_levels[-1]]
                                
                                if isinstance(resolution, h5py.Dataset):
                                    img_data = resolution[()]
                                    
                                    # Process based on dimensionality
                                    if img_data.ndim == 3:
                                        # Take middle slice of 3D volume
                                        middle_slice = img_data.shape[0] // 2
                                        return img_data[middle_slice]
                                    elif img_data.ndim == 2:
                                        return img_data
                        
                    break  # Just use the first channel
        
        return None
        
    except Exception as e:
        print(f"Error extracting thumbnail: {str(e)}")
        return None

def extract_imaris_image_stack(ims_file: h5py.File, channel: int = 0, timepoint: int = 0, max_size: int = 512) -> Optional[np.ndarray]:
    """
    Extract an image stack from an Imaris file for a specific channel and timepoint.
    
    Parameters
    ----------
    ims_file : h5py.File
        Open HDF5 file handle for the IMS file
    channel : int
        Channel index to extract (default: 0)
    timepoint : int
        Timepoint index to extract (default: 0)
    max_size : int
        Maximum size for any dimension (images will be downsampled if larger)
        
    Returns
    -------
    np.ndarray or None
        Image stack as a NumPy array, or None if not available
    """
    try:
        if 'Scene8/Content/Image' in ims_file:
            image_group = ims_file['Scene8/Content/Image']
            
            # Find the requested channel
            channel_keys = [k for k in image_group.keys() if k.startswith('Channel')]
            
            if not channel_keys:
                return None
                
            # Make sure channel index is valid
            if channel >= len(channel_keys):
                channel = 0
                
            channel_key = channel_keys[channel]
            channel_group = image_group[channel_key]
            
            # Check if Data group exists
            if 'Data' not in channel_group:
                return None
                
            data_group = channel_group['Data']
            
            # Find the requested timepoint
            timepoint_keys = [k for k in data_group.keys() if k.startswith('TimePoint')]
            
            if not timepoint_keys:
                return None
                
            # Make sure timepoint index is valid
            if timepoint >= len(timepoint_keys):
                timepoint = 0
                
            timepoint_key = timepoint_keys[timepoint]
            timepoint_group = data_group[timepoint_key]
            
            # Find the appropriate resolution level
            resolution_keys = [k for k in timepoint_group.keys() if k.startswith('Resolution')]
            
            if not resolution_keys:
                return None
                
            # Select resolution level
            # For simplicity, use the highest resolution (first one)
            resolution_key = resolution_keys[0]
            resolution_dataset = timepoint_group[resolution_key]
            
            # Load the image data
            img_data = resolution_dataset[()]
            
            # Downsample if needed
            if max(img_data.shape) > max_size:
                from skimage.transform import resize
                
                # Calculate scaling factor to maintain aspect ratio
                scale = max_size / max(img_data.shape)
                
                # For 3D data
                if img_data.ndim == 3:
                    new_shape = (
                        int(img_data.shape[0] * scale),
                        int(img_data.shape[1] * scale),
                        int(img_data.shape[2] * scale)
                    )
                    img_data = resize(img_data, new_shape, preserve_range=True).astype(img_data.dtype)
                # For 2D data
                elif img_data.ndim == 2:
                    new_shape = (
                        int(img_data.shape[0] * scale),
                        int(img_data.shape[1] * scale)
                    )
                    img_data = resize(img_data, new_shape, preserve_range=True).astype(img_data.dtype)
            
            return img_data
            
        return None
        
    except Exception as e:
        print(f"Error extracting image stack: {str(e)}")
        return None

def construct_tracks_from_spots(spots_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct tracks DataFrame from spots data with track IDs.
    
    Parameters
    ----------
    spots_df : pd.DataFrame
        DataFrame containing spot data with track_id column
        
    Returns
    -------
    pd.DataFrame
        DataFrame formatted for track analysis
    """
    if spots_df is None or 'track_id' not in spots_df.columns:
        return None
    
    # Keep only essential columns for tracking
    track_columns = ['track_id', 'frame', 'time', 'x', 'y', 'z']
    available_columns = [col for col in track_columns if col in spots_df.columns]
    
    # Create a copy with needed columns
    tracks_df = spots_df[available_columns].copy()
    
    # Remove any rows with null track_id
    tracks_df = tracks_df[~tracks_df['track_id'].isna()].reset_index(drop=True)
    
    # Ensure each track is sorted by frame or time
    sort_by = 'frame' if 'frame' in tracks_df.columns else 'time'
    tracks_df = tracks_df.sort_values(['track_id', sort_by]).reset_index(drop=True)
    
    return tracks_df
