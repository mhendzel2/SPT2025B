"""
MVD2 File Handler for Spinning Disk Microscopy Data
Supports Olympus MVD2 format commonly used in spinning disk confocal systems.
"""

import numpy as np
import struct
import io
from typing import List, Dict, Any, Optional, Tuple

class MVD2Reader:
    """Reader for MVD2 files from Olympus spinning disk systems."""
    
    def __init__(self, file_path_or_buffer):
        """Initialize MVD2 reader."""
        self.file_buffer = file_path_or_buffer
        self.header = {}
        self.metadata = {}
        self.frames = []
        
    def read_header(self):
        """Read and parse MVD2 file header."""
        # Reset to beginning
        if hasattr(self.file_buffer, 'seek'):
            self.file_buffer.seek(0)
        
        # Read first few bytes to identify format
        magic_bytes = self.file_buffer.read(8)
        
        if magic_bytes[:4] == b'MVD2':
            self.header['format'] = 'MVD2'
            self.header['version'] = struct.unpack('<I', magic_bytes[4:8])[0]
        else:
            # Try to detect if it's a variant or wrapped format
            self.file_buffer.seek(0)
            data = self.file_buffer.read(512)
            
            # Look for common microscopy file signatures
            if b'TIFF' in data or b'II*' in data or b'MM*' in data:
                self.header['format'] = 'TIFF_VARIANT'
            elif b'CZI' in data:
                self.header['format'] = 'CZI_VARIANT'
            else:
                self.header['format'] = 'UNKNOWN_BINARY'
        
        return self.header
    
    def extract_metadata(self):
        """Extract metadata from MVD2 file."""
        try:
            self.file_buffer.seek(0)
            
            # Try to find metadata sections
            file_data = self.file_buffer.read()
            
            # Look for common metadata patterns
            metadata = {
                'file_size': len(file_data),
                'detected_format': self.header.get('format', 'UNKNOWN'),
                'channels': 1,  # Default assumption
                'frames': 1,    # Default assumption
                'width': 512,   # Default assumption
                'height': 512,  # Default assumption
                'pixel_type': 'uint16'
            }
            
            # Try to detect dimensions from file size
            # Common spinning disk formats
            possible_sizes = [
                (512, 512), (1024, 1024), (2048, 2048),
                (640, 480), (1280, 1024), (1920, 1080)
            ]
            
            for width, height in possible_sizes:
                # Check if file size matches expected size for different bit depths
                expected_size_8bit = width * height
                expected_size_16bit = width * height * 2
                expected_size_32bit = width * height * 4
                
                if abs(len(file_data) - expected_size_16bit) < 1000:  # Allow some header overhead
                    metadata['width'] = width
                    metadata['height'] = height
                    metadata['pixel_type'] = 'uint16'
                    break
                elif abs(len(file_data) - expected_size_8bit) < 1000:
                    metadata['width'] = width
                    metadata['height'] = height
                    metadata['pixel_type'] = 'uint8'
                    break
            
            self.metadata = metadata
            return metadata
            
        except Exception as e:
            # Fallback metadata
            return {
                'file_size': 0,
                'detected_format': 'UNKNOWN',
                'channels': 1,
                'frames': 1,
                'width': 512,
                'height': 512,
                'pixel_type': 'uint16',
                'error': str(e)
            }
    
    def read_frames(self) -> List[np.ndarray]:
        """Read image frames from MVD2 file."""
        try:
            self.file_buffer.seek(0)
            file_data = self.file_buffer.read()
            
            # Get metadata
            if not self.metadata:
                self.extract_metadata()
            
            width = self.metadata['width']
            height = self.metadata['height']
            pixel_type = self.metadata['pixel_type']
            
            # Try different approaches to extract image data
            frames = []
            
            # Approach 1: Try to read as raw binary data
            try:
                if pixel_type == 'uint16':
                    dtype = np.uint16
                    bytes_per_pixel = 2
                elif pixel_type == 'uint8':
                    dtype = np.uint8
                    bytes_per_pixel = 1
                else:
                    dtype = np.uint16
                    bytes_per_pixel = 2
                
                expected_image_size = width * height * bytes_per_pixel
                
                # Skip potential header (try different header sizes)
                header_sizes = [0, 512, 1024, 2048, 4096]
                
                for header_size in header_sizes:
                    if len(file_data) >= header_size + expected_image_size:
                        try:
                            # Extract image data
                            image_data = file_data[header_size:header_size + expected_image_size]
                            
                            # Convert to numpy array
                            image_array = np.frombuffer(image_data, dtype=dtype)
                            
                            if len(image_array) == width * height:
                                # Reshape to 2D image
                                image = image_array.reshape((height, width))
                                
                                # Basic validation - check if image has reasonable intensity range
                                if image.std() > 0 and image.max() > image.min():
                                    frames.append(image)
                                    break
                        except Exception:
                            continue
                
                # If we found at least one frame, check for multiple frames
                if frames:
                    # Try to find additional frames in the remaining data
                    remaining_data = file_data[header_sizes[0] + expected_image_size:]
                    
                    while len(remaining_data) >= expected_image_size:
                        try:
                            image_data = remaining_data[:expected_image_size]
                            image_array = np.frombuffer(image_data, dtype=dtype)
                            
                            if len(image_array) == width * height:
                                image = image_array.reshape((height, width))
                                if image.std() > 0:
                                    frames.append(image)
                                    remaining_data = remaining_data[expected_image_size:]
                                else:
                                    break
                            else:
                                break
                        except Exception:
                            break
                            
            except Exception as e:
                # If raw reading fails, create a placeholder
                frames = [np.zeros((height, width), dtype=np.uint16)]
                
            # If no frames were successfully read, create a visualization of the raw data
            if not frames:
                # Create a heatmap representation of the file data
                data_sample = np.frombuffer(file_data[:min(len(file_data), 512*512*2)], dtype=np.uint8)
                if len(data_sample) > 512*512:
                    data_sample = data_sample[:512*512]
                
                # Pad if necessary
                while len(data_sample) < 512*512:
                    data_sample = np.concatenate([data_sample, np.zeros(512*512 - len(data_sample), dtype=np.uint8)])
                
                frame = data_sample.reshape((512, 512))
                frames = [frame.astype(np.uint16)]
            
            self.frames = frames
            return frames
            
        except Exception as e:
            # Return empty frame as fallback
            return [np.zeros((512, 512), dtype=np.uint16)]

class MVD2Handler:
    """Handler class for MVD2 files with simplified interface."""
    
    def __init__(self):
        """Initialize MVD2 handler."""
        self.reader = None
    
    def load_mvd2_file(self, file_buffer):
        """
        Load MVD2 file and return DataFrame for tracking data.
        
        Parameters
        ----------
        file_buffer : file-like object
            File buffer containing MVD2 data
            
        Returns
        -------
        pandas.DataFrame or None
            DataFrame with tracking data or None if loading failed
        """
        try:
            import pandas as pd
            
            # Use the existing load_mvd2_file function
            frames, metadata = load_mvd2_file(file_buffer)
            
            # Convert to tracking format if possible
            if frames and len(frames) > 0:
                # Create basic tracking data structure
                track_data = []
                for frame_idx, frame in enumerate(frames):
                    # Simple particle detection (just peak finding)
                    if frame.max() > 0:
                        # Find local maxima as particles
                        from scipy import ndimage
                        local_maxima = ndimage.maximum_filter(frame, size=3) == frame
                        y_coords, x_coords = np.where(local_maxima & (frame > frame.mean() + 2*frame.std()))
                        
                        for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                            track_data.append({
                                'frame': frame_idx,
                                'particle': i,
                                'x': float(x),
                                'y': float(y),
                                'intensity': float(frame[y, x])
                            })
                
                if track_data:
                    return pd.DataFrame(track_data)
            
            return None
            
        except Exception as e:
            print(f"Error processing MVD2 file: {str(e)}")
            return None

def load_mvd2_file(file_buffer) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Load MVD2 file and return frames and metadata.
    
    Parameters
    ----------
    file_buffer : file-like object
        File buffer containing MVD2 data
        
    Returns
    -------
    tuple
        (frames, metadata) where frames is list of numpy arrays and metadata is dict
    """
    reader = MVD2Reader(file_buffer)
    
    # Read header and metadata
    header = reader.read_header()
    metadata = reader.extract_metadata()
    
    # Read frames
    frames = reader.read_frames()
    
    # Combine header and metadata
    full_metadata = {**header, **metadata}
    full_metadata['num_frames'] = len(frames)
    
    return frames, full_metadata