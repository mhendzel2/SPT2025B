"""
Security Utilities for SPT Analysis Application
Provides secure file handling, validation, and sanitization.
"""

from pathlib import Path, PurePosixPath
import os
import tempfile
import zipfile
import io
from typing import Optional, List, Dict, Any, Tuple
import mimetypes
import re


class SecureFileHandler:
    """Secure file handling utilities with validation and sanitization."""
    
    # Allowed file extensions for upload
    ALLOWED_EXTENSIONS = {
        '.csv', '.xlsx', '.xls', '.xml', '.mvd2', 
        '.tif', '.tiff', '.png', '.jpg', '.jpeg',
        '.h5', '.hdf5', '.xtc', '.dcd', '.trr',
        '.gro', '.pdb', '.xyz', '.aiix', '.aisf'
    }
    
    # Maximum file sizes
    MAX_FILE_SIZE = 1024 * 1024 * 500  # 500 MB
    MAX_IMAGE_SIZE = 1024 * 1024 * 200  # 200 MB for images
    MAX_ZIP_SIZE = 1024 * 1024 * 1000  # 1 GB for zip files
    
    # String limits
    MAX_FILENAME_LENGTH = 255
    MAX_PATH_LENGTH = 4096
    
    # Dangerous patterns
    DANGEROUS_PATTERNS = [
        r'\.\./',  # Directory traversal
        r'\.\.',   # Parent directory
        r'[\x00-\x1f]',  # Control characters
        r'[<>:"|?*]',  # Windows forbidden characters (in addition to normal sanitization)
    ]
    
    @staticmethod
    def validate_filename(filename: str, check_extension: bool = True) -> str:
        """
        Validate and sanitize filename.
        
        Parameters
        ----------
        filename : str
            Original filename
        check_extension : bool
            Whether to validate file extension
        
        Returns
        -------
        str
            Sanitized filename
        
        Raises
        ------
        ValueError
            If filename is invalid or dangerous
        """
        if not filename:
            raise ValueError("Filename cannot be empty")
        
        # Remove null bytes
        filename = filename.replace('\x00', '')
        
        # Check for dangerous patterns
        for pattern in SecureFileHandler.DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                raise ValueError(f"Dangerous pattern detected in filename: {filename}")
        
        # Get just the base name (prevent directory traversal)
        filename = os.path.basename(filename)
        
        # Check length
        if len(filename) > SecureFileHandler.MAX_FILENAME_LENGTH:
            raise ValueError(
                f"Filename too long: {len(filename)} chars "
                f"(max: {SecureFileHandler.MAX_FILENAME_LENGTH})"
            )
        
        # Check extension
        if check_extension:
            ext = Path(filename).suffix.lower()
            if ext not in SecureFileHandler.ALLOWED_EXTENSIONS:
                raise ValueError(
                    f"File type not allowed: {ext}. "
                    f"Allowed types: {', '.join(sorted(SecureFileHandler.ALLOWED_EXTENSIONS))}"
                )
        
        # Sanitize remaining characters
        # Allow alphanumeric, spaces, underscores, hyphens, and dots
        safe_name = re.sub(r'[^\w\s\-\.]', '_', filename)
        
        # Collapse multiple underscores
        safe_name = re.sub(r'_+', '_', safe_name)
        
        # Trim spaces
        safe_name = safe_name.strip()
        
        if not safe_name:
            raise ValueError("Filename contains no valid characters")
        
        return safe_name
    
    @staticmethod
    def validate_file_size(file, max_size: Optional[int] = None) -> bool:
        """
        Check file size before processing.
        
        Parameters
        ----------
        file : file-like object
            File to validate
        max_size : int, optional
            Maximum allowed size in bytes
        
        Returns
        -------
        bool
            True if size is valid
        
        Raises
        ------
        ValueError
            If file is too large
        """
        if max_size is None:
            max_size = SecureFileHandler.MAX_FILE_SIZE
        
        file_size = 0
        
        if hasattr(file, 'size'):
            file_size = file.size
        elif hasattr(file, 'seek') and hasattr(file, 'tell'):
            current_pos = file.tell()
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(current_pos)  # Restore position
        
        if file_size > max_size:
            raise ValueError(
                f"File too large: {file_size / 1e6:.1f} MB "
                f"(max: {max_size / 1e6:.1f} MB)"
            )
        
        return True
    
    @staticmethod
    def validate_path(path: str, base_dir: Optional[str] = None) -> Path:
        """
        Validate path to prevent directory traversal.
        
        Parameters
        ----------
        path : str
            Path to validate
        base_dir : str, optional
            Base directory that path must be within
        
        Returns
        -------
        Path
            Validated Path object
        
        Raises
        ------
        ValueError
            If path is invalid or escapes base_dir
        """
        if not path:
            raise ValueError("Path cannot be empty")
        
        # Check length
        if len(path) > SecureFileHandler.MAX_PATH_LENGTH:
            raise ValueError(f"Path too long: {len(path)} chars")
        
        # Convert to Path and resolve (handles .., symlinks, etc.)
        try:
            path_obj = Path(path).resolve()
        except (ValueError, OSError) as e:
            raise ValueError(f"Invalid path: {e}")
        
        # If base_dir specified, ensure path is within it
        if base_dir:
            base_path = Path(base_dir).resolve()
            try:
                path_obj.relative_to(base_path)
            except ValueError:
                raise ValueError(
                    f"Path escapes base directory: {path} not in {base_dir}"
                )
        
        return path_obj
    
    @staticmethod
    def extract_zip_safely(
        zip_bytes: bytes,
        max_files: int = 100,
        max_total_size: Optional[int] = None
    ) -> Dict[str, bytes]:
        """
        Safely extract zip file with validation.
        
        Parameters
        ----------
        zip_bytes : bytes
            Zip file content
        max_files : int
            Maximum number of files allowed in zip
        max_total_size : int, optional
            Maximum total extracted size
        
        Returns
        -------
        dict
            Dictionary mapping sanitized filenames to content
        
        Raises
        ------
        ValueError
            If zip file is dangerous or exceeds limits
        """
        if max_total_size is None:
            max_total_size = SecureFileHandler.MAX_ZIP_SIZE
        
        extracted_files = {}
        
        try:
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                # Check file count
                if len(zf.namelist()) > max_files:
                    raise ValueError(
                        f"Too many files in zip: {len(zf.namelist())} "
                        f"(max: {max_files})"
                    )
                
                # Check total extracted size
                total_size = sum(zinfo.file_size for zinfo in zf.infolist())
                if total_size > max_total_size:
                    raise ValueError(
                        f"Zip extraction would exceed size limit: "
                        f"{total_size / 1e6:.1f} MB (max: {max_total_size / 1e6:.1f} MB)"
                    )
                
                # Extract and validate each file
                for zinfo in zf.infolist():
                    if zinfo.is_dir():
                        continue
                    
                    # Get original filename
                    original_name = zinfo.filename
                    
                    # Remove directory components (prevent path traversal)
                    base_name = os.path.basename(original_name)
                    
                    if not base_name:
                        continue  # Skip if no filename
                    
                    # Validate and sanitize filename
                    try:
                        safe_name = SecureFileHandler.validate_filename(
                            base_name, 
                            check_extension=False  # Allow any extension in zip
                        )
                    except ValueError:
                        # Skip files with invalid names
                        continue
                    
                    # Read content
                    content = zf.read(zinfo)
                    
                    # Store with safe name
                    # If duplicate, append number
                    if safe_name in extracted_files:
                        name_parts = safe_name.rsplit('.', 1)
                        if len(name_parts) == 2:
                            base, ext = name_parts
                            counter = 1
                            while f"{base}_{counter}.{ext}" in extracted_files:
                                counter += 1
                            safe_name = f"{base}_{counter}.{ext}"
                        else:
                            counter = 1
                            while f"{safe_name}_{counter}" in extracted_files:
                                counter += 1
                            safe_name = f"{safe_name}_{counter}"
                    
                    extracted_files[safe_name] = content
                
        except zipfile.BadZipFile as e:
            raise ValueError(f"Invalid zip file: {e}")
        except Exception as e:
            raise ValueError(f"Error extracting zip: {e}")
        
        return extracted_files
    
    @staticmethod
    def sanitize_for_filename(text: str, max_length: int = 50) -> str:
        """
        Sanitize text for use in filename.
        
        Parameters
        ----------
        text : str
            Text to sanitize
        max_length : int
            Maximum length of result
        
        Returns
        -------
        str
            Sanitized text suitable for filename
        """
        # Remove/replace dangerous characters
        safe = re.sub(r'[^\w\s\-]', '_', text)
        
        # Collapse whitespace
        safe = re.sub(r'\s+', '_', safe)
        
        # Collapse underscores
        safe = re.sub(r'_+', '_', safe)
        
        # Trim
        safe = safe.strip('_')
        
        # Limit length
        if len(safe) > max_length:
            safe = safe[:max_length].rstrip('_')
        
        return safe or "unnamed"


def validate_csv_content(content: bytes, max_size_mb: int = 100) -> bool:
    """
    Validate CSV file content for safety.
    
    Parameters
    ----------
    content : bytes
        CSV file content
    max_size_mb : int
        Maximum size in MB
    
    Returns
    -------
    bool
        True if valid
    
    Raises
    ------
    ValueError
        If content is invalid
    """
    # Check size
    size_mb = len(content) / 1024 / 1024
    if size_mb > max_size_mb:
        raise ValueError(f"CSV too large: {size_mb:.1f} MB (max: {max_size_mb} MB)")
    
    # Try to decode as text
    try:
        text = content.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = content.decode('latin-1')
        except UnicodeDecodeError:
            raise ValueError("CSV file encoding not supported")
    
    # Basic sanity checks
    if not text.strip():
        raise ValueError("CSV file is empty")
    
    # Check for extremely long lines (potential attack)
    lines = text.split('\n')
    for i, line in enumerate(lines[:100]):  # Check first 100 lines
        if len(line) > 100000:  # 100k chars per line
            raise ValueError(f"Line {i+1} is too long (potential security issue)")
    
    return True


if __name__ == "__main__":
    # Test security utilities
    handler = SecureFileHandler()
    
    # Test filename validation
    try:
        safe = handler.validate_filename("test_file.csv")
        print(f"✓ Safe filename: {safe}")
        
        safe = handler.validate_filename("../../etc/passwd")
        print("✗ Should have rejected directory traversal")
    except ValueError as e:
        print(f"✓ Rejected dangerous filename: {e}")
    
    # Test sanitization
    text = "My Analysis Results (2025)!"
    safe = handler.sanitize_for_filename(text)
    print(f"✓ Sanitized: '{text}' -> '{safe}'")
    
    print("\nSecurity utilities test completed.")
