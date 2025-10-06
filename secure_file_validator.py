"""
Secure File Upload Validation Module

Provides comprehensive security checks for file uploads including:
- File size limits
- File type validation
- Path sanitization
- Malicious content detection
- Memory-safe file processing
"""

import os
import re
import mimetypes
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import tempfile

from logging_config import get_logger

logger = get_logger(__name__)


class FileValidationError(Exception):
    """Custom exception for file validation failures."""
    pass


class SecureFileValidator:
    """
    Comprehensive file upload validation and sanitization.
    
    Features:
    - File size limits (prevent DoS)
    - File type whitelisting
    - Path traversal prevention
    - Filename sanitization
    - Memory-safe file handling
    """
    
    # File size limits (in bytes)
    MAX_FILE_SIZES = {
        'csv': 100 * 1024 * 1024,      # 100 MB
        'xlsx': 50 * 1024 * 1024,      # 50 MB
        'xls': 50 * 1024 * 1024,       # 50 MB
        'json': 50 * 1024 * 1024,      # 50 MB
        'xml': 50 * 1024 * 1024,       # 50 MB
        'h5': 500 * 1024 * 1024,       # 500 MB
        'hdf5': 500 * 1024 * 1024,     # 500 MB
        'ims': 1024 * 1024 * 1024,     # 1 GB (Imaris files can be large)
        'tif': 500 * 1024 * 1024,      # 500 MB
        'tiff': 500 * 1024 * 1024,     # 500 MB
        'png': 100 * 1024 * 1024,      # 100 MB
        'jpg': 50 * 1024 * 1024,       # 50 MB
        'jpeg': 50 * 1024 * 1024,      # 50 MB
        'mvd2': 200 * 1024 * 1024,     # 200 MB
        'uic': 200 * 1024 * 1024,      # 200 MB
        'aisf': 200 * 1024 * 1024,     # 200 MB
        'aiix': 200 * 1024 * 1024,     # 200 MB
    }
    
    # Allowed file extensions
    ALLOWED_EXTENSIONS = {
        'track_data': ['.csv', '.xlsx', '.xls', '.json', '.xml', '.h5', '.hdf5', 
                      '.ims', '.mvd2', '.uic', '.aisf', '.aiix'],
        'image_data': ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.nd2', '.czi'],
    }
    
    # Dangerous filename patterns
    DANGEROUS_PATTERNS = [
        r'\.\.',           # Path traversal
        r'[\x00-\x1f]',    # Control characters
        r'[<>:"|?*]',      # Windows reserved chars
        r'^\.+$',          # Only dots
        r'^\s+$',          # Only whitespace
    ]
    
    def __init__(self, max_total_size: int = 2 * 1024 * 1024 * 1024):  # 2 GB default
        """
        Initialize file validator.
        
        Parameters
        ----------
        max_total_size : int
            Maximum total size for batch uploads in bytes
        """
        self.max_total_size = max_total_size
        logger.info(f"SecureFileValidator initialized with max_total_size={max_total_size / 1024 / 1024:.0f}MB")
    
    def validate_file(self, 
                     file_obj,
                     file_type: str = 'track_data',
                     check_content: bool = False) -> Dict[str, any]:
        """
        Validate uploaded file for security and compatibility.
        
        Parameters
        ----------
        file_obj
            Streamlit UploadedFile or file-like object
        file_type : str
            Type of file: 'track_data', 'image_data'
        check_content : bool
            Whether to perform content inspection (slower)
        
        Returns
        -------
        dict
            Validation results with 'valid', 'errors', 'warnings' keys
        """
        errors = []
        warnings = []
        
        try:
            # Get file properties
            filename = getattr(file_obj, 'name', 'unknown')
            file_size = getattr(file_obj, 'size', None)
            
            if file_size is None:
                # Try to get size from the object
                try:
                    file_obj.seek(0, 2)  # Seek to end
                    file_size = file_obj.tell()
                    file_obj.seek(0)  # Seek back to beginning
                except (AttributeError, OSError):
                    errors.append("Cannot determine file size")
                    return {'valid': False, 'errors': errors, 'warnings': warnings}
            
            logger.info(f"Validating file: {filename} ({file_size / 1024:.1f} KB)")
            
            # Validate filename
            filename_errors = self._validate_filename(filename)
            errors.extend(filename_errors)
            
            # Validate file extension
            ext_valid, ext_msg = self._validate_extension(filename, file_type)
            if not ext_valid:
                errors.append(ext_msg)
            
            # Validate file size
            size_valid, size_msg = self._validate_size(filename, file_size)
            if not size_valid:
                errors.append(size_msg)
            
            # Optional: Check file content
            if check_content and not errors:
                content_warnings = self._check_content(file_obj, filename)
                warnings.extend(content_warnings)
            
            # Determine overall validity
            valid = len(errors) == 0
            
            result = {
                'valid': valid,
                'errors': errors,
                'warnings': warnings,
                'filename': filename,
                'file_size': file_size,
                'sanitized_filename': self.sanitize_filename(filename)
            }
            
            if valid:
                logger.info(f"File validation passed: {filename}")
            else:
                logger.warning(f"File validation failed: {filename} - {errors}")
            
            return result
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            return {
                'valid': False,
                'errors': [f"Validation error: {str(e)}"],
                'warnings': [],
                'filename': 'unknown',
                'file_size': 0
            }
    
    def _validate_filename(self, filename: str) -> List[str]:
        """Check filename for dangerous patterns."""
        errors = []
        
        if not filename:
            errors.append("Empty filename")
            return errors
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, filename):
                errors.append(f"Filename contains dangerous pattern: {pattern}")
        
        # Check length
        if len(filename) > 255:
            errors.append("Filename too long (max 255 characters)")
        
        # Check for null bytes
        if '\x00' in filename:
            errors.append("Filename contains null bytes")
        
        return errors
    
    def _validate_extension(self, filename: str, file_type: str) -> Tuple[bool, str]:
        """Validate file extension against whitelist."""
        ext = Path(filename).suffix.lower()
        
        if not ext:
            return False, "File has no extension"
        
        allowed = self.ALLOWED_EXTENSIONS.get(file_type, [])
        
        if ext not in allowed:
            return False, f"File type '{ext}' not allowed for {file_type}. Allowed: {', '.join(allowed)}"
        
        return True, ""
    
    def _validate_size(self, filename: str, file_size: int) -> Tuple[bool, str]:
        """Validate file size against limits."""
        if file_size <= 0:
            return False, "File is empty"
        
        ext = Path(filename).suffix.lower().replace('.', '')
        max_size = self.MAX_FILE_SIZES.get(ext, 100 * 1024 * 1024)  # Default 100MB
        
        if file_size > max_size:
            return False, f"File size ({file_size / 1024 / 1024:.1f} MB) exceeds limit ({max_size / 1024 / 1024:.0f} MB)"
        
        if file_size > self.max_total_size:
            return False, f"File size exceeds total upload limit ({self.max_total_size / 1024 / 1024 / 1024:.1f} GB)"
        
        return True, ""
    
    def _check_content(self, file_obj, filename: str) -> List[str]:
        """
        Check file content for potential issues.
        
        Returns warnings (not errors) for suspicious content.
        """
        warnings = []
        
        try:
            # Read first few bytes to check
            file_obj.seek(0)
            header = file_obj.read(min(1024, getattr(file_obj, 'size', 1024)))
            file_obj.seek(0)
            
            # Check for binary vs text
            try:
                header.decode('utf-8')
                is_text = True
            except (UnicodeDecodeError, AttributeError):
                is_text = False
            
            ext = Path(filename).suffix.lower()
            
            # Text files should be text
            if ext in ['.csv', '.json', '.xml', '.txt'] and not is_text:
                warnings.append(f"File extension suggests text but content appears binary")
            
            # Check for executable signatures
            exe_signatures = [
                b'MZ',          # Windows executable
                b'\x7fELF',     # Linux executable
                b'#!',          # Script with shebang
            ]
            
            for sig in exe_signatures:
                if header.startswith(sig):
                    warnings.append("File contains executable signature - this may not be a valid data file")
            
        except Exception as e:
            warnings.append(f"Content check failed: {str(e)}")
        
        return warnings
    
    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent security issues.
        
        Parameters
        ----------
        filename : str
            Original filename
        
        Returns
        -------
        str
            Sanitized filename safe for filesystem operations
        """
        if not filename:
            return "unnamed_file"
        
        # Get just the filename without path
        filename = Path(filename).name
        
        # Remove or replace dangerous characters
        filename = re.sub(r'[<>:"|?*\x00-\x1f]', '_', filename)
        
        # Remove path traversal attempts
        filename = filename.replace('..', '_')
        
        # Remove leading/trailing dots and spaces
        filename = filename.strip('. ')
        
        # Ensure filename is not too long
        if len(filename) > 200:
            stem = Path(filename).stem[:190]
            ext = Path(filename).suffix
            filename = stem + ext
        
        # If empty after sanitization, provide default
        if not filename or filename == '_':
            filename = "unnamed_file"
        
        return filename
    
    def create_safe_temp_file(self, file_obj, prefix: str = "spt_upload_") -> str:
        """
        Create a temporary file with secure permissions.
        
        Parameters
        ----------
        file_obj
            Source file object
        prefix : str
            Prefix for temp filename
        
        Returns
        -------
        str
            Path to temporary file
        """
        try:
            # Get extension
            filename = getattr(file_obj, 'name', '')
            ext = Path(filename).suffix
            
            # Create temp file with restricted permissions
            fd, temp_path = tempfile.mkstemp(suffix=ext, prefix=prefix)
            
            # Write data
            file_obj.seek(0)
            os.write(fd, file_obj.read())
            os.close(fd)
            
            # Set restrictive permissions (owner read/write only)
            os.chmod(temp_path, 0o600)
            
            logger.info(f"Created secure temp file: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Failed to create secure temp file: {str(e)}")
            raise FileValidationError(f"Cannot create temporary file: {str(e)}")


# Global validator instance
_validator = None

def get_file_validator() -> SecureFileValidator:
    """Get singleton file validator instance."""
    global _validator
    if _validator is None:
        _validator = SecureFileValidator()
    return _validator
