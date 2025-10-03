from __future__ import annotations
import io
import os
import zipfile
from pathlib import PurePosixPath
from typing import Dict, List, Tuple
import mimetypes
from logging_config import get_logger
from security_utils import SecureFileHandler

# Initialize logger
logger = get_logger(__name__)

ALLOWED_EXTS = {".csv", ".xml", ".mvd2", ".xlsx"}

def _sanitize_name(name: str) -> str:
    """Sanitize name using secure handler."""
    return SecureFileHandler.sanitize_for_filename(name, max_length=80)

def _infer_mime(name: str) -> str:
    mt, _ = mimetypes.guess_type(name)
    return mt or "application/octet-stream"

def parse_zip_group_map(zip_bytes: bytes) -> Dict[str, List[Tuple[str, bytes]]]:
    """Parse zip file with security validation."""
    logger.info(f"Parsing zip file, size: {len(zip_bytes) / 1024:.1f} KB")
    
    groups: Dict[str, List[Tuple[str, bytes]]] = {}
    
    try:
        # Use secure extraction
        extracted_files = SecureFileHandler.extract_zip_safely(
            zip_bytes, 
            max_files=100,
            max_total_size=1024 * 1024 * 1000  # 1 GB
        )
        
        logger.info(f"Extracted {len(extracted_files)} files from zip")
        
        # Group files by directory structure
        for safe_filename, data in extracted_files.items():
            ext = os.path.splitext(safe_filename)[1].lower()
            if ext not in ALLOWED_EXTS:
                logger.debug(f"Skipping file with unsupported extension: {safe_filename}")
                continue
            
            # Use first part of name as group, or 'root'
            group = "root"
            groups.setdefault(group, []).append((safe_filename, data))
        
        return groups
        
    except ValueError as e:
        logger.error(f"Zip validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Error parsing zip file: {e}", exc_info=True)
        raise ValueError(f"Failed to parse zip file: {e}")

def conditions_from_zip_bytes(zip_bytes: bytes, ConditionClass):
    groups = parse_zip_group_map(zip_bytes)
    conditions = []
    for group_name, file_list in groups.items():
        cond = ConditionClass(name=group_name, description=f"Imported from ZIP group '{group_name}'")
        cond.files = []
        for fname, data in file_list:
            cond.files.append({
                "name": fname,
                "type": _infer_mime(fname),
                "size": len(data),
                "data": data
            })
        conditions.append(cond)
    return conditions
