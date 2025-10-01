"""
Logging utilities for SPT2025B application.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Global logger instance
_logger: Optional[logging.Logger] = None


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    console_output: bool = True,
    log_to_file: bool = True
) -> Path:
    """
    Setup application-wide logging.
    
    Parameters
    ----------
    log_dir : str, default "logs"
        Directory to store log files
    log_level : str, default "INFO"
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    console_output : bool, default True
        Whether to output logs to console
    log_to_file : bool, default True
        Whether to output logs to file
    
    Returns
    -------
    Path
        Path to the created log file (if log_to_file is True)
    """
    global _logger
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    log_file = log_path / f"spt2025b_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers = []
    
    # File handler with detailed format
    if log_to_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        root_logger.addHandler(file_handler)
    
    # Console handler with simpler format
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_format = logging.Formatter(
            '%(levelname)s - %(name)s - %(message)s'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
    
    _logger = root_logger
    logging.info(f"Logging initialized. Log file: {log_file}")
    
    return log_file


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Parameters
    ----------
    name : str, optional
        Name for the logger (typically __name__)
    
    Returns
    -------
    logging.Logger
        Logger instance
    """
    if name:
        return logging.getLogger(name)
    return logging.getLogger()


def log_dataframe_info(df, name: str = "DataFrame"):
    """
    Log detailed information about a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to log information about
    name : str, default "DataFrame"
        Name to use in log messages
    """
    logger = get_logger(__name__)
    
    if df is None:
        logger.warning(f"{name} is None")
        return
    
    logger.info(f"{name} - Shape: {df.shape}")
    logger.info(f"{name} - Columns: {list(df.columns)}")
    logger.info(f"{name} - Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for null values
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning(f"{name} - Null values found:")
        for col, count in null_counts[null_counts > 0].items():
            logger.warning(f"  {col}: {count} nulls")
    
    # Log data types
    logger.debug(f"{name} - Data types:")
    for col, dtype in df.dtypes.items():
        logger.debug(f"  {col}: {dtype}")


def log_function_call(func):
    """
    Decorator to log function calls.
    
    Parameters
    ----------
    func : callable
        Function to decorate
    
    Returns
    -------
    callable
        Decorated function
    """
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.debug(f"Calling {func.__name__} with args={args[:3]}..., kwargs={list(kwargs.keys())}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"{func.__name__} completed successfully")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} failed with error: {str(e)}")
            raise
    return wrapper


# Initialize logging with default settings on import
try:
    setup_logging(log_level="INFO", console_output=False, log_to_file=True)
except Exception:
    # If setup fails (e.g., no write permissions), continue without logging
    pass
