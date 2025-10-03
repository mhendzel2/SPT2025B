"""
Centralized Logging Configuration for SPT Analysis Application
Provides structured logging with file rotation, performance tracking, and debug support.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional
import os


class SPTLogger:
    """Centralized logger for SPT Analysis application."""
    
    _instance: Optional['SPTLogger'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logger (singleton pattern)."""
        if not SPTLogger._initialized:
            self.setup_logging()
            SPTLogger._initialized = True
    
    def setup_logging(
        self, 
        log_level: int = logging.INFO,
        log_dir: str = 'logs',
        max_bytes: int = 10 * 1024 * 1024,  # 10 MB
        backup_count: int = 5
    ):
        """
        Configure application-wide logging.
        
        Parameters
        ----------
        log_level : int
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir : str
            Directory for log files
        max_bytes : int
            Maximum size of each log file before rotation
        backup_count : int
            Number of backup files to keep
        """
        # Create logs directory
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)
        
        # File handler with rotation
        log_file = log_path / f'spt_{datetime.now():%Y%m%d}.log'
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter('%(levelname)s: %(message)s')
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        # Root logger configuration
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        root_logger.handlers.clear()
        
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Log startup
        root_logger.info("=" * 60)
        root_logger.info("SPT Analysis Application Started")
        root_logger.info("=" * 60)
        
        return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Parameters
    ----------
    name : str
        Logger name (typically __name__ from calling module)
    
    Returns
    -------
    logging.Logger
        Configured logger instance
    
    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("Track data loaded: %d tracks", n_tracks)
    >>> logger.debug("MSD params: max_lag=%d, pixel_size=%.3f", max_lag, pixel_size)
    """
    # Ensure logging is initialized
    SPTLogger()
    return logging.getLogger(name)


class PerformanceLogger:
    """Track performance metrics for analysis functions."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_timing(self, operation: str, duration: float, **kwargs):
        """
        Log timing information for an operation.
        
        Parameters
        ----------
        operation : str
            Name of the operation
        duration : float
            Duration in seconds
        **kwargs
            Additional context (e.g., n_tracks, n_points)
        """
        context = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.info(
            f"PERFORMANCE | {operation} | {duration:.3f}s | {context}"
        )
    
    def log_memory(self, operation: str, memory_mb: float, **kwargs):
        """Log memory usage information."""
        context = ', '.join(f'{k}={v}' for k, v in kwargs.items())
        self.logger.info(
            f"MEMORY | {operation} | {memory_mb:.2f}MB | {context}"
        )


def log_dataframe_info(logger: logging.Logger, df, name: str = "DataFrame"):
    """
    Log information about a DataFrame.
    
    Parameters
    ----------
    logger : logging.Logger
        Logger instance
    df : pd.DataFrame
        DataFrame to log
    name : str
        Descriptive name for the DataFrame
    """
    if df is None:
        logger.warning(f"{name} is None")
        return
    
    try:
        logger.debug(
            f"{name} shape: {df.shape}, "
            f"columns: {list(df.columns)}, "
            f"memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f}MB"
        )
    except Exception as e:
        logger.error(f"Error logging DataFrame info: {e}")


def log_analysis_start(logger: logging.Logger, analysis_type: str, **params):
    """Log the start of an analysis operation."""
    params_str = ', '.join(f'{k}={v}' for k, v in params.items())
    logger.info(f"Starting {analysis_type} analysis | Parameters: {params_str}")


def log_analysis_end(logger: logging.Logger, analysis_type: str, success: bool, duration: float, **results):
    """Log the completion of an analysis operation."""
    status = "SUCCESS" if success else "FAILED"
    results_str = ', '.join(f'{k}={v}' for k, v in results.items())
    logger.info(
        f"{status} | {analysis_type} analysis | {duration:.3f}s | {results_str}"
    )


# Initialize logging on module import
_spt_logger = SPTLogger()


if __name__ == "__main__":
    # Test logging configuration
    logger = get_logger(__name__)
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test performance logging
    perf_logger = PerformanceLogger(logger)
    perf_logger.log_timing("test_operation", 1.234, n_tracks=100, n_points=5000)
    perf_logger.log_memory("test_memory", 125.5, data_loaded=True)
    
    print("\nLogging test completed. Check logs/ directory for output.")
