"""
Utility functions for the SAM2 YOLO annotation project.
"""

import logging
import os
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR
    }
    
    log_level = level_map.get(level.upper(), logging.INFO)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def validate_directory(path: str, purpose: str, create: bool = False) -> bool:
    """
    Validate if a directory exists and is accessible.
    
    Args:
        path: Directory path to validate
        purpose: Purpose of the directory (for error messages)
        create: Whether to create the directory if it doesn't exist
        
    Returns:
        True if directory is valid, False otherwise
    """
    try:
        dir_path = Path(path)
        
        if not dir_path.exists():
            if create:
                dir_path.mkdir(parents=True, exist_ok=True)
                logging.info(f"Created {purpose} directory: {path}")
            else:
                logging.error(f"{purpose.capitalize()} directory does not exist: {path}")
                return False
        
        if not dir_path.is_dir():
            logging.error(f"{purpose.capitalize()} path is not a directory: {path}")
            return False
        
        # Check if directory is readable
        if not os.access(dir_path, os.R_OK):
            logging.error(f"{purpose.capitalize()} directory is not readable: {path}")
            return False
        
        # Check if directory is writable (for output directories)
        if purpose == "output" and not os.access(dir_path, os.W_OK):
            logging.error(f"{purpose.capitalize()} directory is not writable: {path}")
            return False
        
        logging.debug(f"{purpose.capitalize()} directory validated: {path}")
        return True
        
    except Exception as e:
        logging.error(f"Error validating {purpose} directory {path}: {e}")
        return False

def get_image_files(directory: str) -> list:
    """
    Get all image files from a directory.
    
    Args:
        directory: Directory to search for images
        
    Returns:
        List of image file paths
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []
    
    try:
        dir_path = Path(directory)
        for file_path in dir_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        logging.info(f"Found {len(image_files)} image files in {directory}")
        return image_files
        
    except Exception as e:
        logging.error(f"Error getting image files from {directory}: {e}")
        return []

def ensure_output_directory(output_dir: str) -> bool:
    """
    Ensure output directory exists and is writable.
    
    Args:
        output_dir: Output directory path
        
    Returns:
        True if successful, False otherwise
    """
    return validate_directory(output_dir, "output", create=True)

def get_file_size_mb(file_path: str) -> Optional[float]:
    """
    Get file size in megabytes.
    
    Args:
        file_path: Path to the file
        
    Returns:
        File size in MB or None if error
    """
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logging.error(f"Error getting file size for {file_path}: {e}")
        return None

def format_file_size(size_mb: float) -> str:
    """
    Format file size in human-readable format.
    
    Args:
        size_mb: File size in megabytes
        
    Returns:
        Formatted file size string
    """
    if size_mb < 1:
        return f"{size_mb * 1024:.1f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.1f} MB"
    else:
        return f"{size_mb / 1024:.1f} GB"

def check_dependencies() -> bool:
    """
    Check if all required dependencies are available.
    
    Returns:
        True if all dependencies are available, False otherwise
    """
    missing_deps = []
    
    try:
        import torch
        logging.debug("PyTorch is available")
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import cv2
        logging.debug("OpenCV is available")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import numpy
        logging.debug("NumPy is available")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        from sklearn.cluster import DBSCAN
        logging.debug("scikit-learn is available")
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        from segment_anything_2 import sam_model_registry
        logging.debug("segment-anything-2 is available")
    except ImportError:
        missing_deps.append("segment-anything-2")
    
    if missing_deps:
        logging.error(f"Missing dependencies: {', '.join(missing_deps)}")
        logging.error("Please install missing dependencies: pip install " + " ".join(missing_deps))
        return False
    
    logging.info("All dependencies are available")
    return True 