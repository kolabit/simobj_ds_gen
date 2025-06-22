"""
Configuration settings for SAM2 YOLO annotation generator.
"""

import os
from pathlib import Path

# Default model settings
DEFAULT_MODEL_TYPE = "vit_h"
DEFAULT_MODEL_PATH = None  # Will auto-download if None
DEFAULT_DEVICE = "cuda"

# SAM2 model configuration
SAM2_CONFIG = {
    "points_per_side": 32,
    "pred_iou_thresh": 0.86,
    "stability_score_thresh": 0.92,
    "crop_n_layers": 1,
    "crop_n_points_downscale_factor": 2,
    "min_mask_region_area": 100,
}

# Object detection settings
DEFAULT_SIMILARITY_THRESHOLD = 0.7
DEFAULT_MIN_AREA = 100
DEFAULT_MAX_AREA = 100000

# Supported image formats
SUPPORTED_IMAGE_FORMATS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'
}

# Logging configuration
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Feature extraction weights (for similarity calculation)
FEATURE_WEIGHTS = {
    'area': 0.3,
    'aspect_ratio': 0.2,
    'circularity': 0.2,
    'color_r': 0.1,
    'color_g': 0.1,
    'color_b': 0.1,
}

# Clustering settings
CLUSTERING_CONFIG = {
    'algorithm': 'dbscan',
    'min_samples': 1,
    'eps_scaling_factor': 1.0,  # Multiplier for eps calculation
}

# Output settings
YOLO_FORMAT_PRECISION = 6  # Number of decimal places in YOLO coordinates
CREATE_CLASSES_FILE = True
VALIDATE_ANNOTATIONS = True

# Performance settings
BATCH_SIZE = 1  # Process one image at a time
MEMORY_LIMIT_MB = 8000  # Memory limit for processing
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use

# File paths
DEFAULT_OUTPUT_DIR = "annotations"
DEFAULT_MODEL_CACHE_DIR = Path.home() / ".cache" / "sam2_models"

# Environment variables
ENV_VARS = {
    'CUDA_VISIBLE_DEVICES': os.getenv('CUDA_VISIBLE_DEVICES', '0'),
    'TORCH_HOME': os.getenv('TORCH_HOME', str(Path.home() / '.torch')),
    'HF_HOME': os.getenv('HF_HOME', str(Path.home() / '.cache' / 'huggingface')),
}

# Validation settings
VALIDATION_CONFIG = {
    'check_image_exists': True,
    'check_annotation_format': True,
    'check_coordinate_bounds': True,
    'check_file_permissions': True,
}

# Error handling
ERROR_CONFIG = {
    'max_retries': 3,
    'retry_delay': 1.0,  # seconds
    'continue_on_error': True,
    'save_failed_images': True,
}

# Debug settings
DEBUG_CONFIG = {
    'save_intermediate_results': False,
    'save_debug_images': False,
    'verbose_feature_extraction': False,
    'log_clustering_details': False,
} 