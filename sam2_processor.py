"""
SAM2 (Segment Anything 2) processor for automatic image segmentation.
"""

import logging
import numpy as np
import cv2
from PIL import Image
import torch
from typing import Tuple, List, Optional

try:
    from segment_anything_2 import sam_model_registry, SamAutomaticMaskGenerator
except ImportError:
    logging.warning("segment-anything-2 not found. Please install it: pip install segment-anything-2")

class SAM2Processor:
    """Handles SAM2 model loading and image segmentation."""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize SAM2 processor.
        
        Args:
            model_path: Path to SAM2 model checkpoint. If None, will try to download.
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self.model = None
        self.mask_generator = None
        self._load_model(model_path)
    
    def _load_model(self, model_path: Optional[str]):
        """Load SAM2 model and mask generator."""
        try:
            if not torch.cuda.is_available() and self.device == "cuda":
                self.logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            # Try to load model
            if model_path is None:
                # Use default model
                model_type = "vit_h"
                checkpoint = "sam2_h.pth"
                self.logger.info(f"Loading default SAM2 model: {model_type}")
            else:
                # Extract model type from path
                if "vit_h" in model_path:
                    model_type = "vit_h"
                elif "vit_l" in model_path:
                    model_type = "vit_l"
                elif "vit_b" in model_path:
                    model_type = "vit_b"
                else:
                    model_type = "vit_h"  # Default
                checkpoint = model_path
                self.logger.info(f"Loading SAM2 model from: {model_path}")
            
            # Load model
            self.model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.model.to(device=self.device)
            
            # Configure mask generator
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.model,
                points_per_side=32,
                pred_iou_thresh=0.86,
                stability_score_thresh=0.92,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )
            
            self.logger.info("SAM2 model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load SAM2 model: {e}")
            raise
    
    def process_image(self, image_path: str) -> Tuple[Optional[List[np.ndarray]], 
                                                     Optional[List[List[float]]], 
                                                     Optional[List[float]]]:
        """
        Process an image and return masks, bounding boxes, and scores.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (masks, bounding_boxes, scores) or (None, None, None) if failed
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return None, None, None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Generate masks
            masks = self.mask_generator.generate(image_rgb)
            
            if not masks:
                self.logger.warning(f"No masks generated for {image_path}")
                return None, None, None
            
            # Extract masks, boxes, and scores
            mask_list = []
            box_list = []
            score_list = []
            
            for mask_data in masks:
                # Get mask
                mask = mask_data["segmentation"]
                mask_list.append(mask.astype(np.uint8))
                
                # Get bounding box
                bbox = mask_data["bbox"]  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                x1, y1, w, h = bbox
                x2, y2 = x1 + w, y1 + h
                box_list.append([x1, y1, x2, y2])
                
                # Get score
                score = mask_data.get("stability_score", 0.0)
                score_list.append(score)
            
            self.logger.debug(f"Generated {len(mask_list)} masks for {image_path}")
            return mask_list, box_list, score_list
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None, None, None
    
    def get_image_info(self, image_path: str) -> Tuple[Optional[int], Optional[int]]:
        """Get image dimensions."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None, None
            height, width = image.shape[:2]
            return width, height
        except Exception as e:
            self.logger.error(f"Error getting image info for {image_path}: {e}")
            return None, None 