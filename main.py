#!/usr/bin/env python3
"""
Main script for image segmentation and YOLO annotation generation using SAM2.
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

from sam2_processor import SAM2Processor
from object_detector import ObjectDetector
from yolo_annotator import YOLOAnnotator
from utils import setup_logging, validate_directory

def main():
    parser = argparse.ArgumentParser(description="Generate YOLO annotations using SAM2 segmentation")
    parser.add_argument("--input_dir", type=str, required=True, 
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save YOLO annotation files")
    parser.add_argument("--model_path", type=str, 
                       help="Path to SAM2 model checkpoint (optional)")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--similarity_threshold", type=float, default=0.7,
                       help="Threshold for object similarity (0.0-1.0)")
    parser.add_argument("--min_area", type=int, default=100,
                       help="Minimum object area in pixels")
    parser.add_argument("--max_area", type=int, default=100000,
                       help="Maximum object area in pixels")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate directories
    if not validate_directory(args.input_dir, "input"):
        return
    if not validate_directory(args.output_dir, "output", create=True):
        return
    
    # Initialize components
    try:
        sam2_processor = SAM2Processor(
            model_path=args.model_path,
            device=args.device
        )
        object_detector = ObjectDetector(
            similarity_threshold=args.similarity_threshold,
            min_area=args.min_area,
            max_area=args.max_area
        )
        yolo_annotator = YOLOAnnotator()
        
        logger.info("All components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return
    
    # Process images
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [
        f for f in input_path.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No image files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(image_files)} image files to process")
    
    # Process each image
    successful_count = 0
    failed_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        try:
            logger.debug(f"Processing {image_file.name}")
            
            # Run SAM2 segmentation
            masks, boxes, scores = sam2_processor.process_image(str(image_file))
            
            if masks is None or len(masks) == 0:
                logger.warning(f"No objects detected in {image_file.name}")
                continue
            
            # Detect similar objects
            object_groups = object_detector.detect_similar_objects(
                masks, boxes, scores, image_file
            )
            
            if not object_groups:
                logger.warning(f"No similar object groups found in {image_file.name}")
                continue
            
            # Generate YOLO annotations
            annotation_file = output_path / f"{image_file.stem}.txt"
            yolo_annotator.create_annotations(
                object_groups, 
                str(annotation_file),
                image_file
            )
            
            successful_count += 1
            logger.debug(f"Successfully processed {image_file.name}")
            
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process {image_file.name}: {e}")
            continue
    
    logger.info(f"Processing complete. Success: {successful_count}, Failed: {failed_count}")

if __name__ == "__main__":
    main() 