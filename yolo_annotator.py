"""
YOLO format annotation generator.
"""

import logging
import cv2
from pathlib import Path
from typing import List, Dict, Any

class YOLOAnnotator:
    """Generates YOLO format annotations from detected objects."""
    
    def __init__(self):
        """Initialize YOLO annotator."""
        self.logger = logging.getLogger(__name__)
    
    def create_annotations(self, object_groups: List[List[Dict[str, Any]]], 
                          output_path: str, image_path: Path) -> bool:
        """
        Create YOLO format annotation file.
        
        Args:
            object_groups: List of object groups, each group contains similar objects
            output_path: Path to output annotation file
            image_path: Path to the original image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get image dimensions
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            img_height, img_width = image.shape[:2]
            
            # Create annotation lines
            annotation_lines = []
            
            for group in object_groups:
                class_id = group[0]['class_id'] if group else 0
                
                for obj in group:
                    # Get bounding box in [x1, y1, x2, y2] format
                    x1, y1, x2, y2 = obj['box']
                    
                    # Convert to YOLO format: [class_id, center_x, center_y, width, height]
                    # All values are normalized to [0, 1]
                    center_x = (x1 + x2) / 2.0 / img_width
                    center_y = (y1 + y2) / 2.0 / img_height
                    width = (x2 - x1) / img_width
                    height = (y2 - y1) / img_height
                    
                    # Ensure values are within [0, 1]
                    center_x = max(0.0, min(1.0, center_x))
                    center_y = max(0.0, min(1.0, center_y))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    # Create YOLO annotation line
                    line = f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"
                    annotation_lines.append(line)
            
            # Write annotation file
            with open(output_path, 'w') as f:
                f.write('\n'.join(annotation_lines))
            
            self.logger.info(f"Created YOLO annotation file: {output_path}")
            self.logger.info(f"Generated {len(annotation_lines)} annotations")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating YOLO annotations: {e}")
            return False
    
    def create_classes_file(self, output_dir: str, class_names: List[str] = None) -> bool:
        """
        Create classes.txt file with class names.
        
        Args:
            output_dir: Directory to save classes.txt
            class_names: List of class names. If None, will use generic names.
            
        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_dir) / "classes.txt"
            
            if class_names is None:
                # Generate generic class names
                class_names = [f"object_{i}" for i in range(100)]  # Support up to 100 classes
            
            with open(output_path, 'w') as f:
                for class_name in class_names:
                    f.write(f"{class_name}\n")
            
            self.logger.info(f"Created classes file: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating classes file: {e}")
            return False
    
    def validate_annotation(self, annotation_path: str, image_path: str) -> bool:
        """
        Validate YOLO annotation file.
        
        Args:
            annotation_path: Path to annotation file
            image_path: Path to corresponding image
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check if image exists
            if not Path(image_path).exists():
                self.logger.error(f"Image file not found: {image_path}")
                return False
            
            # Get image dimensions
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return False
            
            img_height, img_width = image.shape[:2]
            
            # Read annotation file
            if not Path(annotation_path).exists():
                self.logger.error(f"Annotation file not found: {annotation_path}")
                return False
            
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            # Validate each annotation line
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split()
                    if len(parts) != 5:
                        self.logger.error(f"Invalid annotation format at line {i+1}: {line}")
                        return False
                    
                    class_id = int(parts[0])
                    center_x = float(parts[1])
                    center_y = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])
                    
                    # Check if values are within valid ranges
                    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 
                           0 <= width <= 1 and 0 <= height <= 1):
                        self.logger.error(f"Invalid coordinate values at line {i+1}: {line}")
                        return False
                    
                    # Check if bounding box is within image bounds
                    x1 = (center_x - width/2) * img_width
                    y1 = (center_y - height/2) * img_height
                    x2 = (center_x + width/2) * img_width
                    y2 = (center_y + height/2) * img_height
                    
                    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
                        self.logger.warning(f"Bounding box extends outside image at line {i+1}: {line}")
                    
                except (ValueError, IndexError) as e:
                    self.logger.error(f"Error parsing annotation at line {i+1}: {e}")
                    return False
            
            self.logger.info(f"Annotation file validated successfully: {annotation_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating annotation: {e}")
            return False 