"""
Object detector for finding similar objects based on volume, shape, and color.
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class ObjectDetector:
    """Detects similar objects based on volume, shape, and color characteristics."""
    
    def __init__(self, similarity_threshold: float = 0.7, 
                 min_area: int = 100, max_area: int = 100000):
        """
        Initialize object detector.
        
        Args:
            similarity_threshold: Threshold for object similarity (0.0-1.0)
            min_area: Minimum object area in pixels
            max_area: Maximum object area in pixels
        """
        self.similarity_threshold = similarity_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.logger = logging.getLogger(__name__)
    
    def detect_similar_objects(self, masks: List[np.ndarray], 
                             boxes: List[List[float]], 
                             scores: List[float],
                             image_path: Path) -> List[Dict[str, Any]]:
        """
        Detect groups of similar objects based on volume, shape, and color.
        
        Args:
            masks: List of binary masks
            boxes: List of bounding boxes [x1, y1, x2, y2]
            scores: List of confidence scores
            image_path: Path to the original image
            
        Returns:
            List of object groups with similar characteristics
        """
        try:
            # Load original image for color analysis
            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error(f"Failed to load image: {image_path}")
                return []
            
            # Filter objects by area
            valid_objects = []
            for i, mask in enumerate(masks):
                area = np.sum(mask)
                if self.min_area <= area <= self.max_area:
                    valid_objects.append({
                        'index': i,
                        'mask': mask,
                        'box': boxes[i],
                        'score': scores[i],
                        'area': area
                    })
            
            if not valid_objects:
                self.logger.warning("No valid objects found after area filtering")
                return []
            
            # Extract features for each object
            features = []
            for obj in valid_objects:
                feature_vector = self._extract_features(obj, image)
                features.append(feature_vector)
            
            # Cluster similar objects
            object_groups = self._cluster_objects(valid_objects, features)
            
            # Assign class IDs to groups
            for group_id, group in enumerate(object_groups):
                for obj in group:
                    obj['class_id'] = group_id
            
            self.logger.info(f"Detected {len(object_groups)} object groups")
            return object_groups
            
        except Exception as e:
            self.logger.error(f"Error detecting similar objects: {e}")
            return []
    
    def _extract_features(self, obj: Dict[str, Any], image: np.ndarray) -> np.ndarray:
        """
        Extract feature vector for an object.
        
        Args:
            obj: Object dictionary with mask, box, etc.
            image: Original image
            
        Returns:
            Feature vector [area, aspect_ratio, circularity, mean_color_r, mean_color_g, mean_color_b]
        """
        mask = obj['mask']
        box = obj['box']
        
        # Area (normalized)
        area = obj['area']
        
        # Aspect ratio
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Circularity (4π * area / perimeter^2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        else:
            circularity = 0
        
        # Color features (mean RGB values)
        mask_3d = np.stack([mask] * 3, axis=-1)
        masked_image = image * mask_3d
        valid_pixels = mask_3d.sum(axis=(0, 1)) > 0
        
        if valid_pixels[0]:  # If there are valid pixels
            mean_color = cv2.mean(masked_image, mask=mask)[:3]  # BGR to RGB
            mean_r, mean_g, mean_b = mean_color[2], mean_color[1], mean_color[0]
        else:
            mean_r = mean_g = mean_b = 0
        
        # Normalize features
        area_norm = area / (image.shape[0] * image.shape[1])  # Normalize by image area
        aspect_ratio_norm = min(aspect_ratio, 1/aspect_ratio)  # Normalize aspect ratio
        color_norm = 255.0  # Normalize color values
        
        return np.array([
            area_norm,
            aspect_ratio_norm,
            circularity,
            mean_r / color_norm,
            mean_g / color_norm,
            mean_b / color_norm
        ])
    
    def _cluster_objects(self, objects: List[Dict[str, Any]], 
                        features: List[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Cluster objects based on their features.
        
        Args:
            objects: List of object dictionaries
            features: List of feature vectors
            
        Returns:
            List of object groups
        """
        if len(objects) < 2:
            return [objects] if objects else []
        
        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Use DBSCAN for clustering
        # Adjust eps based on similarity threshold
        eps = 1.0 - self.similarity_threshold
        
        clustering = DBSCAN(eps=eps, min_samples=1).fit(features_scaled)
        labels = clustering.labels_
        
        # Group objects by cluster
        groups = defaultdict(list)
        for i, label in enumerate(labels):
            groups[label].append(objects[i])
        
        # Convert to list and filter out noise (label -1)
        object_groups = []
        for label, group in groups.items():
            if label != -1 and len(group) > 0:
                object_groups.append(group)
        
        # If no groups found, create individual groups
        if not object_groups:
            object_groups = [[obj] for obj in objects]
        
        return object_groups
    
    def calculate_similarity(self, obj1: Dict[str, Any], 
                           obj2: Dict[str, Any], 
                           image: np.ndarray) -> float:
        """
        Calculate similarity between two objects.
        
        Args:
            obj1: First object
            obj2: Second object
            image: Original image
            
        Returns:
            Similarity score (0.0-1.0)
        """
        features1 = self._extract_features(obj1, image)
        features2 = self._extract_features(obj2, image)
        
        # Calculate cosine similarity
        dot_product = np.dot(features1, features2)
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return max(0.0, min(1.0, similarity)) 