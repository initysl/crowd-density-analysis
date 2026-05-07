import numpy as np
import cv2
from typing import Tuple, Dict, List
import supervision as sv

class DensityEstimator:
    def __init__(
        self, 
        frame_width: int, 
        frame_height: int,
        grid_size: Tuple[int, int] = (10, 10),
        pixels_per_meter: float = 100.0  # Calibration parameter
    ):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.grid_rows, self.grid_cols = grid_size
        self.pixels_per_meter = pixels_per_meter
        
        # Calculate zone dimensions, floor division
        self.zone_width = frame_width // self.grid_cols
        self.zone_height = frame_height // self.grid_rows
        
        # Density thresholds (people per square meter)
        self.SAFE_THRESHOLD = 2.0
        self.CAUTION_THRESHOLD = 4.0
        
    def estimate_density(self, detections: sv.Detections) -> np.ndarray:
        """
        Calculate density map from detections.
        Returns: 2D array with people count per zone
        """
        density_map = np.zeros((self.grid_rows, self.grid_cols), dtype=int)
        
        if len(detections) == 0:
            return density_map
        
        # Get centroids of bounding boxes
        centroids = self._get_centroids(detections)
        
        # Assign each person to a grid zone
        for x, y in centroids:
            col = int(x // self.zone_width)
            row = int(y // self.zone_height)
            
            # Boundary check
            col = min(col, self.grid_cols - 1)
            row = min(row, self.grid_rows - 1)
            
            density_map[row, col] += 1
            
        return density_map
    
    def calculate_density_per_sqm(self, density_map: np.ndarray) -> np.ndarray:
        """Convert people count to people per square meter."""
        zone_area_pixels = self.zone_width * self.zone_height
        zone_area_sqm = zone_area_pixels / (self.pixels_per_meter ** 2)
        
        return density_map / zone_area_sqm
    
    def get_risk_levels(self, density_sqm: np.ndarray) -> np.ndarray:
        """
        Classify zones by risk level.
        Returns: 2D array with 0=safe, 1=caution, 2=critical
        """
        risk_map = np.zeros_like(density_sqm, dtype=int)
        risk_map[density_sqm >= self.SAFE_THRESHOLD] = 1
        risk_map[density_sqm >= self.CAUTION_THRESHOLD] = 2
        
        return risk_map
    
    def _get_centroids(self, detections: sv.Detections) -> np.ndarray:
        """Extract centroids from bounding boxes."""
        if detections.xyxy is None or len(detections.xyxy) == 0:
            return np.array([])
        
        centroids = []
        for box in detections.xyxy:
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroids.append([cx, cy])
            
        return np.array(centroids)
    
    def get_zone_stats(self, density_map: np.ndarray) -> Dict:
        """Get summary statistics."""
        density_sqm = self.calculate_density_per_sqm(density_map)
        risk_levels = self.get_risk_levels(density_sqm)
        
        return {
            'total_people': int(density_map.sum()),
            'max_density': float(density_sqm.max()),
            'avg_density': float(density_sqm.mean()),
            'critical_zones': int((risk_levels == 2).sum()),
            'caution_zones': int((risk_levels == 1).sum()),
            'safe_zones': int((risk_levels == 0).sum())
        }