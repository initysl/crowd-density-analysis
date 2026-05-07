import numpy as np
from typing import Dict, List, Tuple
import supervision as sv
from collections import defaultdict, deque

class FlowAnalyzer:
    def __init__(self, history_length: int = 30):
        """
        Args:
            history_length: Number of frames to track for movement analysis
        """
        self.history_length = history_length
        
        # Store position history for each tracked ID
        # Format: {tracker_id: deque([(x, y, frame_idx), ...])}
        self.track_history = defaultdict(lambda: deque(maxlen=history_length))
        
        # Movement thresholds
        self.STATIONARY_THRESHOLD = 5.0  # pixels per frame
        self.BOTTLENECK_DENSITY = 3.0    # people per m²
        self.BOTTLENECK_SPEED = 10.0     # pixels per frame
        
    def update(self, detections: sv.Detections, frame_idx: int) -> Dict[int, np.ndarray]:
        """
        Update tracking history and calculate movement vectors.
        
        Returns:
            Dictionary mapping tracker_id to movement vector [dx, dy, speed, angle]
        """
        movement_vectors = {}
        
        if detections.tracker_id is None or len(detections) == 0:
            return movement_vectors
        
        # Get centroids
        centroids = self._get_centroids(detections)
        
        # Update history for each tracked person
        for tid, centroid in zip(detections.tracker_id, centroids):
            self.track_history[tid].append((centroid[0], centroid[1], frame_idx))
            
            # Calculate movement vector if we have history
            if len(self.track_history[tid]) >= 2:
                vector = self._calculate_movement_vector(tid)
                movement_vectors[tid] = vector
        
        return movement_vectors
    
    def _calculate_movement_vector(self, tracker_id: int) -> np.ndarray:
        """Calculate movement vector for a tracked person."""
        history = list(self.track_history[tracker_id])
        
        if len(history) < 2:
            return np.array([0.0, 0.0, 0.0, 0.0])  # [dx, dy, speed, angle]
        
        # Use first and last position for direction
        x1, y1, _ = history[0]
        x2, y2, _ = history[-1]
        
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate speed (pixels per frame)
        frames_elapsed = len(history)
        speed = np.sqrt(dx**2 + dy**2) / frames_elapsed if frames_elapsed > 0 else 0
        
        # Calculate angle (radians)
        angle = np.arctan2(dy, dx)
        
        return np.array([dx, dy, speed, angle])
    
    def detect_bottlenecks(
        self, 
        movement_vectors: Dict[int, np.ndarray],
        density_map: np.ndarray,
        density_sqm: np.ndarray,
        zone_assignments: Dict[int, Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Detect bottleneck zones: high density + low movement speed.
        
        Returns:
            List of (row, col) tuples for bottleneck zones
        """
        bottlenecks = []
        
        # Calculate average speed per zone
        zone_speeds = defaultdict(list)
        
        for tid, vector in movement_vectors.items():
            if tid in zone_assignments:
                row, col = zone_assignments[tid]
                speed = vector[2]  # Extract speed from vector
                zone_speeds[(row, col)].append(speed)
        
        # Check each zone for bottleneck conditions
        rows, cols = density_map.shape
        for i in range(rows):
            for j in range(cols):
                # Skip empty zones
                if density_map[i, j] == 0:
                    continue
                
                # Check density threshold
                if density_sqm[i, j] < self.BOTTLENECK_DENSITY:
                    continue
                
                # Check average speed in zone
                if (i, j) in zone_speeds:
                    avg_speed = np.mean(zone_speeds[(i, j)])
                    if avg_speed < self.BOTTLENECK_SPEED:
                        bottlenecks.append((i, j))
        
        return bottlenecks
    
    def detect_reverse_flow(
        self,
        movement_vectors: Dict[int, np.ndarray],
        zone_assignments: Dict[int, Tuple[int, int]]
    ) -> Dict[Tuple[int, int], float]:
        """
        Detect reverse flow: people moving against dominant direction.
        
        Returns:
            Dictionary mapping zone (row, col) to reverse flow ratio (0-1)
        """
        # Calculate dominant direction per zone
        zone_angles = defaultdict(list)
        
        for tid, vector in movement_vectors.items():
            if tid in zone_assignments and vector[2] > self.STATIONARY_THRESHOLD:
                row, col = zone_assignments[tid]
                angle = vector[3]  # Extract angle from vector
                zone_angles[(row, col)].append(angle)
        
        reverse_flow = {}
        
        for zone, angles in zone_angles.items():
            if len(angles) < 3:  # Need minimum sample
                continue
            
            # Calculate dominant direction (circular mean)
            angles_array = np.array(angles)
            mean_angle = np.arctan2(
                np.mean(np.sin(angles_array)),
                np.mean(np.cos(angles_array))
            )
            
            # Count people moving opposite to dominant direction
            reverse_count = 0
            for angle in angles:
                # Angular difference
                angle_diff = abs(((angle - mean_angle + np.pi) % (2 * np.pi)) - np.pi)
                if angle_diff > np.pi / 2:  # More than 90 degrees off
                    reverse_count += 1
            
            reverse_ratio = reverse_count / len(angles)
            if reverse_ratio > 0.2:  # More than 20% reverse flow
                reverse_flow[zone] = reverse_ratio
        
        return reverse_flow
    
    def calculate_zone_throughput(self,movement_vectors: Dict[int, np.ndarray],
        zone_row: int,
        zone_col: int,
        direction: str = "horizontal"  # or "vertical"
    ) -> float: # type: ignore
        """
        Calculate people per minute crossing a zone boundary.
        Not fully implemented - placeholder for exit/entry counting.
        """
        # This would require LineZone detection from supervision
        # Simplified version: count people with significant movement in direction
        pass
    
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
    
    def get_zone_assignments(
        self, 
        detections: sv.Detections,
        grid_rows: int,
        grid_cols: int,
        frame_width: int,
        frame_height: int
    ) -> Dict[int, Tuple[int, int]]:
        """Map each tracker_id to its current zone (row, col)."""
        assignments = {}
        
        if detections.tracker_id is None or len(detections) == 0:
            return assignments
        
        centroids = self._get_centroids(detections)
        zone_width = frame_width // grid_cols
        zone_height = frame_height // grid_rows
        
        for tid, centroid in zip(detections.tracker_id, centroids):
            col = int(centroid[0] // zone_width)
            row = int(centroid[1] // zone_height)
            
            # Boundary check
            col = min(col, grid_cols - 1)
            row = min(row, grid_rows - 1)
            
            assignments[tid] = (row, col)
        
        return assignments