import cv2
import numpy as np
from typing import Dict, List, Tuple

class FlowVisualizer:
    def __init__(self):
        self.COLOR_BOTTLENECK = (255, 0, 255)  # Magenta
        self.COLOR_REVERSE = (0, 165, 255)     # Orange
        
    def draw_movement_vectors(
        self,
        frame: np.ndarray,
        movement_vectors: Dict[int, np.ndarray],
        track_history: Dict,
        scale: float = 3.0
    ) -> np.ndarray:
        """Draw movement arrows for each tracked person."""
        annotated = frame.copy()
        
        for tid, vector in movement_vectors.items():
            if tid not in track_history or len(track_history[tid]) < 2:
                continue
            
            # Get current position
            x, y, _ = track_history[tid][-1]
            
            dx, dy, speed, angle = vector
            
            # Skip if stationary
            if speed < 2.0:
                continue
            
            # Draw arrow
            end_x = int(x + dx * scale)
            end_y = int(y + dy * scale)
            
            cv2.arrowedLine(
                annotated,
                (int(x), int(y)),
                (end_x, end_y),
                (0, 255, 0),
                2,
                tipLength=0.3
            )
        
        return annotated
    
    def highlight_bottlenecks(self, frame: np.ndarray,
        bottlenecks: List[Tuple[int, int]],
        grid_rows: int,
        grid_cols: int
    ) -> np.ndarray:
        """Highlight bottleneck zones."""
        annotated = frame.copy()
        
        frame_h, frame_w = frame.shape[:2]
        zone_h = frame_h // grid_rows
        zone_w = frame_w // grid_cols
        
        for row, col in bottlenecks:
            x1 = col * zone_w
            y1 = row * zone_h
            x2 = x1 + zone_w
            y2 = y1 + zone_h
            
            # Draw thick border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_BOTTLENECK, 4)
            
            # Add warning text
            cv2.putText(
                annotated,
                "BOTTLENECK",
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                self.COLOR_BOTTLENECK,
                2
            )
        
        return annotated
    
    def highlight_reverse_flow(
        self,
        frame: np.ndarray,
        reverse_zones: Dict[Tuple[int, int], float],
        grid_rows: int,
        grid_cols: int
    ) -> np.ndarray:
        """Highlight zones with reverse flow."""
        annotated = frame.copy()
        
        frame_h, frame_w = frame.shape[:2]
        zone_h = frame_h // grid_rows
        zone_w = frame_w // grid_cols
        
        for (row, col), ratio in reverse_zones.items():
            x1 = col * zone_w
            y1 = row * zone_h
            x2 = x1 + zone_w
            y2 = y1 + zone_h
            
            # Draw border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), self.COLOR_REVERSE, 3)
            
            # Add warning
            text = f"REVERSE {ratio*100:.0f}%"
            cv2.putText(
                annotated,
                text,
                (x1 + 5, y2 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.COLOR_REVERSE,
                2
            )
        
        return annotated