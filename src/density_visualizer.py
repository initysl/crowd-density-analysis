import numpy as np
import cv2

class DensityVisualizer:
    def __init__(self):
        # Color map: Green -> Yellow -> Red
        self.COLOR_SAFE = (0, 255, 0)      # Green
        self.COLOR_CAUTION = (0, 255, 255) # Yellow
        self.COLOR_CRITICAL = (0, 0, 255)  # Red
        
    def draw_density_heatmap(self, frame: np.ndarray, density_map: np.ndarray,risk_levels: np.ndarray,alpha: float = 0.4
    ) -> np.ndarray:
        """Overlay density heatmap on frame."""
        overlay = frame.copy()
        rows, cols = density_map.shape
        
        frame_h, frame_w = frame.shape[:2]
        zone_h = frame_h // rows
        zone_w = frame_w // cols
        
        for i in range(rows):
            for j in range(cols):
                # Skip empty zones
                if density_map[i, j] == 0:
                    continue
                
                # Get color based on risk level
                risk = risk_levels[i, j]
                if risk == 0:
                    color = self.COLOR_SAFE
                elif risk == 1:
                    color = self.COLOR_CAUTION
                else:
                    color = self.COLOR_CRITICAL
                
                # Draw filled rectangle for zone
                x1 = j * zone_w
                y1 = i * zone_h
                x2 = x1 + zone_w
                y2 = y1 + zone_h
                
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                
                # Draw people count
                text = str(int(density_map[i, j]))
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                text_x = x1 + (zone_w - text_size[0]) // 2
                text_y = y1 + (zone_h + text_size[1]) // 2
                
                cv2.putText(overlay, text, (text_x, text_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay with original frame
        return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    
    def draw_stats_panel(self, frame: np.ndarray, stats: dict) -> np.ndarray:
        """Draw statistics panel on frame."""
        panel_height = 150 
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)

        y_offset = 20
        line_height = 25

        texts = [
            f"Total People: {stats['total_people']} | Max Density: {stats['max_density']:.2f} p/m²",
            f"Critical: {stats['critical_zones']} | Caution: {stats['caution_zones']} | Safe: {stats['safe_zones']}",
            f"Bottlenecks: {stats.get('bottlenecks', 0)} | Reverse Flow Zones: {stats.get('reverse_flow_zones', 0)}"
        ]

        for i, text in enumerate(texts):
            color = (255, 255, 255)

            # Highlight warnings
            if i == 2 and (stats.get('bottlenecks', 0) > 0 or stats.get('reverse_flow_zones', 0) > 0):
                color = (0, 165, 255)  # Orange for warnings
    
            cv2.putText(
                panel, text, (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
            )

        # Add alert indicator if critical
        if stats.get('bottlenecks', 0) > 0:
            cv2.putText(
                panel, "ALERT: BOTTLENECK DETECTED", 
                (10, y_offset + 3 * line_height + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
            )

        return np.vstack([panel, frame])