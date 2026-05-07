import os
import cv2
import supervision as sv
import numpy as np
from src.detector import CrowdDetector
from src.tracker import CrowdTracker
from src.visualizer import CrowdVisualizer
from src.density_estimator import DensityEstimator
from src.density_visualizer import DensityVisualizer
from src.flow_analyzer import FlowAnalyzer
from src.flow_visualizer import FlowVisualizer

class VideoProcessor:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.4, use_slicer: bool = False):
        self.detector = CrowdDetector(model_path, conf_threshold, use_slicer=use_slicer, 
            slice_wh=(640, 640),
            overlap_ratio=(0.2, 0.2))
        self.tracker = CrowdTracker()
        self.visualizer = CrowdVisualizer()
        
        # Density components
        self.density_estimator = None
        self.density_visualizer = DensityVisualizer()
        
        # Flow components (NEW)
        self.flow_analyzer = FlowAnalyzer(history_length=30)
        self.flow_visualizer = FlowVisualizer()
        
    def process_frame(self, frame: np.ndarray, index: int) -> np.ndarray:
        """Process a single frame: detect -> track -> density -> flow -> visualize."""
        # Initialize density estimator on first frame
        if self.density_estimator is None:
            h, w = frame.shape[:2]
            self.density_estimator = DensityEstimator(w, h, grid_size=(8, 8))
        
        # 1. Detect people
        detections = self.detector.detect(frame)
        
        # 2. Update tracks
        detections = self.tracker.update(detections)
        
        # 3. Calculate density
        density_map = self.density_estimator.estimate_density(detections)
        density_sqm = self.density_estimator.calculate_density_per_sqm(density_map)
        risk_levels = self.density_estimator.get_risk_levels(density_sqm)
        stats = self.density_estimator.get_zone_stats(density_map)
        
        # 4. Analyze flow (NEW)
        movement_vectors = self.flow_analyzer.update(detections, index)
        
        zone_assignments = self.flow_analyzer.get_zone_assignments(
            detections,
            self.density_estimator.grid_rows,
            self.density_estimator.grid_cols,
            self.density_estimator.frame_width,
            self.density_estimator.frame_height
        )
        
        bottlenecks = self.flow_analyzer.detect_bottlenecks(
            movement_vectors,
            density_map,
            density_sqm,
            zone_assignments
        )
        
        reverse_flow = self.flow_analyzer.detect_reverse_flow(
            movement_vectors,
            zone_assignments
        )
        
        # 5. Visualize everything
        # Start with tracking visualization
        annotated = self.visualizer.annotate(frame, detections, self._create_labels(detections))
        
        # Overlay density heatmap
        annotated = self.density_visualizer.draw_density_heatmap(
            annotated, density_map, risk_levels, alpha=0.3
        )
        
        # Draw movement vectors lines (NEW)
        # annotated = self.flow_visualizer.draw_movement_vectors(
        #     annotated,
        #     movement_vectors,
        #     self.flow_analyzer.track_history
        # )
        
        # Highlight bottlenecks (NEW)
        if bottlenecks:
            annotated = self.flow_visualizer.highlight_bottlenecks(
                annotated,
                bottlenecks,
                self.density_estimator.grid_rows,
                self.density_estimator.grid_cols
            )
        
        # Highlight reverse flow
        if reverse_flow:
            annotated = self.flow_visualizer.highlight_reverse_flow(
                annotated,
                reverse_flow,
                self.density_estimator.grid_rows,
                self.density_estimator.grid_cols
            )
        
        # Update stats with flow info 
        stats['bottlenecks'] = len(bottlenecks)
        stats['reverse_flow_zones'] = len(reverse_flow)
        
        # Add stats panel
        annotated = self.density_visualizer.draw_stats_panel(annotated, stats)
        
        return annotated
    
    def _create_labels(self, detections: sv.Detections):
        """Create labels for detections."""
        if detections.tracker_id is not None:
            return [f"ID:{tid}" for tid in detections.tracker_id]
        return None
    

    def process_video(self, source_path: str, target_path: str):
        """Process entire video."""
        import os
        import cv2

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        print(f"Processing {source_path}...")

        # Open input video
        cap = cv2.VideoCapture(source_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Process first frame to get actual output dimensions
        ret, first_frame = cap.read()
        if not ret:
            print("Error: Could not read video")
            return

        processed_first = self.process_frame(first_frame, 0)
        output_height, output_width = processed_first.shape[:2]

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Setup output with ACTUAL dimensions
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        out = cv2.VideoWriter(target_path, fourcc, fps, (output_width, output_height))

        if not out.isOpened():
            print(f"Error: Could not open video writer for {target_path}")
            return

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed = self.process_frame(frame, frame_idx)
            out.write(processed)

            if frame_idx % 30 == 0:
                print(f"Progress: {frame_idx}/{total_frames} frames")

            frame_idx += 1

        cap.release()
        out.release()
        print(f"✓ Complete! Saved to {target_path}")