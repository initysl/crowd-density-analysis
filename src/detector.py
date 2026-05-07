from ultralytics import YOLO
import supervision as sv
from typing import Tuple
import numpy as np

class CrowdDetector:
    def __init__(
        self, 
        model_path: str = "yolov8n.pt", 
        conf_threshold: float = 0.4,
        use_slicer: bool = False,
        slice_wh: Tuple[int, int] = (640, 640),
        overlap_ratio: Tuple[float, float] = (0.2, 0.2)
    ):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_slicer = use_slicer
        
        # Setup slicer if enabled
        if use_slicer:
            def slice_callback(image_slice: np.ndarray) -> sv.Detections:
                results = self.model(image_slice)[0]
                detections = sv.Detections.from_ultralytics(results)
                return detections
            
            self.slicer = sv.InferenceSlicer(
                callback=slice_callback,
                slice_wh=slice_wh,
                overlap_wh=overlap_ratio # type: ignore
            )
        
    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Detect people in a single frame."""
        if self.use_slicer:
            # Use slicer for small object detection
            detections = self.slicer(frame)
        else:
            # Standard detection
            results = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
        
        # Filter for persons only
        detections = detections[detections.class_id == 0]
        detections = detections[detections.confidence > self.conf_threshold] # type: ignore
        
        return detections # type: ignore