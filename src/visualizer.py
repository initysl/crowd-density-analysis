import supervision as sv
import numpy as np
from typing import List, Optional

class CrowdVisualizer:
    def __init__(self):
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
        self.trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)
        
    def annotate(
        self, 
        frame: np.ndarray, 
        detections: sv.Detections,
        labels: Optional[List[str]] = None
    ) -> np.ndarray:
        """Annotate frame with detections and tracks."""
        annotated = self.box_annotator.annotate(frame.copy(), detections=detections)
        
        if labels:
            annotated = self.label_annotator.annotate(annotated, detections=detections, labels=labels) # type: ignore
        
        # Add movement trails if tracker_id exists
        if detections.tracker_id is not None:
            annotated = self.trace_annotator.annotate(annotated, detections=detections) # type: ignore
            
        return annotated # type: ignore