import supervision as sv
import numpy as np

class CrowdTracker:
    def __init__(self):
        # Updated to new API
        self.byte_track = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )
        
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Update tracks with new detections."""
        return self.byte_track.update_with_detections(detections)