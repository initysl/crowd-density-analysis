from src.video_processor import VideoProcessor
import os

if __name__ == "__main__":
    os.makedirs("outputs/flow", exist_ok=True)
    
    processor = VideoProcessor(
        conf_threshold=0.4,
        use_slicer=True 
    )
    
    processor.process_video(
        source_path="data/raw_videos/smallo.mp4",
        target_path="outputs/flow/smallo_flow_analysis.mp4"
    )


# Project 2 - Facial recognition in video
# FutureWarning: The `ByteTrack` was deprecated since v0.28.0. It will be removed in v0.30.0.