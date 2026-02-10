import cv2
import numpy as np
from typing import List, Tuple, Optional
from video_reader import VideoReader


class FrameExtractor:
    
    
    def __init__(self, video_reader: VideoReader):
        """
        Initialize frame extractor
        
        Args:
            video_reader: VideoReader instance
        """
        self.video_reader = video_reader
    
    def extract_uniform(self, num_frames: int) -> List[Tuple[int, np.ndarray]]:
        """
        Extract uniformly distributed frames
        
        Args:
            num_frames: Number of frames to extract
        
        Returns:
            List of (frame_number, frame) tuples
        """
        total_frames = self.video_reader.frame_count
        
        if num_frames >= total_frames:
            # Extract all frames
            indices = list(range(total_frames))
        else:
            # Uniformly sample
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            frame = self.video_reader.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    
    def extract_by_interval(self, interval: int) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames at regular intervals
        
        Args:
            interval: Frame interval (e.g., every 30 frames)
        
        Returns:
            List of (frame_number, frame) tuples
        """
        frames = []
        for frame_num, frame in self.video_reader.read_frames(step=interval):
            frames.append((frame_num, frame))
        
        return frames
    
    def extract_by_fps(self, target_fps: float) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames to achieve target FPS
        
        Args:
            target_fps: Desired frames per second (e.g., 1.0 = 1 frame/sec)
        
        Returns:
            List of (frame_number, frame) tuples
        """
        video_fps = self.video_reader.fps
        
        if target_fps >= video_fps:
            # Extract all frames
            interval = 1
        else:
            # Calculate interval
            interval = int(video_fps / target_fps)
        
        return self.extract_by_interval(interval)
    
    def extract_by_time(self, time_interval: float) -> List[Tuple[int, np.ndarray]]:
        """
        Extract frames at specific time intervals
        
        Args:
            time_interval: Time interval in seconds (e.g., 0.5 = every 0.5 sec)
        
        Returns:
            List of (frame_number, frame) tuples
        """
        fps = self.video_reader.fps
        frame_interval = int(time_interval * fps)
        
        return self.extract_by_interval(max(1, frame_interval))
    
    def extract_keyframes(self, threshold: float = 30.0) -> List[Tuple[int, np.ndarray]]:
        """
        Extract keyframes based on scene change detection
        
        Args:
            threshold: Scene change threshold (higher = fewer keyframes)
        
        Returns:
            List of (frame_number, frame) tuples
        """
        frames = []
        prev_frame = None
        
        for frame_num, frame in self.video_reader.read_frames():
            if prev_frame is None:
                # Always include first frame
                frames.append((frame_num, frame))
                prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                continue
            
            # Calculate frame difference
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(prev_frame, curr_gray)
            mean_diff = np.mean(diff)
            
            # If significant change, consider it a keyframe
            if mean_diff > threshold:
                frames.append((frame_num, frame))
                prev_frame = curr_gray
        
        return frames
    
    def extract_custom(self, frame_indices: List[int]) -> List[Tuple[int, np.ndarray]]:
        """
        Extract specific frames by index
        
        Args:
            frame_indices: List of frame indices to extract
        
        Returns:
            List of (frame_number, frame) tuples
        """
        frames = []
        for idx in frame_indices:
            frame = self.video_reader.read_frame(idx)
            if frame is not None:
                frames.append((idx, frame))
        
        return frames
    # Add this method to FrameExtractor class

def extract_adaptive(self, 
                    target_frames: int = 10,
                    min_interval: int = 5) -> List[Tuple[int, np.ndarray]]:
    """
    Adaptively extract frames based on video duration
    Good for datasets with varying video lengths
    
    Args:
        target_frames: Desired number of frames
        min_interval: Minimum frames between extractions
    
    Returns:
        List of (frame_number, frame) tuples
    """
    total_frames = self.video_reader.frame_count
    
    if total_frames <= target_frames * min_interval:
        # Short video - extract uniformly
        return self.extract_uniform(target_frames)
    else:
        # Long video - use interval
        interval = max(min_interval, total_frames // target_frames)
        return self.extract_by_interval(interval)

# ============ USAGE EXAMPLE ============
if __name__ == "__main__":
    with VideoReader('Hollow Knight Silksong 2025-12-27 23-14-16.mp4') as vr:
        extractor = FrameExtractor(vr)
        
        # Extract 10 uniform frames
        frames = extractor.extract_uniform(10)
        print(f"Extracted {len(frames)} uniform frames")
        
        # Extract 1 frame per second
        frames = extractor.extract_by_fps(1.0)
        print(f"Extracted {len(frames)} frames at 1 FPS")
        
        # Extract every 30 frames
        frames = extractor.extract_by_interval(30)
        print(f"Extracted {len(frames)} frames (every 30th)")
        
        # Extract keyframes
        frames = extractor.extract_keyframes(threshold=30.0)
        print(f"Extracted {len(frames)} keyframes")