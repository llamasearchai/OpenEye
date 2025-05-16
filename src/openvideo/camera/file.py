"""
File-based video source.
"""

import logging
import time
from typing import Optional
import cv2
import numpy as np

from ..core.types import FrameSource, VideoFrame

logger = logging.getLogger(__name__)


class FileVideoSource(FrameSource):
    """
    Video source that reads frames from a video file.
    """
    
    def __init__(self, file_path: str, loop: bool = False, frame_rate: Optional[float] = None):
        """
        Initialize a file video source.
        
        Args:
            file_path: Path to video file
            loop: Whether to loop the video when it ends
            frame_rate: Override the video's frame rate (frames per second)
        """
        self.file_path = file_path
        self.loop = loop
        self.frame_rate = frame_rate
        self.cap = None
        self.fps = 0
        self.frame_time = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.open()
        
    def open(self) -> bool:
        """
        Open the video file.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cap = cv2.VideoCapture(self.file_path)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file: {self.file_path}")
                return False
                
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.frame_rate:
                self.fps = self.frame_rate
                
            self.frame_time = 1.0 / self.fps if self.fps > 0 else 0.033
            self.last_frame_time = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Error opening video file: {e}", exc_info=True)
            return False
            
    def get_frame(self) -> Optional[VideoFrame]:
        """
        Get the next frame from the video.
        
        Returns:
            VideoFrame: Frame data and metadata, or None if no frame is available
        """
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None
                
        # Control frame rate
        now = time.time()
        elapsed = now - self.last_frame_time
        if elapsed < self.frame_time:
            # Not time for the next frame yet
            time.sleep(max(0, self.frame_time - elapsed))
            
        # Read frame
        ret, frame = self.cap.read()
        
        if not ret:
            # End of video
            if self.loop:
                # Reopen the video and try again
                self.cap.release()
                self.cap = None
                if self.open():
                    return self.get_frame()
            return None
            
        self.last_frame_time = time.time()
        self.frame_count += 1
        
        # Get current position in the video
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        pos_frames = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        
        return VideoFrame(
            data=frame,
            timestamp=self.last_frame_time,
            sequence_number=self.frame_count,
            metadata={
                "source": "file",
                "path": self.file_path,
                "position_ms": pos_msec,
                "position_frames": pos_frames
            }
        )
        
    def release(self) -> None:
        """Release video file resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None