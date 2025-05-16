"""
Network stream video source.
"""

import logging
import time
from typing import Optional, Dict, Any
import cv2
import numpy as np

from ..core.types import FrameSource, VideoFrame

logger = logging.getLogger(__name__)


class NetworkStreamSource(FrameSource):
    """
    Video source that reads frames from a network stream (RTSP, HTTP, etc.).
    """
    
    def __init__(self, url: str, reconnect_attempts: int = 3, 
                 reconnect_delay: float = 5.0, buffer_size: int = 10,
                 options: Optional[Dict[str, Any]] = None):
        """
        Initialize a network stream source.
        
        Args:
            url: URL of the network stream
            reconnect_attempts: Number of reconnection attempts
            reconnect_delay: Delay between reconnection attempts in seconds
            buffer_size: OpenCV buffer size
            options: Additional options for OpenCV VideoCapture
        """
        self.url = url
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay = reconnect_delay
        self.buffer_size = buffer_size
        self.options = options or {}
        
        self.cap = None
        self.reconnect_count = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.connected = False
        
            # Try to connect immediately
            self.open()
        
        def open(self) -> bool:
            """
            Open the network stream.
        
            Returns:
                bool: True if successful, False otherwise
            """
            try:
                # Apply options to OpenCV
                if self.cap is not None:
                    self.cap.release()
                
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            
                # Set buffer size and other options
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            
                for key, value in self.options.items():
                    self.cap.set(key, value)
                
                if not self.cap.isOpened():
                    logger.error(f"Failed to open network stream: {self.url}")
                    return False
                
                self.connected = True
                self.reconnect_count = 0
                logger.info(f"Successfully connected to stream: {self.url}")
            
                return True
            
            except Exception as e:
                logger.error(f"Error opening network stream {self.url}: {e}", exc_info=True)
                self.connected = False
                return False
            
        def get_frame(self) -> Optional[VideoFrame]:
            """
            Get the next frame from the network stream.
        
            Returns:
                VideoFrame: Frame data and metadata, or None if no frame is available
            """
            if self.cap is None or not self.cap.isOpened():
                if not self._try_reconnect():
                    return None
                
            # Read frame
            ret, frame = self.cap.read()
        
            if not ret:
                logger.warning(f"Failed to read frame from {self.url}")
            
                # Try to reconnect
                if not self._try_reconnect():
                    return None
                
                # Try again after reconnection
                ret, frame = self.cap.read()
                if not ret:
                    return None
                
            self.last_frame_time = time.time()
            self.frame_count += 1
        
            return VideoFrame(
                data=frame,
                timestamp=self.last_frame_time,
                sequence_number=self.frame_count,
                metadata={
                    "source": "network",
                    "url": self.url
                }
            )
        
        def _try_reconnect(self) -> bool:
            """
            Try to reconnect to the stream.
        
            Returns:
                bool: True if reconnection successful, False otherwise
            """
            if self.reconnect_count >= self.reconnect_attempts:
                logger.error(f"Max reconnection attempts ({self.reconnect_attempts}) reached for {self.url}")
                return False
            
            logger.info(f"Attempting to reconnect to {self.url} (attempt {self.reconnect_count + 1}/{self.reconnect_attempts})")
            self.reconnect_count += 1
        
            # Wait before reconnecting
            time.sleep(self.reconnect_delay)
        
            # Try to open the stream again
            return self.open()
        
        def release(self) -> None:
            """Release network stream resources."""
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.connected = False        self