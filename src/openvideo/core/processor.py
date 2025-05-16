import logging
import threading
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from contextlib import contextmanager
from dataclasses import dataclass

from ..utils.logging import get_logger
from ..vision.detector import BaseDetector

@dataclass
class VideoFrame:
    data: np.ndarray
    timestamp: float
    metadata: dict

class VideoProcessor:
    """
    Core video processing class that handles camera/video input, 
    processing pipelines, and output streaming.
    """
    
    def __init__(self, 
                 detector: Optional[BaseDetector] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the video processor.
        
        Args:
            detector: Optional object detector
            config: Configuration dictionary
        """
        self.logger = get_logger("VideoProcessor")
        self.config = config or {}
        
        # Video source
        self.video_source = None
        self.video_path = None
        self.camera_id = None
        self.is_camera = False
        
        # Video properties
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frame_count = 0
        self.duration = 0
        self.codec = ""
        
        # Processing
        self.detector = detector
        self.processing_thread = None
        self.is_running = False
        self.pause = False
        
        # Output handlers
        self.outputs = []
        
        # Frame buffer
        self.current_frame = None
        self.current_frame_idx = 0
        self.current_frame_time = 0
        self.frame_lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            "fps": 0,
            "processing_time": 0,
            "detection_count": 0,
            "dropped_frames": 0
        }
        self.metrics_lock = threading.Lock()
        
        self.logger.info("VideoProcessor initialized")
    
    def __enter__(self) -> "VideoProcessor":
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
    
    def open(self, source: Union[str, int]) -> bool:
        """
        Open a video source (file or camera).
        
        Args:
            source: Path to video file or camera ID
            
        Returns:
            bool: Success status
        """
        try:
            # Close any existing source
            self.close()
            
            # Check if source is a camera or file
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                self.is_camera = True
                self.camera_id = int(source)
                self.video_source = cv2.VideoCapture(self.camera_id)
                self.logger.info(f"Opened camera {self.camera_id}")
            else:
                self.is_camera = False
                self.video_path = source
                self.video_source = cv2.VideoCapture(self.video_path)
                self.logger.info(f"Opened video file: {self.video_path}")
            
            # Check if source is opened successfully
            if not self.video_source.isOpened():
                self.logger.error(f"Failed to open video source: {source}")
                return False
            
            # Get video properties
            self.width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.video_source.get(cv2.CAP_PROP_FPS)
            
            if not self.is_camera:
                self.frame_count = int(self.video_source.get(cv2.CAP_PROP_FRAME_COUNT))
                self.duration = self.frame_count / self.fps if self.fps > 0 else 0
                
                # Get codec information
                fourcc = int(self.video_source.get(cv2.CAP_PROP_FOURCC))
                self.codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            else:
                # Set camera properties if specified in config
                if "camera" in self.config:
                    camera_config = self.config["camera"]
                    if "width" in camera_config and "height" in camera_config:
                        self.video_source.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
                        self.video_source.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
                        
                    if "fps" in camera_config:
                        self.video_source.set(cv2.CAP_PROP_FPS, camera_config["fps"])
                        
                    # Refresh properties after setting
                    self.width = int(self.video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
                    self.height = int(self.video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.fps = self.video_source.get(cv2.CAP_PROP_FPS)
                
                self.codec = "MJPG"  # Default for camera
            
            self.logger.info(f"Video properties: {self.width}x{self.height}, {self.fps} fps, codec: {self.codec}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error opening video source: {e}", exc_info=True)
            return False
    
    def close(self) -> None:
        """Close the video source."""
        self.stop()
        
        if self.video_source is not None:
            self.video_source.release()
            self.video_source = None
            self.logger.info("Video source closed")
    
    def start(self) -> bool:
        """
        Start video processing in a separate thread.
        
        Returns:
            bool: Success status
        """
        if self.is_running:
            self.logger.warning("Video processing already running")
            return False
            
        if self.video_source is None or not self.video_source.isOpened():
            self.logger.error("Video source not opened")
            return False
            
        self.is_running = True
        self.pause = False
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Video processing started")
        return True
    
    def stop(self) -> None:
        """Stop video processing."""
        if not self.is_running:
            return
            
        self.is_running = False
        
        if self.processing_thread is not None:
            self.processing_thread.join(timeout=1.0)
            self.processing_thread = None
            
        self.logger.info("Video processing stopped")
    
    def pause_processing(self, pause: bool = True) -> None:
        """Pause or resume processing."""
        self.pause = pause
        self.logger.info(f"Video processing {'paused' if pause else 'resumed'}")
    
    def seek(self, frame_idx: int) -> bool:
        """
        Seek to a specific frame (for video files only).
        
        Args:
            frame_idx: Frame index to seek to
            
        Returns:
            bool: Success status
        """
        if self.is_camera or self.video_source is None:
            return False
            
        with self.frame_lock:
            success = self.video_source.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            if success:
                self.current_frame_idx = frame_idx
                self.logger.debug(f"Seeked to frame {frame_idx}")
            else:
                self.logger.error(f"Failed to seek to frame {frame_idx}")
            return success
    
    def get_frame(self, frame_idx: Optional[int] = None) -> Optional[np.ndarray]:
        """
        Get a specific frame or the current frame.
        
        Args:
            frame_idx: Optional frame index to retrieve
            
        Returns:
            np.ndarray: Frame data or None if not available
        """
        if self.video_source is None:
            return None
            
        if frame_idx is not None and not self.is_camera:
            # Save current position
            current_pos = int(self.video_source.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Seek to requested frame
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.video_source.read()
            
            # Restore position
            self.video_source.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            
            if ret:
                return frame
            else:
                return None
        else:
            # Return current frame
            with self.frame_lock:
                if self.current_frame is not None:
                    return self.current_frame.copy()
            return None
    
    def get_metadata(self) -> Dict:
        """
        Get video metadata.
        
        Returns:
            Dict: Video properties
        """
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "codec": self.codec,
            "is_camera": self.is_camera,
        }
    
    def get_metrics(self) -> Dict:
        """
        Get processing metrics.
        
        Returns:
            Dict: Current metrics
        """
        with self.metrics_lock:
            return self.metrics.copy()
    
    def add_output(self, output_handler) -> None:
        """
        Add an output handler (e.g., RTSPServer).
        
        Args:
            output_handler: Handler object with frame_callback method
        """
        if hasattr(output_handler, 'frame_callback'):
            self.outputs.append(output_handler)
            self.logger.info(f"Added output handler: {output_handler.__class__.__name__}")
        else:
            self.logger.error(f"Invalid output handler: {output_handler}")
    
    def remove_output(self, output_handler) -> None:
        """
        Remove an output handler.
        
        Args:
            output_handler: Handler to remove
        """
        if output_handler in self.outputs:
            self.outputs.remove(output_handler)
            self.logger.info(f"Removed output handler: {output_handler.__class__.__name__}")
    
    def _processing_loop(self) -> None:
        """Main processing loop running in a separate thread."""
        last_frame_time = time.time()
        frame_count = 0
        fps_update_time = last_frame_time
        
        try:
            while self.is_running:
                if self.pause:
                    time.sleep(0.01)
                    continue
                
                # Read frame
                start_time = time.time()
                ret, frame = self.video_source.read()
                
                if not ret:
                    if self.is_camera:
                        self.logger.error("Failed to read frame from camera")
                        time.sleep(0.1)
                        continue
                    else:
                        self.logger.info("End of video reached")
                        break
                
                # Store current frame with lock
                with self.frame_lock:
                    self.current_frame = frame
                    self.current_frame_idx = int(self.video_source.get(cv2.CAP_PROP_POS_FRAMES))
                    self.current_frame_time = start_time
                
                # Process frame if detector is available
                detections = []
                if self.detector is not None:
                    try:
                        detections = self.detector.detect(frame)
                    except Exception as e:
                        self.logger.error(f"Error in detector: {e}", exc_info=True)
                
                # Calculate processing time
                processing_time = (time.time() - start_time) * 1000  # ms
                
                # Update metrics
                with self.metrics_lock:
                    self.metrics["processing_time"] = processing_time
                    self.metrics["detection_count"] = len(detections)
                
                # Send frame to outputs
                for output in self.outputs:
                    try:
                        output.frame_callback(frame, detections)
                    except Exception as e:
                        self.logger.error(f"Error in output handler: {e}", exc_info=True)
                
                # Calculate FPS
                frame_count += 1
                current_time = time.time()
                time_diff = current_time - fps_update_time
                
                if time_diff >= 1.0:  # Update FPS every second
                    with self.metrics_lock:
                        self.metrics["fps"] = frame_count / time_diff
                    
                    fps_update_time = current_time
                    frame_count = 0
                
                # Sleep to maintain target FPS for file playback
                if not self.is_camera and self.fps > 0:
                    elapsed = time.time() - last_frame_time
                    frame_time = 1.0 / self.fps
                    
                    if elapsed < frame_time:
                        time.sleep(frame_time - elapsed)
                
                last_frame_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Error in processing loop: {e}", exc_info=True)
        finally:
            self.is_running = False
            self.logger.info("Processing loop ended")

    def process_frame(self, frame: VideoFrame) -> VideoFrame:
        """Process a single video frame with optional detection.
        
        Args:
            frame: Input video frame with metadata
            
        Returns:
            Processed frame with updated metadata
            
        Raises:
            ProcessingError: If frame processing fails
        """
        if self.detector:
            frame.metadata["detections"] = self.detector.detect(frame.data)
        return frame