import os
import socket
import subprocess
import threading
import time
from typing import Dict, Optional, Union, List, Callable

import cv2
import numpy as np

from ..utils.logging import get_logger


class RTSPServer:
    """RTSP server for streaming video frames."""
    
    def __init__(
        self, 
        port: int = 8554, 
        stream_name: str = "stream",
        codec: str = "h264",
        fps: int = 30,
        width: Optional[int] = None,
        height: Optional[int] = None,
        config: Optional[Dict] = None
    ):
        """
        Initialize RTSP server.
        
        Args:
            port: RTSP server port
            stream_name: Name of the stream
            codec: Video codec (h264, h265)
            fps: Frames per second
            width: Frame width (if None, determined from first frame)
            height: Frame height (if None, determined from first frame)
            config: Additional configuration
        """
        self.logger = get_logger("RTSPServer")
        self.config = config or {}
        
        self.port = port
        self.stream_name = stream_name
        self.codec = codec
        self.fps = fps
        self.width = width
        self.height = height
        
        self.url = f"rtsp://localhost:{port}/{stream_name}"
        self.process = None
        self.running = False
        self.frame_count = 0
        
        # Check if ffmpeg is available
        try:
            subprocess.run(
                ["ffmpeg", "-version"], 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                check=True
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.error("ffmpeg not found. Please install ffmpeg.")
            raise RuntimeError("ffmpeg not found. Please install ffmpeg.")
    
    def start(self) -> bool:
        """
        Start the RTSP server using ffmpeg.
        
        Returns:
            bool: Success status
        """
        if self.running:
            return True
        
        try:
            # Find free port if specified port is in use
            if not self._is_port_available(self.port):
                for port in range(8554, 8654):
                    if self._is_port_available(port):
                        self.port = port
                        self.url = f"rtsp://localhost:{port}/{self.stream_name}"
                        break
                else:
                    self.logger.error("No available ports found for RTSP server")
                    return False
            
            # Prepare FFmpeg command
            cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pixel_format", "bgr24",
                "-video_size", f"{self.width or 640}x{self.height or 480}",
                "-framerate", str(self.fps),
                "-i", "pipe:0",
                "-c:v", self.codec,
                "-preset", "ultrafast",
                "-tune", "zerolatency",
                "-f", "rtsp",
                "-rtsp_transport", "tcp",
                f"rtsp://0.0.0.0:{self.port}/{self.stream_name}"
            ]
            
            # Start FFmpeg process
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=10**8
            )
            
            # Start stderr reader thread to avoid blocking
            threading.Thread(
                target=self._read_stderr,
                daemon=True
            ).start()
            
            self.running = True
            self.logger.info(f"RTSP server started at {self.url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start RTSP server: {e}", exc_info=True)
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the RTSP server."""
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
            
            self.process = None
        
        self.running = False
        self.logger.info("RTSP server stopped")
    
    def frame_callback(self, frame: np.ndarray, detections: Optional[List] = None) -> None:
        """
        Process a frame from the video source.
        
        Args:
            frame: BGR frame
            detections: Optional list of detections to visualize
        """
        if not self.running or self.process is None:
            return
        
        # Set frame dimensions if not already set
        if self.width is None or self.height is None:
            self.height, self.width = frame.shape[:2]
        
        # Resize frame if necessary
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Draw detections if available
        if detections:
            frame = frame.copy()  # Create a copy to avoid modifying the original
            for det in detections:
                try:
                    # Extract detection data
                    if hasattr(det, 'bbox'):
                        x1, y1, x2, y2 = map(int, det.bbox)
                        class_name = getattr(det, 'class_name', 'Object')
                        confidence = getattr(det, 'confidence', 1.0)
                    else:
                        # Assume dictionary format
                        x1, y1, x2, y2 = map(int, det.get('bbox', [0, 0, 0, 0]))
                        class_name = det.get('class_name', 'Object')
                        confidence = det.get('confidence', 1.0)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                except Exception as e:
                    self.logger.warning(f"Error drawing detection: {e}")
        
        try:
            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1
            
        except (BrokenPipeError, IOError) as e:
            self.logger.error(f"Error sending frame to ffmpeg: {e}")
            self.stop()
            # Try to restart
            self.start()
    
    def _read_stderr(self) -> None:
        """Read and log stderr output from FFmpeg."""
        while self.process and self.process.stderr:
            line = self.process.stderr.readline()
            if not line:
                break
                
            line = line.decode('utf-8', errors='replace').strip()
            if line and not line.startswith('frame='):  # Filter out progress messages
                self.logger.debug(f"FFmpeg: {line}")
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) != 0
    
    def __del__(self):
        """Cleanup on deletion."""
        self.stop()
