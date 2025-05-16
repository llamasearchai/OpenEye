"""
MIPI camera interface for embedded systems.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .interface import Camera, CameraCapabilities, CameraControl, ControlType
from ..core.types import VideoFrame

logger = logging.getLogger(__name__)

# Import PyO3 Rust extension for MIPI low-level operations
try:
    from ..rust_extensions import mipi_ext
    RUST_EXTENSION_AVAILABLE = True
except ImportError:
    logger.warning("Rust extension for MIPI not available, using pure Python implementation")
    RUST_EXTENSION_AVAILABLE = False
    
    # Try to import libcamera for pure Python implementation
    try:
        import libcamera
        LIBCAMERA_AVAILABLE = True
    except ImportError:
        logger.warning("libcamera package not available for Python MIPI implementation")
        LIBCAMERA_AVAILABLE = False


class MIPICamera(Camera):
    """
    Implementation of Camera interface for MIPI CSI-2 devices.
    """
    
    def __init__(self, device_id: int = 0, width: int = 1280, height: int = 720,
                 fps: float = 30.0, format: str = "RGB"):
        """
        Initialize MIPI camera.
        
        Args:
            device_id: Camera device ID
            width: Desired frame width
            height: Desired frame height
            fps: Desired frames per second
            format: Pixel format
        """
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        
        self._camera = None
        self._running = False
        self._capabilities = None
        
        # Use Rust extension if available
        self.use_rust_ext = RUST_EXTENSION_AVAILABLE
        
        # Statistics
        self.frames_captured = 0
        self.start_time = 0
        
        # Try to open the camera immediately
        self.open()
        
    def open(self) -> bool:
        """
        Open the MIPI camera.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.use_rust_ext:
                self.camera_handle = mipi_ext.open_camera(
                    self.device_id, self.width, self.height, self.fps, self.format
                )
                return self.camera_handle is not None
                
            elif LIBCAMERA_AVAILABLE:
                # Initialize libcamera
                camera_manager = libcamera.CameraManager()
                camera_id = list(camera_manager.cameras.keys())[self.device_id]
                self._camera = camera_manager.get(camera_id)
                
                # Configure camera
                config = self._camera.generate_configuration([{
                    'format': {'width': self.width, 'height': self.height},
                    'buffer_count': 4
                }])
                
                # Set framerate
                stream_config = config.at(0)
                controls = self._camera.controls
                if 'FrameDurationLimits' in controls:
                    frame_time = int(1000000 / self.fps)  # Convert fps to duration in microseconds
                    stream_config['controls.FrameDurationLimits'] = (frame_time, frame_time)
                
                # Validate and apply configuration
                config.validate()
                self._camera.configure(config)
                
                # Query capabilities
                self._query_capabilities()
                
                return True
            else:
                logger.error("No MIPI camera implementation available")
                return False
                
        except Exception as e:
            logger.error(f"Error opening MIPI camera: {e}", exc_info=True)
            return False
            
    def start(self) -> bool:
        """
        Start video streaming.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self._running:
            return True
            
        try:
            if self.use_rust_ext:
                result = mipi_ext.start_streaming(self.camera_handle)
                if result:
                    self._running = True
                    self.start_time = time.time()
                return result
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                self._camera.start()
                self._running = True
                self.start_time = time.time()
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error starting MIPI camera: {e}", exc_info=True)
            return False
            
    def stop(self) -> None:
        """Stop video streaming."""
        if not self._running:
            return
            
        try:
            if self.use_rust_ext:
                mipi_ext.stop_streaming(self.camera_handle)
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                self._camera.stop()
                
            self._running = False
                
        except Exception as e:
            logger.error(f"Error stopping MIPI camera: {e}", exc_info=True)
            
    def release(self) -> None:
        """Release all resources."""
        try:
            if self._running:
                self.stop()
                
            if self.use_rust_ext and hasattr(self, 'camera_handle'):
                mipi_ext.close_camera(self.camera_handle)
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                self._camera.release()
                self._camera = None
                
        except Exception as e:
            logger.error(f"Error releasing MIPI camera: {e}", exc_info=True)
            
    def is_running(self) -> bool:
        """Check if the camera is running."""
        return self._running
            
    def get_frame(self) -> Optional[VideoFrame]:
        """
        Get the next frame from the camera.
        
        Returns:
            VideoFrame: Frame data and metadata, or None if no frame is available
        """
        if not self._running:
            if not self.start():
                return None
                
        try:
            if self.use_rust_ext:
                frame_data = mipi_ext.get_frame(self.camera_handle, timeout_ms=100)
                if frame_data is None:
                    return None
                    
                self.frames_captured += 1
                return VideoFrame(
                    data=frame_data['data'],
                    timestamp=frame_data['timestamp'],
                    sequence_number=self.frames_captured,
                    metadata=frame_data.get('metadata', {})
                )
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                request = self._camera.create_request()
                if not request:
                    return None
                    
                self._camera.queue_request(request)
                completed_request = self._camera.wait_for_request(timeout_ms=100)
                if not completed_request:
                    return None
                    
                buffer = completed_request.buffers[0]
                data = np.array(buffer.data, copy=True)
                
                self.frames_captured += 1
                return VideoFrame(
                    data=data.reshape((self.height, self.width, -1)),
                    timestamp=time.time(),
                    sequence_number=self.frames_captured,
                    metadata={"camera_id": self.device_id}
                )
                
            return None
                
        except Exception as e:
            logger.error(f"Error getting frame from MIPI camera: {e}", exc_info=True)
            return None
            
    def set_control(self, control_id: int, value: Any) -> bool:
        """Set a camera control parameter."""
        try:
            if self.use_rust_ext:
                return mipi_ext.set_control(self.camera_handle, control_id, value)
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                controls = {}
                controls[control_id] = value
                self._camera.set_controls(controls)
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error setting MIPI control {control_id}: {e}", exc_info=True)
            return False
            
    def get_control(self, control_id: int) -> Any:
        """Get a camera control parameter value."""
        try:
            if self.use_rust_ext:
                return mipi_ext.get_control(self.camera_handle, control_id)
                
            elif LIBCAMERA_AVAILABLE and self._camera:
                return self._camera.get_controls().get(control_id)
                
            return None
                
        except Exception as e:
            logger.error(f"Error getting MIPI control {control_id}: {e}", exc_info=True)
            return None
            
    def get_capabilities(self) -> CameraCapabilities:
        """Get the camera capabilities."""
        if self._capabilities is None:
            self._query_capabilities()
        return self._capabilities
            
    def _query_capabilities(self) -> None:
        """Query the camera capabilities."""
        # Basic capabilities for demonstration
        caps = CameraCapabilities(
            resolutions=[(1920, 1080), (1280, 720), (640, 480)],
            formats=["RGB", "YUV420"],
            framerates={(1920, 1080): [30.0, 24.0], (1280, 720): [60.0, 30.0], (640, 480): [90.0, 60.0, 30.0]},
            controls=[]
        )
        self._capabilities = caps