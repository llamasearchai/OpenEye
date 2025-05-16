"""
Core types for the OpenVideo system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import time


@dataclass
class VideoFrame:
    """Represents a video frame with associated metadata."""
    # Image data as numpy array (height, width, channels)
    data: np.ndarray
    
    # Frame timestamp in seconds (epoch time)
    timestamp: float = field(default_factory=lambda: time.time())
    
    # Frame sequence number
    sequence_number: Optional[int] = None
    
    # Metadata dictionary
    metadata: Dict = field(default_factory=dict)
    
    # Frame dimensions
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def channels(self) -> int:
        if len(self.data.shape) == 2:
            return 1
        return self.data.shape[2]
    
    def copy(self) -> 'VideoFrame':
        """Create a copy of the frame."""
        return VideoFrame(
            data=self.data.copy(),
            timestamp=self.timestamp,
            sequence_number=self.sequence_number,
            metadata=self.metadata.copy()
        )


@dataclass
class GeoLocation:
    """Geographic location data."""
    latitude: float
    longitude: float
    altitude: Optional[float] = None
    # Accuracy in meters
    accuracy: Optional[float] = None


@dataclass
class CameraParameters:
    """Camera intrinsic and extrinsic parameters."""
    # Intrinsic parameters
    focal_length: Union[float, Tuple[float, float]]
    principal_point: Tuple[float, float]
    distortion_coefficients: List[float] = field(default_factory=list)
    
    # Extrinsic parameters (camera position and orientation)
    position: Optional[Tuple[float, float, float]] = None
    rotation: Optional[Tuple[float, float, float]] = None


class FrameSource:
    """Base class for any component that produces frames."""
    
    def get_frame(self) -> Optional[VideoFrame]:
        """Get the next frame from the source."""
        raise NotImplementedError("Subclasses must implement get_frame()")
    
    def release(self) -> None:
        """Release any resources."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class FrameSink:
    """Base class for any component that consumes frames."""
    
    def process_frame(self, frame: VideoFrame) -> Optional[VideoFrame]:
        """Process a frame and optionally return a modified version."""
        raise NotImplementedError("Subclasses must implement process_frame()")
    
    def release(self) -> None:
        """Release any resources."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()