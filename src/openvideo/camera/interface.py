"""
Abstract camera interface definitions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.types import FrameSource, VideoFrame


class ControlType(Enum):
    """Types of camera controls."""
    INTEGER = auto()
    BOOLEAN = auto()
    MENU = auto()
    FLOAT = auto()
    STRING = auto()


@dataclass
class CameraControl:
    """Represents a camera control parameter."""
    id: int
    name: str
    type: ControlType
    minimum: Optional[Any] = None
    maximum: Optional[Any] = None
    step: Optional[Any] = None
    default: Optional[Any] = None
    value: Optional[Any] = None
    options: Optional[List[str]] = None  # For MENU type


@dataclass
class CameraCapabilities:
    """Camera capabilities and supported features."""
    # Supported resolutions as (width, height) tuples
    resolutions: List[Tuple[int, int]]
    
    # Supported pixel formats (e.g., "MJPG", "YUYV", "RGB")
    formats: List[str]
    
    # Supported framerates for each resolution
    framerates: Dict[Tuple[int, int], List[float]]
    
    # Available controls
    controls: List[CameraControl]


class Camera(FrameSource, ABC):
    """Abstract base class for camera interfaces."""
    
    @abstractmethod
    def get_capabilities(self) -> CameraCapabilities:
        """Get the camera capabilities."""
        pass
    
    @abstractmethod
    def set_control(self, control_id: int, value: Any) -> bool:
        """
        Set a camera control parameter.
        
        Args:
            control_id: Control identifier
            value: New control value
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_control(self, control_id: int) -> Any:
        """
        Get a camera control parameter value.
        
        Args:
            control_id: Control identifier
            
        Returns:
            Any: Current control value
        """
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """
        Start the camera capturing.
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop the camera capturing."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """
        Check if the camera is currently running.
        
        Returns:
            bool: True if the camera is running, False otherwise
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[VideoFrame]:
        """
        Get the next frame from the camera.
        
        Returns:
            VideoFrame: The next frame, or None if no frame is available
        """
        pass
        
    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass
    
    def set_resolution(self, width: int, height: int) -> bool:
        """
        Set the camera resolution.
        
        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Default implementation using controls
        return False
        
    def set_framerate(self, fps: float) -> bool:
        """
        Set the camera framerate.
        
        Args:
            fps: Frames per second
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Default implementation using controls
        return False
        
    def set_format(self, format: str) -> bool:
        """
        Set the camera pixel format.
        
        Args:
            format: Pixel format string
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Default implementation using controls
        return False