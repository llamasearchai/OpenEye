"""
Camera interfaces for OpenVideo.
"""

from typing import Dict, Optional

from .interface import Camera, CameraCapabilities
from .v4l2 import V4L2Camera
from .mipi import MIPICamera
from .file import FileVideoSource
from .network import NetworkStreamSource


def create_camera(config: Dict) -> Camera:
    """
    Factory function to create a camera based on configuration.
    
    Args:
        config: Camera configuration dictionary
        
    Returns:
        Camera: Configured camera instance
    """
    camera_type = config.get("camera_type", "v4l2")
    
    if camera_type == "v4l2":
        return V4L2Camera(
            device_path=config.get("device_path", "/dev/video0"),
            width=config.get("width", 1280),
            height=config.get("height", 720),
            fps=config.get("fps", 30),
            format=config.get("format", "MJPG")
        )
    elif camera_type == "mipi":
        return MIPICamera(
            device_id=config.get("device_id", 0),
            width=config.get("width", 1280),
            height=config.get("height", 720),
            fps=config.get("fps", 30)
        )
    elif camera_type == "file":
        return FileVideoSource(config.get("path"))
    elif camera_type == "network":
        return NetworkStreamSource(config.get("url"))
    else:
        raise ValueError(f"Unknown camera type: {camera_type}")


__all__ = [
    'Camera', 'CameraCapabilities', 'V4L2Camera', 'MIPICamera',
    'FileVideoSource', 'NetworkStreamSource', 'create_camera'
]