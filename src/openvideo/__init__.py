"""
OpenVideo - Comprehensive Python and C++/Rust repository for real-time and asynchronous
video processing from drone camera systems to web, VR clients, and computer vision pipelines.
"""

__version__ = "1.0.0"

# Import core modules for easier access
from .core import VideoProcessor
from .streaming import RTSPServer, WebRTCServer
from .vision import YOLODetector