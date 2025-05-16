"""
Base encoder interface and factory.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Any

from ..core.types import FrameSink, VideoFrame

logger = logging.getLogger(__name__)


class Encoder(FrameSink, ABC):
    """Base class for video encoders."""
    
    @abstractmethod
    def set_bitrate(self, bitrate: int) -> bool:
        """
        Set the encoder bitrate.
        
        Args:
            bitrate: Target bitrate in bits per second
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_keyframe_interval(self, interval: int) -> bool:
        """
        Set the keyframe (I-frame) interval.
        
        Args:
            interval: Number of frames between keyframes
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def set_quality(self, quality: int) -> bool:
        """
        Set the encoding quality (0-100).
        
        Args:
            quality: Quality level, higher is better
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_encoded_data(self) -> Optional[bytes]:
        """
        Get the most recently encoded data.
        
        Returns:
            bytes: Encoded video data, or None if no data is available
        """
        pass
    
    @abstractmethod
    def encode(self, frame: VideoFrame) -> bool:
        """
        Encode a video frame.
        
        Args:
            frame: Video frame to encode
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    def process_frame(self, frame: VideoFrame) -> VideoFrame:
        """
        Process a frame by encoding it and passing through.
        
        Args:
            frame: Video frame to encode
            
        Returns:
            VideoFrame: The original frame (for pipeline chaining)
        """
        self.encode(frame)
        return frame


def create_encoder(config: Dict[str, Any]) -> Optional[Encoder]:
    """
    Factory function to create an encoder based on configuration.
    
    Args:
        config: Encoder configuration dictionary
        
    Returns:
        Encoder: Configured encoder instance
    """
    encoder_type = config.get("encoder_type", "h264")
    
    try:
        if encoder_type == "h264":
            from .h264 import H264Encoder
            return H264Encoder(
                width=config.get("width", 1280),
                height=config.get("height", 720),
                bitrate=config.get("bitrate", 2000000),
                fps=config.get("fps", 30),
                keyframe_interval=config.get("keyframe_interval", 30),
                use_gpu=config.get("use_gpu", True),
                gpu_device=config.get("gpu_device", 0),
                profile=config.get("profile", "main"),
                preset=config.get("preset", "medium")
            )
        elif encoder_type == "h265":
            from .h265 import H265Encoder
            return H265Encoder(
                width=config.get("width", 1280),
                height=config.get("height", 720),
                bitrate=config.get("bitrate", 1500000),
                fps=config.get("fps", 30),
                keyframe_interval=config.get("keyframe_interval", 30),
                use_gpu=config.get("use_gpu", True),
                gpu_device=config.get("gpu_device", 0),
                profile=config.get("profile", "main"),
                preset=config.get("preset", "medium")
            )
        else:
            logger.error(f"Unknown encoder type: {encoder_type}")
            return None
    except Exception as e:
        logger.error(f"Error creating encoder: {e}", exc_info=True)
        return None        Get the most recently encoded data.