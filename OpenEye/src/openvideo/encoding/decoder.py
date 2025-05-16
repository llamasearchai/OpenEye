"""
Base decoder interface and factory.
"""

from abc import ABC, abstractmethod
import logging
from typing import Dict, Optional, Any

from ..core.types import FrameSink, FrameSource, VideoFrame

logger = logging.getLogger(__name__)


class Decoder(FrameSource, ABC):
    """Base class for video decoders."""
    
    @abstractmethod
    def decode(self, data: bytes) -> Optional[VideoFrame]:
        """
        Decode video data into a frame.
        
        Args:
            data: Encoded video data
            
        Returns:
            VideoFrame: Decoded video frame, or None if decoding failed
        """
        pass
    
    @abstractmethod
    def get_frame(self) -> Optional[VideoFrame]:
        """
        Get the next decoded frame if available.
        
        Returns:
            VideoFrame: Decoded video frame, or None if no frame is available
        """
        pass


def create_decoder(config: Dict[str, Any]) -> Optional[Decoder]:
    """
    Factory function to create a decoder based on configuration.
    
    Args:
        config: Decoder configuration dictionary
        
    Returns:
        Decoder: Configured decoder instance
    """
    decoder_type = config.get("decoder_type", "h264")
    
    try:
        if decoder_type == "h264":
            from .h264 import H264Decoder
            return H264Decoder(
                width=config.get("width", 1280),
                height=config.get("height", 720),
                use_gpu=config.get("use_gpu", True),
                gpu_device=config.get("gpu_device", 0),
                output_format=config.get("output_format", "rgb24")
            )
        elif decoder_type == "h265":
            from .h265 import H265Decoder
            return H265Decoder(
                width=config.get("width", 1280),
                height=config.get("height", 720),
                use_gpu=config.get("use_gpu", True),
                gpu_device=config.get("gpu_device", 0),
                output_format=config.get("output_format", "rgb24")
            )
        else:
            logger.error(f"Unknown decoder type: {decoder_type}")
            return None
    except Exception as e:
        logger.error(f"Error creating decoder: {e}", exc_info=True)
        return None