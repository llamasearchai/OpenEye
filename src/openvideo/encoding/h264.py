"""
H.264 encoding and decoding implementation.
"""

import logging
import time
from typing import Dict, Optional, Any, List, Tuple
import threading
import numpy as np
import ctypes

from ..core.types import VideoFrame
from .encoder import Encoder
from .decoder import Decoder

logger = logging.getLogger(__name__)

# Try to import hardware acceleration libraries
try:
    import av  # PyAV for FFmpeg integration
    PYAV_AVAILABLE = True
except ImportError:
    logger.warning("PyAV not available, some encoding features will be limited")
    PYAV_AVAILABLE = False

try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    CUDA_AVAILABLE = True
except ImportError:
    logger.warning("PyCUDA not available, GPU acceleration disabled")
    CUDA_AVAILABLE = False

# Try to import Rust extension for optimized encoding
try:
    from ..rust_extensions import h264_ext
    RUST_EXTENSION_AVAILABLE = True
except ImportError:
    logger.warning("Rust extension for H264 not available")
    RUST_EXTENSION_AVAILABLE = False


class H264Encoder(Encoder):
    """
    H.264 (AVC) video encoder implementation.
    """
    
    def __init__(self, width: int = 1280, height: int = 720, bitrate: int = 2000000,
                 fps: float = 30.0, keyframe_interval: int = 30, 
                 use_gpu: bool = True, gpu_device: int = 0,
                 profile: str = "main", preset: str = "medium"):
        """
        Initialize H.264 encoder.
        
        Args:
            width: Video width in pixels
            height: Video height in pixels
            bitrate: Target bitrate in bits per second
            fps: Frames per second
            keyframe_interval: Number of frames between keyframes (GOP size)
            use_gpu: Whether to use GPU acceleration if available
            gpu_device: GPU device index to use
            profile: H.264 profile (baseline, main, high)
            preset: Encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
        """
        self.width = width
        self.height = height
        self.bitrate = bitrate
        self.fps = fps
        self.keyframe_interval = keyframe_interval
        self.use_gpu = use_gpu and (CUDA_AVAILABLE or RUST_EXTENSION_AVAILABLE)
        self.gpu_device = gpu_device
        self.profile = profile
        self.preset = preset
        
        self.encoder = None
        self.encoding_thread = None
        self.frame_queue = []
        self.output_queue = []
        self.lock = threading.Lock()
        self.running = False
        self.frame_count = 0
        
        # Initialize the encoder
        self._initialize()
        
    def _initialize(self) -> bool:
        """
        Initialize the encoder.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if RUST_EXTENSION_AVAILABLE and self.use_gpu:
            try:
                self.encoder = h264_ext.create_encoder(
                    width=self.width,
                    height=self.height,
                    bitrate=self.bitrate,
                    fps=self.fps,
                    keyframe_interval=self.keyframe_interval,
                    gpu_device=self.gpu_device,
                    profile=self.profile,
                    preset=self.preset
                )
                self.running = True
                return True
            except Exception as e:
                logger.error(f"Failed to initialize Rust H264 encoder: {e}", exc_info=True)
                self.running = False
                
        # Fallback to PyAV if available
        if PYAV_AVAILABLE:
            try:
                codec_name = 'h264_nvenc' if self.use_gpu and CUDA_AVAILABLE else 'libx264'
                
                # Create an output container for encoding
                self.output = av.open(f'null.mp4', mode='w')
                
                # Create a stream
                self.stream = self.output.add_stream(codec_name, rate=self.fps)
                self.stream.width = self.width
                self.stream.height = self.height
                self.stream.pix_fmt = 'yuv420p'
                self.stream.bit_rate = self.bitrate
                self.stream.options = {
                    'profile': self.profile,
                    'preset': self.preset,
                    'g': str(self.keyframe_interval)
                }
                
                # Add codec-specific options for hardware encoding
                if self.use_gpu and CUDA_AVAILABLE:
                    self.stream.options.update({
                        'gpu': str(self.gpu_device),
                        'delay': '0',  # Low-latency mode
                        'zerolatency': '1'
                    })
                
                self.running = True
                return True
                
            except Exception as e:
                logger.error(f"Failed to initialize PyAV H264 encoder: {e}", exc_info=True)
                self.running = False
                
        logger.error("No suitable H264 encoder available")
        return False
        
    def encode(self, frame: VideoFrame) -> bool:
        """
        Encode a video frame.
        
        Args:
            frame: Video frame to encode
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.running:
            if not self._initialize():
                return False
                
        try:
            if RUST_EXTENSION_AVAILABLE and self.encoder is not None:
                # Direct encoding with Rust extension
                result = h264_ext.encode_frame(
                    self.encoder,
                    frame.data,
                    is_keyframe=self.frame_count % self.keyframe_interval == 0
                )
                
                self.frame_count += 1
                
                if result:
                    with self.lock:
                        self.output_queue.append(result)
                    return True
                return False
                
            elif PYAV_AVAILABLE:
                # Convert numpy array to AVFrame
                av_frame = av.VideoFrame.from_ndarray(frame.data, format='rgb24')
                av_frame.pts = self.frame_count
                
                # Encode the frame
                packets = self.stream.encode(av_frame)
                
                self.frame_count += 1
                
                # Store encoded packets
                if packets:
                    with self.lock:
                        for packet in packets:
                            self.output_queue.append(packet.to_bytes())
                    return True
                return False
                
            return False
                
        except Exception as e:
            logger.error(f"Error encoding frame: {e}", exc_info=True)
            return False
            
    def get_encoded_data(self) -> Optional[bytes]:
        """
        Get the most recently encoded data.
        
        Returns:
            bytes: Encoded video data, or None if no data is available
        """
        with self.lock:
            if not self.output_queue:
                return None
            return self.output_queue.pop(0)
            
    def set_bitrate(self, bitrate: int) -> bool:
        """Set the encoder bitrate."""
        try:
            self.bitrate = bitrate
            
            if RUST_EXTENSION_AVAILABLE and self.encoder is not None:
                return h264_ext.set_encoder_parameter(self.encoder, "bitrate", bitrate)
                
            elif PYAV_AVAILABLE and hasattr(self, 'stream'):
                self.stream.bit_rate = bitrate
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error setting bitrate: {e}", exc_info=True)
            return False
            
    def set_keyframe_interval(self, interval: int) -> bool:
        """Set the keyframe interval."""
        try:
            self.keyframe_interval = interval
            
            if RUST_EXTENSION_AVAILABLE and self.encoder is not None:
                return h264_ext.set_encoder_parameter(self.encoder, "keyframe_interval", interval)
                
            elif PYAV_AVAILABLE and hasattr(self, 'stream'):
                self.stream.options['g'] = str(interval)
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error setting keyframe interval: {e}", exc_info=True)
            return False
            
    def set_quality(self, quality: int) -> bool:
        """Set encoding quality."""
        try:
            # Convert quality (0-100) to CRF value (0-51, lower is better)
            crf = max(0, min(51, int(51 - (quality / 100.0 * 51))))
            
            if RUST_EXTENSION_AVAILABLE and self.encoder is not None:
                return h264_ext.set_encoder_parameter(self.encoder, "crf", crf)
                
            elif PYAV_AVAILABLE and hasattr(self, 'stream'):
                self.stream.options['crf'] = str(crf)
                return True
                
            return False
                
        except Exception as e:
            logger.error(f"Error setting quality: {e}", exc_info=True)
            return False
            
    def release(self) -> None:
        """Release encoder resources."""
        try:
            self.running = False
            
            if RUST_EXTENSION_AVAILABLE and self.encoder is not None:
                h264_ext.destroy_encoder(self.encoder)
                self.encoder = None
                
            elif PYAV_AVAILABLE:
                # Flush encoder
                if hasattr(self, 'stream'):
                    packets = self.stream.encode(None)
                    with self.lock:
                        for packet in packets:
                            self.output_queue.append(packet.to_bytes())
                            
                # Close output
                if hasattr(self, 'output'):
                    self.output.close()
                    
            self.frame_queue = []
                
        except Exception as e:
            logger.error(f"Error releasing encoder: {e}", exc_info=True)                    self.output.close