"""
Video encoding and decoding components.
"""

from typing import Dict, Optional

from .encoder import Encoder, create_encoder
from .decoder import Decoder, create_decoder
from .h264 import H264Encoder, H264Decoder
from .h265 import H265Encoder, H265Decoder
from .klv import KLVMetadataEmbed

__all__ = [
    'Encoder', 'Decoder', 'create_encoder', 'create_decoder',
    'H264Encoder', 'H264Decoder', 'H265Encoder', 'H265Decoder',
    'KLVMetadataEmbed'
]