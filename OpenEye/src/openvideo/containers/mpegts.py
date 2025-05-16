"""
MPEG-TS containerization for video and metadata.
"""

import logging
import io
import time
import struct
from typing import Dict, Optional, Any, List, Tuple, Union, BinaryIO
import threading

from ..core.types import VideoFrame
from ..encoding.klv import KLVMetadataEmbed

logger = logging.getLogger(__name__)

# Try to import PyAV for MPEG-TS handling
try:
    import av
    PYAV_AVAILABLE = True
except ImportError:
    logger.warning("PyAV not available, MPEG-TS functionality limited")
    PYAV_AVAILABLE = False

# Try to import Rust extension for optimized TS handling
try:
    from ..rust_extensions import mpegts_ext
    RUST_EXTENSION_AVAILABLE = True
except ImportError:
    logger.warning("Rust extension for MPEG-TS not available")
    RUST_EXTENSION_AVAILABLE = False