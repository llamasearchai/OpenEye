import struct
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ..utils.logging import get_logger


class KLVParser:
    """Parser for KLV (Key-Length-Value) metadata format