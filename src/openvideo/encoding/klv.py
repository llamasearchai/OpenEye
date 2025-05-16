"""
MISB KLV (Key-Length-Value) metadata embedding for video streams.
"""

import logging
import struct
import time
from typing import Dict, Optional, Any, List, Tuple, Union
import uuid
import numpy as np

logger = logging.getLogger(__name__)


class KLVMetadataEmbed:
    """
    MISB KLV metadata embedding for video streams.
    
    Implements MISB Standard 0601.X for embedding sensor metadata in
    motion imagery streams.
    """
    
    # MISB Universal Label (UL) - 16 bytes
    MISB_UL = bytes([
        0x06, 0x0E, 0x2B, 0x34, 0x02, 0x0B, 0x01, 0x01,
        0x0E, 0x01, 0x03, 0x01, 0x01, 0x00, 0x00, 0x00
    ])
    
    # KLV tag IDs for common metadata elements
    TAG_UNIX_TIMESTAMP = 2
    TAG_MISSION_ID = 3
    TAG_PLATFORM_HEADING = 5
    TAG_PLATFORM_PITCH = 6
    TAG_PLATFORM_ROLL = 7
    TAG_PLATFORM_DESIGNATION = 10
    TAG_IMAGE_SOURCE_SENSOR = 11
    TAG_IMAGE_COORDINATE_SYSTEM = 12
    TAG_SENSOR_LAT = 13
    TAG_SENSOR_LON = 14
    TAG_SENSOR_TRUE_ALT = 15
    TAG_SENSOR_HORIZONTAL_FOV = 16
    TAG_SENSOR_VERTICAL_FOV = 17
    TAG_SENSOR_REL_AZ_ANGLE = 18
    TAG_SENSOR_REL_EL_ANGLE = 19
    TAG_SENSOR_REL_ROLL_ANGLE = 20
    TAG_SLANT_RANGE = 21
    TAG_TARGET_WIDTH = 22
    TAG_FRAME_CENTER_LAT = 23
    TAG_FRAME_CENTER_LON = 24
    TAG_FRAME_CENTER_ELEV = 25
    TAG_OFFSET_CORNER_LAT_1 = 26  # ... through 29 for corners 1-4
    TAG_OFFSET_CORNER_LON_1 = 30  # ... through 33 for corners 1-4
    TAG_ICING_DETECTED = 34
    TAG_WIND_DIRECTION = 35
    TAG_WIND_SPEED = 36
    TAG_STATIC_PRESSURE = 37
    TAG_DENSITY_ALTITUDE = 38
    TAG_OUTSIDE_AIR_TEMP = 39
    TAG_TARGET_LAT = 40
    TAG_TARGET_LON = 41
    TAG_TARGET_ELEV = 42
    TAG_TARGET_LOCATION_ACCURACY = 43
    TAG_DEVICE_ID = 72
    TAG_CHECKSUM = 73
    
    def __init__(self):
        """Initialize KLV metadata handler."""
        self.metadata = {}
        
    def set_metadata(self, tag_id: int, value: Any) -> None:
        """
        Set metadata value for a specific tag.
        
        Args:
            tag_id: KLV tag ID
            value: Value to set
        """
        self.metadata[tag_id] = value
        
    def set_geo_location(self, lat: float, lon: float, alt: float) -> None:
        """
        Set the platform geolocation.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters
        """
        self.set_metadata(self.TAG_SENSOR_LAT, lat)
        self.set_metadata(self.TAG_SENSOR_LON, lon)
        self.set_metadata(self.TAG_SENSOR_TRUE_ALT, alt)
        
    def set_orientation(self, heading: float, pitch: float, roll: float) -> None:
        """
        Set the platform orientation.
        
        Args:
            heading: Heading in degrees (0-360)
            pitch: Pitch in degrees
            roll: Roll in degrees
        """
        self.set_metadata(self.TAG_PLATFORM_HEADING, heading)
        self.set_metadata(self.TAG_PLATFORM_PITCH, pitch)
        self.set_metadata(self.TAG_PLATFORM_ROLL, roll)
        
    def set_frame_center(self, lat: float, lon: float, elevation: float) -> None:
        """
        Set the center coordinates of the current frame.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            elevation: Elevation in meters
        """
        self.set_metadata(self.TAG_FRAME_CENTER_LAT, lat)
        self.set_metadata(self.TAG_FRAME_CENTER_LON, lon)
        self.set_metadata(self.TAG_FRAME_CENTER_ELEV, elevation)
        
    def set_frame_corners(self, corners: List[Tuple[float, float]]) -> None:
        """
        Set the corner coordinates of the current frame.
        
        Args:
            corners: List of (lat, lon) tuples for the four corners
        """
        if len(corners) != 4:
            raise ValueError("Must provide exactly 4 corners")
            
        for i, (lat, lon) in enumerate(corners):
            self.set_metadata(self.TAG_OFFSET_CORNER_LAT_1 + i, lat)
            self.set_metadata(self.TAG_OFFSET_CORNER_LON_1 + i, lon)
            
    def set_mission_id(self, mission_id: str) -> None:
        """
        Set the mission ID.
        
        Args:
            mission_id: Mission identifier string
        """
        self.set_metadata(self.TAG_MISSION_ID, mission_id)
        
    def set_platform_designation(self, designation: str) -> None:
        """
        Set the platform designation.
        
        Args:
            designation: Platform designation (e.g., "UAV-1")
        """
        self.set_metadata(self.TAG_PLATFORM_DESIGNATION, designation)
        
    def set_device_id(self, device_id: str) -> None:
        """
        Set the device ID.
        
        Args:
            device_id: Device identifier
        """
        self.set_metadata(self.TAG_DEVICE_ID, device_id)
        
    def encode(self) -> bytes:
        """
        Encode the metadata as KLV bytes.
        
        Returns:
            bytes: KLV encoded metadata
        """
        # Ensure timestamp is present
        if self.TAG_UNIX_TIMESTAMP not in self.metadata:
            self.set_metadata(self.TAG_UNIX_TIMESTAMP, time.time())
            
        # Start with the Universal Label
        klv_bytes = bytearray(self.MISB_UL)
        
        # Calculate the total length of the value part
        value_length = 0
        encoded_tags = []
        
        for tag_id, value in self.metadata.items():
            encoded_value = self._encode_value(tag_id, value)
            encoded_tags.append((tag_id, encoded_value))
            value_length += 1  # Tag ID byte
            value_length += len(self._encode_length(len(encoded_value)))  # Length bytes
            value_length += len(encoded_value)  # Value bytes
            
        # Encode the length of the value part
        klv_bytes.extend(self._encode_length(value_length))
        
        # Add all tags
        for tag_id, encoded_value in encoded_tags:
            klv_bytes.append(tag_id)
            klv_bytes.extend(self._encode_length(len(encoded_value)))
            klv_bytes.extend(encoded_value)
            
        # Calculate and append checksum if needed
        if self.TAG_CHECKSUM not in self.metadata:
            checksum = sum(klv_bytes) % 256
            klv_bytes.append(self.TAG_CHECKSUM)
            klv_bytes.append(1)  # Length
            klv_bytes.append(checksum)
            
        return bytes(klv_bytes)
        
    def _encode_length(self, length: int) -> bytes:
        """
        Encode a length value according to BER encoding rules.
        
        Args:
            length: Length value to encode
            
        Returns:
            bytes: BER encoded length
        """
        if length < 128:
            return bytes([length])
        else:
            # Calculate how many bytes needed to represent the length
            byte_count = (length.bit_length() + 7) // 8
            result = bytearray([128 | byte_count])
            
            # Add the length bytes
            for i in range(byte_count - 1, -1, -1):
                result.append((length >> (i * 8)) & 0xFF)
                
            return bytes(result)
            
    def _encode_value(self, tag_id: int, value: Any) -> bytes:
        """
        Encode a value based on its tag ID.
        
        Args:
            tag_id: KLV tag ID
            value: Value to encode
            
        Returns:
            bytes: Encoded value
        """
        # String types
        if tag_id in [self.TAG_MISSION_ID, self.TAG_PLATFORM_DESIGNATION, 
                      self.TAG_IMAGE_SOURCE_SENSOR, self.TAG_DEVICE_ID]:
            if isinstance(value, str):
                return value.encode('utf-8')
            else:
                return str(value).encode('utf-8')
                
        # Timestamp (UNIX time)
        elif tag_id == self.TAG_UNIX_TIMESTAMP:
            return struct.pack('>Q', int(value * 1000000))  # Microseconds
            
        # Lat/lon values - encoded as 4-byte integers (scaled)
        elif tag_id in [self.TAG_SENSOR_LAT, self.TAG_FRAME_CENTER_LAT, 
                       self.TAG_OFFSET_CORNER_LAT_1, self.TAG_OFFSET_CORNER_LAT_1 + 1,
                       self.TAG_OFFSET_CORNER_LAT_1 + 2, self.TAG_OFFSET_CORNER_LAT_1 + 3,
                       self.TAG_TARGET_LAT]:
            # Convert to scaled integer (-90 to 90 degrees mapped to -2^31 to 2^31-1)
            scaled = int((value / 90.0) * 2147483647)
            return struct.pack('>i', scaled)
            
        elif tag_id in [self.TAG_SENSOR_LON, self.TAG_FRAME_CENTER_LON,
                       self.TAG_OFFSET_CORNER_LON_1, self.TAG_OFFSET_CORNER_LON_1 + 1,
                       self.TAG_OFFSET_CORNER_LON_1 + 2, self.TAG_OFFSET_CORNER_LON_1 + 3,
                       self.TAG_TARGET_LON]:
            # Convert to scaled integer (-180 to 180 degrees mapped to -2^31 to 2^31-1)
            scaled = int((value / 180.0) * 2147483647)
            return struct.pack('>i', scaled)
            
        # Altitude values - encoded as 2-byte integers (scaled)
        elif tag_id in [self.TAG_SENSOR_TRUE_ALT, self.TAG_FRAME_CENTER_ELEV, self.TAG_TARGET_ELEV]:
            # Convert to scaled integer (-900 to 19000 meters mapped to 0 to 2^16-1)
            scaled = int(((value + 900) / 19900.0) * 65535)
            return struct.pack('>H', min(65535, max(0, scaled)))
            
        # Angle values - encoded as 2-byte integers (scaled)
        elif tag_id in [self.TAG_PLATFORM_HEADING, self.TAG_PLATFORM_PITCH, self.TAG_PLATFORM_ROLL,
                       self.TAG_SENSOR_REL_AZ_ANGLE, self.TAG_SENSOR_REL_EL_ANGLE, 
                       self.TAG_SENSOR_REL_ROLL_ANGLE]:
            # Convert to scaled integer (0-360 or -180 to 180 mapped to 0 to 2^16-1)
            if tag_id == self.TAG_PLATFORM_HEADING:
                # 0-360 degrees
                scaled = int((value / 360.0) * 65535)
            else:
                # -180 to 180 degrees
                scaled = int(((value + 180) / 360.0) * 65535)
            return struct.pack('>H', min(65535, max(0, scaled)))
            
        # FOV values - encoded as 2-byte unsigned integers (scaled)
        elif tag_id in [self.TAG_SENSOR_HORIZONTAL_FOV, self.TAG_SENSOR_VERTICAL_FOV]:
            # Convert to scaled integer (0-180 degrees mapped to 0 to 2^16-1)
            scaled = int((value / 180.0) * 65535)
            return struct.pack('>H', min(65535, max(0, scaled)))
            
        # Default - try to encode as a 4-byte float
        else:
            try:
                return struct.pack('>f', float(value))
            except:
                # Fallback to string encoding
                return str(value).encode('utf-8')
                
    @classmethod
    def decode(cls, klv_bytes: bytes) -> Dict[int, Any]:
        """
        Decode KLV bytes into a metadata dictionary.
        
        Args:
            klv_bytes: KLV encoded metadata
            
        Returns:
            Dict[int, Any]: Decoded metadata dictionary
        """
        # Implement KLV decoding here
        pass