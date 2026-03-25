"""
Bitstream Package for LeWM-VC

Implements NAL/OBU bitstream serialization and deserialization.
Contains writer and reader classes for video bitstream handling.
"""

from .writer import BitstreamWriter, NALUnitType
from .reader import BitstreamReader

__all__ = [
    "BitstreamWriter",
    "BitstreamReader",
    "NALUnitType",
]
