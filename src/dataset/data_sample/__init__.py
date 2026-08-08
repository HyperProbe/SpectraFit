"""
Data sample classes for different dataset types.

This module provides dataclass representations of dataset samples that offer
both attribute-style and dictionary-style access for backward compatibility.
"""

from .base_data_sample import BaseDataSample
from .concentration_sample import ConcentrationSample
from .helicoid_sample import HelicoidSample
from .fused_helicoid_sample import FusedHelicoidSample

__all__ = [
    "BaseDataSample",
    "ConcentrationSample",
    "HelicoidSample", 
    "FusedHelicoidSample",
]