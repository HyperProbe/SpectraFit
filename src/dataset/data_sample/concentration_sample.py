"""
Data sample class for concentration dataset samples.
"""

import typing
from dataclasses import dataclass
import numpy as np

from .base_data_sample import BaseDataSample


@dataclass
class ConcentrationSample(BaseDataSample):
    """
    Dataclass representing a single sample from the ConcentrationsDataset.
    
    This provides easy property access and type hints for concentration data components.
    Supports __getitem__ for backward compatibility with dictionary-based access.
    Maintains the exact same structure as the original dictionary to ensure compatibility.
    
    Attributes
    ----------
    id : str
        Sample ID (inherited from BaseDataSample).
    patient_id : str
        Patient ID (inherited from BaseDataSample).
    fov : str
        FOV number as string (inherited from BaseDataSample).
    reduced_wl : dict
        Dictionary with reduced wavelength concentration data.
        Contains 'coef_list', 'scatter_params', and optionally 'errors_scatter'.
    gt : dict
        Dictionary with ground truth concentration data.
        Contains 'coef_list', 'scatter_params', and optionally 'errors_scatter'.
    """
    
    reduced_wl: dict
    gt: dict
