"""
Data sample class for HELICOID dataset samples.
"""

import typing
from dataclasses import dataclass
import numpy as np
from PIL import Image

from .base_data_sample import BaseDataSample


@dataclass
class HelicoidSample(BaseDataSample):
    """
    Dataclass representing a single sample from the HelicoidDataset.
    
    This provides easy property access and type hints for HELICOID data components.
    Supports __getitem__ for backward compatibility with dictionary-based access.
    Maintains the exact same structure as the original dictionary to ensure compatibility.
    
    Attributes
    ----------
    id : str
        Patient identifier (e.g., "004-02") - inherited from BaseDataSample.
    patient_id : str
        Patient ID (e.g., "004") - inherited from BaseDataSample.
    fov : str
        FOV number (e.g., "02") - inherited from BaseDataSample.
    hsi_cube : np.ndarray
        Hyperspectral cube data (C, H, W).
    gt_map : np.ndarray
        Ground truth segmentation map (H, W).
    rgb_image : Image or None
        RGB image as PIL Image (if with_rgb=True).
    wavelengths : np.ndarray
        Wavelengths corresponding to the spectral dimension.
    delta_A : np.ndarray, optional
        Delta attenuation (if with_delta_A=True).
    reference_pixel : tuple, optional
        Reference pixel coordinates (if with_delta_A=True).
    """
    
    hsi_cube: np.ndarray
    gt_map: np.ndarray
    rgb_image: typing.Optional[Image.Image]
    wavelengths: np.ndarray
    delta_A: typing.Optional[np.ndarray] = None
    reference_pixel: typing.Optional[typing.Tuple[int, int]] = None
