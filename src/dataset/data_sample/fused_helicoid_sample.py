import typing

from dataclasses import dataclass
import numpy as np

from .base_data_sample import BaseDataSample

@dataclass
class FusedHelicoidSample(BaseDataSample):
    """
    Dataclass representing a single sample from the FusedHelicoidConcentrationDataset.

    This provides easy property access and type hints for all sample components.
    Supports __getitem__ for backward compatibility with dictionary-based access.

    Attributes
    ----------
    id : str
        Sample ID (e.g., "004-02") - inherited from BaseDataSample.
    patient_id : str
        Patient ID (e.g., "004") - inherited from BaseDataSample.
    fov : str
        FOV number (e.g., "02") - inherited from BaseDataSample.
    hsi_cube_original : np.ndarray
        Original HSI cube data (C, H, W) with all wavelengths.
    hsi_cube : np.ndarray
        HSI cube data (C_selected, H, W) with selected wavelengths.
    wavelengths : np.ndarray
        Selected wavelengths corresponding to the filtered HSI cube.
    gt_map : np.ndarray
        Ground truth segmentation map (H, W).
    concentration_data : dict
        Concentration maps from both reduced wavelength and ground truth processing.
    """

    hsi_cube_original: np.ndarray
    hsi_cube: np.ndarray
    wavelengths: np.ndarray
    gt_map: np.ndarray
    concentration_data: dict