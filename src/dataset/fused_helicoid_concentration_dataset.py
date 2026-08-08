"""
Fused HELICOID dataset combining HSI data and concentration maps.

This dataset loads both the original hyperspectral imaging data from HELICOID
and the corresponding concentration maps from spectral unmixing results.
Each sample contains both the HSI cube (with selected wavelengths) and
the concentration coefficients.
"""

import os
import re
import typing
from pathlib import Path
import numpy as np
import torch
from loguru import logger

from .base_dataset import BaseHSIDataset
from .helicoid_dataset import HelicoidDataset
from .concentrations_dataset import ConcentrationsDataset, Signal
from ..wavelength_selection.enums import SampleType
from .data_sample.fused_helicoid_sample import FusedHelicoidSample


class FusedHelicoidConcentrationDataset(BaseHSIDataset):
    """
    A fused dataset that combines HELICOID HSI data with concentration maps.

    This dataset provides access to both the original hyperspectral cubes
    and their corresponding concentration maps from spectral unmixing.
    The concentration data includes both reduced wavelength and ground truth
    versions, loaded from separate directories. The HSI data can be filtered
    to specific wavelengths either by left/right cuts or by providing a
    selected wavelength array.
    """

    def __init__(
        self,
        helicoid_data_dir: typing.Union[str, Path],
        reduced_wl_dir: str,
        gt_dir: str,
        left_cut: int = 530,
        right_cut: int = 750,
        selected_wavelengths: typing.Optional[
            typing.Union[str, Path, np.ndarray]
        ] = None,
        coarseness: int = 1,
        normalize_image: bool = True,
        downsample_factor: typing.Optional[
            typing.Union[int, typing.Tuple[int, int]]
        ] = None,
        center_crop_size: typing.Optional[
            typing.Union[int, typing.Tuple[int, int]]
        ] = None,
        normalization_dir: typing.Optional[str] = None,
        transform: typing.Optional[typing.Callable] = None,
        channels_first: bool = True,
        inference_mode: bool = False,
        merge_channels_to_signals: typing.List[Signal]= None,
        image_wise_normalization: bool = False,
        ids_subset: typing.Optional[typing.List[int]] = None,
    ):
        """
        Initialize the FusedHelicoidConcentrationDataset.

        Parameters
        ----------
        helicoid_data_dir : str or Path
            Path to the directory containing HELICOID patient folders with HSI data.
        reduced_wl_dir : str
            Directory containing concentration maps computed with reduced wavelength sets.
            Expected to contain subdirectories with concentration map files.
        gt_dir : str
            Directory containing ground truth concentration maps computed with full wavelength range.
            Expected to contain subdirectories with concentration map files.
        left_cut : int, optional
            Lower wavelength bound (nm) for cropping the spectral axis.
            Only used if selected_wavelengths is None. Defaults to 530.
        right_cut : int, optional
            Upper wavelength bound (nm) for cropping the spectral axis.
            Only used if selected_wavelengths is None. Defaults to 750.
        selected_wavelengths : str, Path, np.ndarray, or None, optional
            Specific wavelengths to select from the HSI cube:
            - If str or Path: path to a .npy file containing wavelength array
            - If np.ndarray: array of wavelengths to select
            - If None: use left_cut and right_cut for wavelength selection
            Example array: [537.994, 548.18, 555.456, 566.369, ...]
            Defaults to None.
        coarseness : int, optional
            Spatial downsampling factor for both spatial dimensions.
            Defaults to 1 (no spatial downsampling).
        normalize_image : bool, optional
            Whether to normalize the raw HSI data using white and dark references.
            Defaults to True.
        downsample_factor : int or tuple of two ints, optional
            Additional spatial downsampling factor for the H×W dimensions.
            Defaults to None.
        center_crop_size : int or tuple of int, optional
            Size for center cropping the spatial dimensions.
            Defaults to None.
        normalization_dir : str, optional
            Directory containing normalization statistics for each channel.
            Run concentration_dataset_demo.ipnyb to generate this directory.
            Expected to contain two subdirectories:
            - 'gt/': with mean_{index}.npy and std_{index}.npy files for ground truth data
            - 'reduced_wl/': with mean_{index}.npy and std_{index}.npy files for reduced wavelength data
            If None, no normalization is applied. Defaults to None.
        transform : callable, optional
            Optional transform to be applied on a sample.
            Defaults to None.
        channels_first : bool, optional
            If True, HSI cubes are returned in (C, H, W) format.
            If False, (H, W, C) format is used. Defaults to True.
        merge_channels_to_signals : list of Signal, optional
            List of Signal enums specifying which channels to merge into single signals.
        inference_mode : bool, optional
            If True, dataset is set for inference, use full image cropped to be divisible by 8.
        image_wise_normalization : bool, optional
            If True, applies image-wise normalization (zero mean, unit variance) to each sample.
        ids_subset : list of int, optional
            Optional list of ids_subset to subset the dataset. If None, use all samples.
            Defaults to None.
        Raises
        ------
        FileNotFoundError
            If helicoid_data_dir, reduced_wl_dir, or gt_dir does not exist.
        ValueError
            If selected_wavelengths file cannot be loaded.
        """
        self.helicoid_data_dir = Path(helicoid_data_dir)
        self.reduced_wl_dir = Path(reduced_wl_dir)
        self.gt_dir = Path(gt_dir)
        self.left_cut = left_cut
        self.right_cut = right_cut
        self.selected_wavelengths = selected_wavelengths
        self.coarseness = coarseness
        self.normalize_image = normalize_image
        self.downsample_factor = downsample_factor
        self.center_crop_size = center_crop_size
        self.normalization_dir = normalization_dir
        self.transform = transform
        self.channels_first = channels_first
        self.inference_mode = inference_mode
        self.ids_subset = ids_subset
        self.image_wise_normalization = image_wise_normalization
        self.merge_channels_to_signals = merge_channels_to_signals

        if not self.helicoid_data_dir.exists():
            raise FileNotFoundError(
                f"HELICOID data directory not found: {self.helicoid_data_dir}"
            )
        if not self.reduced_wl_dir.exists():
            raise FileNotFoundError(
                f"Reduced wavelength directory not found: {self.reduced_wl_dir}"
            )
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {self.gt_dir}")

        # Load wavelength selection if provided
        self.wavelength_array = self._load_wavelength_selection()

        # Initialize the HelicoidDataset for HSI data
        self.helicoid_dataset = HelicoidDataset(
            helicoid_data_dir=helicoid_data_dir,
            left_cut=left_cut,
            right_cut=right_cut,
            coarseness=coarseness,
            normalize_image=normalize_image,
            with_delta_A=False,
            downsample_factor=downsample_factor,
            center_crop_size=center_crop_size,
            transform=None,  # We'll apply transforms later
            inference_mode=inference_mode,
            ids_subset=ids_subset,
        )

        self.concentrations_dataset = ConcentrationsDataset(
            reduced_wl_dir=str(self.reduced_wl_dir),
            gt_dir=str(self.gt_dir),
            sample_type=SampleType.HELICOID,
            center_crop_size=center_crop_size,
            normalization_dir=normalization_dir,
            inference_mode=inference_mode,
            channels_first=self.channels_first,
            merge_channels_to_signals=merge_channels_to_signals,
            image_wise_normalization=image_wise_normalization,
            ids_subset=ids_subset,
        )

        # Find common samples between both datasets
        self.samples = self._find_common_samples()

        # Create mapping for faster lookup
        self.sample_map = {sample["id"]: idx for idx, sample in enumerate(self.samples)}

        # Get wavelength indices for filtering if using selected wavelengths
        self.wavelength_indices = self._get_wavelength_indices()

        logger.info(
            f"Initialized FusedHelicoidConcentrationDataset with {len(self.samples)} samples"
        )

        super().__init__()

    def _load_wavelength_selection(self) -> typing.Optional[np.ndarray]:
        """
        Load wavelength selection from file or array.

        Returns
        -------
        np.ndarray or None
            Array of selected wavelengths or None if not specified.
        """
        if self.selected_wavelengths is None:
            return None

        if isinstance(self.selected_wavelengths, np.ndarray):
            return self.selected_wavelengths

        elif isinstance(self.selected_wavelengths, (str, Path)):
            wavelength_path = Path(self.selected_wavelengths)
            if not wavelength_path.exists():
                raise FileNotFoundError(
                    f"Wavelength selection file not found: {wavelength_path}"
                )

            try:
                wavelength_array = np.load(wavelength_path)
                logger.info(
                    f"Loaded {len(wavelength_array)} selected wavelengths from {wavelength_path}"
                )
                return wavelength_array
            except Exception as e:
                raise ValueError(
                    f"Error loading wavelength selection from {wavelength_path}: {e}"
                )

        else:
            raise ValueError(
                f"selected_wavelengths must be None, np.ndarray, or path to .npy file, "
                f"got {type(self.selected_wavelengths)}"
            )

    def set_inference_mode(self, inference_mode: bool):
        """
        Set the inference mode for the dataset.

        Parameters
        ----------
        inference_mode : bool
            If True, dataset is set for inference, use full image cropped to be divisible by 8.
        """
        self.inference_mode = inference_mode
        self.helicoid_dataset.inference_mode = inference_mode
        self.concentrations_dataset.inference_mode = inference_mode

    def _find_common_samples(self) -> typing.List[dict]:
        """
        Find samples that exist in both HELICOID and concentration datasets.

        Returns
        -------
        list of dict
            List of common sample information.
        """
        # Get sample IDs from both datasets
        helicoid_ids = set(self.helicoid_dataset.get_sample_ids())
        concentration_ids = set(
            sample["id"] for sample in self.concentrations_dataset.samples
        )

        # Find intersection
        common_ids = helicoid_ids.intersection(concentration_ids)

        if not common_ids:
            raise ValueError(
                "No common samples found between HELICOID and concentration datasets"
            )

        # Create sample list with metadata from both datasets
        samples = []
        for sample_id in sorted(common_ids):
            # Get patient_id and fov from sample_id (e.g., "004-02" -> "004", "02")
            parts = sample_id.split("-")
            patient_id = parts[0]
            fov = parts[1]

            sample = {
                "id": sample_id,
                "patient_id": patient_id,
                "fov": fov,
            }
            samples.append(sample)

        logger.info(f"Found {len(samples)} common samples between datasets")
        return samples

    def _get_wavelength_indices(self) -> typing.Optional[np.ndarray]:
        """
        Get indices of wavelengths to select from the HSI cube.

        Returns
        -------
        np.ndarray or None
            Indices of wavelengths to select, or None if using left/right cuts.
        """
        if self.wavelength_array is None:
            return None

        # Get all wavelengths from HELICOID dataset
        all_wavelengths = self.helicoid_dataset.cut_wavelengths

        # Find indices of selected wavelengths
        indices = []
        for target_wl in self.wavelength_array:
            # Find closest wavelength
            closest_idx = np.argmin(np.abs(all_wavelengths - target_wl))
            closest_wl = all_wavelengths[closest_idx]

            # Check if it's close enough (within 1 nm tolerance)
            if np.abs(closest_wl - target_wl) <= 1.0:
                indices.append(closest_idx)
            else:
                logger.warning(
                    f"Warning: Could not find close match for wavelength {target_wl:.3f} nm. "
                    f"Closest available: {closest_wl:.3f} nm"
                )

        if not indices:
            raise ValueError("No matching wavelengths found in HSI data")

        logger.info(
            f"Selected {len(indices)} wavelength bands from {len(all_wavelengths)} available"
        )
        return np.array(indices)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> FusedHelicoidSample:
        """
        Retrieve a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        FusedHelicoidSample
            Sample dataclass containing:
            - id: Sample ID (e.g., "004-02")
            - patient_id: Patient ID (e.g., "004")
            - fov: FOV number (e.g., "02")
            - hsi_cube_original: Original HSI cube data (C, H, W) with all wavelengths
            - wavelengths_original: All wavelengths corresponding to original HSI cube
            - hsi_cube: HSI cube data (C_selected, H, W) with selected wavelengths
            - wavelengths: Selected wavelengths corresponding to the filtered HSI cube
            - gt_map: Ground truth segmentation map (H, W)
            - concentration_data: ConcentrationData with both reduced_wl and gt concentration data
        """
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        sample_info = self.samples[idx]
        sample_id = sample_info["id"]

        # Get HSI data from HELICOID dataset
        hsi_sample = self.helicoid_dataset.get_sample_by_id(sample_id)
        if hsi_sample is None:
            raise ValueError(f"HSI sample {sample_id} not found in HELICOID dataset")

        # Get concentration data from concentrations dataset
        concentration_sample = self.concentrations_dataset.get_sample_by_id(sample_id)
        if concentration_sample is None:
            raise ValueError(
                f"Concentration sample {sample_id} not found in concentrations dataset"
            )

        # Extract HSI cube and apply wavelength selection if specified
        hsi_cube_original = hsi_sample["hsi_cube"]  # Shape: (C, H, W)
        wavelengths_original = hsi_sample["wavelengths"]

        # Always include the original HSI cube
        hsi_cube_selected = hsi_cube_original
        wavelengths_selected = wavelengths_original

        if self.wavelength_indices is not None:
            # Select specific wavelength bands for the selected version
            hsi_cube_selected = hsi_cube_original[self.wavelength_indices, :, :]
            wavelengths_selected = wavelengths_original[self.wavelength_indices]

        # Create concentration dict
        concentration_data = {
            "reduced_wl": concentration_sample["reduced_wl"],
            "gt": concentration_sample["gt"],
        }

        # Create fused sample dataclass
        fused_sample = FusedHelicoidSample(
            id=sample_id,
            patient_id=sample_info["patient_id"],
            fov=sample_info["fov"],
            hsi_cube_original=hsi_cube_original,
            hsi_cube=hsi_cube_selected,
            wavelengths=wavelengths_selected,
            gt_map=hsi_sample["gt_map"],
            concentration_data=concentration_data,
        )

        # Apply transform if provided
        if self.transform:
            fused_sample = self.transform(fused_sample)

        return fused_sample

    def get_patient_ids(self) -> typing.List[str]:
        """
        Get a list of all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        patient_ids = set(sample["patient_id"] for sample in self.samples)
        return sorted(patient_ids)

    def get_sample_by_id(
        self, sample_id: str
    ) -> typing.Union[FusedHelicoidSample, None]:
        """
        Retrieve a sample by its ID.

        Parameters
        ----------
        sample_id : str
            The sample ID (e.g., "004-02").

        Returns
        -------
        FusedHelicoidSample or None
            The sample data or None if not found.
        """
        if sample_id in self.sample_map:
            return self.__getitem__(self.sample_map[sample_id])

        logger.warning(f"Sample with ID '{sample_id}' not found.")
        return None

    def get_samples_by_patient_id(
        self, patient_id: str
    ) -> typing.List[FusedHelicoidSample]:
        """
        Retrieve all samples for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "004").

        Returns
        -------
        list of FusedHelicoidSample
            A list of all samples for the given patient, empty if none found.
        """
        samples = []

        for idx, sample_info in enumerate(self.samples):
            if sample_info["patient_id"] == patient_id:
                samples.append(self.__getitem__(idx))

        if not samples:
            logger.warning(f"No samples found for patient ID '{patient_id}'.")

        return samples

    def get_sample_by_patient_and_fov(
        self, patient_id: str, fov: str
    ) -> typing.Union[FusedHelicoidSample, None]:
        """
        Retrieve a specific sample by patient ID and FOV number.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "004").
        fov : str
            The FOV number as a string (e.g., "02").

        Returns
        -------
        FusedHelicoidSample or None
            The sample data or None if not found.
        """
        sample_id = f"{patient_id}-{fov}"
        return self.get_sample_by_id(sample_id)

    def get_fovs_for_patient(self, patient_id: str) -> typing.List[str]:
        """
        Get a list of FOV numbers available for a specific patient.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "004").

        Returns
        -------
        list of str
            Sorted list of FOV numbers for the patient.
        """
        fovs = []

        for sample_info in self.samples:
            if sample_info["patient_id"] == patient_id:
                fovs.append(sample_info["fov"])

        return sorted(fovs)

    def load_reference_params(
        self, id: str, load_from_path: str
    ) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Load reference spectrum parameters (a_t1, b_t1) from a specified path.

        Parameters
        ----------
        id : str
            Sample ID (e.g., "004-02").
        load_from_path : str
            Path to the .npz file containing 'a_t1' and 'b_t1' arrays.

        Returns
        -------
        tuple of np.ndarray or None
            Tuple containing (a_t1, b_t1) arrays, or None if loading fails.
        """
        return self.helicoid_dataset.load_reference_params(id, load_from_path)
    
    def compute_delta_A(
        self, id: str, hsi_cube: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
        """
        Compute the delta_A for a given HSI cube and ground truth map.

        Parameters
        ----------
        id : str
            Sample ID (e.g., "004-02").
        hsi_cube : np.ndarray
            The HSI cube to process.
        gt_map : np.ndarray
            The ground truth map.

        Returns
        -------
        np.ndarray
            The computed delta_A array.
        """
        # Compute the delta_A
        return self.helicoid_dataset.compute_delta_A(id, hsi_cube, gt_map)

    def get_wavelength_info(self) -> dict:
        """
        Get information about the wavelength selection.

        Returns
        -------
        dict
            Dictionary containing wavelength information.
        """
        if self.wavelength_array is not None:
            return {
                "wavelength_selection_method": "selected_wavelengths",
                "selected_wavelengths": self.wavelength_array.copy(),
                "num_selected_bands": len(self.wavelength_array),
                "wavelength_range": f"{self.wavelength_array.min():.3f} - {self.wavelength_array.max():.3f} nm",
            }
        else:
            return {
                "wavelength_selection_method": "left_right_cut",
                "left_cut": self.left_cut,
                "right_cut": self.right_cut,
                "num_selected_bands": len(self.helicoid_dataset.cut_wavelengths),
                "wavelength_range": f"{self.helicoid_dataset.cut_wavelengths.min():.3f} - {self.helicoid_dataset.cut_wavelengths.max():.3f} nm",
            }

    def get_summary(self) -> dict:
        """
        Get a summary of the dataset.

        Returns
        -------
        dict
            Summary containing dataset statistics.
        """
        patient_ids = self.get_patient_ids()
        total_samples = len(self.samples)

        patient_fov_counts = {}
        for patient_id in patient_ids:
            fovs = self.get_fovs_for_patient(patient_id)
            patient_fov_counts[patient_id] = len(fovs)

        summary = {
            "total_samples": total_samples,
            "num_patients": len(patient_ids),
            "patient_ids": patient_ids,
            "patient_fov_counts": patient_fov_counts,
            "helicoid_data_dir": str(self.helicoid_data_dir),
            "reduced_wl_dir": str(self.reduced_wl_dir),
            "gt_dir": str(self.gt_dir),
            "wavelength_info": self.get_wavelength_info(),
            "coarseness": self.coarseness,
            "normalize_image": self.normalize_image,
            "center_crop_size": self.center_crop_size,
            "channels_first": self.channels_first,
        }

        return summary
