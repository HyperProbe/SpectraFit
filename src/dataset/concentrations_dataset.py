"""
PyTorch dataset for loading concentration maps from spectral unmixing results.

This dataset handles the loading of concentration maps with both reduced wavelength
and ground truth maps, providing similar functionality to the biopsy2_dataset.
Supports both Biopsy and HELICOID sample types.
"""

from enum import Enum
import os
import re
from loguru import logger
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import torch
from torchvision.transforms import (
    CenterCrop,
)
from monai.transforms import NormalizeIntensity

from ..wavelength_selection.enums import SampleType
from ..constants import _normalize_id
from .base_dataset import BaseHSIDataset
from .data_sample import ConcentrationSample


class Signal(str, Enum):
    HbT = "HbT"
    diffCCO = "diffCCO"
    b = "b"


class ConcentrationsDataset(BaseHSIDataset):
    """
    A PyTorch dataset for loading concentration maps from spectral unmixing results.

    This dataset loads concentration maps from two different directories:
    - reduced_wl_dir: Contains concentration maps computed using reduced wavelength sets
    - gt_dir: Contains ground truth concentration maps computed using full wavelength range

    Supports both Biopsy and HELICOID sample types with different directory patterns.
    The dataset provides functionality similar to biopsy2_dataset for sample retrieval
    by ID, patient ID, and patient+FOV combinations.
    """

    def __init__(
        self,
        reduced_wl_dir: str,
        gt_dir: str,
        sample_type: SampleType = SampleType.BIOPSY2,
        file_pattern: Optional[str] = None,
        center_crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        normalization_dir: Optional[str] = None,
        inference_mode: bool = False,
        channels_first: bool = False,
        merge_channels_to_signals: Optional[List[Signal]] = None,
        image_wise_normalization: bool = False,
        ids_subset: Optional[List[int]] = None,
    ):
        """
        Initialize the ConcentrationsDataset.

        Parameters
        ----------
        reduced_wl_dir : str
            Directory containing concentration maps computed with reduced wavelength sets.
            Expected to contain subdirectories with concentration map files.
        gt_dir : str
            Directory containing ground truth concentration maps computed with full wavelength range.
            Expected to contain subdirectories with concentration map files.
        sample_type : SampleType, optional
            Type of samples to handle (BIOPSY or HELICOID). Defaults to SampleType.BIOPSY2.
        file_pattern : str, optional
            Custom regex pattern to extract patient ID and FOV from directory names.
            If None, uses default pattern based on sample_type.
        center_crop_size : int or tuple of int, optional
            Size for center cropping the concentration maps (coef_list).
            If int, crops to (center_crop_size, center_crop_size).
            If tuple, crops to (height, width).
            If None, no cropping is applied. Defaults to None.
        normalization_dir : str, optional
            deprecated use image_wise_normalization instead
            Directory containing normalization statistics for each channel.
            Run concentration_dataset_demo.ipnyb to generate this directory.
            Expected to contain two subdirectories:
            - 'gt/': with mean_{index}.npy and std_{index}.npy files for ground truth data
            - 'reduced_wl/': with mean_{index}.npy and std_{index}.npy files for reduced wavelength data
            If None, no normalization is applied. Defaults to None.
        inference_mode : bool, optional
            If True, the concentration maps will be loaded in their original form but cropped that their H and W are divisible by 8.
            Defaults to False.
        channels_first : bool, optional
            If True, the concentration maps will be returned in CHW format instead of HWC.
            This applies to both 'coef_list' and 'scatter_params' arrays.
            Defaults to False.
        merge_channels_to_signals : list of Signal, optional
            List of signals to merge channels into. If None, no merging is applied.
            Defaults to None.
        image_wise_normalization : bool, optional
            If True, applies image-wise normalization (zero mean, unit variance) to each sample.
            This is done after loading and cropping and does not apply normalization across the whole dataset
            Defaults to False.
        ids_subset : list of int, optional
            Optional list of ids_subset to subset the dataset. If None, use all samples.
            Defaults to None.

        Raises
        ------
        FileNotFoundError
            If either reduced_wl_dir or gt_dir does not exist, or if normalization_dir
            is specified but doesn't exist or doesn't contain required normalization files.
        """
        self.reduced_wl_dir = Path(reduced_wl_dir)
        self.gt_dir = Path(gt_dir)
        self.sample_type = sample_type
        self.inference_mode = inference_mode
        self.channels_first = channels_first
        self.merge_channels_to_signals = merge_channels_to_signals
        self.ids_subset = ids_subset

        # Set up center crop parameters
        if center_crop_size is not None:
            if isinstance(center_crop_size, int):
                self.center_crop_size = (center_crop_size, center_crop_size)
            else:
                self.center_crop_size = tuple(center_crop_size)
        else:
            self.center_crop_size = None

        if self.center_crop_size is not None and self.inference_mode:
            logger.warning(
                "Warning: Both center_crop_size and inference_mode are set. "
                "Center cropping will be ignored in inference mode."
            )
            self.center_crop_size = None

        # Set default file pattern based on sample type
        if file_pattern is None:
            if sample_type == SampleType.BIOPSY2:
                # Pattern for biopsy: HyperProbe1.1_Biopsy_S1.2_FOV3
                self.file_pattern = re.compile(
                    r"HyperProbe1\.1_Biopsy_(S\d+(?:\.\d+)?)(?:_FOV(\d+))?"
                )
            elif sample_type == SampleType.HELICOID:
                # Pattern for HELICOID: 004-02 (patient 4, FOV 2)
                self.file_pattern = re.compile(r"(\d{3})-(\d{2})")
            else:
                raise ValueError(f"Unknown sample type: {sample_type}")
        else:
            self.file_pattern = re.compile(file_pattern)

        if not self.reduced_wl_dir.exists():
            raise FileNotFoundError(
                f"Reduced wavelength directory not found: {reduced_wl_dir}"
            )
        if not self.gt_dir.exists():
            raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

        # Set up normalization parameters
        self.normalization_dir = Path(normalization_dir) if normalization_dir else None
        self.normalization_coef_means_gt = None
        self.normalization_coef_stds_gt = None
        self.normalization_coef_means_reduced_wl = None
        self.normalization_coef_stds_reduced_wl = None

        self.normalization_scatter_means_gt = None
        self.normalization_scatter_stds_gt = None
        self.normalization_scatter_means_reduced_wl = None
        self.normalization_scatter_stds_reduced_wl = None

        if self.normalization_dir is not None:
            self._load_normalization_stats()

        self.image_wise_normalization = image_wise_normalization
        if self.image_wise_normalization:
            logger.info(
                "Image-wise normalization enabled using MONAI NormalizeIntensity"
            )
        self.image_normalization_transform = NormalizeIntensity(channel_wise=True)

        # Load sample information
        self.samples = self._discover_samples()

        # Create mapping for faster lookup
        self.sample_map = {sample["id"]: idx for idx, sample in enumerate(self.samples)}

        # Call parent constructor
        super().__init__()

    def _discover_samples(self) -> List[dict]:
        """
        Discover available samples by scanning both directories and matching them.

        Returns
        -------
        list of dict
            List of sample dictionaries containing sample information.
        """
        # Discover samples in reduced wavelength directory
        reduced_samples = self._discover_samples_in_dir(self.reduced_wl_dir)

        # Discover samples in ground truth directory
        gt_samples = self._discover_samples_in_dir(self.gt_dir)

        # Find intersection of samples (only keep samples present in both directories)
        reduced_ids = {sample["id"] for sample in reduced_samples}
        gt_ids = {sample["id"] for sample in gt_samples}

        if self.ids_subset is not None:
            reduced_ids = reduced_ids.intersection(set(self.ids_subset))
            gt_ids = gt_ids.intersection(set(self.ids_subset))

        common_ids = reduced_ids.intersection(gt_ids)

        if not common_ids:
            raise ValueError(
                "No common samples found between reduced wavelength and ground truth directories"
            )

        # Create sample list with both paths
        samples = []
        reduced_map = {sample["id"]: sample for sample in reduced_samples}
        gt_map = {sample["id"]: sample for sample in gt_samples}

        for sample_id in sorted(common_ids):
            reduced_sample = reduced_map[sample_id]
            gt_sample = gt_map[sample_id]

            sample = {
                "id": sample_id,
                "patient_id": reduced_sample["patient_id"],
                "fov": reduced_sample["fov"],
                "reduced_wl_path": reduced_sample["dir_path"],
                "gt_path": gt_sample["dir_path"],
            }
            samples.append(sample)

        print(f"Found {len(samples)} common samples between directories")
        return samples

    def _discover_samples_in_dir(self, base_dir: Path) -> List[dict]:
        """
        Discover samples in a single directory.

        Parameters
        ----------
        base_dir : Path
            Directory to scan for sample subdirectories.

        Returns
        -------
        list of dict
            List of sample information dictionaries.
        """
        samples = []

        for dir_path in base_dir.iterdir():
            if not dir_path.is_dir():
                continue

            # Extract patient ID and FOV from directory name
            match = self.file_pattern.match(dir_path.name)
            if not match:
                continue

            if self.sample_type == SampleType.BIOPSY2:
                patient_id = match.group(1)  # e.g., "S1.2"
                fov_str = match.group(2)  # e.g., "3" or None

                # Default FOV to "1" if not specified
                fov = fov_str if fov_str is not None else "1"

                # Create normalized ID for consistent lookup
                # Remove "S" prefix for internal ID format
                if patient_id.startswith("S"):
                    patient_number = patient_id[1:]
                else:
                    patient_number = patient_id

                sample_id = f"{patient_number}_{fov}"

            elif self.sample_type == SampleType.HELICOID:
                # For HELICOID: 004-02 -> patient_id="004", fov="02"
                patient_id = match.group(1)  # e.g., "004"
                fov = match.group(2)  # e.g., "02"

                # Use full directory name as sample ID
                sample_id = dir_path.name  # e.g., "004-02"

            else:
                raise ValueError(f"Unknown sample type: {self.sample_type}")

            sample = {
                "id": sample_id,
                "patient_id": patient_id,
                "fov": fov,
                "dir_path": dir_path,
            }
            samples.append(sample)

        return samples

    def _load_concentration_files(self, dir_path: Path) -> Dict[str, np.ndarray]:
        """
        Load concentration and scattering parameter files from a directory.

        Parameters
        ----------
        dir_path : Path
            Directory containing the .npy files.

        Returns
        -------
        dict
            Dictionary containing loaded arrays with keys:
            - 'coef_list': Concentration coefficients (H, W, num_molecules)
            - 'scatter_params': Scattering parameters (H, W, 2)
            - 'errors_scatter': Optimization errors (H, W) [if available]

        Raises
        ------
        FileNotFoundError
            If required files are not found in the directory.
        """
        data = {}

        # Required files
        coef_path = dir_path / "coef_list.npy"
        scatter_path = dir_path / "scatter_params.npy"

        if not coef_path.exists():
            raise FileNotFoundError(f"coef_list.npy not found in {dir_path}")
        if not scatter_path.exists():
            raise FileNotFoundError(f"scatter_params.npy not found in {dir_path}")

        data["coef_list"] = np.load(coef_path)
        data["scatter_params"] = np.load(scatter_path)

        # Optional files
        errors_path = dir_path / "errors_scatter.npy"
        if errors_path.exists():
            data["errors_scatter"] = np.load(errors_path)

        return data

    def _apply_center_crop(
        self, data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply center cropping to concentration data using torchvision CenterCrop.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing arrays to potentially crop.

        Returns
        -------
        dict
            Dictionary with cropped arrays. Only 'coef_list' is cropped if center_crop_size is set.
        """
        if self.inference_mode:
            # In inference mode, crop to ensure dimensions are divisible by 8
            h, w = data_dict["coef_list"].shape[:2]
            crop_h = h - (h % 8)
            crop_w = w - (w % 8)
            self.center_crop_size = (crop_h, crop_w)

        if self.center_crop_size is None:
            return data_dict

        cropped_data = data_dict.copy()

        # Only apply center crop to coef_list
        coef_list = data_dict["coef_list"]
        h, w = coef_list.shape[:2]
        crop_h, crop_w = self.center_crop_size

        # Check if cropping is necessary and possible
        if crop_h > h or crop_w > w:
            logger.warning(
                f"Warning: Requested crop size {self.center_crop_size} is than or equal to image size ({h}, {w}). No cropping applied."
            )
            return data_dict

        # Create CenterCrop transform
        center_crop = CenterCrop(self.center_crop_size)

        # Convert to tensor, apply transform, and convert back to numpy
        # Shape: (H, W, C) -> (C, H, W) -> crop -> (C, H, W) -> (H, W, C)
        coef_tensor = torch.from_numpy(coef_list).permute(2, 0, 1).float()  # (C, H, W)
        cropped_coef_tensor = center_crop(coef_tensor)  # (C, crop_h, crop_w)
        cropped_data["coef_list"] = cropped_coef_tensor.permute(
            1, 2, 0
        ).numpy()  # (crop_h, crop_w, C)

        # Also crop scatter_params and errors_scatter if they have the same spatial dimensions
        if "scatter_params" in data_dict:
            scatter_params = data_dict["scatter_params"]
            if scatter_params.shape[:2] == (h, w):
                scatter_tensor = (
                    torch.from_numpy(scatter_params).permute(2, 0, 1).float()
                )
                cropped_scatter_tensor = center_crop(scatter_tensor)
                cropped_data["scatter_params"] = cropped_scatter_tensor.permute(
                    1, 2, 0
                ).numpy()

        if "errors_scatter" in data_dict:
            errors_scatter = data_dict["errors_scatter"]
            if errors_scatter.shape[:2] == (h, w):
                # errors_scatter is 2D, add a channel dimension temporarily
                errors_tensor = (
                    torch.from_numpy(errors_scatter).unsqueeze(0).float()
                )  # (1, H, W)
                cropped_errors_tensor = center_crop(
                    errors_tensor
                )  # (1, crop_h, crop_w)
                cropped_data["errors_scatter"] = cropped_errors_tensor.squeeze(
                    0
                ).numpy()  # (crop_h, crop_w)

        return cropped_data

    def _load_normalization_stats(self) -> None:
        """
        Load normalization statistics (mean and std) for each channel from the normalization directory.

        Expected structure:
        normalization_dir/
        ├── gt/
        │   ├── mean_0.npy, std_0.npy
        │   ├── mean_1.npy, std_1.npy
        │   └── ...
        └── reduced_wl/
            ├── mean_0.npy, std_0.npy
            ├── mean_1.npy, std_1.npy
            └── ...

        Raises
        ------
        FileNotFoundError
            If normalization directory doesn't exist or required subdirectories/files are missing.
        ValueError
            If the normalization files don't contain valid data.
        """
        if not self.normalization_dir.exists():
            raise FileNotFoundError(
                f"Normalization directory not found: {self.normalization_dir}"
            )

        # Check for required subdirectories
        gt_dir = self.normalization_dir / "gt"
        reduced_wl_dir = self.normalization_dir / "reduced_wl"

        if not gt_dir.exists():
            raise FileNotFoundError(
                f"Ground truth normalization directory not found: {gt_dir}"
            )
        if not reduced_wl_dir.exists():
            raise FileNotFoundError(
                f"Reduced wavelength normalization directory not found: {reduced_wl_dir}"
            )

        # Load GT normalization statistics
        (
            self.normalization_coef_means_gt,
            self.normalization_coef_stds_gt,
            self.normalization_scatter_means_gt,
            self.normalization_scatter_stds_gt,
        ) = self._load_stats_from_dir(gt_dir, "ground truth")

        # Load reduced wavelength normalization statistics
        (
            self.normalization_coef_means_reduced_wl,
            self.normalization_coef_stds_reduced_wl,
            self.normalization_scatter_means_reduced_wl,
            self.normalization_scatter_stds_reduced_wl,
        ) = self._load_stats_from_dir(reduced_wl_dir, "reduced wavelength")

        logger.info(f"Loaded normalization statistics from {self.normalization_dir}")
        logger.info(
            f"GT: {len(self.normalization_coef_means_gt)} coef channels"
            + (
                f", {len(self.normalization_scatter_means_gt)} scatter channels"
                if self.normalization_scatter_means_gt is not None
                else ", no scatter channels"
            )
        )
        logger.info(
            f"Reduced WL: {len(self.normalization_coef_means_reduced_wl)} coef channels"
            + (
                f", {len(self.normalization_scatter_means_reduced_wl)} scatter channels"
                if self.normalization_scatter_means_reduced_wl is not None
                else ", no scatter channels"
            )
        )

    def _load_stats_from_dir(
        self, stats_dir: Path, data_type: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load mean and std statistics from a specific directory for both coef_list and scatter_params.

        Parameters
        ----------
        stats_dir : Path
            Directory containing mean_*.npy, std_*.npy, scatter_mean_*.npy and scatter_std_*.npy files.
        data_type : str
            Description of the data type (for error messages).

        Returns
        -------
        tuple of np.ndarray
            (coef_means, coef_stds, scatter_means, scatter_stds) arrays for all channels.

        Raises
        ------
        FileNotFoundError
            If required files are missing.
        ValueError
            If the files don't contain valid data.
        """
        # Get all mean files to determine the number of channels for coef_list
        mean_files = list(stats_dir.glob("mean_*.npy"))
        if not mean_files:
            raise FileNotFoundError(
                f"No mean_*.npy files found in {data_type} directory: {stats_dir}"
            )

        # Extract channel indices from filenames
        channel_indices = []
        for mean_file in mean_files:
            # Extract index from filename like "mean_5.npy"
            match = re.match(r"mean_(\d+)\.npy", mean_file.name)
            if match:
                channel_indices.append(int(match.group(1)))

        if not channel_indices:
            raise ValueError(
                f"Could not extract channel indices from mean files in {data_type} directory: {stats_dir}"
            )

        # Sort indices to ensure consistent ordering
        channel_indices.sort()
        n_channels = max(channel_indices) + 1

        # Initialize arrays for coef means and stds
        coef_means = np.zeros(n_channels, dtype=np.float32)
        coef_stds = np.ones(
            n_channels, dtype=np.float32
        )  # Default to 1 to avoid division by zero

        # Load mean and std for each channel (coef_list)
        loaded_channels = []
        for channel_idx in channel_indices:
            mean_path = stats_dir / f"mean_{channel_idx}.npy"
            std_path = stats_dir / f"std_{channel_idx}.npy"

            if not mean_path.exists():
                raise FileNotFoundError(f"Mean file not found: {mean_path}")
            if not std_path.exists():
                raise FileNotFoundError(f"Std file not found: {std_path}")

            try:
                mean_val = np.load(mean_path).item()  # Load scalar value
                std_val = np.load(std_path).item()  # Load scalar value

                # Validate loaded values
                if not np.isfinite(mean_val):
                    raise ValueError(f"Invalid mean value in {mean_path}: {mean_val}")
                if not np.isfinite(std_val) or std_val <= 0:
                    raise ValueError(f"Invalid std value in {std_path}: {std_val}")

                coef_means[channel_idx] = mean_val
                coef_stds[channel_idx] = std_val
                loaded_channels.append(channel_idx)

            except Exception as e:
                raise ValueError(
                    f"Error loading {data_type} coef normalization stats for channel {channel_idx}: {e}"
                )

        logger.debug(f"{data_type} - Loaded coef channels: {loaded_channels}")
        logger.debug(
            f"{data_type} - Coef means range: [{coef_means.min():.6f}, {coef_means.max():.6f}]"
        )
        logger.debug(
            f"{data_type} - Coef stds range: [{coef_stds.min():.6f}, {coef_stds.max():.6f}]"
        )

        # Load scatter_params statistics (optional)
        scatter_mean_files = list(stats_dir.glob("scatter_mean_*.npy"))

        # Extract scatter channel indices
        scatter_channel_indices = []
        for scatter_mean_file in scatter_mean_files:
            # Extract index from filename like "scatter_mean_0.npy"
            match = re.match(r"scatter_mean_(\d+)\.npy", scatter_mean_file.name)
            if match:
                scatter_channel_indices.append(int(match.group(1)))

            # Sort indices to ensure consistent ordering
            scatter_channel_indices.sort()
            n_scatter_channels = max(scatter_channel_indices) + 1

            # Initialize arrays for scatter means and stds
            scatter_means = np.zeros(n_scatter_channels, dtype=np.float32)
            scatter_stds = np.ones(
                n_scatter_channels, dtype=np.float32
            )  # Default to 1 to avoid division by zero

            # Load scatter mean and std for each channel
            loaded_scatter_channels = []

            for scatter_idx in scatter_channel_indices:
                scatter_mean_path = stats_dir / f"scatter_mean_{scatter_idx}.npy"
                scatter_std_path = stats_dir / f"scatter_std_{scatter_idx}.npy"

                scatter_mean_val = np.load(
                    scatter_mean_path
                ).item()  # Load scalar value
                scatter_std_val = np.load(scatter_std_path).item()  # Load scalar value
                scatter_means[scatter_idx] = scatter_mean_val
                scatter_stds[scatter_idx] = scatter_std_val
                loaded_scatter_channels.append(scatter_idx)

        # All scatter channels loaded successfully
        logger.debug(
            f"{data_type} - Loaded scatter channels: {loaded_scatter_channels}"
        )
        logger.debug(
            f"{data_type} - Scatter means range: [{scatter_means.min():.6f}, {scatter_means.max():.6f}]"
        )
        logger.debug(
            f"{data_type} - Scatter stds range: [{scatter_stds.min():.6f}, {scatter_stds.max():.6f}]"
        )

        return coef_means, coef_stds, scatter_means, scatter_stds

    def _apply_image_wise_normalization(self, coef_list: np.ndarray) -> np.ndarray:
        """
        Apply image-wise normalization to concentration coefficients.

        Parameters
        ----------
        coef_list : np.ndarray
            Concentration coefficients of shape (H, W, C) where C is the number of channels.

        Returns
        -------
        np.ndarray
            Normalized concentration coefficients of the same shape.
        """
        if not self.image_wise_normalization:
            return coef_list
        # Convert to tensor and permute to (C, H, W)
        coef_tensor = torch.from_numpy(coef_list).permute(2, 0, 1).float()  # (C, H, W)

        # Apply MONAI NormalizeIntensity
        normalized_tensor = self.image_normalization_transform(coef_tensor)

        # Permute back to (H, W, C) and convert to numpy
        return normalized_tensor.permute(1, 2, 0).numpy()  # (H, W, C)

    def _apply_global_normalization(
        self, data_dict: dict, data_type: str
    ) -> np.ndarray:
        """
        Apply channel-wise normalization to concentration coefficients.

        Performs (coef_list - mean) / std for each channel.
        Performs (scatter_params - mean) / std for each channel if scatter_params are present.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing the data to be normalized. Must include 'coef_list' and may include 'scatter_params'.
        data_type : str
            Type of data: 'gt' for ground truth or 'reduced_wl' for reduced wavelength.

        Returns
        -------
        np.ndarray
            Normalized concentration coefficients of the same shape.
        """
        if data_type == "gt":
            coef_means = self.normalization_coef_means_gt
            coef_stds = self.normalization_coef_stds_gt
            scatter_means = self.normalization_scatter_means_gt
            scatter_stds = self.normalization_scatter_stds_gt
        elif data_type == "reduced_wl":
            coef_means = self.normalization_coef_means_reduced_wl
            coef_stds = self.normalization_coef_stds_reduced_wl
            scatter_means = self.normalization_scatter_means_reduced_wl
            scatter_stds = self.normalization_scatter_stds_reduced_wl
        else:
            raise ValueError(
                f"Unknown data type: {data_type}. Must be 'gt' or 'reduced_wl'"
            )

        if coef_means is None or coef_stds is None or self.image_wise_normalization:
            return data_dict

        normalized_data_dict = data_dict.copy()
        coef_list = data_dict["coef_list"]

        # Normalize coef_list
        normalized_coef_list = coef_list.copy()
        n_channels = min(coef_list.shape[2], len(coef_means))

        for channel_idx in range(n_channels):
            normalized_coef_list[:, :, channel_idx] = (
                coef_list[:, :, channel_idx] - coef_means[channel_idx]
            ) / coef_stds[channel_idx]

        normalized_data_dict["coef_list"] = normalized_coef_list

        scatter_params = data_dict["scatter_params"]
        normalized_scatter_params = scatter_params.copy()
        n_scatter_channels = min(scatter_params.shape[2], len(scatter_means))

        for channel_idx in range(n_scatter_channels):
            normalized_scatter_params[:, :, channel_idx] = (
                scatter_params[:, :, channel_idx] - scatter_means[channel_idx]
            ) / scatter_stds[channel_idx]

        normalized_data_dict["scatter_params"] = normalized_scatter_params

        return normalized_data_dict

    def denormalize(
        self, normalized_coef_list: np.ndarray, data_type: str
    ) -> np.ndarray:
        """
        Reverse the normalization applied to concentration coefficients.

        Performs (normalized_coef_list * std) + mean for each channel.

        Parameters
        ----------
        normalized_coef_list : np.ndarray
            Normalized concentration coefficients of shape (H, W, C) where C is the number of channels.
        data_type : str
            Type of data: 'gt' for ground truth or 'reduced_wl' for reduced wavelength.

        Returns
        -------
        np.ndarray
            Denormalized concentration coefficients of the same shape.

        Raises
        ------
        ValueError
            If normalization is not enabled or normalization statistics are not available.
        """
        if data_type == "gt":
            means = self.normalization_coef_means_gt
            stds = self.normalization_coef_stds_gt
        elif data_type == "reduced_wl":
            means = self.normalization_coef_means_reduced_wl
            stds = self.normalization_coef_stds_reduced_wl
        else:
            raise ValueError(
                f"Unknown data type: {data_type}. Must be 'gt' or 'reduced_wl'"
            )

        if means is None or stds is None:
            raise ValueError(
                f"Cannot denormalize {data_type} data: normalization statistics not available"
            )

        denormalized = normalized_coef_list.copy()
        n_channels = min(normalized_coef_list.shape[2], len(means))

        for channel_idx in range(n_channels):
            denormalized[:, :, channel_idx] = (
                normalized_coef_list[:, :, channel_idx] * stds[channel_idx]
            ) + means[channel_idx]

        return denormalized

    def _apply_channels_first(
        self, data_dict: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """
        Apply channels-first permutation to concentration data.

        Converts arrays from HWC format to CHW format if channels_first is enabled.

        Parameters
        ----------
        data_dict : dict
            Dictionary containing arrays to potentially permute.
            Expected to contain 'coef_list' and optionally 'scatter_params'.

        Returns
        -------
        dict
            Dictionary with permuted arrays if channels_first is True, otherwise unchanged.
        """
        if not self.channels_first:
            return data_dict

        permuted_data = data_dict.copy()

        # Permute coef_list from (H, W, C) to (C, H, W)
        if "coef_list" in data_dict:
            coef_list = data_dict["coef_list"]
            if len(coef_list.shape) == 3:  # Ensure it's 3D (H, W, C)
                permuted_data["coef_list"] = np.transpose(coef_list, (2, 0, 1))

        # Permute scatter_params from (H, W, C) to (C, H, W) if present
        if "scatter_params" in data_dict:
            scatter_params = data_dict["scatter_params"]
            if len(scatter_params.shape) == 3:  # Ensure it's 3D (H, W, C)
                permuted_data["scatter_params"] = np.transpose(
                    scatter_params, (2, 0, 1)
                )

        return permuted_data

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> ConcentrationSample:
        """
        Retrieve a sample by index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        ConcentrationSample
            Sample dataclass containing:
            - id: Sample ID
            - patient_id: Patient ID (e.g., "S1.2")
            - fov: FOV number as string
            - reduced_wl: Dictionary with reduced wavelength concentration data
            - gt: Dictionary with ground truth concentration data

            Note: Both 'reduced_wl' and 'gt' dictionaries contain 'coef_list' arrays
            that are normalized using channel-wise statistics if normalization_dir was provided.
            If channels_first is True, arrays will be in CHW format instead of HWC format.
        """
        if idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        sample_info = self.samples[idx]

        # Load data from both directories
        reduced_wl_data = self._load_concentration_files(sample_info["reduced_wl_path"])
        gt_data = self._load_concentration_files(sample_info["gt_path"])

        # Apply center cropping if enabled
        reduced_wl_data = self._apply_center_crop(reduced_wl_data)
        gt_data = self._apply_center_crop(gt_data)

        # Apply global normalization if enabled
        reduced_wl_data = self._apply_global_normalization(
            reduced_wl_data, "reduced_wl"
        )
        gt_data = self._apply_global_normalization(gt_data, "gt")

        # Merge channels if specified
        reduced_wl_data["coef_list"] = self._merge_channels(reduced_wl_data)
        gt_data["coef_list"] = self._merge_channels(gt_data)

        # Apply normalization to coef_list
        reduced_wl_data["coef_list"] = self._apply_image_wise_normalization(
            reduced_wl_data["coef_list"],
        )
        gt_data["coef_list"] = self._apply_image_wise_normalization(
            gt_data["coef_list"],
        )

        # Apply channels-first permutation if enabled
        reduced_wl_data = self._apply_channels_first(reduced_wl_data)
        gt_data = self._apply_channels_first(gt_data)

        return ConcentrationSample(
            id=sample_info["id"],
            patient_id=sample_info["patient_id"],
            fov=sample_info["fov"],
            reduced_wl=reduced_wl_data,
            gt=gt_data,
        )

    def _merge_channels(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Merge channels from the input data dictionary into a single array.

        Parameters
        ----------
        data_dict : Dict[str, np.ndarray]
            Dictionary containing the input arrays to merge.

        Returns
        -------
        np.ndarray
            Merged array with shape (H, W, num_signals) if merge_channels_to_signals is specified,
            otherwise returns the original 'coef_list' array.
        """
        if self.merge_channels_to_signals is None:
            return data_dict["coef_list"]

        merged = np.zeros(
            (
                data_dict["coef_list"].shape[0],
                data_dict["coef_list"].shape[1],
                len(self.merge_channels_to_signals),
            )
        )
        for i, signal in enumerate(self.merge_channels_to_signals):
            if signal == Signal.HbT:
                merged[:, :, i] = (
                    data_dict["coef_list"][:, :, 0] + data_dict["coef_list"][:, :, 1]
                )  # HbO2 + Hb
            elif signal == Signal.diffCCO:
                merged[:, :, i] = (
                    data_dict["coef_list"][:, :, 2] - data_dict["coef_list"][:, :, 3]
                )  # oxCCO - redCCO
            elif signal == Signal.b:
                merged[:, :, i] = data_dict["scatter_params"][
                    :, :, -1
                ]  # scattering parameter b
            else:
                raise ValueError(f"Unknown signal for merging: {signal}")
        return merged

    def get_sample_by_id(self, sample_id: str) -> Union[ConcentrationSample, None]:
        """
        Retrieve a sample by its ID.

        Parameters
        ----------
        sample_id : str
            The sample ID (e.g., "1.2_3" for patient S1.2 FOV 3).

        Returns
        -------
        ConcentrationSample or None
            The sample data or None if not found.
        """
        if sample_id in self.sample_map:
            return self.__getitem__(self.sample_map[sample_id])

        print(f"Sample with ID '{sample_id}' not found.")
        return None

    def get_samples_by_patient_id(self, patient_id: str) -> List[ConcentrationSample]:
        """
        Retrieve all samples (FOVs) for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID. Format depends on sample type:
            - BIOPSY: "S1.2"
            - HELICOID: "004" (patient number only)

        Returns
        -------
        list of ConcentrationSample
            A list of all samples for the given patient, empty if none found.
        """
        samples = []

        if self.sample_type == SampleType.BIOPSY2:
            # Normalize the patient ID
            normalized_id = _normalize_id(patient_id)

            # Strip "S" prefix if present for matching with internal ID format
            if normalized_id.startswith("S"):
                patient_number = normalized_id[1:]
            else:
                patient_number = normalized_id

            for idx, sample_info in enumerate(self.samples):
                if sample_info["id"].startswith(f"{patient_number}_"):
                    samples.append(self.__getitem__(idx))

        elif self.sample_type == SampleType.HELICOID:
            # For HELICOID, match by patient ID
            for idx, sample_info in enumerate(self.samples):
                if sample_info["patient_id"] == patient_id:
                    samples.append(self.__getitem__(idx))

        if not samples:
            print(f"No samples found for patient ID '{patient_id}'.")

        return samples

    def get_sample_by_patient_and_fov(
        self, patient_id: str, fov: str
    ) -> Union[ConcentrationSample, None]:
        """
        Retrieve a specific sample by patient ID and FOV number.

        Parameters
        ----------
        patient_id : str
            The patient ID. Format depends on sample type:
            - BIOPSY: "S1.2"
            - HELICOID: "004" (patient number only)
        fov : str
            The FOV number as a string.

        Returns
        -------
        ConcentrationSample or None
            The sample data or None if not found.
        """
        if self.sample_type == SampleType.BIOPSY2:
            # Normalize the patient ID
            normalized_id = _normalize_id(patient_id)

            # Strip "S" prefix if present for matching with internal ID format
            if normalized_id.startswith("S"):
                patient_number = normalized_id[1:]
            else:
                patient_number = normalized_id

            sample_id = f"{patient_number}_{fov}"

        elif self.sample_type == SampleType.HELICOID:
            # For HELICOID, construct the full directory name
            # Ensure patient_id is 3 digits and fov is 2 digits
            patient_padded = patient_id.zfill(3)
            fov_padded = fov.zfill(2)
            sample_id = f"{patient_padded}-{fov_padded}"

        else:
            raise ValueError(f"Unknown sample type: {self.sample_type}")

        return self.get_sample_by_id(sample_id)

    def get_patient_ids(self) -> List[str]:
        """
        Get a list of all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        patient_ids = set(sample["patient_id"] for sample in self.samples)
        return sorted(patient_ids)

    def get_fovs_for_patient(self, patient_id: str) -> List[str]:
        """
        Get a list of FOV numbers available for a specific patient.

        Parameters
        ----------
        patient_id : str
            The patient ID. Format depends on sample type:
            - BIOPSY: "S1.2"
            - HELICOID: "004" (patient number only)

        Returns
        -------
        list of str
            Sorted list of FOV numbers for the patient.
        """
        fovs = []

        if self.sample_type == SampleType.BIOPSY2:
            # Normalize the patient ID
            normalized_id = _normalize_id(patient_id)

            # Strip "S" prefix if present for matching with internal ID format
            if normalized_id.startswith("S"):
                patient_number = normalized_id[1:]
            else:
                patient_number = normalized_id

            for sample_info in self.samples:
                if sample_info["id"].startswith(f"{patient_number}_"):
                    fovs.append(sample_info["fov"])

        elif self.sample_type == SampleType.HELICOID:
            # For HELICOID, match by patient number
            for sample_info in self.samples:
                if sample_info["patient_id"] == patient_id:
                    fovs.append(sample_info["fov"])

        return sorted(fovs)

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
            "sample_type": self.sample_type.value,
            "reduced_wl_dir": str(self.reduced_wl_dir),
            "gt_dir": str(self.gt_dir),
            "center_crop_size": self.center_crop_size,
            "normalization_enabled": self.normalization_dir is not None,
            "channels_first": self.channels_first,
            "merge_channels_to_signals": self.merge_channels_to_signals,
            "image_wise_normalization": self.image_wise_normalization,
        }

        if self.normalization_dir is not None:
            summary["normalization_dir"] = str(self.normalization_dir)
            summary["normalization_channels_gt"] = (
                len(self.normalization_coef_means_gt)
                if self.normalization_coef_means_gt is not None
                else 0
            )
            summary["normalization_channels_reduced_wl"] = (
                len(self.normalization_coef_means_reduced_wl)
                if self.normalization_coef_means_reduced_wl is not None
                else 0
            )

        return summary

    def get_unique_patient_numbers(self) -> List[str]:
        """
        Get a list of unique patient numbers (useful for HELICOID data).

        Returns
        -------
        list of str
            Sorted list of unique patient numbers.
        """
        if self.sample_type == SampleType.HELICOID:
            patient_numbers = set(sample["patient_number"] for sample in self.samples)
            return sorted(patient_numbers)
        else:
            # For biopsy data, extract patient numbers from patient IDs
            patient_numbers = set()
            for sample in self.samples:
                patient_id = sample["patient_id"]
                if patient_id.startswith("S"):
                    patient_numbers.add(patient_id[1:])
                else:
                    patient_numbers.add(patient_id)
            return sorted(patient_numbers)

    def set_inference_mode(self, inference: bool = True) -> None:
        """
        Enable or disable inference mode.

        In inference mode, center cropping is applied to ensure dimensions are divisible by 8.

        Parameters
        ----------
        inference : bool
            If True, enable inference mode. If False, disable it.
        """
        self.inference_mode = inference

    def get_normalization_info(
        self,
    ) -> Dict[str, Union[str, bool, np.ndarray, None]]:
        """
        Get information about the normalization settings.

        Returns
        -------
        dict
            Dictionary containing normalization information:
            - 'enabled': Whether normalization is enabled
            - 'normalization_dir': Path to normalization directory (if enabled)
            - 'gt_means': Array of mean values per channel for ground truth coef_list (if enabled)
            - 'gt_stds': Array of std values per channel for ground truth coef_list (if enabled)
            - 'reduced_wl_means': Array of mean values per channel for reduced wavelength coef_list (if enabled)
            - 'reduced_wl_stds': Array of std values per channel for reduced wavelength coef_list (if enabled)
            - 'gt_scatter_means': Array of mean values per channel for ground truth scatter_params (if enabled)
            - 'gt_scatter_stds': Array of std values per channel for ground truth scatter_params (if enabled)
            - 'reduced_wl_scatter_means': Array of mean values per channel for reduced wavelength scatter_params (if enabled)
            - 'reduced_wl_scatter_stds': Array of std values per channel for reduced wavelength scatter_params (if enabled)
            - 'num_channels_gt': Number of channels with GT coef normalization statistics
            - 'num_channels_reduced_wl': Number of channels with reduced WL coef normalization statistics
            - 'num_scatter_channels_gt': Number of channels with GT scatter normalization statistics
            - 'num_scatter_channels_reduced_wl': Number of channels with reduced WL scatter normalization statistics
        """

        info = {
            "enabled": self.normalization_dir is not None,
            "normalization_dir": (
                str(self.normalization_dir) if self.normalization_dir else None
            ),
            "gt_means": (
                self.normalization_coef_means_gt.copy()
                if self.normalization_coef_means_gt is not None
                else None
            ),
            "gt_stds": (
                self.normalization_coef_stds_gt.copy()
                if self.normalization_coef_stds_gt is not None
                else None
            ),
            "reduced_wl_means": (
                self.normalization_coef_means_reduced_wl.copy()
                if self.normalization_coef_means_reduced_wl is not None
                else None
            ),
            "reduced_wl_stds": (
                self.normalization_coef_stds_reduced_wl.copy()
                if self.normalization_coef_stds_reduced_wl is not None
                else None
            ),
            "gt_scatter_means": (
                self.normalization_scatter_means_gt.copy()
                if self.normalization_scatter_means_gt is not None
                else None
            ),
            "gt_scatter_stds": (
                self.normalization_scatter_stds_gt.copy()
                if self.normalization_scatter_stds_gt is not None
                else None
            ),
            "reduced_wl_scatter_means": (
                self.normalization_scatter_means_reduced_wl.copy()
                if self.normalization_scatter_means_reduced_wl is not None
                else None
            ),
            "reduced_wl_scatter_stds": (
                self.normalization_scatter_stds_reduced_wl.copy()
                if self.normalization_scatter_stds_reduced_wl is not None
                else None
            ),
            "num_channels_gt": (
                len(self.normalization_coef_means_gt)
                if self.normalization_coef_means_gt is not None
                else 0
            ),
            "num_channels_reduced_wl": (
                len(self.normalization_coef_means_reduced_wl)
                if self.normalization_coef_means_reduced_wl is not None
                else 0
            ),
            "num_scatter_channels_gt": (
                len(self.normalization_scatter_means_gt)
                if self.normalization_scatter_means_gt is not None
                else 0
            ),
            "num_scatter_channels_reduced_wl": (
                len(self.normalization_scatter_means_reduced_wl)
                if self.normalization_scatter_means_reduced_wl is not None
                else 0
            ),
        }
        return info
