import json
import os
import re
import typing
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import spectral
from spectral import open_image
from torch.utils.data import Dataset
from torchvision.transforms import CenterCrop
from loguru import logger

spectral.settings.envi_support_nonlowercase_params = True

from ..constants import (
    HELICOID_DATA_DIR,
    HELICOID_REFERENCE_PIXEL_DIR,
)
from .base_dataset import BaseHSIDataset
from .data_sample import HelicoidSample


class HelicoidDataset(BaseHSIDataset):
    """
    A PyTorch dataset class for HELICOID hyperspectral imaging data.

    This dataset loads hyperspectral cubes from .hdr files located in patient
    directories within the HELICOID data directory. Each sample corresponds to
    a single patient with their hyperspectral cube, ground truth map, and RGB image.
    """

    def __init__(
        self,
        helicoid_data_dir: typing.Union[str, Path] = HELICOID_DATA_DIR,
        left_cut: int = 530,
        right_cut: int = 750,
        coarseness: int = 1,
        normalize_image: bool = True,
        with_delta_A: bool = False,
        reference_pixel_strategy: str = "center_blood",
        downsample_factor: typing.Optional[
            typing.Union[int, typing.Tuple[int, int]]
        ] = None,
        center_crop_size: typing.Optional[
            typing.Union[int, typing.Tuple[int, int]]
        ] = None,
        transform: typing.Optional[typing.Callable] = None,
        with_rgb: bool = False,
        inference_mode: bool = False,
        ids_subset: typing.Optional[typing.List[int]] = None,
    ):
        """
        Initialize the HelicoidDataset.

        Parameters
        ----------
        helicoid_data_dir : str or Path, optional
            Path to the directory containing HELICOID patient folders.
            Defaults to `HELICOID_DATA_DIR`.
        left_cut : int, optional
            Lower wavelength bound (nm) for cropping the spectral axis.
            Bands at wavelengths >= `left_cut` will be included.
            Defaults to LEFT_CUT_HELICOID.
        right_cut : int, optional
            Upper wavelength bound (nm) for cropping the spectral axis.
            Bands at wavelengths < `right_cut` will be included.
            Defaults to RIGHT_CUT_HELICOID.
        coarseness : int, optional
            Spatial downsampling factor for both spatial dimensions.
            A coarseness > 1 reduces the spatial resolution.
            Defaults to 1 (no spatial downsampling).
        normalize_image : bool, optional
            Whether to normalize the raw HSI data using white and dark references.
            Defaults to True.
        with_delta_A : bool, optional
            If True, computes delta A (relative attenuation) by comparing to
            a reference pixel. Defaults to False.
        reference_pixel_strategy : str, optional
            Strategy for selecting reference pixel when computing delta A.
            Options: "center_blood", "mean_blood", "first_blood", "random_blood".
            Defaults to "center_blood".
        downsample_factor : int or tuple of two ints, optional
            Additional spatial downsampling factor for the H×W dimensions:
            - If None, no additional downsampling is applied.
            - If int f, applies average‐pooling with kernel=(f,f) and stride=(f,f).
            - If (fh, fw), applies average‐pooling with kernel=(fh,fw) and stride=(fh,fw).
            Applied after coarseness. Defaults to None.
        center_crop_size : int or tuple of int, optional
            Size for center cropping the spatial dimensions of the HSI cube and ground truth map.
            If int, crops to (center_crop_size, center_crop_size).
            If tuple, crops to (height, width).
            If None, no cropping is applied. Defaults to None.
        transform : callable, optional
            Optional transform to be applied on a sample.
            Should be a function that takes a sample dict and returns a transformed version.
            Defaults to None.
        with_rgb : bool, optional
            Whether to load and return RGB images for each sample. Defaults to False.
        inference_mode : bool, optional
            If True, dataset is set for inference, use full image cropped to be divisible by 8.
        indices : list of int, optional
            Optional list of indices to subset the dataset. If None, use all samples.
            Defaults to None.
        Raises
        ------
        FileNotFoundError
            If `helicoid_data_dir` does not exist.
        """
        self.helicoid_data_dir = Path(helicoid_data_dir)
        self.left_cut = left_cut
        self.right_cut = right_cut
        self.coarseness = coarseness
        self.normalize_image = normalize_image
        self.with_delta_A = with_delta_A
        self.reference_pixel_strategy = reference_pixel_strategy
        self.downsample_factor = downsample_factor
        self.transform = transform
        self.with_rgb = with_rgb
        self.inference_mode = inference_mode
        self.ids_subset = ids_subset

        # Set up center crop parameters
        if center_crop_size is not None:
            if isinstance(center_crop_size, int):
                self.center_crop_size = (center_crop_size, center_crop_size)
            else:
                self.center_crop_size = tuple(center_crop_size)
        else:
            self.center_crop_size = None

        if not self.helicoid_data_dir.exists():
            raise FileNotFoundError(
                f"HELICOID data directory not found: {self.helicoid_data_dir}"
            )

        # Find all patient samples
        self.samples = self._find_patient_samples()
        self.samples.sort(key=lambda x: x["id"])

        # Dictionary for faster sample lookups by id
        self.sample_map = {s["id"]: i for i, s in enumerate(self.samples)}

        # Cache for wavelengths (loaded from first sample)
        self._wavelengths = None
        self._cut_wavelengths = None
        self._left_cut_index = None
        self._right_cut_index = None

        # Initialize wavelength indices
        self._initialize_wavelength_indices()

        super().__init__()

    def _find_patient_samples(self) -> typing.List[dict]:
        """
        Scan the HELICOID data directory to find all patient samples.

        Returns
        -------
        list of dict
            List of dictionaries containing patient information.
        """
        samples = []

        # Pattern to match patient IDs like "004-02", "016-05", etc.
        pattern = re.compile(r"^\d{3}-\d{2}$")

        for item in self.helicoid_data_dir.iterdir():
            if item.is_dir() and pattern.match(item.name):
                id = item.name

                # Check if required files exist
                required_files = ["raw.hdr", "gtMap.hdr", "image.jpg"]
                if self.normalize_image:
                    required_files.extend(["whiteReference.hdr", "darkReference.hdr"])

                if all((item / file).exists() for file in required_files):
                    # Extract patient_id and fov from the directory name
                    # For "004-02": patient_id="004", fov="02"
                    parts = id.split("-")
                    patient_id = parts[0]  # "004"
                    fov = parts[1]  # "02"

                    if self.ids_subset is not None:
                        if id not in self.ids_subset:
                            continue

                    samples.append(
                        {
                            "id": id,
                            "patient_id": patient_id,
                            "fov": fov,
                            "patient_dir": item,
                            "raw_path": item / "raw.hdr",
                            "gt_path": item / "gtMap.hdr",
                            "rgb_path": item / "image.jpg",
                            "white_ref_path": (
                                item / "whiteReference.hdr"
                                if self.normalize_image
                                else None
                            ),
                            "dark_ref_path": (
                                item / "darkReference.hdr"
                                if self.normalize_image
                                else None
                            ),
                        }
                    )
                else:
                    logger.warning(
                        f"Warning: Patient {id} missing required files, skipping."
                    )

        logger.debug(f"Found {len(samples)} valid HELICOID patient samples")
        return samples

    def _initialize_wavelength_indices(self):
        """Initialize wavelength-related attributes by loading from the first sample."""
        if len(self.samples) == 0:
            raise ValueError("No samples found to initialize wavelengths")

        # Load wavelengths from the first sample
        first_sample = self.samples[0]
        hdr_img = open_image(first_sample["raw_path"])
        self._wavelengths = np.array(hdr_img.metadata["wavelength"]).astype(float)

        # Calculate cut indices
        self._left_cut_index = np.where(self._wavelengths >= self.left_cut)[0][0]
        right_cut_indices = np.where(self._wavelengths > self.right_cut)[0]

        if len(right_cut_indices) == 0:
            self._right_cut_index = len(self._wavelengths)
        else:
            self._right_cut_index = right_cut_indices[0]

        self._cut_wavelengths = self._wavelengths[
            self._left_cut_index : self._right_cut_index
        ]
        print(
            f"Initialized wavelengths: {len(self._cut_wavelengths)} bands in range [{self.left_cut}, {self.right_cut})"
        )

    @property
    def wavelengths(self) -> np.ndarray:
        """Get all wavelengths from the dataset."""
        return self._wavelengths

    @property
    def cut_wavelengths(self) -> np.ndarray:
        """Get wavelengths after applying left and right cuts."""
        return self._cut_wavelengths

    def __len__(self) -> int:
        """Return the number of patient samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> HelicoidSample:
        """
        Retrieve a single patient sample by its index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        HelicoidSample
            A dataclass containing:
            - id: Patient identifier (e.g., "004-02")
            - patient_id: Patient ID (e.g., "004")
            - fov: FOV number (e.g., "02")
            - hsi_cube: Hyperspectral cube data (C, H, W)
            - gt_map: Ground truth segmentation map (H, W)
            - rgb_image: RGB image as PIL Image
            - wavelengths: Wavelengths corresponding to the spectral dimension
            - delta_A: Delta attenuation (if with_delta_A=True), otherwise None
            - reference_pixel: Reference pixel coordinates (if with_delta_A=True), otherwise None
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Sample index out of range")

        sample_info = self.samples[idx]

        # Load HSI cube
        hsi_cube = self._load_hsi_cube(sample_info["raw_path"])

        # Apply spectral cropping
        hsi_cube = hsi_cube[:, :, self._left_cut_index : self._right_cut_index]

        # Apply spatial coarsening
        if self.coarseness > 1:
            hsi_cube = hsi_cube[:: self.coarseness, :: self.coarseness, :]

        # Convert to (C, H, W) format for PyTorch
        hsi_cube = np.transpose(hsi_cube, (2, 0, 1))

        # Apply additional downsampling if specified
        if self.downsample_factor is not None:
            if isinstance(self.downsample_factor, int):
                kh = kw = self.downsample_factor
            else:
                kh, kw = self.downsample_factor

            x = torch.from_numpy(hsi_cube).float().unsqueeze(0)  # (1, C, H, W)
            x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
            hsi_cube = x.squeeze(0).numpy()  # back to (C, H, W)

        # Load ground truth map
        gt_map = self._load_gt_map(sample_info["gt_path"])
        # Apply spatial coarsening to gt_map
        if self.coarseness > 1:
            gt_map = gt_map[:: self.coarseness, :: self.coarseness]

        # Apply additional downsampling to gt_map if specified
        if self.downsample_factor is not None:
            if isinstance(self.downsample_factor, int):
                kh = kw = self.downsample_factor
            else:
                kh, kw = self.downsample_factor

            # For ground truth, use nearest neighbor to preserve labels
            gt_tensor = (
                torch.from_numpy(gt_map).float().unsqueeze(0).unsqueeze(0)
            )  # (1, 1, H, W)
            gt_tensor = F.avg_pool2d(gt_tensor, kernel_size=(kh, kw), stride=(kh, kw))
            gt_map = gt_tensor.squeeze().numpy().astype(gt_map.dtype)

        # Load RGB image
        rgb_image = Image.open(sample_info["rgb_path"]) if self.with_rgb else None

        # Compute delta A if requested
        delta_A = None
        reference_pixel = None
        if self.with_delta_A:
            delta_A, reference_pixel = self.compute_delta_A(
                sample_info["id"], hsi_cube, gt_map
            )

        sample = HelicoidSample(
            id=sample_info["id"],
            patient_id=sample_info["patient_id"],
            fov=sample_info["fov"],
            hsi_cube=hsi_cube,
            gt_map=gt_map,
            rgb_image=rgb_image,
            wavelengths=self._cut_wavelengths,
            delta_A=delta_A,
            reference_pixel=reference_pixel,
        )

        # Apply center crop if specified
        sample = self._apply_center_crop(sample)

        # Apply transform if provided
        if self.transform:
            sample = self.transform(sample)

        return sample

    def compute_delta_A(self, id, hsi_cube, gt_map):
        reference_pixel = self._get_reference_pixel(gt_map, id)
        delta_A = None
        if reference_pixel is not None:
            # Convert back to (H, W, C) for reference pixel selection
            hsi_cube_hwc = np.transpose(hsi_cube, (1, 2, 0))
            reference_spectrum = hsi_cube_hwc[reference_pixel[0], reference_pixel[1], :]

            # Compute delta A
            hsi_cube_safe = np.where(hsi_cube_hwc <= 0, 1e-3, hsi_cube_hwc)
            delta_A = -np.log(hsi_cube_safe / reference_spectrum)
        return delta_A, reference_pixel

    def _load_hsi_cube(self, hdr_path: Path) -> np.ndarray:
        """
        Load and optionally normalize an HSI cube from an HDR file.

        Parameters
        ----------
        hdr_path : Path
            Path to the HDR file.

        Returns
        -------
        np.ndarray
            HSI cube data as (H, W, C) array.
        """
        hdr_img = open_image(hdr_path)
        hsi_cube = hdr_img.load()

        if self.normalize_image:
            # Get paths for white and dark references
            sample_info = None
            for sample in self.samples:
                if sample["raw_path"] == hdr_path:
                    sample_info = sample
                    break

            if (
                sample_info
                and sample_info["white_ref_path"]
                and sample_info["dark_ref_path"]
            ):
                hsi_cube = self._normalize_hsi_cube(
                    hsi_cube,
                    sample_info["white_ref_path"],
                    sample_info["dark_ref_path"],
                )

        return hsi_cube

    def _normalize_hsi_cube(
        self, hsi_cube: np.ndarray, white_ref_path: Path, dark_ref_path: Path
    ) -> np.ndarray:
        """
        Normalize HSI cube using white and dark references.

        Parameters
        ----------
        hsi_cube : np.ndarray
            Raw HSI cube data.
        white_ref_path : Path
            Path to white reference HDR file.
        dark_ref_path : Path
            Path to dark reference HDR file.

        Returns
        -------
        np.ndarray
            Normalized HSI cube.
        """
        white_ref = open_image(white_ref_path).load()
        dark_ref = open_image(dark_ref_path).load()

        # Tile references to match full image dimensions
        white_full = np.tile(white_ref, (hsi_cube.shape[0], 1, 1))
        dark_full = np.tile(dark_ref, (hsi_cube.shape[0], 1, 1))

        hsi_normalized = (hsi_cube - dark_full) / (white_full - dark_full) + 0.1
        hsi_normalized[hsi_normalized <= 0] = 1e-2

        return hsi_normalized

    def _load_gt_map(self, gt_path: Path) -> np.ndarray:
        """
        Load ground truth map from HDR file.

        Parameters
        ----------
        gt_path : Path
            Path to the ground truth HDR file.

        Returns
        -------
        np.ndarray
            Ground truth map as (H, W) or (H, W, 1) array.
        """
        gt_img = open_image(gt_path)
        return gt_img.load()

    def _get_reference_pixel(
        self,
        gt_map: np.ndarray,
        sample_id: typing.Optional[str] = None,
        reference_pixel_path: typing.Optional[typing.Union[str, Path]] = None,
    ) -> typing.Optional[typing.Tuple[int, int]]:
        """
        Get reference pixel coordinates based on the specified strategy.

        Parameters
        ----------
        gt_map : np.ndarray
            Ground truth map.
        sample_id : str, optional
            The sample ID for loading reference pixel from file if no blood pixels found.
        reference_pixel_path : str or Path, optional
            Path to directory containing reference pixel JSON files.
            If not provided, defaults to data/helicoid_reference_pixel/.

        Returns
        -------
        tuple of int or None
            (row, col) coordinates of reference pixel, or None if no blood pixels found
            and no reference pixel file available.
        """
        # Find blood pixels (assuming label 3 is blood, as in helicoid.py)
        blood_pixels = np.argwhere(gt_map == 3)

        if len(blood_pixels) == 0:
            logger.warning(
                "Warning: No blood pixels found for reference pixel selection"
            )

            # Try to load reference pixel from file as fallback
            if sample_id is not None:
                if reference_pixel_path is None:
                    # Default to the helicoid_reference_pixel directory
                    reference_pixel_path = HELICOID_REFERENCE_PIXEL_DIR

                reference_pixel = self._load_reference_pixel_from_path(
                    sample_id, reference_pixel_path
                )
                if reference_pixel is not None:
                    logger.info(
                        f"Using reference pixel from file for sample {sample_id}: {reference_pixel}"
                    )
                    return reference_pixel

            return None

        if self.reference_pixel_strategy == "center_blood":
            # Select the center blood pixel
            idx = len(blood_pixels) // 2
            return tuple(blood_pixels[idx][:2])
        elif self.reference_pixel_strategy == "random_blood":
            # Select a random blood pixel using the active NumPy RNG state
            idx = np.random.randint(len(blood_pixels))
            return tuple(blood_pixels[idx][:2])
        elif self.reference_pixel_strategy == "first_blood":
            # Select the first blood pixel
            return tuple(blood_pixels[0][:2])
        elif self.reference_pixel_strategy == "mean_blood":
            # Select the pixel closest to the mean position of blood pixels
            mean_pos = np.mean(blood_pixels, axis=0)[:2]
            distances = np.sum((blood_pixels[:, :2] - mean_pos) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            return tuple(blood_pixels[closest_idx][:2])
        else:
            raise ValueError(
                f"Unknown reference pixel strategy: {self.reference_pixel_strategy}"
            )

    def get_sample_by_id(self, id: str) -> typing.Optional[HelicoidSample]:
        """
        Retrieve a sample by its patient ID.

        Parameters
        ----------
        id : str
            The patient ID (e.g., "004-02").

        Returns
        -------
        HelicoidSample or None
            The sample data or None if not found.
        """
        if id in self.sample_map:
            return self.__getitem__(self.sample_map[id])

        logger.warning(f"Patient with ID '{id}' not found")
        return None

    def get_patient_ids(self) -> typing.List[str]:
        """
        Get all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            List of all unique patient IDs (e.g., ["004", "016", ...]).
        """
        return self.get_unique_patient_ids()

    def get_sample_ids(self) -> typing.List[str]:
        """
        Get all sample IDs in the dataset.

        Returns
        -------
        list of str
            List of all sample IDs (e.g., ["004-01", "004-02", "016-05", ...]).
        """
        return [sample["id"] for sample in self.samples]

    def get_samples_by_patient_id(self, patient_id: str) -> typing.List[HelicoidSample]:
        """
        Retrieve all samples for a specific patient ID.

        For HelicoidDataset, this will return all FOVs for a given patient.
        For example, for patient_id="004", this might return samples for
        "004-01", "004-02", etc.

        Parameters
        ----------
        patient_id : str
            The patient ID to get samples for (e.g., "004").

        Returns
        -------
        list of HelicoidSample
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
    ) -> typing.Union[HelicoidSample, None]:
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
        HelicoidSample or None
            The sample data or None if not found.
        """
        # Construct the full sample ID
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

    def get_unique_patient_ids(self) -> typing.List[str]:
        """
        Get a list of unique patient IDs (without FOV) in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        patient_ids = set(sample["patient_id"] for sample in self.samples)
        return sorted(patient_ids)

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
            "left_cut": self.left_cut,
            "right_cut": self.right_cut,
            "coarseness": self.coarseness,
            "normalize_image": self.normalize_image,
            "with_delta_A": self.with_delta_A,
            "center_crop_size": self.center_crop_size,
        }

        return summary

    def get_label_statistics(self) -> dict:
        """
        Compute statistics about ground truth labels across all samples.

        Returns
        -------
        dict
            Dictionary with label statistics.
        """
        label_counts = {}
        total_pixels = 0

        for idx in range(len(self)):
            sample = self.__getitem__(idx)
            gt_map = sample["gt_map"]

            unique_labels, counts = np.unique(gt_map, return_counts=True)

            for label, count in zip(unique_labels, counts):
                if label not in label_counts:
                    label_counts[label] = 0
                label_counts[label] += count
                total_pixels += count

        # Convert to percentages
        label_percentages = {
            label: (count / total_pixels) * 100 for label, count in label_counts.items()
        }

        return {
            "label_counts": label_counts,
            "label_percentages": label_percentages,
            "total_pixels": total_pixels,
        }

    def load_delta_c_and_scatter_params(
        self, id: str, load_from_path: str
    ) -> typing.Tuple[typing.Optional[np.ndarray], typing.Optional[np.ndarray]]:
        """
        Load delta C and scatter parameters from .npy files for a given HELICOID sample.

        Parameters
        ----------
        id : str
            The ID in HELICOID format (e.g., "004-02", "016-05").
        load_from_path : str
            Path to the directory containing the results of the spectral unmixing.

        Returns
        -------
        tuple of np.ndarray or None
            (coef_list, scatter_params), or (None, None) if not found.
            Expects directory structure: load_from_path/patient_id/coef_list.npy
                                                           /patient_id/scatter_params.npy
        """
        # Direct path to patient directory
        patient_dir = os.path.join(load_from_path, id)

        if not os.path.isdir(patient_dir):
            logger.warning(f"Patient directory not found: {patient_dir}")
            return None, None

        # Direct paths to required files
        coef_path = os.path.join(patient_dir, "coef_list.npy")
        scatter_path = os.path.join(patient_dir, "scatter_params.npy")

        # Check if files exist
        if not os.path.exists(coef_path):
            logger.warning(f"coef_list.npy not found in {patient_dir}")
            return None, None

        if not os.path.exists(scatter_path):
            logger.warning(f"scatter_params.npy not found in {patient_dir}")
            return None, None

        try:
            # Load the files
            coef_list = np.load(coef_path)
            scatter_params = np.load(scatter_path)

            logger.debug(f"Successfully loaded delta C and scatter params for {id}")
            return coef_list, scatter_params

        except Exception as e:
            logger.error(f"Error loading files for patient {id}: {e}")
            return None, None

    def load_reference_params(
        self, id: str, load_from_path: str
    ) -> typing.Optional[typing.Tuple[np.ndarray, np.ndarray]]:
        """
        Load the reference parameters (a_t1, b_t1) for a given HELICOID sample.

        Parameters
        ----------
        id : str
            The patient ID in HELICOID format (e.g., "004-02", "016-05").
        load_from_path : str
            Path to the directory containing the reference parameter files.

        Returns
        -------
        tuple of np.ndarray or None
            (a_t1, b_t1) as numpy arrays, or None if not found.
            Expects directory structure: load_from_path/id/a_t1.npy
                                                           /id/b_t1.npy
        """
        # Direct path to patient directory
        patient_dir = os.path.join(load_from_path, id)

        if not os.path.isdir(patient_dir):
            logger.warning(f"Patient directory not found: {patient_dir}")
            return None

        # Direct paths to reference parameter files
        a_t1_path = os.path.join(patient_dir, "a_t1.npy")
        b_t1_path = os.path.join(patient_dir, "b_t1.npy")

        # Check if files exist
        if not os.path.exists(a_t1_path):
            logger.warning(f"a_t1.npy not found in {patient_dir}")
            return None

        if not os.path.exists(b_t1_path):
            logger.warning(f"b_t1.npy not found in {patient_dir}")
            return None

        try:
            # Load the files
            a_t1 = np.load(a_t1_path)
            b_t1 = np.load(b_t1_path)

            logger.debug(f"Successfully loaded reference parameters for {id}")
            return a_t1, b_t1

        except Exception as e:
            logger.error(f"Error loading reference parameters for patient {id}: {e}")
            return None

    def _load_reference_pixel_from_path(
        self, id: str, reference_pixel_path: typing.Union[str, Path]
    ) -> typing.Optional[typing.Tuple[int, int]]:
        """
        Load reference pixel coordinates from a JSON file.

        Parameters
        ----------
        id : str
            The patient ID (e.g., "004-02").
        reference_pixel_path : str or Path
            Path to the directory containing reference pixel JSON files.

        Returns
        -------
        tuple of int or None
            (row, col) coordinates of reference pixel, or None if file not found or invalid.
            Note: JSON stores {"x": col, "y": row}, but we return (row, col).
        """
        reference_pixel_path = Path(reference_pixel_path)
        json_file_path = reference_pixel_path / f"{id}.json"

        if not json_file_path.exists():
            logger.warning(f"Reference pixel file not found: {json_file_path}")
            return None

        try:
            with open(json_file_path, "r") as f:
                data = json.load(f)

            # JSON stores as {"x": col, "y": row}, convert to (row, col)
            x = data.get("x")  # This is column
            y = data.get("y")  # This is row

            if x is None or y is None:
                logger.warning(
                    f"Invalid reference pixel data in {json_file_path}: missing x or y"
                )
                return None

            # Return as (row, col) to match the format expected by the rest of the code
            return (y, x)

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"Error loading reference pixel from {json_file_path}: {e}")
            return None

    def plot_sample_with_reference_pixel(self, id: str) -> None:
        """
        Plot the RGB image with reference pixel and its spectral signature.

        Parameters
        ----------
        id : str
            The patient ID (e.g., "004-02").

        Raises
        ------
        ValueError
            If the sample ID is not found in the dataset.
        """
        # Get the sample
        sample = self.get_sample_by_id(id)
        if sample is None:
            raise ValueError(f"Sample with ID '{id}' not found")

        # Get sample info for RGB path
        sample_info = None
        for s in self.samples:
            if s["id"] == id:
                sample_info = s
                break

        if sample_info is None:
            raise ValueError(f"Sample info for ID '{id}' not found")

        # Load RGB image
        rgb_image = Image.open(sample_info["rgb_path"])
        rgb_array = np.array(rgb_image)

        reference_pixel = sample["reference_pixel"]

        if reference_pixel is None:
            logger.warning(f"No reference pixel found for sample {id}")
            return

        # Get the HSI cube and extract spectrum at reference pixel
        hsi_cube = sample["hsi_cube"]  # Shape: (C, H, W)
        wavelengths = sample["wavelengths"]

        # Convert HSI cube back to (H, W, C) for indexing
        hsi_cube_hwc = np.transpose(hsi_cube, (1, 2, 0))

        # Extract spectrum at reference pixel
        row, col = reference_pixel
        reference_spectrum = hsi_cube_hwc[row, col, :]

        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Left subplot: RGB image with reference pixel marked
        ax1.imshow(rgb_array)

        # Calculate scaling factors
        rgb_height, rgb_width = rgb_array.shape[:2]

        # Mark the reference pixel with a red cross
        ax1.scatter(col, row, color="red", s=100, edgecolor=[], linewidth=2, marker="x")
        ax1.grid(False)
        ax1.set_title(f"RGB Image - Sample {id}\nReference Pixel: ({row}, {col})")

        # Right subplot: Spectral signature
        ax2.plot(wavelengths, reference_spectrum, "b-", linewidth=2)
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Intensity")
        ax2.set_title(f"Spectral Signature at Reference Pixel\nSample {id}")
        ax2.grid(True, alpha=0.3)

        # Add some styling
        ax2.set_xlim(wavelengths[0], wavelengths[-1])

        plt.tight_layout()
        plt.show()

        # Log information about the reference pixel
        logger.info(f"Plotted sample {id}:")
        logger.info(f"  Reference pixel (processed coordinates): ({row}, {col})")
        logger.info(
            f"  Spectral range: {wavelengths[0]:.1f} - {wavelengths[-1]:.1f} nm"
        )
        logger.info(f"  Number of spectral bands: {len(wavelengths)}")

    def _apply_center_crop(self, sample: HelicoidSample) -> HelicoidSample:
        """
        Apply center cropping to the HSI cube and ground truth map using torchvision CenterCrop.

        Parameters
        ----------
        sample : HelicoidSample
            Sample dataclass containing hsi_cube and gt_map.

        Returns
        -------
        HelicoidSample
            Dataclass with cropped arrays.
        """
        if self.center_crop_size is None:
            return sample

        # Get the HSI cube and gt_map
        hsi_cube = sample.hsi_cube  # Shape: (C, H, W)
        gt_map = sample.gt_map  # Shape: (H, W)

        # If gt_map is (H, W, 1), convert to (H, W)
        if len(gt_map.shape) == 3 and gt_map.shape[2] == 1:
            gt_map = gt_map.squeeze(-1)  # Remove the singleton dimension to get (H, W)

        # Get current dimensions
        c, h, w = hsi_cube.shape
        if self.inference_mode:
            # In inference mode, crop to largest size divisible by 8
            new_h = (h // 8) * 8
            new_w = (w // 8) * 8
            crop_h, crop_w = new_h, new_w
        else:
            crop_h, crop_w = self.center_crop_size

        # Check if cropping is necessary and possible
        if crop_h > h or crop_w > w:
            logger.warning(
                f"Warning: Requested crop size {self.center_crop_size} is larger than image size ({h}, {w}). No cropping applied."
            )
            return sample

        # Create CenterCrop transform
        center_crop = CenterCrop((crop_h, crop_w))

        # Apply center crop to HSI cube
        hsi_tensor = torch.from_numpy(hsi_cube).float()  # (C, H, W)
        cropped_hsi_tensor = center_crop(hsi_tensor)  # (C, crop_h, crop_w)
        sample.hsi_cube = cropped_hsi_tensor.numpy()

        # Apply center crop to ground truth map
        gt_tensor = torch.from_numpy(gt_map).float().unsqueeze(0)  # (1, H, W)
        cropped_gt_tensor = center_crop(gt_tensor)  # (1, crop_h, crop_w)
        sample.gt_map = cropped_gt_tensor.squeeze(0).numpy().astype(gt_map.dtype)

        # Also crop delta_A if it exists
        if sample.delta_A is not None:
            delta_A = sample.delta_A  # Shape: (H, W, C)
            # Convert to (C, H, W) for cropping
            delta_A_tensor = (
                torch.from_numpy(delta_A).permute(2, 0, 1).float()
            )  # (C, H, W)
            cropped_delta_A_tensor = center_crop(delta_A_tensor)  # (C, crop_h, crop_w)
            sample.delta_A = cropped_delta_A_tensor.permute(
                1, 2, 0
            ).numpy()  # (crop_h, crop_w, C)

        return sample
