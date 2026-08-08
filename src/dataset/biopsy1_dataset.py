from torch.utils.data import Dataset
from ..constants import HYPERPROBE_DATA_DIR
from .base_dataset import BaseHSIDataset
import numpy as np
import os
import re
import typing
from pathlib import Path


class Biopsy1Dataset(BaseHSIDataset):
    """
    Dataset class for loading and processing hyperprobe biopsy data.

    This class reads preprocessed hyperspectral data files from a specified directory,
    and provides methods to access them by ID or index.
    """

    def __init__(
        self,
        data_dir: str = HYPERPROBE_DATA_DIR,
        left_cut: int = 510,
        right_cut: int = 900,
        coarseness: int = 1,
        reference_spectrum_path: str = "../data/reference/S1_mean_roi.npy",
    ):
        """
        Initialize the HyperprobeDataset.

        Parameters
        ----------
        data_dir : str
            Directory containing the hyperprobe data files.
        left_cut : int
            Left wavelength cutoff in nm.
        right_cut : int
            Right wavelength cutoff in nm.
        """
        self.data_dir = data_dir
        self.left_cut = left_cut
        self.right_cut = right_cut
        self.coarseness = coarseness
        self.reference_spectrum = np.load(reference_spectrum_path)

        # Create wavelength array based on the specified range
        self.wavelengths = np.linspace(
            510,
            900,
            79,  # wl every 5 nm, max range is 510 - 900
        )

        # Find the index positions corresponding to the wavelength cuts
        self.left_cut_index = np.where(self.wavelengths >= self.left_cut)[0][0]
        right_cut_indices = np.where(self.wavelengths > self.right_cut)[0]

        # Handle right cut differently since upper bound is exclusive while lower bound is inclusive
        if len(right_cut_indices) == 0:
            self.right_cut_index = len(self.wavelengths)
        else:
            self.right_cut_index = right_cut_indices[0]

        self.cut_wavelengths = self.wavelengths[
            self.left_cut_index : self.right_cut_index
        ]

        # Load all available samples
        self.samples = self._load_data_files()
        self.samples.sort(key=lambda x: x["id"])

        # Dictionary for faster sample lookups by id
        self.sample_map = {s["id"]: i for i, s in enumerate(self.samples)}

        # ensure proper Dataset behavior
        super().__init__()

    def _load_data_files(self) -> list:
        """
        Load the data files from the specified directory.

        Returns
        -------
        list
            List of dictionaries containing sample information.
        """
        samples = []
        # Two patterns: one for standard samples and one for samples with FOV
        standard_pattern = re.compile(r"^Biopsy_(S\d+)_reflectance_preprocessed\.npy$")
        fov_pattern = re.compile(
            r"^Biopsy_(S\d+)_fov(\d+)_reflectance_preprocessed\.npy$"
        )

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            # First try to match the FOV pattern
            fov_match = fov_pattern.match(filename)
            if fov_match:
                sample_base_id = fov_match.group(1)  # e.g., 'S4'
                fov_num = fov_match.group(2)  # e.g., '1'
                sample_id = f"{sample_base_id}_{fov_num}"  # e.g., 'S4_1'
                file_path = os.path.join(self.data_dir, filename)

                samples.append(
                    {
                        "id": sample_id,
                        "file_path": file_path,
                    }
                )
                continue

            # If not a FOV pattern, try the standard pattern
            std_match = standard_pattern.match(filename)
            if std_match:
                sample_id = std_match.group(1)
                file_path = os.path.join(self.data_dir, filename)

                samples.append(
                    {
                        "id": sample_id,
                        "file_path": file_path,
                    }
                )

        if not samples:
            raise ValueError(f"No valid data files found in {self.data_dir}")

        return samples

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a sample by its index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing sample id and HSI cube data.
        """
        if idx < 0 or idx >= len(self.samples):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self.samples)}"
            )

        sample_info = self.samples[idx]

        # Lazy loading of HSI cube data
        hsi_cube = self._load_hsi_cube(sample_info["file_path"])

        delta_A = self._compute_delta_A(hsi_cube)

        return {
            "id": sample_info["id"],
            "file_path": sample_info["file_path"],
            "hsi_cube": hsi_cube,
            "delta_A": delta_A,
        }

    def _load_hsi_cube(self, file_path: str) -> np.ndarray:
        """
        Load a hyperspectral cube from a file.

        Parameters
        ----------
        file_path : str
            Path to the HSI cube file.

        Returns
        -------
        np.ndarray
            The loaded HSI cube data.
        """
        try:
            # Load the numpy array from the .npy file
            hsi_cube = np.load(file_path)

            # Apply wavelength cut if needed
            if self.left_cut_index > 0 or self.right_cut_index < hsi_cube.shape[-1]:
                hsi_cube = hsi_cube[..., self.left_cut_index : self.right_cut_index]

            return hsi_cube

        except Exception as e:
            raise IOError(f"Error loading HSI cube from {file_path}: {str(e)}")

    def get_sample_by_id(self, id: str) -> typing.Union[dict, None]:
        """
        Retrieve a sample by its ID.

        Parameters
        ----------
        id : str
            ID of the sample to retrieve.

        Returns
        -------
        dict or None
            Sample dictionary if found, None otherwise.
        """
        if id in self.sample_map:
            return self.__getitem__(self.sample_map[id])
        return None

    def _compute_delta_A(self, hsi_cube: np.ndarray) -> np.ndarray:
        """
        Compute relative image from absolute image and reference spectrum, apply coarseness.

        Returns:
            NDArray: Relative image (delta A)
        """
        # Set all zero values to 1e-3 to avoid log(0)
        hsi_cube = np.where(hsi_cube <= 0, 1e-3, hsi_cube)
        delta_A = -np.log(
            hsi_cube[
                :: self.coarseness,
                :: self.coarseness,
                :,
            ]
            / self.reference_spectrum
        )
        return delta_A[:, :, self.left_cut_index : self.right_cut_index]

    def load_delta_c_and_scatter_params(
        self, id: str, load_from_path: str
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Load delta C and scatter parameters from .npy files for a given sample.

        Parameters
        ----------
        id : str
            The ID in the format "S_patient_number_fov_number" e.g. "S1" or "S4_1".
        load_from_path : str
            Path to the directory containing the results of the spectral unmixing.

        Returns
        -------
        tuple of np.ndarray
            (coef_list, scatter_params), or (None, None) if not found.
            Looks for:
            - coef_list.npy
            - scatter_params.npy OR scattering_params.npy
        """
        # Build the base directory name
        if "_" in id:
            sample_id, fov_num = id.split("_")
            base_dir = f"Biopsy_{sample_id}_fov{fov_num}"
        else:
            base_dir = f"Biopsy_{id}"

        # Look for the right folder suffix
        for suffix in ("reflectance_preprocessed", "reflectance"):
            dir_path = os.path.join(load_from_path, f"{base_dir}_{suffix}")
            if os.path.isdir(dir_path):
                break
        else:
            print(f"Error: No directory found for {id} (tried "
                f"{base_dir}_reflectance_preprocessed and {base_dir}_reflectance)")
            return None, None

        # Paths for coef_list and scatter
        coef_path = os.path.join(dir_path, "coef_list.npy")

        # Try both possible scatter filenames
        scatter_candidates = ["scatter_params.npy", "scattering_params.npy"]
        scatter_path = None
        for fname in scatter_candidates:
            p = os.path.join(dir_path, fname)
            if os.path.exists(p):
                scatter_path = p
                break

        if not os.path.exists(coef_path) or scatter_path is None:
            missing = []
            if not os.path.exists(coef_path):
                missing.append("coef_list.npy")
            if scatter_path is None:
                missing.append("scatter_params.npy or scattering_params.npy")
            print(f"Error: {', '.join(missing)} not found in {dir_path}")
            return None, None

        # Load and return
        coef_list = np.load(coef_path)
        scatter_params = np.load(scatter_path)
        return coef_list, scatter_params

    def get_patient_ids(self) -> typing.List[str]:
        """
        Get a list of all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        raise NotImplementedError("get_patient_ids() method not yet implemented for Biopsy1Dataset")

    def get_samples_by_patient_id(self, patient_id: str) -> typing.List[dict]:
        """
        Retrieve all samples for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID to get samples for.

        Returns
        -------
        list of dict
            A list of all samples for the given patient, empty if none found.
        """
        raise NotImplementedError("get_samples_by_patient_id() method not yet implemented for Biopsy1Dataset")
