import pandas as pd
import os
import re
from scipy.io import loadmat
import numpy as np
import typing
import h5py
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

from ..constants import (
    HSI_DATA_DIR,
    METADATA_CSV_PATH,
    HSI_PREPROCESSED_DIR,
    WAVELENGTHS,
)
from .base_dataset import BaseHSIDataset


class Biopsy2Dataset(BaseHSIDataset):
    """
    A dataset class to load and manage HSI biopsy data.

    It reads hyperspectral cubes from .mat files located in a specified directory
    and associates them with metadata from a provided CSV file.
    Each sample is a single HSI cube, making it memory-efficient for training.
    """

    def __init__(
        self,
        hsi_data_dir: str = HSI_DATA_DIR,
        metadata_csv_path: str = METADATA_CSV_PATH,
        left_cut: int = 385,
        right_cut: int = 1015,
        coarseness: int = 1,
        reference_spectrum_path: str = "../data/reference/S1_4_1_mean_roi.npy",
        use_preprocessed: bool = True,
        with_delta_A: bool = False,
        downsample_factor: typing.Optional[
            typing.Union[int, typing.Tuple[int, int]]
        ] = None,
    ):
        """
        Initializes the Biopsy2Dataset.

        Parameters
        ----------
        hsi_data_dir : str, optional
            Path to the directory containing raw HSI `.mat` files.
            Defaults to `HSI_DATA_DIR`.
        metadata_csv_path : str, optional
            Path to the processed metadata CSV file.
            Defaults to `METADATA_CSV_PATH`.
        left_cut : int, optional
            Lower wavelength bound (nm) for cropping the spectral axis.
            Bands at wavelengths >= `left_cut` will be included.
            Defaults to 385.
        right_cut : int, optional
            Upper wavelength bound (nm) for cropping the spectral axis.
            Bands at wavelengths < `right_cut` will be included.
            Defaults to 1015.
        coarseness : int, optional
            Step size (in number of bands) between adjacent spectral channels.
            A coarseness > 1 subsamples the spectral axis.
            Defaults to 1 (no spectral subsampling).
        reference_spectrum_path : str, optional
            Path to a NumPy `.npy` file containing the reference spectrum used
            for ΔA computation. Only bands in the [left_cut, right_cut) range
            are loaded.
            Defaults to `"../data/reference/S1_4_1_mean_roi.npy"`.
        use_preprocessed : bool, optional
            If True, loads preprocessed HSI data from `HSI_PREPROCESSED_DIR`;
            otherwise loads raw data from `hsi_data_dir`.
            Defaults to True.
        with_delta_A : bool, optional
            If True, computes ΔA for each cube by comparing to `reference_spectrum`.
            Defaults to False.
        downsample_factor : int or tuple of two ints, optional
            Spatial downsampling factor for the H×W dimensions:
            - If None, no downsampling is applied.
            - If int f, applies average‐pooling with kernel=(f,f) and stride=(f,f).
            - If (fh, fw), applies average‐pooling with kernel=(fh,fw) and stride=(fh,fw).
            Defaults to None.

        Raises
        ------
        FileNotFoundError
            If `metadata_csv_path` does not exist (caught internally, leaves
            `metadata_df` empty).
        """
        self.use_preprocessed = use_preprocessed
        self.coarseness = coarseness
        self.left_cut = left_cut
        self.right_cut = right_cut
        self.with_delta_A = with_delta_A
        self.downsample_factor = downsample_factor

        self.left_cut_index = np.where(WAVELENGTHS >= self.left_cut)[0][0]
        right_cut_indices = np.where(WAVELENGTHS > self.right_cut)[0]

        # handle right cut differently since upper bound is exclusive while lower bound is inclusive
        if len(right_cut_indices) == 0:
            self.right_cut_index = len(WAVELENGTHS)  # set to the end if no cut
        else:
            self.right_cut_index = right_cut_indices[0]

        self.cut_wavelengths = WAVELENGTHS[self.left_cut_index : self.right_cut_index]

        self.reference_spectrum = np.load(reference_spectrum_path)[
            self.left_cut_index : self.right_cut_index
        ]
        if use_preprocessed:
            self.hsi_data_dir = HSI_PREPROCESSED_DIR
        else:
            self.hsi_data_dir = hsi_data_dir
        try:
            self.metadata_df = pd.read_csv(metadata_csv_path)
        except FileNotFoundError:
            print(f"Error: Metadata CSV file not found at {metadata_csv_path}")
            self.metadata_df = pd.DataFrame()  # Empty DataFrame
        except Exception as e:
            print(f"Error reading metadata CSV {metadata_csv_path}: {e}")
            self.metadata_df = pd.DataFrame()

        if not self.metadata_df.empty and "id" in self.metadata_df.columns:
            self.metadata_df["normalized_id"] = self.metadata_df["id"].apply(
                self._normalize_id_csv
            )
            self.metadata_df.set_index(
                "normalized_id", inplace=True, drop=False
            )  # Keep 'id' column too
        else:
            print(
                "Warning: Metadata DataFrame is empty or 'id' column is missing. No metadata will be loaded."
            )
            # Ensure 'normalized_id' column exists even if empty for consistent access
            if "normalized_id" not in self.metadata_df.columns:
                self.metadata_df["normalized_id"] = pd.Series(dtype="str")

        # Each sample is a single HSI cube now, rather than a patient with multiple FOVs
        self.cube_samples = self._find_cube_samples()
        self.cube_samples.sort(key=lambda x: x["id"])

        # Dictionary for faster sample lookups by id
        self.sample_map = {s["id"]: i for i, s in enumerate(self.cube_samples)}

        # ensure proper Dataset behavior
        super().__init__()

    def _normalize_id_csv(self, sample_id: any) -> typing.Union[str, any]:
        """Normalizes sample IDs from the CSV for consistent matching."""
        if isinstance(sample_id, str):
            return sample_id.replace("S.", "S").replace(" ", "").strip()
        return sample_id

    def _normalize_id_filename(self, filename_id_part: str) -> str:
        """Normalizes sample IDs extracted from filenames."""
        fid = filename_id_part.strip()
        # replace leading ‘s’ or ‘S’ with uppercase ‘S’
        return re.sub(r"^[Ss]", "S", fid)

    def _extract_patient_number(self, sample_id: str) -> str:
        """
        Extracts the patient number from a sample ID (e.g., 'S1.2' -> '1.2').
        """
        if sample_id.startswith("S"):
            return sample_id[1:]
        return sample_id

    def _find_cube_samples(self) -> list:
        """
        Scans the HSI data directory and treats each HSI cube as a separate sample.
        Each sample gets a unique id of format "patient_number_fov_number".
        """
        cube_samples = []
        # Regex: HyperProbe1.1_Biopsy_ (S<digits>.<digits>) (_FOV(<digits>))? .mat or .npy
        pattern = re.compile(
            r"^HyperProbe1\.1_Biopsy_([Ss]\d+\.\d+)(?:_FOV(\d+))?(?:_BIS)?\.(mat|npy)$"
        )

        if not os.path.isdir(self.hsi_data_dir):
            print(f"Error: HSI data directory not found at {self.hsi_data_dir}")
            return []

        for filename in os.listdir(self.hsi_data_dir):
            match = pattern.match(filename)
            if match:
                raw_sample_id_part = match.group(1)  # e.g., "S1.2"
                fov_number_str = match.group(2)  # e.g., "1" or None

                # Default FOV is "1" if not specified
                fov_key = fov_number_str if fov_number_str else "1"

                normalized_file_id = self._normalize_id_filename(raw_sample_id_part)
                patient_number = self._extract_patient_number(normalized_file_id)

                # Create a ID like "1.2_3" for patient S1.2 FOV 3
                id = f"{patient_number}_{fov_key}"

                if (
                    not self.metadata_df.empty
                    and "normalized_id" in self.metadata_df.index.names
                    and normalized_file_id in self.metadata_df.index
                ):
                    file_path = os.path.join(self.hsi_data_dir, filename)

                    # Ensure metadata is a dict, even if multiple rows match (should not happen with set_index)
                    meta_entry = self.metadata_df.loc[normalized_file_id]
                    if isinstance(
                        meta_entry, pd.DataFrame
                    ):  # if somehow index is not unique
                        metadata = meta_entry.iloc[0].to_dict()
                        print(
                            f"Warning: Multiple metadata entries for {normalized_file_id}, using first."
                        )
                    else:
                        metadata = meta_entry.to_dict()

                    # Each sample is a single HSI cube
                    cube_samples.append(
                        {
                            "id": id,  # Unique ID for each cube (patient_fov)
                            "patient_id": normalized_file_id,  # Original patient ID (e.g., S1.2)
                            "fov": fov_key,  # FOV number
                            "file_path": file_path,  # Path to the .mat file
                            "metadata": metadata,  # Associated metadata
                        }
                    )
                else:
                    print(
                        f"Warning: Metadata not found for sample ID '{normalized_file_id}' from file '{filename}' (normalized from '{raw_sample_id_part}')"
                    )

        return cube_samples

    def __len__(self) -> int:
        """Returns the number of individual HSI cubes in the dataset."""
        return len(self.cube_samples)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieves a single HSI cube sample by its index.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            A dictionary containing:
            - id: Unique identifier of format "patient_number_fov_number" e.g., "1.2_3"
            - patient_id: The patient identifier (e.g., "S1.2")
            - fov: The field of view number
            - hsi_cube: The hyperspectral cube data
            - metadata: Associated metadata from the CSV
        """
        # Validate index
        if idx < 0 or idx >= len(self.cube_samples):
            raise IndexError("Sample index out of range.")

        sample_info = self.cube_samples[idx]

        # loading of the HSI cube data
        hsi_cube = self._load_hsi_cube(sample_info["file_path"])[
            self.left_cut_index : self.right_cut_index, :, :
        ]

        if self.downsample_factor is not None:
            # ensure kernel is a 2‐tuple
            if isinstance(self.downsample_factor, int):
                kh = kw = self.downsample_factor
            else:
                kh, kw = self.downsample_factor

            # convert to float-tensor, add batch-dim
            x = torch.from_numpy(hsi_cube).float().unsqueeze(0)  # (1, C, H, W)
            x = F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))
            hsi_cube = x.squeeze(0).numpy()  # back to (C, H', W')

        delta_A = (
            self.compute_delta_A(hsi_cube)
            if self.with_delta_A
            and hsi_cube.shape[0] == self.reference_spectrum.shape[0]
            else None
        )

        return {
            "id": sample_info["id"],
            "patient_id": sample_info["patient_id"],
            "fov": sample_info["fov"],
            "file_path": sample_info["file_path"],
            "hsi_cube": hsi_cube,
            "metadata": sample_info["metadata"],
            "delta_A": delta_A,
        }

    def compute_delta_A(self, hsi_cube: np.ndarray) -> np.ndarray:
        """
        Compute relative image from absolute image and reference spectrum, apply coarseness.

        Returns:
            NDArray: Relative image (delta A)
        """
        # Set all zero values to 1e-3 to avoid log(0)
        hsi_cube = np.where(hsi_cube <= 0, 1e-3, hsi_cube)
        delta_A = -np.log(
            np.transpose(hsi_cube)[
                :: self.coarseness,
                :: self.coarseness,
                :,
            ]
            / self.reference_spectrum
        )
        return delta_A

    def _load_hsi_cube(self, file_path: str) -> np.ndarray:
        """
        Loads a single HSI cube from a .mat file or preprocessed .npy file.

        Parameters
        ----------
        file_path : str
            Path to the .mat file containing the HSI cube.

        Returns
        -------
        np.ndarray or None
            The HSI cube data as a numpy array, or None if loading fails.
        """
        if self.use_preprocessed:
            return np.load(file_path)
        try:
            with h5py.File(file_path, "r") as f:
                dset = f["Ref_hyper"]
                return dset[...]
        except Exception as e:
            print(f"Error loading HSI data from {file_path}: {e}")
            return None

    def get_sample_by_id(self, id: str) -> typing.Union[dict, None]:
        """
        Retrieves a sample by its ID (patient_number_fov).

        Parameters
        ----------
        id : str
            The ID in the format "patient_number_fov_number" (e.g., "1.2_3").

        Returns
        -------
        dict or None
            The sample data or None if not found.
        """
        if id in self.sample_map:
            return self.__getitem__(self.sample_map[id])

        print(f"Sample with ID '{id}' not found.")
        return None

    def get_samples_by_patient_id(self, patient_id: str) -> typing.List[dict]:
        """
        Retrieves all samples (FOVs) for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "S1.2").

        Returns
        -------
        list of dict
            A list of all samples for the given patient, empty if none found.
        """
        normalized_id = self._normalize_id_csv(patient_id)

        # Strip "S" prefix if present for matching with id format
        if normalized_id.startswith("S"):
            patient_number = normalized_id[1:]
        else:
            patient_number = normalized_id

        samples = []
        for idx, sample in enumerate(self.cube_samples):
            if sample["id"].startswith(f"{patient_number}_"):
                samples.append(self.__getitem__(idx))

        if not samples:
            print(
                f"No samples found for patient ID '{patient_id}' (normalized: '{normalized_id}')."
            )

        return samples

    def get_patient_ids(self) -> typing.List[str]:
        """
        Get a list of all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        patient_ids = set()
        for sample in self.cube_samples:
            # Extract patient ID from sample id (e.g., "1.2_3" -> "S1.2")
            parts = sample["id"].split("_")
            if len(parts) >= 1:
                patient_number = parts[0]
                patient_id = f"S{patient_number}"
                patient_ids.add(patient_id)
        return sorted(patient_ids)

    def get_sample_by_patient_and_fov(
        self, patient_id: str, fov: str
    ) -> typing.Union[dict, None]:
        """
        Retrieves a specific sample by patient ID and FOV number.

        Parameters
        ----------
        patient_id : str
            The patient ID (e.g., "S1.2").
        fov : str
            The FOV number as a string.

        Returns
        -------
        dict or None
            The sample data or None if not found.
        """
        normalized_id = self._normalize_id_csv(patient_id)

        # Strip "S" prefix if present for matching with id format
        if normalized_id.startswith("S"):
            patient_number = normalized_id[1:]
        else:
            patient_number = normalized_id

        id = f"{patient_number}_{fov}"
        return self.get_sample_by_id(id)

    def load_delta_c_and_scatter_params(
        self, id: str, load_from_path: str
    ) -> typing.Tuple[np.ndarray, np.ndarray]:
        """
        Load delta C and scatter parameters from .npy files for a given sample.

        Parameters
        ----------
        id : str
            The ID in the format "patient_number_fov_number" (e.g., "1.2_3").
        load_from_path : str
            Path to the directory containing the results of the spectral unmixing.

        Returns
        -------
        tuple of np.ndarray
            Tuple containing delta C (coef_list) and scatter parameters (scatter_params), or (None, None) if not found.
        """
        # Find the directory that matches the id
        for dirname in os.listdir(load_from_path):
            dir_path = os.path.join(load_from_path, dirname)
            if not os.path.isdir(dir_path):
                continue
            # Remove 'HyperProbe1.1_Biopsy_' prefix and file ending for matching
            # Directory names are like 'HyperProbe1.1_Biopsy_S1.2_FOV3' or 'HyperProbe1.1_Biopsy_S1.2'
            name = dirname.replace("HyperProbe1.1_Biopsy_", "").replace("_BIS", "")
            if "_FOV" in name:
                patient, fov = name.split("_FOV")
                dir_id = f"{patient[1:] if patient.startswith('S') else patient}_{fov}"
            else:
                patient = name
                dir_id = patient[1:] if patient.startswith("S") else patient
                dir_id = f"{dir_id}_1"  # Default FOV is 1 if not specified
            if dir_id == id:
                coef_path = os.path.join(dir_path, "coef_list.npy")
                scatter_path = os.path.join(dir_path, "scatter_params.npy")
                if os.path.exists(coef_path) and os.path.exists(scatter_path):
                    coef_list = np.load(coef_path)
                    scatter_params = np.load(scatter_path)
                    return coef_list, scatter_params
                else:
                    print(
                        f"Error: coef_list.npy or scatter_params.npy not found in {dir_path}"
                    )
                    return None, None
        print(f"Error: No directory found for id {id} in {load_from_path}")
        return None, None
