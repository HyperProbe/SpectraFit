import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, ClassVar
from spectral import open_image

# Constants for HSI data directory and metadata CSV path, loaded from environment
HSI_DATA_DIR = os.environ.get("HSI_DATA_DIR")
HSI_PREPROCESSED_DIR = os.environ.get("HSI_PREPROCESSED_DIR")
METADATA_CSV_PATH = os.environ.get("METADATA_CSV_PATH")
MOLECULES_DIR = Path(os.environ.get("MOLECULES_DIR"))
SPECTRA_UCL_NIR_DATA = MOLECULES_DIR / "UCL-NIR-Spectra/spectra/"
HYPERPROBE_DATA_DIR = os.environ.get("HYPERPROBE_DATA_DIR")
HELICOID_DATA_DIR = Path(os.environ.get("HELICOID_DATA_DIR", ""))
HELICOID_REFERENCE_PIXEL_DIR = os.environ.get(
    "HELICOID_REFERENCE_PIXEL_DIR", "../data/helicoid_reference_pixel"
)
HELICOID_SELECTED_WAVELENGTHS_CP = "../results/wl_selection/helicoid/cp_coarseness_8_530_750_nohup/iteration_286/wl.npy"

SPECTRA_UCL_NIR_DATA_DICT = {
    "cytoa_oxy": SPECTRA_UCL_NIR_DATA / "moody cyt aa3 oxidised.txt",
    "cytoa_red": SPECTRA_UCL_NIR_DATA / "moody cyt aa3 reduced.txt",
    "hbo2": SPECTRA_UCL_NIR_DATA / "hb02.txt",
    "hbo2_450": SPECTRA_UCL_NIR_DATA / "z_adult_hbo2_450_630.txt",
    "hbo2_600": SPECTRA_UCL_NIR_DATA / "z_adult_hbo2_600_800.txt",
    "hb": SPECTRA_UCL_NIR_DATA / "hb.txt",
    "hb_450": SPECTRA_UCL_NIR_DATA / "z_adult_hb_450_630.txt",
    "hb_600": SPECTRA_UCL_NIR_DATA / "z_adult_hb_600_800.txt",
    "water": SPECTRA_UCL_NIR_DATA / "matcher94_nir_water_37.txt",
    "fat": SPECTRA_UCL_NIR_DATA / "fat.txt",
    "water_hale": SPECTRA_UCL_NIR_DATA / "water_hale73.txt",
    "cytoa_diff": SPECTRA_UCL_NIR_DATA / "cytoxidase_diff_odmMcm.txt",
    "cytoc_oxy": SPECTRA_UCL_NIR_DATA / "cooper pig c oxidised.txt",
    "cytoc_red": SPECTRA_UCL_NIR_DATA / "cooper pig c reduced.txt",
    "cytob_oxy": SPECTRA_UCL_NIR_DATA / "cope cyt b oxidised.txt",
    "cytob_red": SPECTRA_UCL_NIR_DATA / "cope cyt b reduced.txt",
}

SPECTRA_CREATIS_DATA = MOLECULES_DIR / "CREATIS-Spectra/spectra/"

WAVELENGTHS = np.linspace(385, 1015, 127)
hdr_img = open_image(HELICOID_DATA_DIR / "004-02" / "raw.hdr")
HELICOID_WAVELENGTHS = np.array(hdr_img.metadata["wavelength"]).astype(float)
LN10 = 2.30258509299
HELICOID_VAL_SET = ["013-01", "014-01", "022-02", "027-02"]
HELICOID_TEST_SET = [
    "008-01",
    "008-02",
    "012-01",
    "012-02",
    "015-01",
    "016-01",
    "016-02",
    "016-03",
    "016-04",
    "016-05",
    "020-01",
    "025-02",
]


# Scattering param range
MAX_A = 100
MAX_B = 5
B_STEPS = 25
""" Number of steps for the scattering power b during brute force grid search, i.e. slice(0, MAX_B, MAX_B/B_STEPS) """
A_STEPS = 3
""" Number of steps for the scattering coefficient a during brute force grid search, i.e. slice(0, MAX_A, MAX_A/A_STEPS) """

if not HSI_DATA_DIR or not METADATA_CSV_PATH:
    raise RuntimeError(
        "Environment variables HSI_DATA_DIR and METADATA_CSV_PATH must be set. "
        "You can set them in a .env file (gitignored) or export in your shell before running the code."
    )


# Normalize patient IDs for consistent usage (same as in dataset.py)
def _normalize_id(sample_id):
    if isinstance(sample_id, str):
        return sample_id.replace("S.", "S").replace(" ", "").strip()
    return sample_id


# Load metadata CSV file and create patient ID mappings
try:
    _metadata_df = pd.read_csv(METADATA_CSV_PATH)
    _metadata_df["normalized_id"] = _metadata_df["id"].apply(_normalize_id)
    _all_patient_ids = _metadata_df["normalized_id"].dropna().unique().tolist()

    # Group patient IDs by tumor type
    _patient_ids_by_type = {}
    for tumor_type in _metadata_df["type_of_tumor"].dropna().unique():
        filtered_ids = (
            _metadata_df[_metadata_df["type_of_tumor"] == tumor_type]["normalized_id"]
            .dropna()
            .tolist()
        )
        _patient_ids_by_type[tumor_type] = filtered_ids

    print(f"Loaded {len(_all_patient_ids)} patient IDs from metadata CSV")
except Exception as e:
    print(f"Warning: Could not load patient IDs: {e}")
    _all_patient_ids = []
    _patient_ids_by_type = {}


# Create an enum-like class with patient IDs as class attributes for autocomplete
class ALL_PATIENT_IDS:
    """
    A class containing all normalized patient IDs from the metadata CSV as attributes.
    This provides autocomplete support for patient IDs in IDEs.

    Usage:
        # Access specific patient IDs (with autocomplete)
        patient_id = ALL_PATIENT_IDS.S1_2  # Returns "S1.2"

        # Get all patient IDs as a list
        all_ids = ALL_PATIENT_IDS.get_all()

        # Get glioma patient IDs
        glioma_ids = ALL_PATIENT_IDS.get_by_type("Glioma")
    """

    # Class variables to store collections of IDs
    _all_ids: ClassVar[List[str]] = _all_patient_ids
    _by_type: ClassVar[Dict[str, List[str]]] = _patient_ids_by_type

    @classmethod
    def get_all(cls) -> List[str]:
        """Returns a list of all patient IDs."""
        return sorted(cls._all_ids)

    @classmethod
    def get_by_type(cls, tumor_type: str) -> List[str]:
        """Returns a list of patient IDs filtered by tumor type."""
        return sorted(cls._by_type.get(tumor_type, []))

    @classmethod
    def get_types(cls) -> List[str]:
        """Returns a list of all tumor types."""
        return sorted(list(cls._by_type.keys()))


# Dynamically add each patient ID as a class attribute for autocomplete
for patient_id in _all_patient_ids:
    if isinstance(patient_id, str):
        # Convert dots to underscores to create valid attribute names
        # For example: "S1.2" becomes ALL_PATIENT_IDS.S1_2
        attr_name = patient_id.replace(".", "_")
        setattr(ALL_PATIENT_IDS, attr_name, patient_id)
