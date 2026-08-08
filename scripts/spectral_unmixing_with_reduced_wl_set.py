import numpy as np
import sys
import gc

# Add project root to path
sys.path.append("..")

# Import project modules
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.molecules import Molecules, REFERENCE_SPECTRUM_T1, MoleculeMode
from src.scattering_model.scatter_optim import single_optim_scattering_with_wl_subset
from tqdm import tqdm
from pathlib import Path

from loguru import logger


logger.enable("src")  # Explicitly enable logging for the src module

# Set logging level to display INFO and above
logger.level("DEBUG")

COARSENESS = 1
LEFT_CUT = 500
RIGHT_CUT = 900

dataset = Biopsy2Dataset(
    reference_spectrum_path="../data/reference/S1_4_1_mean_roi.npy",
    coarseness=COARSENESS,
    left_cut=LEFT_CUT,
    right_cut=RIGHT_CUT,
    with_delta_A=True,  # We want it to be false so that we can downsample it first
    downsample_factor=4,
)

wl_set = np.load(
    "../results/wl_selection/biopsy2/concentration_preservation_500_900/iteration_65/wl.npy"
)
x_chosen = np.isin(dataset.cut_wavelengths, wl_set)

save_dir = Path(
    f"../results/spectral_unmixing_avgpool_{4}_coarseness_{COARSENESS}_left_{LEFT_CUT}_right_{RIGHT_CUT}_reduced_wl_set_500_900"
)
save_dir.mkdir(parents=True, exist_ok=True)
molecules = Molecules(left_cut=dataset.left_cut, right_cut=dataset.right_cut)
t1_params = REFERENCE_SPECTRUM_T1[MoleculeMode.ALL]
print(x_chosen)

if __name__ == "__main__":
    for sample in tqdm(dataset.cube_samples):
        print("Processing sample:", sample["id"])
        orig_path = Path(sample["file_path"])
        sample_dir = save_dir / orig_path.stem

        if sample_dir.exists():
            print(f"Sample {orig_path.stem} processed. Skipping.")
            continue

        sample = dataset.get_sample_by_id(sample["id"])

        # Use the shared executor to avoid creating new processes
        errors, coef_list, scatter_params = single_optim_scattering_with_wl_subset(
            data=sample,
            dataset=dataset,
            gt_path=None,
            x_chosen=x_chosen,
            use_parallel=True,
        )

        # --- Store results for each sample ---
        # Get original file name without extension
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save errors, coef_list, scatter_params as .npy files
        np.save(sample_dir / "errors.npy", errors)
        np.save(sample_dir / "coef_list.npy", coef_list)
        np.save(sample_dir / "scatter_params.npy", scatter_params)

        # Force garbage collection after each sample
        import gc

        gc.collect()
