import numpy as np
import sys
import gc

# Add project root to path
sys.path.append("..")

# Import project modules
from src.dataset.helicoid_dataset import HelicoidDataset
from src.molecules import MoleculeIndex
from src.scattering_model.scatter_optim import (
    single_optim_scattering_with_wl_subset,
)
from tqdm import tqdm
from pathlib import Path


COARSENESS = 1
LEFT_CUT = 530
RIGHT_CUT = 750

dataset = HelicoidDataset(
    left_cut=LEFT_CUT,
    right_cut=RIGHT_CUT,
    coarseness=COARSENESS,
    with_delta_A=True,  # We want it to be false so that we can down
)


wl_set = np.load(
    "../results/wl_selection/helicoid/cp_coarseness_8_530_750_nohup/iteration_286/wl.npy"
)
x_chosen = np.isin(dataset.cut_wavelengths, wl_set)
save_dir = Path(
    f"../results/helicoid_spectral_unmixing/coarseness_{COARSENESS}_left_{LEFT_CUT}_right_{RIGHT_CUT}_reduced"
)
save_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Create a shared executor to avoid repeatedly creating/destroying processes
    for id in tqdm(dataset.sample_map):
        print("Processing sample:", id)
        sample_dir = save_dir / id

        if sample_dir.exists():
            print(f"Sample {id} processed. Skipping.")
            continue
        # Use the shared executor to avoid creating new processes
        try:
            sample = dataset.get_sample_by_id(id)
            errors, params_found, _ = single_optim_scattering_with_wl_subset(
                data=sample,
                dataset=dataset,
                gt_path="../results/helicoid_spectral_unmixing/coarseness_1_left_530_right_750",
                x_chosen=x_chosen,
                use_parallel=True,
            )
            # --- Store results for each sample ---
            # Get original file name without extension
            sample_dir.mkdir(parents=True, exist_ok=True)
            coef_list, scatter_params = np.split(params_found, [len(MoleculeIndex)], axis=2)

            # Save errors, coef_list, scatter_params as .npy files
            np.save(sample_dir / "errors.npy", errors)
            np.save(sample_dir / "coef_list.npy", coef_list)
            np.save(sample_dir / "scatter_params.npy", scatter_params)

            # Force garbage collection after each sample
            import gc

            gc.collect()
        except TypeError as e:
            print(f"Error processing sample {id}: {e}, skipping.")
            continue
