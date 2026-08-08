import numpy as np
import sys
import gc

# Add project root to path
sys.path.append("..")

# Import project modules
from src.dataset.helicoid_dataset import HelicoidDataset
from src.molecules import MoleculeMode
from src.scattering_model.scatter_optim import (
    optim_reference_spectrum_scatter_params,
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


save_dir = Path(
    f"../results/helicoid_spectral_unmixing/coarseness_{COARSENESS}_left_{LEFT_CUT}_right_{RIGHT_CUT}"
)
save_dir.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    # Create a shared executor to avoid repeatedly creating/destroying processes
    for id in tqdm(dataset.sample_map.keys()):
        print("Processing sample:", id)
        sample_dir = save_dir / id

        if sample_dir.exists():
            print(f"Sample {id} processed. Skipping.")
            continue
        data = dataset.get_sample_by_id(id)
        try:
            (
                coef_list,
                scatter_params,
                errors,
                a_t1,
                b_t1,
                delta_A,
                reference_pxl,
                _,
                _,
            ) = optim_reference_spectrum_scatter_params(
                data=data,
                dataset=dataset,
                molecule_mode=MoleculeMode.ALL,
                load_a_b_from_path=None,
                coarseness=8,
            )

            # --- Store results for each sample ---
            # Get original file name without extension
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save errors, coef_list, scatter_params as .npy files
            np.save(sample_dir / "errors.npy", errors)
            np.save(sample_dir / "coef_list.npy", coef_list)
            np.save(sample_dir / "scatter_params.npy", scatter_params)
            np.save(sample_dir / "a_t1.npy", a_t1)
            np.save(sample_dir / "b_t1.npy", b_t1)
            np.save(sample_dir / "reference_pxl.npy", reference_pxl)

            # Force garbage collection after each sample
            import gc

            gc.collect()
        except ValueError as e:
            print(f"Error processing sample {id}: {e}")
            continue
