import numpy as np
import sys
import gc

# Add project root to path
sys.path.append("..")

# Import project modules
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.molecules import Molecules, REFERENCE_SPECTRUM_T1, MoleculeMode
from src.scattering_model.scatter_optim import optimize_image_parallel
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor


COARSENESS = 1
LEFT_CUT = 500
RIGHT_CUT = 900

dataset = Biopsy2Dataset(
    reference_spectrum_path="../data/reference/S1_1_1_glioma_ref.npy",
    coarseness=COARSENESS,
    left_cut=LEFT_CUT,
    right_cut=RIGHT_CUT,
    with_delta_A=False,  # We want it to be false so that we can downsample it first
)


save_dir = Path(
    f"../results/biopsy2_spectral_unmixing/glioma_avgpool_{4}_coarseness_{COARSENESS}_left_{LEFT_CUT}_right_{RIGHT_CUT}"
)
save_dir.mkdir(parents=True, exist_ok=True)
molecules = Molecules(left_cut=dataset.left_cut, right_cut=dataset.right_cut)
t1_params = REFERENCE_SPECTRUM_T1[MoleculeMode.ALL]

if __name__ == "__main__":
    # Create a shared executor to avoid repeatedly creating/destroying processes
    with ProcessPoolExecutor(max_workers=16) as shared_executor:
        for sample in tqdm(dataset.cube_samples):
            if sample["metadata"]["type_of_tumor"] != "Glioma":
                print(f"Sample {sample['id']} is not a glioma. Skipping.")
                continue
            print("Processing sample:", sample["id"])
            orig_path = Path(sample["file_path"])
            sample_dir = save_dir / orig_path.stem

            if sample_dir.exists():
                print(f"Sample {orig_path.stem} processed. Skipping.")
                continue

            hsi_cube = np.load(sample["file_path"])[
                dataset.left_cut_index : dataset.right_cut_index, :, :
            ]
            hsi_cube[hsi_cube <= 0] = 10**-3

            downsampled = (
                F.avg_pool2d(
                    torch.tensor(hsi_cube).unsqueeze(0),
                    kernel_size=4,
                    stride=4,
                )
                .squeeze(0)
                .numpy()
            )

            delta_A = dataset.compute_delta_A(downsampled)
            print("Delta A shape:", delta_A.shape)
            delta_A_flat = delta_A.reshape(-1, delta_A.shape[-1])

            # Use the shared executor to avoid creating new processes
            errors, coef_list, scatter_params, _ = optimize_image_parallel(
                t1_params,
                delta_A_flat,
                molecules.M,
                molecules.cut_wavelengths,
                8,
                executor=shared_executor,
            )

            # --- Store results for each sample ---
            # Get original file name without extension
            sample_dir.mkdir(parents=True, exist_ok=True)
            coef_list = coef_list.reshape(delta_A.shape[0], delta_A.shape[1], -1)
            scatter_params = scatter_params.reshape(
                delta_A.shape[0], delta_A.shape[1], -1
            )

            # Save errors, coef_list, scatter_params as .npy files
            np.save(sample_dir / "errors.npy", errors)
            np.save(sample_dir / "coef_list.npy", coef_list)
            np.save(sample_dir / "scatter_params.npy", scatter_params)

            # Force garbage collection after each sample
            import gc

            gc.collect()
