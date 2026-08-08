import numpy as np
import sys
import gc

sys.path.append("..")

from src.wavelength_selection.enums import SampleType

# Add project root to path

# Import project modules
from src.dataset.helicoid_dataset import HelicoidDataset
from src.constants import HELICOID_TEST_SET
from src.molecules import Molecules
from src.scattering_model.scatter_optim import (
    optimize_image_parallel,
)
from src.models.pipeline.model_pipeline import load_model_pipeline
from tqdm import tqdm
from pathlib import Path


COARSENESS = 1
LEFT_CUT = 530
RIGHT_CUT = 750

# dataset = HelicoidDataset(
#     left_cut=LEFT_CUT,
#     right_cut=RIGHT_CUT,
#     coarseness=COARSENESS,
#     with_delta_A=True,  # We want it to be false so that we can down
# )


molecules = Molecules(
    left_cut=LEFT_CUT, right_cut=RIGHT_CUT, sample_type=SampleType.HELICOID
)

model_path = (
    "../results/models/sweep/ssr-autoencoder-helicoid/unet_15_wl_ssr/ancient-sweep-4"
)

model_pipeline = load_model_pipeline(path=model_path, device="cuda:0")

dataset = model_pipeline.data_manager.dataset
dataset.set_inference_mode(True)

save_dir = Path(
    f"../results/helicoid_spectral_unmixing/ssr_coarseness_{COARSENESS}_left_{LEFT_CUT}_right_{RIGHT_CUT}"
)
save_dir.mkdir(parents=True, exist_ok=True)
if __name__ == "__main__":
    # Create a shared executor to avoid repeatedly creating/destroying processes
    for id in tqdm(HELICOID_TEST_SET):
        print("Processing sample:", id)
        sample_dir = save_dir / id

        if sample_dir.exists():
            print(f"Sample {id} processed. Skipping.")
            continue
        sample = dataset.get_sample_by_id(id)
        hsi_cube_enhanced = (
            model_pipeline.predict_sample(sample).squeeze().cpu().numpy()
        )

        t1_params = dataset.load_reference_params(
            id=id,
            load_from_path="../results/helicoid_spectral_unmixing/coarseness_1_left_530_right_750",
        )
        try:
            delta_A = dataset.compute_delta_A(
                id=id, hsi_cube=hsi_cube_enhanced, gt_map=sample.gt_map
            )[0]
            print("delta_A shape:", delta_A.shape)
            delta_A_flat = delta_A.reshape(-1, delta_A.shape[-1])

            errors, coef_list, scatter_params, _ = optimize_image_parallel(
                t1_params, delta_A_flat, molecules.M, molecules.cut_wavelengths, 8
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
        except ValueError as e:
            print(f"Error processing sample {id}: {e}")
            continue
