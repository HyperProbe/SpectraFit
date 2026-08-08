from pathlib import Path

from loguru import logger

import sys

sys.path.append("..")


from src.molecules import MoleculeMode
from src.wavelength_selection.criterion import ConcentrationPreservation, SpectralFit
from src.wavelength_selection.enums import LightModel, SampleType
from src.wavelength_selection.wavelength_selector import WavelengthSelector

if __name__ == "__main__":

    # Enable logging to see progress and debug information
    # logger.disable("src")  # Comment out this line to enable logging
    logger.enable("src")  # Explicitly enable logging for the src module

    # Set logging level to display INFO and above
    logger.level("INFO")

    logger.info("Starting wavelength optimization with spectral_fit criterion")

    # samples = ["1.2_3", "1.2_4", "1.3_1", "1.3_2", "1.41_1", "1.41_2"]

    shared_kwargs = {
        "model": LightModel.SCATTERING,
        "sample_ids": ["1.2_3", "1.2_4", "1.3_1", "1.3_2", "1.41_1", "1.41_2"],
        "sample_type": SampleType.BIOPSY2,
        "gt_path": "../results/biopsy2_spectral_unmixing/spectral_unmixing_avgpool_4_coarseness_1_left_500_right_900_t1_all_molecules",
        "img_coarseness": 8,
        "wl_coarseness": 1,
        "target_N": 15,
        "molecule_mode": MoleculeMode.ALL,
        "max_workers": 8,
        "left_cut": 500,
        "right_cut": 900,
    }

    logger.info(f"Using parameters: {shared_kwargs}")

    sf_selector = WavelengthSelector(criterion=SpectralFit, **shared_kwargs)

    run_dir = Path(f"../results/wl_selection/biopsy2")
    logger.info(f"Starting sequential backward selection to directory: {run_dir}")

    # Set the target number of iterations to match target_N - this is how many wavelengths will be removed
    # For example, if we start with all wavelengths and target_N is 10, we'll have (len(all_wavelengths) - 10) iterations
    n_iterations = (
        len(sf_selector.criterion.all_wavelengths) - shared_kwargs["target_N"]
    )
    logger.info(
        f"Will run {n_iterations} iterations to get down to {shared_kwargs['target_N']} wavelengths"
    )

    sf_selector.sequential_backward_selection(
        save_dir=run_dir / "spectral_fit_500_900",
    )

    logger.info("Sequential backward selection complete, computing metrics...")
    logger.success("Process completed successfully!")
