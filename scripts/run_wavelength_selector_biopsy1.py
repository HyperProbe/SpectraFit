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

    logger.info(
        "Starting wavelength optimization with concentration_preservation criterion"
    )

    shared_kwargs = {
        "model": LightModel.SCATTERING,
        "sample_ids": ["S7", "S10", "S4_1", "S5"],
        "sample_type": SampleType.BIOPSY1,
        "gt_path": "/home/home/tim_ivan/thesis/marcel_thesis/master_thesis/data/biopsy/hyperprobe_biopsies_optim_marcel/5_50_t1",
        "img_coarseness": 8,
        "wl_coarseness": 1,
        "target_N": 10,
        "molecule_mode": MoleculeMode.ALL,
        "max_workers": 8,
    }

    logger.info(f"Using parameters: {shared_kwargs}")

    cp_selector = WavelengthSelector(
        criterion=ConcentrationPreservation, **shared_kwargs
    )

    run_dir = Path(
        f"../results/wl_selection/sbs_c8_M_{shared_kwargs['molecule_mode'].name}"
    )
    logger.info(f"Starting sequential backward selection to directory: {run_dir}")

    # Set the target number of iterations to match target_N - this is how many wavelengths will be removed
    # For example, if we start with all wavelengths and target_N is 10, we'll have (len(all_wavelengths) - 10) iterations
    n_iterations = (
        len(cp_selector.criterion.all_wavelengths) - shared_kwargs["target_N"]
    )
    logger.info(
        f"Will run {n_iterations} iterations to get down to {shared_kwargs['target_N']} wavelengths"
    )

    cp_selector.sequential_backward_selection(
        save_dir=run_dir / "concentration_preservation",
    )

    logger.info("Sequential backward selection complete, computing metrics...")
    logger.success("Process completed successfully!")
