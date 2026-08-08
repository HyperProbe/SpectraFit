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
    logger.remove(0)

    logger.add(sys.stderr, level="INFO")

    logger.info("Starting wavelength optimization with spectral_fit criterion")
    coarseness = 8

    shared_kwargs = {
        "model": LightModel.SCATTERING,
        "sample_ids": ["008-01"],
        "sample_type": SampleType.HELICOID,
        "gt_path": "../results/helicoid_spectral_unmixing/coarseness_1_left_530_right_750",
        "img_coarseness": coarseness,
        "wl_coarseness": 1,
        "target_N": 15,
        "molecule_mode": MoleculeMode.ALL,
        "max_workers": 8,
        "left_cut": 530,
        "right_cut": 750,
    }

    logger.info(f"Using parameters: {shared_kwargs}")

    cp_selector = WavelengthSelector(
        criterion=ConcentrationPreservation, **shared_kwargs
    )

    run_dir = Path(f"../results/wl_selection/helicoid")
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
        save_dir=run_dir / f"cp_coarseness_{coarseness}_530_750_nohup", disable_tqdm=True
    )

    logger.info("Sequential backward selection complete, computing metrics...")
    logger.success("Process completed successfully!")
