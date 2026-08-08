import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, IntEnum
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, differential_evolution
from tqdm import tqdm

from src import constants
from src.dataset.biopsy1_dataset import Biopsy1Dataset
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.dataset.helicoid_dataset import HelicoidDataset
from src.molecules import MoleculeMode, Molecules
from src.wavelength_selection.criterion import Criterion
from src.wavelength_selection.enums import CriterionMetric
from src.wavelength_selection.enums import LightModel, SampleType


class WavelengthSelector:

    def __init__(
        self,
        model: LightModel,
        sample_ids: List[str],
        sample_type: SampleType,
        criterion: Criterion,
        gt_path: str,
        img_coarseness: int,
        wl_coarseness: int,
        target_N: int = 10,
        max_workers: int = 48,
        initial_wl_set: Optional[NDArray] = None,
        molecule_mode: MoleculeMode = MoleculeMode.ALL,
        criterion_metric: Optional[CriterionMetric] = None,
        left_cut: int = 500,
        right_cut: int = 900,
    ):
        """
        Initialize the WavelengthSelector

        Args:
            model (LightModel): Light model to use
            sample_ids (List[str]): List of sample IDs
            sample_type (SampleType): Type of sample
            criterion (Criterion): Criterion to use for selection
            gt_path : Path to ground-truth data
            img_coarseness (int): Image coarseness
            wl_coarseness (int): Wavelength coarseness
            target_N (int, optional): Target number of wavelengths. Defaults to 10.
            max_workers (int, optional): Maximum number of workers for parallel processing. Defaults to 48.
            initial_wl_set (Optional[NDArray], optional): Initial set of wavelengths if not the full range should be used to start with. Defaults to None, i.e. full range will be used to start.
            molecule_mode (MoleculeMode, optional): Molecule mode for biopsy samples. Defaults to MoleculeMode.ALL.
            criterion_metric (Optional[CriterionMetric], optional): Criterion metric to use. Defaults to None. Only relevant for ModelOrthogonality and DataOrthogonality.

        """
        self.light_model = model
        self.sample_ids = sample_ids
        self.sample_type = sample_type
        self.img_coarseness = img_coarseness
        self.wl_coarseness = wl_coarseness
        self.target_N = target_N
        self.max_workers = max_workers
        self.initial_wl_set = initial_wl_set
        self.molecule_mode = molecule_mode
        self.criterion_metric = criterion_metric
        self.left_cut = left_cut
        self.right_cut = right_cut

        # initial set
        logger.info("Loading initial data to get wavelengths and absorption matrix")

        if self.sample_type == SampleType.BIOPSY1:
            match self.molecule_mode:
                case MoleculeMode.ALL:
                    self.dataset = Biopsy1Dataset(coarseness=self.img_coarseness)
                case MoleculeMode.OXCCO2_WATER:
                    self.dataset = Biopsy1Dataset(
                        left_cut=740,
                        right_cut=900,
                        coarseness=self.img_coarseness,
                    )
                case _:
                    raise NotImplementedError()
        elif self.sample_type == SampleType.BIOPSY2:
            self.dataset = Biopsy2Dataset(
                coarseness=self.img_coarseness,
                left_cut=self.left_cut,
                right_cut=self.right_cut,
                with_delta_A=True,
                downsample_factor=4,
            )

        elif self.sample_type == SampleType.HELICOID:
            self.dataset = HelicoidDataset(
                left_cut=self.left_cut,
                right_cut=self.right_cut,
                coarseness=self.img_coarseness,
                with_delta_A=True,
            )
        else:
            raise NotImplementedError()

        self.all_wavelengths = self.dataset.cut_wavelengths[
            :: self.wl_coarseness
        ].copy()

        self.samples = [self._load_sample(sample_id) for sample_id in self.sample_ids]

        self.criterion = criterion(
            light_model=self.light_model,
            samples=self.samples,
            sample_type=self.sample_type,
            dataset=self.dataset,
            img_coarseness=self.img_coarseness,
            wl_coarseness=self.wl_coarseness,
            all_wavelengths=self.all_wavelengths,
            gt_path=gt_path,
            molecule_mode=self.molecule_mode,
            metric=self.criterion_metric,
        )

    def _load_sample(self, sample_id: str) -> dict:
        return self.dataset.get_sample_by_id(sample_id)

    def _get_num_to_remove(self, wavelength_idx: NDArray, rm_percent: float) -> int:
        """Get number of wavelengths to remove based on percentage

        Args:
            wavelength_idx (NDArray): Current set of wavelength indices
            rm_percent (float): Fraction of wavelengths to remove, e.g. 0.05 for 5%

        Returns:
            int: Number of wavelengths to remove (>=1)
        """

        if not rm_percent:
            return 1

        num_to_remove = int(len(wavelength_idx) * rm_percent)

        # dont drop below target N
        num_to_remove = min(num_to_remove, len(wavelength_idx) - self.target_N)

        if num_to_remove < 1:
            num_to_remove = 1
        return num_to_remove

    def sequential_backward_selection(
        self,
        rm_percent: Optional[float] = None,
        save_dir: Optional[Path] = None,
        disable_tqdm: bool = False,
    ) -> Tuple[List[NDArray], List[NDArray], List[float]]:
        """
        Run Sequential Backward Selection (SBS) to reduce the number of wavelengths in the set based on the specified criterion.

        Args:
            rm_percent (Optional[float], optional): Fraction of wavelengths to remove in each iteration, e.g. 0.05 for 5% (if percentage yields removal of < 1, 1 wl will be removed). Defaults to None.
            save_dir (Optional[Path], optional): Directory to save sets, runtimes and metrics. Defaults to None.

        Returns:
            Tuple[List[NDArray], List[NDArray], List[float]]: Found sets, metrics, compute times for each iteration.
        """

        # determine initial and available set
        if self.initial_wl_set is None:
            # start set is idx of all wavelengths
            current_wavelength_idx = np.array(range(len(self.all_wavelengths)))
        else:
            # Limit initial set idx for reduction to the specified wl set, obtained e.g. by previous Proj. Max.
            current_wavelength_idx = np.argwhere(
                np.isin(self.all_wavelengths, self.initial_wl_set)
            ).flatten()
            assert len(current_wavelength_idx) == len(self.initial_wl_set)

        found_metrics = []

        logger.info(
            f"Running SBS with model={self.light_model}, criterion={self.criterion}, target N={self.target_N}, wl_coarseness={self.wl_coarseness} img_coarseness={self.img_coarseness}"
        )

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        iteration = 0
        while len(current_wavelength_idx) > self.target_N:
            if os.path.exists(save_dir / f"iteration_{iteration}"):
                logger.info(
                    f"Iteration {iteration} already exists in {save_dir}, skipping..."
                )
                iteration += 1
                continue

            current_wavelengths = (
                np.load(save_dir / f"iteration_{iteration-1}" / "wl.npy")
                if iteration > 0
                else self.all_wavelengths[current_wavelength_idx]
            )
            current_wavelength_idx = np.argwhere(
                np.isin(self.all_wavelengths, current_wavelengths)
            ).flatten()
            assert len(current_wavelength_idx) == len(current_wavelengths)

            num_to_remove = self._get_num_to_remove(
                wavelength_idx=current_wavelength_idx, rm_percent=rm_percent
            )

            metrics = np.zeros(len(current_wavelength_idx))
            step_start_time = time.time()

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks for each wavelength subset and map results back to the metrics array
                futures = {
                    executor.submit(
                        self.criterion.compute_metric,
                        np.delete(current_wavelength_idx, i),
                    ): i
                    for i in range(len(current_wavelength_idx))
                }
                for future in tqdm(futures, disable=disable_tqdm):
                    i = futures[future]
                    metrics[i] = future.result()

            # remove the wl that has the least impact on optimality of criterion
            # Find the indices of the 'num_to_remove' wavelengths with the lowest impact on optimality  of criterion
            indices_to_remove = self.criterion.get_idx_to_remove(
                metrics=metrics, num_to_remove=num_to_remove
            )

            current_wavelength_idx = np.delete(
                current_wavelength_idx, indices_to_remove
            )
            logger.info(
                f"Removed {num_to_remove} WL(s). New set: {current_wavelength_idx}"
            )

            found_wl = self.all_wavelengths[current_wavelength_idx]

            # this can be multiple, is this an issue?
            found_metrics.append(metrics[indices_to_remove])
            step_compute_time = time.time() - step_start_time
            logger.debug(
                f"Iteration {iteration} runtime: {step_compute_time}"
            )

            # save intermediate results
            if save_dir is not None:
                iteration_dir = save_dir / f"iteration_{iteration}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                print(f"Rm index: {indices_to_remove}")
                np.save(
                    iteration_dir / "wl.npy",
                    found_wl,
                )
                np.save(
                    iteration_dir / "all_metrics.npy",
                    metrics,
                )
                np.save(
                    iteration_dir / f"times.npy",
                    step_compute_time,
                )

            # increment iteration
            iteration += 1

    def sequential_forward_selection(
        self, save_dir: Optional[Path] = None
    ) -> Tuple[List[NDArray], List[NDArray], List[float]]:
        """
        Run Sequential Forward Selection (SFS) to reduce the number of wavelengths in the set based on the specified criterion.

        Args:
            save_dir (Optional[Path], optional): Directory to save sets, runtimes and metrics. Defaults to None.

        Returns:
            Tuple[List[NDArray], List[NDArray], List[float]]: Found sets, metrics, compute times for each iteration.
        """

        # determine initial and available set
        if self.initial_wl_set is None:
            # start set is empty and all wavelengths are available
            current_wavelength_idx = np.array([], dtype=int)
            available_wavelength_idx = np.array(range(len(self.all_wavelengths)))
        else:
            # Start set is idx of initial wl set and available set is all wl - initial set
            current_wavelength_idx = np.argwhere(
                np.isin(self.all_wavelengths, self.initial_wl_set)
            ).flatten()
            assert len(current_wavelength_idx) == len(self.initial_wl_set)

            available_wavelength_idx = np.delete(
                np.array(range(len(self.all_wavelengths)), dtype=int),
                current_wavelength_idx,
            )

        found_set = []
        found_metrics = []
        step_compute_times = []

        # logger.info(
        #     f"Running SFS with model={self.light_model}, criterion={self.criterion}, target N={self.target_N}, wl_coarseness={self.wl_coarseness} img_coarseness={self.img_coarseness}"
        # )

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)

        iteration = 0
        while len(current_wavelength_idx) < self.target_N:

            metrics = np.zeros(len(available_wavelength_idx))
            step_start_time = time.time()

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit tasks for each wavelength subset and map results back to the metrics array
                futures = {
                    executor.submit(
                        self.criterion.compute_metric,
                        np.append(
                            current_wavelength_idx,
                            available_wavelength_idx[i],
                        ),  # add wl i to current set
                    ): i
                    for i in range(len(available_wavelength_idx))
                }
                for future in tqdm(futures):
                    i = futures[future]
                    metrics[i] = future.result()

            # add the wl that has the most impact on optimality of criterion
            index_to_add = self.criterion.get_idx_to_add(metrics=metrics)

            # add wl to set and remove from available options
            current_wavelength_idx = np.append(
                current_wavelength_idx, available_wavelength_idx[index_to_add]
            )
            logger.info(
                f"Added WL {available_wavelength_idx[index_to_add]}. New set: {current_wavelength_idx}"
            )
            # delete option
            available_wavelength_idx = np.delete(available_wavelength_idx, index_to_add)

            found_set.append(self.all_wavelengths[current_wavelength_idx])

            # this can be multiple, is this an issue?
            found_metrics.append(metrics[index_to_add])
            step_compute_times.append(time.time() - step_start_time)
            logger.debug(
                f"Iteration {iteration} runtime: {step_compute_times[iteration]}"
            )

            # save intermediate results
            if save_dir is not None:
                iteration_dir = save_dir / f"iteration_{iteration}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                np.save(
                    iteration_dir / "wl.npy",
                    found_set[iteration],
                )
                np.save(
                    iteration_dir / "metrics.npy",
                    found_metrics[iteration],
                )
                np.save(
                    iteration_dir / f"times.npy",
                    step_compute_times[iteration],
                )

            # increment iteration
            iteration += 1

        return found_set, found_metrics, step_compute_times

    def differential_evolution_selection(
        self,
        save_dir: Optional[Path],
    ) -> None:
        """
        Run Differential Evolution to find the optimal set of wavelengths based on the specified criterion.

        Args:
            save_dir (Optional[Path]): Directory to save results
        """
        save_dir.mkdir(parents=True, exist_ok=True)

        def objective(indices: NDArray) -> float:
            valid_indices = np.unique(np.round(indices).astype(int))
            return self.criterion.compute_metric(
                wl_subset_idx=valid_indices,
            )

        def intermediate_result_callback(intermediate_result: OptimizeResult):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            intermediate_folder = save_dir / f"intermediate_results_{timestamp}"
            intermediate_folder.mkdir(parents=True, exist_ok=True)
            np.save(
                intermediate_folder / f"x.npy",
                intermediate_result.x,
            )
            np.save(
                intermediate_folder / f"fun.npy",
                intermediate_result.fun,
            )

        wavelength_count = len(self.all_wavelengths)
        bounds = [(0, wavelength_count - 1)] * self.target_N

        result = differential_evolution(
            objective,
            bounds,
            strategy="best1bin",
            popsize=15,  # Population size
            maxiter=40,  # Maximum number of generations
            # TODO play with mutation and recombination
            seed=constants.SEED,
            callback=intermediate_result_callback,
            integrality=[True] * self.target_N,
        )

        # save final result
        np.save(save_dir / "final_result_x.npy", result.x)
        np.save(save_dir / "final_result_fun.npy", result.fun)
