from abc import ABC, abstractmethod
from typing import List, Optional, Union

import numpy as np
from numpy.typing import NDArray

from src import constants
from src.dataset.helicoid_dataset import HelicoidDataset
from src.scattering_model import scatter
from src.wavelength_selection import metrics
from src.wavelength_selection.enums import LightModel, SampleType, CriterionMetric
from src.molecules import MoleculeMode, Molecules, REFERENCE_SPECTRUM_T1
from src.dataset.biopsy1_dataset import Biopsy1Dataset
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.scattering_model.scatter_optim import single_optim_scattering_with_wl_subset


class Criterion(ABC):

    def __init__(
        self,
        light_model: LightModel,
        sample_type: SampleType,
        samples: List[dict],
        dataset: Union[Biopsy1Dataset, Biopsy2Dataset, HelicoidDataset],
        img_coarseness: int,
        wl_coarseness: int,
        all_wavelengths: NDArray,
        gt_path: str,
        molecule_mode: MoleculeMode = MoleculeMode.ALL,
        metric: Optional[CriterionMetric] = None,
    ):
        self.light_model = light_model
        self.sample_type = sample_type
        self.samples = samples
        self.img_coarseness = img_coarseness
        self.wl_coarseness = wl_coarseness
        self.all_wavelengths = all_wavelengths
        self.molecule_mode = molecule_mode
        self.gt_path = gt_path
        self.metric = metric if metric is not None else CriterionMetric.SV_PROD
        self.dataset = dataset

    @abstractmethod
    def compute_metric(self, wl_subset_idx: NDArray) -> float:
        """
        Compute the metric for a given wavelength subset.

        Args:
            wl_subset_idx (NDArray): Indices of the wavelengths to consider

        Returns:
            float: The computed metric
        """
        pass

    @abstractmethod
    def get_idx_to_remove(self, metrics: List, num_to_remove: int) -> List:
        """
        Get the indices of the wavelengths to remove based on the computed metrics in a SBS strategy.

        Args:
            metrics (List): List of metrics computed for each wavelength subset
            num_to_remove (int): Number of wavelengths to remove

        Returns:
            List: List of indices to remove
        """
        pass

    @abstractmethod
    def get_idx_to_add(self, metrics: List) -> int:
        """
        Get the index of the wavelength to add based on the computed metrics in a SFS strategy.

        Args:
            metrics (List): List of metrics computed for each wavelength subset

        Returns:
            int: Index to add
        """
        pass


class ConcentrationPreservation(Criterion):
    """
    Base the selection process on the preservation of inferred molecular concentrations.
    Computes the NRMSE between the inferred molecular concentrations using the complete wavelength range
    and the inferred molecular concentrations using the subset of wavelengths.

    .. math::
        NRMSE (delta_c - delta_c_reduced)
    """

    def compute_metric(self, wl_subset_idx: NDArray) -> float:
        delta_c_errors_for_wl_subset = []

        for sample in self.samples:
            data = sample

            wavelengths_chosen_mask = np.isin(
                self.dataset.cut_wavelengths, self.all_wavelengths[wl_subset_idx]
            )
            # compute error between inferred and GT delta_c using the subset of wavelengths and chosen light model

            if self.light_model == LightModel.ABSORPTION:
                raise NotImplementedError(
                    "Concentration preservation for absorption model is not implemented."
                )

            elif self.light_model == LightModel.SCATTERING:
                # TODO move error computation in here rather than encapsulated in function

                # TODO: include scatter params in error?
                subset_error, _, _ = single_optim_scattering_with_wl_subset(
                    data=data,
                    dataset=self.dataset,
                    gt_path=self.gt_path,
                    x_chosen=wavelengths_chosen_mask,
                    use_parallel=True,
                    molecule_mode=self.molecule_mode,
                    disable_tqdm=True,
                )
            delta_c_errors_for_wl_subset.append(subset_error)

        mean_error = np.mean(delta_c_errors_for_wl_subset)
        return mean_error

    def get_idx_to_remove(self, metrics: List, num_to_remove: int) -> List:
        return np.argsort(metrics)[:num_to_remove]

    def get_idx_to_add(self, metrics: List) -> int:
        return np.argsort(metrics)[0]


class SpectralFit(Criterion):
    """
    Base the selection process on maximizing the spectral fit.
    Similar to ConcentrationPreservation, but uses the RMSE between the inferred and GT delta A images and is a different mathematical operation.

    .. math::
        RMSE (delta_A - delta_A_reduced)
    """

    def _compute_delta_A_scattering(
        self,
        data: dict,
        wavelengths_chosen_mask: NDArray,
    ) -> NDArray:
        # compute params using only the subset of wavelengths
        _, params_found, _ = single_optim_scattering_with_wl_subset(
            data=data,
            dataset=self.dataset,
            gt_path=self.gt_path,
            x_chosen=wavelengths_chosen_mask,
            use_parallel=True,
            molecule_mode=self.molecule_mode,
            disable_tqdm=True,
        )

        if self.sample_type == SampleType.HELICOID:
            reference_pxl_scatter_a, reference_pxl_scatter_b = (
                self.dataset.load_reference_params(
                    data["id"], load_from_path=self.gt_path
                )
            )
        else:
            # compute delta A for the whole image using the found params with ALL wavelength bands (considering initial coarsening) for M and x
            reference_pxl_scatter_a, reference_pxl_scatter_b = REFERENCE_SPECTRUM_T1[
                self.molecule_mode
            ]

        molecules = Molecules(
            left_cut=self.dataset.left_cut,
            right_cut=self.dataset.right_cut,
            molecule_mode=self.molecule_mode,
            sample_type=self.sample_type,
        )
        num_molecules = molecules.M.shape[1]
        computed_delta_A_img = scatter.compute_delta_A_img(
            M=molecules.M,  # use all wavelengths
            delta_c=params_found[:, :, :num_molecules],  # first n are delta_c
            a=params_found[:, :, num_molecules],  # next is a
            b=params_found[:, :, num_molecules + 1],  # next is b
            a_0=reference_pxl_scatter_a,
            b_0=reference_pxl_scatter_b,
            x=self.all_wavelengths,  # use all wavelengths (not just the subset), using subset is proj. max. approach!
        )

        return computed_delta_A_img

    def compute_metric(self, wl_subset_idx: NDArray) -> float:
        delta_A_errors_for_wl_subset = []

        for sample in self.samples:
            data = sample

            wavelengths_chosen_mask = np.isin(
                self.dataset.cut_wavelengths,
                self.all_wavelengths[wl_subset_idx],
            )

            # compute delta A based on the light model
            if self.light_model == LightModel.ABSORPTION:
                raise NotImplementedError(
                    "Spectral fit for absorption model is not implemented."
                )
            elif self.light_model == LightModel.SCATTERING:
                computed_delta_A_img = self._compute_delta_A_scattering(
                    data=data,
                    wavelengths_chosen_mask=wavelengths_chosen_mask,
                )

            # rmse error between the computed delta A and the gt delta A (relative) image
            # apply same wavelength coarseness as in the optimization (dont confuse with subset of wavelengths! this is the 'full' wavelength range)
            # coarseness is done to reduce the starting set X of available wl for speed up during testing, wl_subset is done to find optimal wavelengths
            coarse_delta_A = data["delta_A"][
                :,
                :,
                :: self.wl_coarseness,
            ]

            rmse_error = metrics.rmse(
                y_true=coarse_delta_A,
                y_pred=computed_delta_A_img,
            )
            delta_A_errors_for_wl_subset.append(rmse_error)

        mean_error = np.mean(delta_A_errors_for_wl_subset)
        return mean_error

    def get_idx_to_remove(self, metrics: List, num_to_remove: int) -> List:
        return np.argsort(metrics)[:num_to_remove]

    def get_idx_to_add(self, metrics):
        return np.argsort(metrics)[0]
