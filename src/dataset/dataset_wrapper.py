from typing import List, Optional, Union
import torch
from torch.utils.data import Dataset


import random

from src.dataset.base_dataset import BaseHSIDataset
from src.dataset.concentrations_dataset import ConcentrationsDataset

from torchvision.transforms import Compose


class MoleculeFilterDataset:
    """Wrapper dataset that filters molecules based on chosen_molecules."""

    def __init__(self, base_dataset, molecule_indices):
        self.base_dataset = base_dataset
        self.molecule_indices = molecule_indices

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        if self.molecule_indices is not None:
            # Filter molecules for both reduced_wl and gt data
            sample["reduced_wl"]["coef_list"] = sample["reduced_wl"]["coef_list"][
                ..., self.molecule_indices
            ]
            sample["gt"]["coef_list"] = sample["gt"]["coef_list"][
                ..., self.molecule_indices
            ]
        return sample


class MultiResReconstructionDataset(Dataset):
    """
    Wraps a base dataset returning (input_img, target_img), each H×W×C,
    and for each sample returns:
      inputs  = [full, crop128, crop64]
      targets = [full, crop128, crop64]
    All crops share the same random location in input and target.
    Pixel values are untouched; only HWC→CHW reordering is done.
    """

    def __init__(self, base_dataset, full_size: int = 224, crop_sizes=(128, 64)):
        self.base_dataset = base_dataset
        self.full_size = full_size
        self.crop_sizes = crop_sizes

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]

        # Extract input and target from the ConcentrationsDataset sample
        input = sample["reduced_wl"]["coef_list"]  # Shape: (H, W, C)
        target = sample["gt"]["coef_list"]  # Shape: (H, W, C)

        # 3) full-resolution CHW tensors
        full_inp = torch.from_numpy(input).permute(2, 0, 1)
        full_tgt = torch.from_numpy(target).permute(2, 0, 1)

        h, w, c = input.shape
        # 4) random crops (same coords for inp & tgt)
        crops_inp = []
        crops_tgt = []
        for sz in self.crop_sizes:
            top = random.randint(0, h - sz)
            left = random.randint(0, w - sz)
            sub_in = input[top : top + sz, left : left + sz, :]
            sub_t = target[top : top + sz, left : left + sz, :]
            crops_inp.append(torch.from_numpy(sub_in).permute(2, 0, 1))
            crops_tgt.append(torch.from_numpy(sub_t).permute(2, 0, 1))

        inputs = [full_inp] + crops_inp
        targets = [full_tgt] + crops_tgt

        # return two parallel lists of tensors
        return inputs, targets


class AugmentedConcentrationsDataset(BaseHSIDataset):
    """
    A wrapper around ConcentrationsDataset that applies random augmentations to the data.

    This dataset wrapper applies the same transformations to both reduced wavelength
    and ground truth concentration maps to maintain consistency between inputs and targets.
    """

    def __init__(
        self,
        base_dataset: ConcentrationsDataset,
        augmentation_transforms: Optional[Compose] = None,
    ):
        """
        Initialize the AugmentedConcentrationsDataset.

        Parameters
        ----------
        base_dataset : Dataset
            The base dataset (usually a Subset of ConcentrationsDataset).
        augmentation_transforms : Compose, optional
            Composed torchvision transforms to apply to the concentration maps.
            If None, no augmentations are applied.
        """
        self.base_dataset = base_dataset
        self.augmentation_transforms = augmentation_transforms

    def get_patient_ids(self) -> List[str]:
        """Get a list of all unique patient IDs in the base dataset."""
        return self.base_dataset.get_patient_ids()

    def get_sample_by_id(self, sample_id: str) -> Union[dict, None]:
        """Get a sample from the base dataset by its ID."""
        return self.base_dataset.get_sample_by_id(sample_id)

    def get_samples_by_patient_id(self, patient_id: str) -> List[dict]:
        """Get all samples for a specific patient ID from the base dataset."""
        return self.base_dataset.get_samples_by_patient_id(patient_id)

    def __len__(self) -> int:
        """Returns the number of samples in the base dataset."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve a sample by index with optional augmentations applied.

        Parameters
        ----------
        idx : int
            Index of the sample to retrieve.

        Returns
        -------
        dict
            Sample dictionary with augmented concentration maps if transforms are enabled.
        """
        # Get the original sample
        sample = self.base_dataset[idx]

        # Apply augmentations if specified
        if self.augmentation_transforms is not None:
            # Set the same random seed for both reduced_wl and gt to ensure consistency
            seed = torch.randint(0, 2**32, (1,)).item()

            # Apply transforms to reduced_wl coef_list
            torch.manual_seed(seed)
            reduced_wl_coef = (
                torch.from_numpy(sample["reduced_wl"]["coef_list"])
                .permute(2, 0, 1)
                .float()
            )  # (C, H, W)
            augmented_reduced_wl_coef = self.augmentation_transforms(reduced_wl_coef)
            sample["reduced_wl"]["coef_list"] = augmented_reduced_wl_coef.permute(
                1, 2, 0
            ).numpy()  # (H, W, C)

            # Apply the same transforms to gt coef_list
            torch.manual_seed(seed)
            gt_coef = (
                torch.from_numpy(sample["gt"]["coef_list"]).permute(2, 0, 1).float()
            )  # (C, H, W)
            augmented_gt_coef = self.augmentation_transforms(gt_coef)
            sample["gt"]["coef_list"] = augmented_gt_coef.permute(
                1, 2, 0
            ).numpy()  # (H, W, C)

            # Apply transforms to scatter_params if available
            if (
                "scatter_params" in sample["reduced_wl"]
                and "scatter_params" in sample["gt"]
            ):
                torch.manual_seed(seed)
                reduced_wl_scatter = (
                    torch.from_numpy(sample["reduced_wl"]["scatter_params"])
                    .permute(2, 0, 1)
                    .float()
                )
                augmented_reduced_wl_scatter = self.augmentation_transforms(
                    reduced_wl_scatter
                )
                sample["reduced_wl"]["scatter_params"] = (
                    augmented_reduced_wl_scatter.permute(1, 2, 0).numpy()
                )

                torch.manual_seed(seed)
                gt_scatter = (
                    torch.from_numpy(sample["gt"]["scatter_params"])
                    .permute(2, 0, 1)
                    .float()
                )
                augmented_gt_scatter = self.augmentation_transforms(gt_scatter)
                sample["gt"]["scatter_params"] = augmented_gt_scatter.permute(
                    1, 2, 0
                ).numpy()

        return sample
