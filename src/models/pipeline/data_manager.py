"""
Data management for the model pipeline.

This module handles dataset creation, splitting, and data loader setup.
"""

import random
from loguru import logger
import numpy
import torch
from torch.utils.data import DataLoader
from typing import Tuple, Optional
from src.constants import HELICOID_DATA_DIR, HELICOID_SELECTED_WAVELENGTHS_CP
import copy


from src.dataset.dataset_wrapper import (
    AugmentedConcentrationsDataset,
    MoleculeFilterDataset,
    MultiResReconstructionDataset,
)
from src.dataset.dataset_utils import (
    collate_dataclass,
    get_random_augmentation_transform,
    get_random_split_ids,
)
from src.dataset.fused_helicoid_concentration_dataset import (
    FusedHelicoidConcentrationDataset,
)
from src.molecules import MoleculeIndex
from src.wavelength_selection.enums import SampleType
from src.dataset.concentrations_dataset import (
    ConcentrationsDataset,
)
from monai.transforms import NormalizeIntensity
from .config import ModelConfig
from torchvision.transforms import RandomCrop, Compose


class DataManager:
    """Manages dataset creation and data loading for the model pipeline."""

    def __init__(self, config: ModelConfig):
        """
        Initialize the data manager.

        Parameters
        ----------
        config : ModelConfig
            Configuration object containing data parameters.
        """
        self.config = config
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.dataset = None

    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Setup datasets and data loaders.

        Returns
        -------
        Tuple[DataLoader, DataLoader, DataLoader]
            Train, validation, and test data loaders.
        """
        # Create base dataset
        self.dataset = self._create_dataset()

        # Print dataset summary
        summary = self.dataset.get_summary()
        logger.info(f"Dataset summary: {summary}")

        # Split dataset
        train_dataset, val_dataset, test_dataset = self._split_dataset()

        # Apply multi-resolution training if configured
        if self.config.multires_training:
            train_dataset = self._apply_multires_training(train_dataset)

        # Validate batch size for original resolution training
        self._validate_batch_size()

        # Apply molecule filtering if needed
        train_dataset, val_dataset, test_dataset = self._apply_molecule_filtering(
            train_dataset, val_dataset, test_dataset
        )

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        # Create data loaders
        self.train_loader, self.val_loader, self.test_loader = (
            self._create_data_loaders(train_dataset, val_dataset, test_dataset)
        )

        self._log_data_loader_info()

        return self.train_loader, self.val_loader, self.test_loader

    def _create_dataset(
        self,
    ) -> ConcentrationsDataset | FusedHelicoidConcentrationDataset:
        """Create the base concentrations dataset."""
        sample_type = (
            SampleType.HELICOID
            if self.config.sample_type == "helicoid"
            else SampleType.BIOPSY2
        )

        dataset = None
        shared_args = {
            "reduced_wl_dir": self.config.reduced_wl_dir,
            "gt_dir": self.config.gt_dir,
            "center_crop_size": self.config.center_crop_size,
            "normalization_dir": self.config.normalization_dir,
            "inference_mode": self.config.train_on_original_res,
            "channels_first": self.config.channels_first,
            "merge_channels_to_signals": self.config.merge_channels_to_signals,
            "image_wise_normalization": self.config.image_wise_normalization,
            "ids_subset": None,
        }

        if self.config.autoencoder_type == "DualEncoder" or self.config.ssr:

            dataset = FusedHelicoidConcentrationDataset(
                **shared_args,
                helicoid_data_dir=self.config.fused_helicoid_config.get(
                    "helicoid_data_dir", HELICOID_DATA_DIR
                ),
                left_cut=self.config.fused_helicoid_config.get("left_cut", 530),
                right_cut=self.config.fused_helicoid_config.get("right_cut", 750),
                selected_wavelengths=self.config.fused_helicoid_config.get(
                    "selected_wavelengths", HELICOID_SELECTED_WAVELENGTHS_CP
                ),
                coarseness=self.config.fused_helicoid_config.get("coarseness", 1),
                normalize_image=self.config.fused_helicoid_config.get(
                    "normalize_image", True
                ),
                downsample_factor=self.config.fused_helicoid_config.get(
                    "downsample_factor", None
                ),
                transform=self.config.fused_helicoid_config.get("transform", None),
            )
        else:
            dataset = ConcentrationsDataset(
                **shared_args,
                sample_type=sample_type,
            )

        return dataset

    def _create_train_dataset(
        self, train_ids
    ) -> (
        ConcentrationsDataset
        | FusedHelicoidConcentrationDataset
        | AugmentedConcentrationsDataset
    ):
        """Create the training dataset."""
        sample_type = (
            SampleType.HELICOID
            if self.config.sample_type == "helicoid"
            else SampleType.BIOPSY2
        )

        if self.config.autoencoder_type == "DualEncoder" or self.config.ssr:
            train_dataset = FusedHelicoidConcentrationDataset(
                helicoid_data_dir=self.config.fused_helicoid_config.get(
                    "helicoid_data_dir", HELICOID_DATA_DIR
                ),
                reduced_wl_dir=self.config.reduced_wl_dir,
                gt_dir=self.config.gt_dir,
                left_cut=self.config.fused_helicoid_config.get("left_cut", 530),
                right_cut=self.config.fused_helicoid_config.get("right_cut", 750),
                selected_wavelengths=self.config.fused_helicoid_config.get(
                    "selected_wavelengths", HELICOID_SELECTED_WAVELENGTHS_CP
                ),
                coarseness=self.config.fused_helicoid_config.get("coarseness", 1),
                normalize_image=self.config.fused_helicoid_config.get(
                    "normalize_image", True
                ),
                downsample_factor=self.config.fused_helicoid_config.get(
                    "downsample_factor", None
                ),
                center_crop_size=self.config.center_crop_size,
                normalization_dir=self.config.normalization_dir,
                transform=self.config.fused_helicoid_config.get("transform", None),
                channels_first=self.config.channels_first,
                inference_mode=self.config.train_on_original_res,
                image_wise_normalization=self.config.image_wise_normalization,
                ids_subset=train_ids,
            )
        else:
            train_dataset = ConcentrationsDataset(
                reduced_wl_dir=self.config.reduced_wl_dir,
                gt_dir=self.config.gt_dir,
                sample_type=sample_type,
                center_crop_size=self.config.center_crop_size,
                normalization_dir=self.config.normalization_dir,
                inference_mode=self.config.train_on_original_res,
                channels_first=self.config.channels_first,
                merge_channels_to_signals=self.config.merge_channels_to_signals,
                image_wise_normalization=self.config.image_wise_normalization,
                ids_subset=train_ids,
            )

            if self.config.augmentation_ratio > 0:
                logger.info(
                    f"Applying data augmentation with ratio: {self.config.augmentation_ratio}"
                )
                random_augmentation_transform = get_random_augmentation_transform(
                    augmentation_ratio=self.config.augmentation_ratio
                )
                train_dataset = AugmentedConcentrationsDataset(
                    base_dataset=train_dataset,
                    augmentation_transform=random_augmentation_transform,
                )
            elif self.config.random_crop_size is not None:
                logger.info(
                    f"Applying random cropping with size: {self.config.random_crop_size} "
                )
                if self.config.center_crop_size:
                    logger.warning(
                        "Both center_crop_size and random_crop_size are set. "
                        "Center cropping will be ignored during training."
                    )
                train_dataset.set_inference_mode(True)
                random_crop_transform_list = [RandomCrop(self.config.random_crop_size)]

                if self.config.image_wise_normalization:
                    # We want to normalize the random crop
                    train_dataset.image_wise_normalization = False
                    random_crop_transform_list.append(
                        NormalizeIntensity(channel_wise=True)
                    )

                random_crop_transform = Compose(random_crop_transform_list)

                train_dataset = AugmentedConcentrationsDataset(
                    base_dataset=train_dataset,
                    augmentation_transforms=random_crop_transform,
                )

        return train_dataset

    def _create_val_and_test_datasets(self, val_ids_set, test_ids_set) -> Tuple[
        ConcentrationsDataset | FusedHelicoidConcentrationDataset,
        ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    ]:
        """Create the validation and test datasets."""
        sample_type = (
            SampleType.HELICOID
            if self.config.sample_type == "helicoid"
            else SampleType.BIOPSY2
        )

        base_dataset_args = {
            "reduced_wl_dir": self.config.reduced_wl_dir,
            "gt_dir": self.config.gt_dir,
            "center_crop_size": self.config.center_crop_size,
            "normalization_dir": self.config.normalization_dir,
            "inference_mode": self.config.evaluate_full_resolution,
            "channels_first": self.config.channels_first,
            "merge_channels_to_signals": self.config.merge_channels_to_signals,
            "image_wise_normalization": self.config.image_wise_normalization,
        }

        if self.config.autoencoder_type == "DualEncoder" or self.config.ssr:
            base_dataset_args.update(
                {
                    "helicoid_data_dir": self.config.fused_helicoid_config.get(
                        "helicoid_data_dir", HELICOID_DATA_DIR
                    ),
                    "left_cut": self.config.fused_helicoid_config.get("left_cut", 530),
                    "right_cut": self.config.fused_helicoid_config.get(
                        "right_cut", 750
                    ),
                    "selected_wavelengths": self.config.fused_helicoid_config.get(
                        "selected_wavelengths", HELICOID_SELECTED_WAVELENGTHS_CP
                    ),
                    "coarseness": self.config.fused_helicoid_config.get(
                        "coarseness", 1
                    ),
                    "normalize_image": self.config.fused_helicoid_config.get(
                        "normalize_image", True
                    ),
                    "downsample_factor": self.config.fused_helicoid_config.get(
                        "downsample_factor", None
                    ),
                    "transform": None,  # No augmentation for val/test
                }
            )

            val_dataset = FusedHelicoidConcentrationDataset(
                **base_dataset_args,
                ids_subset=val_ids_set,
            )

            test_dataset = FusedHelicoidConcentrationDataset(
                **base_dataset_args,
                ids_subset=test_ids_set,
            )

        else:
            val_dataset = ConcentrationsDataset(
                **base_dataset_args,
                ids_subset=val_ids_set,
                sample_type=sample_type,
            )

            test_dataset = ConcentrationsDataset(
                **base_dataset_args,
                ids_subset=test_ids_set,
                sample_type=sample_type,
            )

        return val_dataset, test_dataset

    def _split_dataset(
        self,
    ) -> Tuple[
        ConcentrationsDataset
        | FusedHelicoidConcentrationDataset
        | AugmentedConcentrationsDataset,
        ConcentrationsDataset | FusedHelicoidConcentrationDataset,
        ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    ]:
        """Split dataset into train, validation, and test sets."""

        if self.config.validation_ids is not None and self.config.test_ids is not None:
            # Use provided patient IDs for validation and testing
            logger.info(
                f"Using provided validation IDs: {self.config.validation_ids} "
                f"and test IDs: {self.config.test_ids}"
            )
            all_ids = set(self.dataset.sample_map)
            val_ids_set = set(self.config.validation_ids)
            test_ids_set = set(self.config.test_ids)
            train_ids_set = all_ids - val_ids_set - test_ids_set
        else:
            # Randomly split dataset based on provided ratios
            logger.info(
                f"Splitting dataset with train/val/test ratios: {self.config.split_ratios}"
            )
            train_ids_set, val_ids_set, test_ids_set = get_random_split_ids(
                self.dataset,
                split_ratios=self.config.split_ratios,
                random_seed=self.config.random_seed,
            )

        train_dataset = self._create_train_dataset(train_ids_set)
        val_dataset, test_dataset = self._create_val_and_test_datasets(
            val_ids_set, test_ids_set
        )
        return train_dataset, val_dataset, test_dataset

    def _apply_multires_training(self, train_dataset) -> MultiResReconstructionDataset:
        """Apply multi-resolution training wrapper."""
        train_dataset = MultiResReconstructionDataset(
            train_dataset,
            full_size=self.config.center_crop_size[0],
            crop_sizes=self.config.multires_crop_sizes,
        )
        logger.info(
            f"Multi-resolution training enabled with crop sizes: {self.config.multires_crop_sizes}"
        )
        return train_dataset

    def _validate_batch_size(self):
        """Validate batch size for original resolution training."""
        if self.config.train_on_original_res and self.config.batch_size != 1:
            raise ValueError(
                "When training on original resolution, batch size must be 1 to avoid memory issues."
            )

    def _apply_molecule_filtering(
        self, train_dataset, val_dataset, test_dataset
    ) -> Tuple:
        """Apply molecule filtering if chosen_molecules is specified."""
        molecule_indices = self._get_molecule_indices()
        if molecule_indices is not None:
            logger.info(
                f"Filtering molecules: {self.config.chosen_molecules} (indices: {molecule_indices})"
            )
            train_dataset = MoleculeFilterDataset(train_dataset, molecule_indices)
            val_dataset = MoleculeFilterDataset(val_dataset, molecule_indices)
            test_dataset = MoleculeFilterDataset(test_dataset, molecule_indices)

        return train_dataset, val_dataset, test_dataset

    def get_full_res_test_loader(self) -> DataLoader:
        """Get a test data loader for full resolution evaluation."""
        if self.test_dataset is None:
            raise ValueError("Data loaders have not been set up yet.")

        full_res_test_dataset = copy.deepcopy(self.test_dataset)

        full_res_test_dataset.set_inference_mode(True)

        full_res_test_loader = DataLoader(
            full_res_test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_dataclass,
            pin_memory=self.config.pin_memory,
        )

        return full_res_test_loader

    def _create_data_loaders(
        self, train_dataset, val_dataset, test_dataset
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create data loaders for train, validation, and test sets."""
        g = torch.Generator()
        g.manual_seed(self.config.random_seed)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            worker_init_fn=seed_worker,
            persistent_workers=False,
            collate_fn=collate_dataclass,
            generator=g,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_dataclass,
            pin_memory=self.config.pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=collate_dataclass,
            pin_memory=self.config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _log_data_loader_info(self):
        """Log information about the created data loaders."""
        logger.info(
            f"Data loaders created - Train: {len(self.train_loader)} batches, "
            f"Val: {len(self.val_loader)} batches, Test: {len(self.test_loader)} batches"
        )
        logger.debug(
            f"Train dataset patient IDs: {self.train_dataset.get_patient_ids()}"
        )
        logger.debug(f"Val dataset patient IDs: {self.val_dataset.get_patient_ids()}")
        logger.debug(f"Test dataset patient IDs: {self.test_dataset.get_patient_ids()}")

    def _get_molecule_indices(self) -> Optional[list]:
        """Get indices of chosen molecules."""
        if self.config.chosen_molecules is None:
            return None

        return [
            MoleculeIndex[mol_name].value for mol_name in self.config.chosen_molecules
        ]
