import torch
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomRotation,
    RandomVerticalFlip,
)
from src.dataset.dataset_wrapper import (
    AugmentedConcentrationsDataset,
    MoleculeFilterDataset,
    MultiResReconstructionDataset,
)
from src.dataset.concentrations_dataset import (
    ConcentrationsDataset,
)

from dataclasses import is_dataclass, fields
from torch.utils.data._utils.collate import default_collate
import numpy as np
from torch.utils.data import Subset


import random
import typing

from src.dataset.fused_helicoid_concentration_dataset import (
    FusedHelicoidConcentrationDataset,
)


def split_dataset(
    dataset: ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    split_ratios: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
) -> typing.Tuple[Subset, Subset, Subset]:
    """
    Split a ConcentrationsDataset into train, validation, and test sets by patient ID.

    This function ensures that all samples from the same patient are kept together
    in the same split to prevent data leakage. The split is deterministic when
    using the same random_seed.

    Parameters
    ----------
    dataset : ConcentrationsDataset
        The dataset to split.
    split_ratios : tuple of float, optional
        The desired ratios for (train, val, test) splits. Default is (0.8, 0.1, 0.1).
        Ratios should sum to 1.0. Actual splits will be approximate due to patient-wise splitting.
    random_seed : int, optional
        Random seed for deterministic splitting. Default is 42.

    Returns
    -------
    tuple of torch.utils.data.Subset
        A tuple containing (train_dataset, val_dataset, test_dataset).

    Raises
    ------
    ValueError
        If split_ratios don't sum to 1.0 or if any ratio is negative.

    Examples
    --------
    >>> train_ds, val_ds, test_ds = split_dataset(dataset, (0.7, 0.15, 0.15))
    >>> print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    """
    # Validate split ratios
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(sum(split_ratios), 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    if any(ratio < 0 for ratio in split_ratios):
        raise ValueError("Split ratios must be non-negative")

    # Get all unique patient IDs
    patient_ids = dataset.get_patient_ids()
    num_patients = len(patient_ids)

    if num_patients < 3:
        raise ValueError(
            f"Need at least 3 patients for train/val/test split, got {num_patients}"
        )

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle patient IDs deterministically
    shuffled_patients = patient_ids.copy()
    random.shuffle(shuffled_patients)

    # Calculate split points based on number of patients
    train_split = int(num_patients * train_ratio)
    val_split = int(num_patients * (train_ratio + val_ratio))

    # Ensure at least one patient in each split
    train_split = max(1, train_split)
    val_split = max(train_split + 1, min(val_split, num_patients - 1))

    # Split patients
    train_patients = shuffled_patients[:train_split]
    val_patients = shuffled_patients[train_split:val_split]
    test_patients = shuffled_patients[val_split:]

    # Get indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for idx, sample in enumerate(dataset.samples):
        patient_id = sample["patient_id"]

        if patient_id in train_patients:
            train_indices.append(idx)
        elif patient_id in val_patients:
            val_indices.append(idx)
        elif patient_id in test_patients:
            test_indices.append(idx)

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Print split summary
    total_samples = len(dataset)
    print(f"Dataset split summary:")
    print(f"  Total patients: {num_patients}")
    print(f"  Total samples: {total_samples}")
    print(
        f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples "
        f"({len(train_dataset)/total_samples:.1%})"
    )
    print(
        f"  Val: {len(val_patients)} patients, {len(val_dataset)} samples "
        f"({len(val_dataset)/total_samples:.1%})"
    )
    print(
        f"  Test: {len(test_patients)} patients, {len(test_dataset)} samples "
        f"({len(test_dataset)/total_samples:.1%})"
    )
    print(f"  Random seed: {random_seed}")

    return train_dataset, val_dataset, test_dataset


def get_random_split_ids(
    dataset: ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    split_ratios: typing.Tuple[float, float, float] = (0.8, 0.1, 0.1),
    random_seed: int = 42,
):
    train_ratio, val_ratio, test_ratio = split_ratios
    if not np.isclose(sum(split_ratios), 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {sum(split_ratios)}")
    if any(ratio < 0 for ratio in split_ratios):
        raise ValueError("Split ratios must be non-negative")

    # Get all unique patient IDs
    patient_ids = dataset.get_patient_ids()
    num_patients = len(patient_ids)

    if num_patients < 3:
        raise ValueError(
            f"Need at least 3 patients for train/val/test split, got {num_patients}"
        )

    # Set random seed for reproducibility
    random.seed(random_seed)

    # Shuffle patient IDs deterministically
    shuffled_patients = patient_ids.copy()
    random.shuffle(shuffled_patients)

    # Calculate split points based on number of patients
    train_split = int(num_patients * train_ratio)
    val_split = int(num_patients * (train_ratio + val_ratio))

    # Ensure at least one patient in each split
    train_split = max(1, train_split)
    val_split = max(train_split + 1, min(val_split, num_patients - 1))

    # Split patients
    train_patients = shuffled_patients[:train_split]
    val_patients = shuffled_patients[train_split:val_split]
    test_patients = shuffled_patients[val_split:]

    train_ids = []
    val_ids = []
    test_ids = []

    for sample in dataset.samples:
        patient_id = sample["patient_id"]
        sample_id = sample["id"]
        if patient_id in train_patients:
            train_ids.append(sample_id)
        elif patient_id in val_patients:
            val_ids.append(sample_id)
        elif patient_id in test_patients:
            test_ids.append(sample_id)

    print(f"Dataset split summary:")
    print(f"  Total patients: {num_patients}")
    print(f"  Total samples: {len(dataset)}")
    print(
        f"  Train: {len(train_patients)} patients, {len(train_ids)} samples "
        f"({len(train_ids)/len(dataset):.1%})"
    )
    print(
        f"  Val: {len(val_patients)} patients, {len(val_ids)} samples "
        f"({len(val_ids)/len(dataset):.1%})"
    )
    print(
        f"  Test: {len(test_patients)} patients, {len(test_ids)} samples "
        f"({len(test_ids)/len(dataset):.1%})"
    )
    print(f"  Random seed: {random_seed}")
    return train_ids, val_ids, test_ids


def get_patient_ids_from_subset(
    dataset: ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    subset: typing.Union[Subset, AugmentedConcentrationsDataset, MoleculeFilterDataset],
) -> set:
    """
    Get patient IDs from a subset or augmented dataset.

    Parameters
    ----------
    dataset : ConcentrationsDataset
        The original dataset.
    subset : Subset or AugmentedConcentrationsDataset
        The subset or augmented dataset to get patient IDs from.

    Returns
    -------
    set
        Set of unique patient IDs in the subset.
    """
    patient_ids = set()

    if (
        isinstance(subset, AugmentedConcentrationsDataset)
        or isinstance(subset, MultiResReconstructionDataset)
        or isinstance(subset, MoleculeFilterDataset)
    ):
        # For augmented datasets, get indices from the base dataset
        if isinstance(subset.base_dataset, Subset):
            indices = subset.base_dataset.indices
        else:
            # If base dataset is the full dataset, use all indices
            indices = range(len(subset.base_dataset))
    elif isinstance(subset, Subset):
        # For regular subsets, use the indices directly
        indices = subset.indices
    else:
        raise TypeError(
            f"Expected Subset or AugmentedConcentrationsDataset, got {type(subset)}"
        )

    for idx in indices:
        sample = dataset.samples[idx]
        patient_ids.add(sample["patient_id"])
    return patient_ids


def split_dataset_with_ids(
    dataset: ConcentrationsDataset | FusedHelicoidConcentrationDataset,
    val_ids: typing.List[str],
    test_ids: typing.List[str],
    indices_only: bool = False,
) -> typing.Tuple[Subset, Subset, Subset]:
    """
    Split a ConcentrationsDataset into train, validation, and test sets using specified sample IDs.

    This function allows you to specify exactly which samples should be in the validation
    and test sets by their IDs. The remaining samples will be used for training. This is
    useful when you want precise control over the dataset splits using human-readable
    sample identifiers.

    Parameters
    ----------
    dataset : ConcentrationsDataset
        The dataset to split.
    val_ids : list of str
        List of sample IDs to include in the validation set.
        For BIOPSY2: Format like "1.2_3" (patient S1.2, FOV 3)
        For HELICOID: Format like "004-02" (patient 004, FOV 2)
    test_ids : list of str
        List of sample IDs to include in the test set.
        Same format as val_ids.

    Returns
    -------
    tuple of torch.utils.data.Subset
        A tuple containing (train_dataset, val_dataset, test_dataset).

    Raises
    ------
    ValueError
        If there are overlapping IDs between val_ids and test_ids,
        or if any sample ID is not found in the dataset.

    Examples
    --------
    >>> # For BIOPSY2 data
    >>> val_ids = ["1.2_1", "1.3_2", "1.4_1"]
    >>> test_ids = ["1.5_1", "1.6_2", "1.7_1"]
    >>> train_ds, val_ds, test_ds = split_dataset_with_ids(dataset, val_ids, test_ids)
    >>>
    >>> # For HELICOID data
    >>> val_ids = ["004-01", "005-02"]
    >>> test_ids = ["006-01", "007-02"]
    >>> train_ds, val_ds, test_ds = split_dataset_with_ids(dataset, val_ids, test_ids)
    """
    # Check for overlapping IDs
    val_set = set(val_ids)
    test_set = set(test_ids)
    overlap = val_set.intersection(test_set)

    if overlap:
        raise ValueError(
            f"Overlapping sample IDs found between validation and test sets: {overlap}"
        )

    # Convert sample IDs to indices
    val_indices = []
    test_indices = []

    # Validate that all specified IDs exist in the dataset
    all_specified_ids = val_set.union(test_set)
    missing_ids = []

    for sample_id in all_specified_ids:
        if sample_id not in dataset.sample_map:
            missing_ids.append(sample_id)

    if missing_ids:
        raise ValueError(
            f"The following sample IDs were not found in the dataset: {missing_ids}"
        )

    # Convert IDs to indices
    for sample_id in val_ids:
        val_indices.append(dataset.sample_map[sample_id])

    for sample_id in test_ids:
        test_indices.append(dataset.sample_map[sample_id])

    # Create train indices as all remaining indices
    specified_indices = set(val_indices + test_indices)
    total_samples = len(dataset)
    train_indices = [i for i in range(total_samples) if i not in specified_indices]

    # Create subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    # Get patient information for summary
    train_patients = get_patient_ids_from_subset(dataset, train_dataset)
    val_patients = get_patient_ids_from_subset(dataset, val_dataset)
    test_patients = get_patient_ids_from_subset(dataset, test_dataset)

    # Print split summary
    print(f"Dataset split summary (custom sample IDs):")
    print(f"  Total samples: {total_samples}")
    print(f"  Validation IDs: {val_ids}")
    print(f"  Test IDs: {test_ids}")
    print(
        f"  Train: {len(train_patients)} patients, {len(train_dataset)} samples "
        f"({len(train_dataset)/total_samples:.1%})"
    )
    print(
        f"  Val: {len(val_patients)} patients, {len(val_dataset)} samples "
        f"({len(val_dataset)/total_samples:.1%})"
    )
    print(
        f"  Test: {len(test_patients)} patients, {len(test_dataset)} samples "
        f"({len(test_dataset)/total_samples:.1%})"
    )
    if indices_only:
        return train_indices, val_indices, test_indices

    return train_dataset, val_dataset, test_dataset


def get_random_augmentation_transform(augmentation_ratio: float):
    """
    Create a random augmentation transform that applies augmentations with a given probability.

    Parameters
    ----------
    augmentation_ratio : float
        Probability of applying augmentations (between 0 and 1).

    Returns
    -------
    function
        A function that takes an image and applies random augmentations based on the probability.
    """
    # Create a transform that randomly applies augmentations based on augmentation_ratio
    available_transforms = [
        RandomHorizontalFlip(p=1.0),
        RandomVerticalFlip(p=1.0),
        RandomRotation(degrees=90),  # 90-degree rotations only
    ]

    # Create a custom transform that applies augmentation with the specified probability
    def random_augmentation_transform(img):
        """Apply random augmentation with specified probability."""
        if torch.rand(1).item() < augmentation_ratio:
            # Randomly select one of the available transforms
            transform_idx = torch.randint(0, len(available_transforms), (1,)).item()
            selected_transform = available_transforms[transform_idx]
            return selected_transform(img)
        return img

    return random_augmentation_transform


def collate_dataclass(batch):
    elem = batch[0]
    if is_dataclass(elem):
        collated = {
            f.name: default_collate([getattr(x, f.name) for x in batch])
            for f in fields(elem)
        }
        return type(elem)(**collated)  # reconstruct same dataclass type
    return default_collate(batch)