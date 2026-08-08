"""
Abstract base class for all HSI datasets.

This module provides a common interface for all dataset classes in the HSI biopsy project,
ensuring consistency across different data types (Biopsy1, Biopsy2, HELICOID, etc.).
"""

from abc import ABC, abstractmethod
import typing
from torch.utils.data import Dataset


class BaseHSIDataset(Dataset, ABC):
    """
    Abstract base class for all HSI datasets.
    
    This class defines the common interface that all dataset implementations should follow.
    It extends PyTorch's Dataset class and provides abstract methods for patient and sample
    management functionality.
    """

    @abstractmethod
    def get_patient_ids(self) -> typing.List[str]:
        """
        Get a list of all unique patient IDs in the dataset.

        Returns
        -------
        list of str
            Sorted list of unique patient IDs.
        """
        raise NotImplementedError("Subclasses must implement get_patient_ids()")

    @abstractmethod
    def get_sample_by_id(self, sample_id: str) -> typing.Union[dict, None]:
        """
        Retrieve a sample by its ID.

        Parameters
        ----------
        sample_id : str
            The sample ID to retrieve.

        Returns
        -------
        dict or None
            The sample data or None if not found.
        """
        raise NotImplementedError("Subclasses must implement get_sample_by_id()")

    @abstractmethod
    def get_samples_by_patient_id(self, patient_id: str) -> typing.List[dict]:
        """
        Retrieve all samples for a specific patient ID.

        Parameters
        ----------
        patient_id : str
            The patient ID to get samples for.

        Returns
        -------
        list of dict
            A list of all samples for the given patient, empty if none found.
        """
        raise NotImplementedError("Subclasses must implement get_samples_by_patient_id()")
