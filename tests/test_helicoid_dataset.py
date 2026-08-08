#!/usr/bin/env python3
"""
Test script for HelicoidDataset
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from src.dataset.helicoid_dataset import HelicoidDataset
from src.constants import HELICOID_WAVELENGTHS


def test_helicoid_dataset():
    """Test basic functionality of HelicoidDataset"""
    print("Testing HelicoidDataset...")

    # Create dataset instance
    dataset = HelicoidDataset(
        coarseness=1,  # Use coarseness to reduce memory usage for testing
        with_delta_A=False,  # Don't compute delta A for initial test
        normalize_image=True,
        with_rgb=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Patient IDs: {dataset.get_patient_ids()[:5]}...")  # Show first 5
    print(HELICOID_WAVELENGTHS.shape)

    # Test getting a sample by index
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nFirst sample:")
        print(f"  Patient ID: {sample['id']}")
        print(f"  HSI cube shape: {sample['hsi_cube'].shape}")
        print(f"  GT map shape: {sample['gt_map'].shape}")
        print(f"  RGB image size: {sample['rgb_image'].size}")
        print(f"  Wavelengths shape: {sample['wavelengths'].shape}")
        print(f"  Delta A: {sample['delta_A'] is not None}")

        # Test getting sample by ID
        sample_by_id = dataset.get_sample_by_id(sample["id"])
        print(
            f"  Retrieved by ID: {sample_by_id['id'] == sample['id']}"
        )

    # Test with delta A
    print("\nTesting with delta A computation...")
    dataset_delta = HelicoidDataset(
        coarseness=8,  # Even more coarseness for delta A test
        with_delta_A=True,
        normalize_image=True,
        with_rgb=True
    )

    if len(dataset_delta) > 0:
        sample_delta = dataset_delta[0]
        print(f"  Delta A computed: {sample_delta['delta_A'] is not None}")
        if sample_delta["delta_A"] is not None:
            print(f"  Delta A shape: {sample_delta['delta_A'].shape}")
            print(f"  Reference pixel: {sample_delta['reference_pixel']}")

    print("\nHelicoidDataset test completed successfully!")


if __name__ == "__main__":
    test_helicoid_dataset()
