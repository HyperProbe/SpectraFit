#!/usr/bin/env python3
"""
Test script for the split_dataset function.
"""

import sys
import os

from src.dataset.dataset_utils import split_dataset, split_dataset_with_ids

sys.path.append("/home/macht/thesis/hsi-biopsy")

from src.dataset.concentrations_dataset import (
    ConcentrationsDataset,
)


def test_split_dataset():
    """Test the split_dataset function with actual data."""

    # Use existing result directories
    reduced_wl_dir = "/home/macht/thesis/hsi-biopsy/results/biopsy2_spectral_unmixing/spectral_unmixing_avgpool_4_coarseness_1_left_500_right_900_reduced_wl_set"
    gt_dir = "/home/macht/thesis/hsi-biopsy/results/biopsy2_spectral_unmixing/spectral_unmixing_avgpool_4_coarseness_1_left_500_right_900_t1_all_molecules"

    # Check if directories exist
    if not os.path.exists(reduced_wl_dir):
        print(f"Reduced wavelength directory not found: {reduced_wl_dir}")
        return False

    if not os.path.exists(gt_dir):
        print(f"Ground truth directory not found: {gt_dir}")
        return False

    try:
        # Create dataset
        print("Creating ConcentrationsDataset...")
        dataset = ConcentrationsDataset(reduced_wl_dir, gt_dir)

        # Get dataset summary
        summary = dataset.get_summary()
        print(f"\nDataset summary:")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Number of patients: {summary['num_patients']}")
        print(f"  Patient IDs: {summary['patient_ids']}")
        print(f"  Patient FOV counts: {summary['patient_fov_counts']}")

        # Test split with default ratios
        print("\n" + "=" * 50)
        print("Testing split with default ratios (0.8, 0.1, 0.1):")
        train_ds, val_ds, test_ds = split_dataset(dataset)

        # Test split with custom ratios
        print("\n" + "=" * 50)
        print("Testing split with custom ratios (0.7, 0.15, 0.15):")
        train_ds2, val_ds2, test_ds2 = split_dataset(dataset, (0.7, 0.15, 0.15))

        # Test reproducibility
        print("\n" + "=" * 50)
        print("Testing reproducibility with same seed:")
        train_ds3, val_ds3, test_ds3 = split_dataset(dataset, random_seed=42)

        # Check if first split is same as third split (same seed)
        train_same = len(train_ds) == len(train_ds3)
        val_same = len(val_ds) == len(val_ds3)
        test_same = len(test_ds) == len(test_ds3)

        print(
            f"Reproducibility check (same sizes): {train_same and val_same and test_same}"
        )

        # Test different seed
        print("\n" + "=" * 50)
        print("Testing with different seed (123):")
        train_ds4, val_ds4, test_ds4 = split_dataset(dataset, random_seed=123)

        # Verify no overlap between splits (check patient IDs)
        print("\n" + "=" * 50)
        print("Verifying no patient overlap between splits...")

        def get_patient_ids_from_subset(subset):
            """Get patient IDs from a subset."""
            patient_ids = set()
            for idx in subset.indices:
                sample = dataset.samples[idx]
                patient_ids.add(sample["patient_id"])
            return patient_ids

        train_patients = get_patient_ids_from_subset(train_ds)
        val_patients = get_patient_ids_from_subset(val_ds)
        test_patients = get_patient_ids_from_subset(test_ds)

        print(f"Train patients: {sorted(train_patients)}")
        print(f"Val patients: {sorted(val_patients)}")
        print(f"Test patients: {sorted(test_patients)}")

        # Check for overlap
        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)

        no_overlap = (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        )
        print(f"No patient overlap between splits: {no_overlap}")

        if not no_overlap:
            print(f"  Train-Val overlap: {train_val_overlap}")
            print(f"  Train-Test overlap: {train_test_overlap}")
            print(f"  Val-Test overlap: {val_test_overlap}")

        print("\n" + "=" * 50)
        print("Split dataset test completed successfully!")

        # Test split_dataset_with_ids function
        print("\n" + "=" * 50)
        print("Testing split_dataset_with_ids function:")

        # Get available sample IDs for testing
        all_sample_ids = [sample["id"] for sample in dataset.samples]
        print(f"Available sample IDs: {all_sample_ids[:10]}...")  # Show first 10

        if len(all_sample_ids) >= 6:
            # Select some sample IDs for validation and test
            val_ids = all_sample_ids[:2]  # First 2 samples for validation
            test_ids = all_sample_ids[2:4]  # Next 2 samples for test

            print(f"Selected validation IDs: {val_ids}")
            print(f"Selected test IDs: {test_ids}")

            try:
                train_ds_ids, val_ds_ids, test_ds_ids = split_dataset_with_ids(
                    dataset, val_ids, test_ids
                )

                # Verify the splits contain the correct samples
                val_subset_ids = [
                    dataset.samples[idx]["id"] for idx in val_ds_ids.indices
                ]
                test_subset_ids = [
                    dataset.samples[idx]["id"] for idx in test_ds_ids.indices
                ]

                val_ids_match = set(val_subset_ids) == set(val_ids)
                test_ids_match = set(test_subset_ids) == set(test_ids)

                print(f"Validation IDs match: {val_ids_match}")
                print(f"Test IDs match: {test_ids_match}")

                # Verify total samples count
                total_split_samples = (
                    len(train_ds_ids) + len(val_ds_ids) + len(test_ds_ids)
                )
                total_matches = total_split_samples == len(dataset)
                print(
                    f"Total samples preserved: {total_matches} ({total_split_samples}/{len(dataset)})"
                )

                # Test error cases
                print("\nTesting error cases:")

                # Test overlapping IDs
                try:
                    overlapping_val = [all_sample_ids[0], all_sample_ids[1]]
                    overlapping_test = [
                        all_sample_ids[0],
                        all_sample_ids[2],
                    ]  # all_sample_ids[0] overlaps
                    split_dataset_with_ids(dataset, overlapping_val, overlapping_test)
                    print("ERROR: Should have raised ValueError for overlapping IDs")
                    return False
                except ValueError as e:
                    print(f"✓ Correctly caught overlapping IDs error: {str(e)[:50]}...")

                # Test non-existent ID
                try:
                    invalid_val = ["non_existent_id"]
                    invalid_test = [all_sample_ids[0]]
                    split_dataset_with_ids(dataset, invalid_val, invalid_test)
                    print("ERROR: Should have raised ValueError for non-existent ID")
                    return False
                except ValueError as e:
                    print(f"✓ Correctly caught non-existent ID error: {str(e)[:50]}...")

                if val_ids_match and test_ids_match and total_matches:
                    print("✓ split_dataset_with_ids function works correctly!")
                else:
                    print("✗ split_dataset_with_ids function has issues")
                    return False

            except Exception as e:
                print(f"Error testing split_dataset_with_ids: {e}")
                import traceback

                traceback.print_exc()
                return False
        else:
            print("Not enough samples to test split_dataset_with_ids (need at least 6)")

        return True

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_split_dataset()
    sys.exit(0 if success else 1)
