#!/usr/bin/env python3
# filepath: /home/home/tim_ivan/thesis/tim_thesis/tests/test_patient_ids.py
"""
This script tests the ALL_PATIENT_IDS class to verify it's working correctly.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import the ALL_PATIENT_IDS class
from src.constants import ALL_PATIENT_IDS, METADATA_CSV_PATH


def main():
    """Test the ALL_PATIENT_IDS functionality."""
    print("Testing ALL_PATIENT_IDS class...")
    print("Loading from ", METADATA_CSV_PATH)

    # Test direct attribute access (the autocomplete feature)
    print("\nTesting direct attribute access (autocomplete feature):")
    try:
        # Access a patient ID directly as an attribute
        s1_2_value = ALL_PATIENT_IDS.S1_2
        print(f"ALL_PATIENT_IDS.S1_2 = {s1_2_value}")

        # Access a few more patient IDs
        s1_6_value = ALL_PATIENT_IDS.S1_6
        print(f"ALL_PATIENT_IDS.S1_6 = {s1_6_value}")

        s1_10_value = ALL_PATIENT_IDS.S1_10
        print(f"ALL_PATIENT_IDS.S1_10 = {s1_10_value}")

        print("✅ Direct attribute access works!")
    except AttributeError as e:
        print(f"❌ Error accessing attributes: {e}")

    # Get all patient IDs
    all_ids = ALL_PATIENT_IDS.get_all()
    print(f"\nAll patient IDs ({len(all_ids)}):")
    print(all_ids)

    # Get all tumor types
    tumor_types = ALL_PATIENT_IDS.get_types()
    print(f"\nAll tumor types ({len(tumor_types)}):")
    print(tumor_types)

    # Get patient IDs by tumor type
    for tumor_type in tumor_types:
        type_ids = ALL_PATIENT_IDS.get_by_type(tumor_type)
        print(f"\nPatient IDs for {tumor_type} ({len(type_ids)}):")
        print(type_ids)


if __name__ == "__main__":
    main()
