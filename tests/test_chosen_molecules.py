#!/usr/bin/env python3
"""
Test script for the chosen_molecules functionality in ModelPipeline.
"""

import sys
sys.path.append('/home/macht/thesis/hsi-biopsy')

from src.models.pipeline.model_pipeline import ModelConfig
from src.molecules import MoleculeIndex

def test_chosen_molecules_config():
    """Test that chosen_molecules correctly updates in_channels."""
    
    print("Testing chosen_molecules functionality...")
    
    # Test 1: Default behavior (all molecules)
    config_default = ModelConfig()
    print(f"Default config - in_channels: {config_default.in_channels}, chosen_molecules: {config_default.chosen_molecules}")
    
    # Test 2: Choose specific molecules
    chosen_mols = ['HBO2', 'HB', 'FAT']
    config_filtered = ModelConfig(chosen_molecules=chosen_mols)
    print(f"Filtered config - in_channels: {config_filtered.in_channels}, chosen_molecules: {config_filtered.chosen_molecules}")
    
    # Test 3: Invalid molecule name (should raise error)
    try:
        config_invalid = ModelConfig(chosen_molecules=['HBO2', 'INVALID_MOL'])
        print("ERROR: Should have raised ValueError for invalid molecule")
    except ValueError as e:
        print(f"Correctly caught error for invalid molecule: {e}")
    
    # Test 4: Show all available molecules
    all_molecules = [name for name, _ in MoleculeIndex.__members__.items()]
    print(f"All available molecules: {all_molecules}")
    
    print("✅ All tests passed!")

if __name__ == "__main__":
    test_chosen_molecules_config()
