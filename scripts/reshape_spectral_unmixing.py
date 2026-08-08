#!/usr/bin/env python3
"""
Reshape coef_list.npy and scatter_params.npy arrays in each sample folder to specified dimensions.
"""

import argparse
from pathlib import Path
import numpy as np
import sys


def reshape_in_directory(results_dir: Path, height: int, width: int) -> None:
    """
    Iterate over sample subdirectories in results_dir, reshape coef_list and scatter_params arrays.

    Parameters
    ----------
    results_dir : Path
        Path to the top-level results directory containing sample subfolders.
    height : int
        Target height dimension for reshaping.
    width : int
        Target width dimension for reshaping.
    """
    if not results_dir.is_dir():
        print(f"Error: '{results_dir}' is not a directory.")
        sys.exit(1)

    for sample_dir in results_dir.iterdir():
        if not sample_dir.is_dir():
            continue
        coef_file = sample_dir / "coef_list.npy"
        scatter_file = sample_dir / "scatter_params.npy"

        if not coef_file.exists() or not scatter_file.exists():
            print(f"Skipping '{sample_dir.name}': missing .npy files.")
            continue

        # Load arrays
        coef = np.load(coef_file)
        scatter = np.load(scatter_file)

        # Compute new shape
        try:
            n_features = coef.size // (height * width)
        except ZeroDivisionError:
            print(f"Invalid dimensions: {height}x{width}.")
            sys.exit(1)

        # Reshape
        coef_reshaped = coef.reshape(height, width, n_features)
        scatter_reshaped = scatter.reshape(height, width, -1)

        # Overwrite files
        np.save(coef_file, coef_reshaped)
        np.save(scatter_file, scatter_reshaped)

        print(f"Reshaped '{sample_dir.name}' to ({height}, {width}, {n_features}).")


def main():
    parser = argparse.ArgumentParser(
        description="Reshape spectral unmixing outputs in result folders"
    )
    parser.add_argument(
        "results_path",
        type=Path,
        help="Path to results directory (e.g., results/spectral_unmixing_...)",
    )
    parser.add_argument(
        "--height", type=int, default=512, help="Target height dimension"
    )
    parser.add_argument("--width", type=int, default=512, help="Target width dimension")

    args = parser.parse_args()
    reshape_in_directory(args.results_path, args.height, args.width)


if __name__ == "__main__":
    main()
