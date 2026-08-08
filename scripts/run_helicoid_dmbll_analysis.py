#!/usr/bin/env python3
"""Run the pathlength-aware IDP dMBLL model on a HELICoiD sample."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataset.helicoid_dataset import HelicoidDataset
from src.molecules import MoleculeMode, Molecules
from src.scattering_model.dmbll import load_pathlength_file, run_helicoid_dmbll_analysis
from src.wavelength_selection.enums import SampleType


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lp", type=Path, default=None)
    parser.add_argument("--ls", type=Path, default=None)
    parser.add_argument("--include-scattering", action="store_true")
    parser.add_argument("--b-ref", type=float, default=None)
    parser.add_argument("--g", type=float, default=0.9)
    parser.add_argument("--reference-pixel", type=int, nargs=2, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataset = HelicoidDataset(coarseness=1, with_delta_A=True, normalize_image=True)
    sample = dataset.get_sample_by_id(args.sample_id) if args.sample_id else dataset[0]
    wavelengths = sample["wavelengths"]
    reflectance_cube = sample["hsi_cube"]

    molecules = Molecules(
        left_cut=dataset.left_cut,
        right_cut=dataset.right_cut,
        molecule_mode=MoleculeMode.ALL,
        sample_type=SampleType.HELICOID,
    )
    M = molecules.M

    Lp = load_pathlength_file(args.lp, wavelengths) if args.lp is not None else None
    Ls = load_pathlength_file(args.ls, wavelengths) if args.ls is not None else None

    reference_pixel = (
        tuple(args.reference_pixel)
        if args.reference_pixel is not None
        else sample["reference_pixel"]
    )
    if reference_pixel is None:
        raise ValueError("A reference pixel is required when no reference_mask is provided")

    result = run_helicoid_dmbll_analysis(
        reflectance_cube=reflectance_cube,
        wavelengths=wavelengths,
        M=M,
        reference_pixel=reference_pixel,
        Lp=Lp,
        Ls=Ls,
        b_ref=args.b_ref,
        g=args.g,
        include_scattering=args.include_scattering,
        save_path=args.output,
    )
    print(f"Saved dMBLL results to {args.output}")
    print(f"delta_c shape: {result['delta_c'].shape}")


if __name__ == "__main__":
    main()
