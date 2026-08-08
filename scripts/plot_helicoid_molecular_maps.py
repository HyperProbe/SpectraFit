#!/usr/bin/env python3
"""Compute and plot HELICOID molecular maps for selected sample IDs.

The script loads raw HELICOID hyperspectral cubes, selects a seeded random
blood-pixel reference for delta-A computation, runs the existing scattering
optimization pipeline, and renders three 4x3 figures.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(repo_root: Path) -> None:
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(PROJECT_ROOT)

# Keep src.constants importable even if the user's shell environment is sparse.
os.environ.setdefault("HSI_DATA_DIR", str(PROJECT_ROOT / "data"))
os.environ.setdefault("METADATA_CSV_PATH", str(PROJECT_ROOT / "metadata.csv"))
os.environ["HELICOID_DATA_DIR"] = "/home/ezhovi/npj_database"

from src.dataset.helicoid_dataset import HelicoidDataset
from src.molecules import MoleculeMode
from src.scattering_model.scatter_optim import (
    optim_reference_spectrum_scatter_params,
)
from src.visualize_data import (
    create_rgb,
    get_HbT_map,
    get_diffCCO_map,
    get_scattering_power_map,
)
from src.wavelength_selection.enums import SampleType


DEFAULT_SAMPLE_IDS = [
    "008-01",
    "008-02",
    "012-01",
    "012-02",
    "015-01",
    "016-04",
    "016-05",
    "020-01",
    "025-02",
]

DISPLAY_LEFT_CUT = 400
DISPLAY_RIGHT_CUT = 1000
UNMIX_LEFT_CUT = 530
UNMIX_RIGHT_CUT = 750
RGB_BANDS = (708.97, 542.03, 479.06)
LABEL_COLORS = {
    1: (0.0, 0.75, 0.0),
    2: (0.05, 0.25, 1.0),
    3: (1.0, 0.9, 0.0),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute HELICOID molecular maps and render 3 montages."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/ezhovi/npj_database"),
        help="Root directory containing HELICOID sample folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "plots" / "helicoid_molecular_maps",
        help="Directory where the rendered figures and manifest will be saved.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "helicoid_molecular_maps",
        help="Directory where reusable inference results will be cached.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Seed for random blood-pixel selection.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="+",
        default=DEFAULT_SAMPLE_IDS,
        help="HELICOID sample IDs to render.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Save DPI for the figures.",
    )
    parser.add_argument(
        "--pathlength-mode",
        type=str,
        default="pathlength_from_wavelength",
        help="Pathlength mode passed to scattering optimization.",
    )
    parser.add_argument(
        "--pathlength-debug-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "pathlengths",
        help="Directory for pathlength debug outputs.",
    )
    return parser


def normalize_rgb(rgb_image: np.ndarray) -> np.ndarray:
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32
    )
    normalized = rgb_image.copy()
    for channel in range(normalized.shape[-1]):
        channel_data = normalized[..., channel]
        min_value = float(np.nanmin(channel_data))
        max_value = float(np.nanmax(channel_data))
        if max_value > min_value:
            normalized[..., channel] = (channel_data - min_value) / (
                max_value - min_value
            )
        else:
            normalized[..., channel] = 0.0
    return normalized


def prepare_gt_map(gt_map: np.ndarray) -> np.ndarray:
    if gt_map.ndim == 3 and gt_map.shape[-1] == 1:
        gt_map = np.squeeze(gt_map, axis=-1)
    return np.asarray(gt_map).astype(int)


def overlay_tissue_labels(rgb_image: np.ndarray, gt_map: np.ndarray) -> np.ndarray:
    overlay = rgb_image.copy()
    alpha = 0.35
    for label, color in LABEL_COLORS.items():
        mask = gt_map == label
        if not np.any(mask):
            continue
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * np.array(color)
    return np.clip(overlay, 0.0, 1.0)


def group_samples(sample_ids: list[str]) -> list[list[str]]:
    if len(sample_ids) % 3 != 0:
        raise ValueError("Expected sample IDs to be divisible into groups of 3.")
    return [sample_ids[index : index + 3] for index in range(0, len(sample_ids), 3)]


def load_display_dataset(data_root: Path) -> HelicoidDataset:
    return HelicoidDataset(
        helicoid_data_dir=data_root,
        left_cut=DISPLAY_LEFT_CUT,
        right_cut=DISPLAY_RIGHT_CUT,
        coarseness=1,
        normalize_image=True,
        with_delta_A=False,
        with_rgb=True,
        inference_mode=False,
    )


def load_unmixing_dataset(data_root: Path) -> HelicoidDataset:
    return HelicoidDataset(
        helicoid_data_dir=data_root,
        left_cut=UNMIX_LEFT_CUT,
        right_cut=UNMIX_RIGHT_CUT,
        coarseness=1,
        normalize_image=True,
        with_delta_A=True,
        reference_pixel_strategy="random_blood",
        with_rgb=False,
        inference_mode=False,
    )


def compute_maps(
    sample_id: str,
    compute_dataset: HelicoidDataset,
    load_a_b_from_path: Path | None = None, pathlength_mode: str = "pathlength_from_wavelength",
    pathlength_debug_dir: Path | None = None
) -> dict[str, Any]:
    sample = compute_dataset.get_sample_by_id(sample_id)
    if sample is None:
        raise ValueError(f"Sample '{sample_id}' was not found in the unmixing dataset")

    (
        coef_list,
        scatter_params,
        errors,
        a_t1,
        b_t1,
        delta_A,
        reference_pixel,
        _,
        _,
    ) = optim_reference_spectrum_scatter_params(
        data=sample,
        dataset=compute_dataset,
        molecule_mode=MoleculeMode.ALL,
        load_a_b_from_path=str(load_a_b_from_path) if load_a_b_from_path else None,
        coarseness=8,
        pathlength_mode=pathlength_mode,
        pathlength_debug_dir=pathlength_debug_dir
    )

    return {
        "coef_list": coef_list,
        "scatter_params": scatter_params,
        "errors": errors,
        "a_t1": a_t1,
        "b_t1": b_t1,
        "delta_A": delta_A,
        "reference_pixel": reference_pixel,
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def sample_cache_path(results_dir: Path, sample_id: str) -> Path:
    return results_dir / "sample_cache" / f"{sample_id}.npz"


def error_cache_path(results_dir: Path, sample_id: str) -> Path:
    return results_dir / "sample_cache" / f"{sample_id}.error.json"


def save_sample_cache(cache_path: Path, sample_id: str, record: dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    reference_pixel = record["reference_pixel"]
    reference_pixel_array = (
        np.asarray(reference_pixel, dtype=np.int64)
        if reference_pixel is not None
        else np.asarray([], dtype=np.int64)
    )
    temp_path = cache_path.with_suffix(".npz.tmp")
    with temp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            sample_id=np.asarray(sample_id),
            reference_pixel=reference_pixel_array,
            rgb=record["rgb"],
            gt_map=record["gt_map"],
            coef_list=record["coef_list"],
            scatter_params=record["scatter_params"],
            hbt_map=record["hbt_map"],
            diffcco_map=record["diffcco_map"],
            b_map=record["b_map"],
            a_t1=np.asarray(record["a_t1"]),
            b_t1=np.asarray(record["b_t1"]),
        )
    temp_path.replace(cache_path)


def load_sample_cache(cache_path: Path) -> dict[str, Any]:
    with np.load(cache_path, allow_pickle=False) as cached:
        reference_pixel_array = np.asarray(cached["reference_pixel"])
        reference_pixel = (
            reference_pixel_array.astype(np.int64).tolist()
            if reference_pixel_array.size
            else None
        )
        rgb = cached["rgb"]
        gt_map = cached["gt_map"]
        coef_list = cached["coef_list"]
        scatter_params = cached["scatter_params"]
        return {
            "sample_id": str(cached["sample_id"][()]),
            "reference_pixel": reference_pixel,
            "rgb": rgb,
            "gt_map": gt_map,
            "coef_list": coef_list,
            "scatter_params": scatter_params,
            "hbt_map": cached["hbt_map"],
            "diffcco_map": cached["diffcco_map"],
            "b_map": cached["b_map"],
            "a_t1": cached["a_t1"],
            "b_t1": cached["b_t1"],
            "shape": {
                "rgb": list(rgb.shape),
                "coef_list": list(coef_list.shape),
                "scatter_params": list(scatter_params.shape),
                "gt_map": list(gt_map.shape),
            },
        }


def build_sample_record(
    sample_id: str,
    display_sample: dict[str, Any],
    map_results: dict[str, Any],
) -> dict[str, Any]:
    rgb_image = normalize_rgb(
        create_rgb(
            display_sample["hsi_cube"],
            r_band=RGB_BANDS[0],
            g_band=RGB_BANDS[1],
            b_band=RGB_BANDS[2],
            sample_type=SampleType.HELICOID,
        )
    )
    gt_map = prepare_gt_map(display_sample["gt_map"])
    coef_list = map_results["coef_list"]
    scatter_params = map_results["scatter_params"]

    return {
        "sample_id": sample_id,
        "reference_pixel": list(map_results["reference_pixel"])
        if map_results["reference_pixel"] is not None
        else None,
        "rgb": rgb_image,
        "gt_map": gt_map,
        "hbt_map": get_HbT_map(coef_list),
        "diffcco_map": get_diffCCO_map(coef_list),
        "b_map": get_scattering_power_map(scatter_params),
        "coef_list": coef_list,
        "scatter_params": scatter_params,
        "a_t1": map_results["a_t1"],
        "b_t1": map_results["b_t1"],
        "shape": {
            "rgb": list(rgb_image.shape),
            "coef_list": list(coef_list.shape),
            "scatter_params": list(scatter_params.shape),
        },
    }


def render_sample_column(
    axis_row_1: plt.Axes,
    axis_row_2: plt.Axes,
    axis_row_3: plt.Axes,
    axis_row_4: plt.Axes,
    rgb_image: np.ndarray,
    gt_map: np.ndarray,
    reference_pixel: list[int] | None,
    hbt_map: np.ndarray,
    diffcco_map: np.ndarray,
    scatter_b_map: np.ndarray,
    title: str,
    row_norms: dict[str, Any],
) -> dict[str, Any]:
    axis_row_1.imshow(overlay_tissue_labels(rgb_image, gt_map))
    if reference_pixel is not None:
        axis_row_1.plot(
            reference_pixel[1],
            reference_pixel[0],
            marker="x",
            color="orange",
            markersize=10,
            markeredgewidth=2.2,
            linestyle="none",
            zorder=10,
        )
    tumor_mask = gt_map == 2
    if np.any(tumor_mask):
        axis_row_1.contour(
            tumor_mask.astype(float), levels=[0.5], colors=["cyan"], linewidths=1.8
        )
    axis_row_1.set_title(title, fontsize=11, fontweight="bold")
    axis_row_1.set_xticks([])
    axis_row_1.set_yticks([])

    hbt_im = axis_row_2.imshow(hbt_map, cmap="Reds", norm=row_norms["hbt"])
    diff_im = axis_row_3.imshow(diffcco_map, cmap="terrain", norm=row_norms["diffcco"])
    b_im = axis_row_4.imshow(scatter_b_map, cmap="seismic_r", norm=row_norms["b"])

    for axis in (axis_row_2, axis_row_3, axis_row_4):
        axis.set_xticks([])
        axis.set_yticks([])

    return {"hbt": hbt_im, "diffcco": diff_im, "b": b_im}


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    display_dataset = load_display_dataset(args.data_root)
    compute_dataset = load_unmixing_dataset(args.data_root)

    grouped_sample_ids = group_samples(args.sample_ids)
    all_figures: list[dict[str, Any]] = []
    reference_params_dir = args.results_dir / "reference_params"
    reference_params_dir.mkdir(parents=True, exist_ok=True)
    sample_cache_dir = args.results_dir / "sample_cache"
    sample_cache_dir.mkdir(parents=True, exist_ok=True)
    results_manifest_path = args.results_dir / "helicoid_molecular_maps_results_manifest.json"

    sample_records_by_id: dict[str, dict[str, Any]] = {}
    reference_params_ready = (reference_params_dir / "a_t1.npy").exists() and (
        reference_params_dir / "b_t1.npy"
    ).exists()

    legend_handles = [
        Patch(facecolor=LABEL_COLORS[3], edgecolor="black", label="Blood vessel"),
        Patch(facecolor=LABEL_COLORS[1], edgecolor="black", label="Normal tissue"),
        Patch(facecolor=LABEL_COLORS[2], edgecolor="black", label="Tumor tissue"),
        Line2D([0], [0], color="cyan", lw=2, label="Tumor contour"),
    ]

    row_titles = ["Synthetic RGB", "HbT", "diffCCO", "Scattering power b"]

    compute_progress = tqdm(total=len(args.sample_ids), desc="Computing and caching samples")
    for sample_id in args.sample_ids:
        cache_path = sample_cache_path(sample_cache_dir, sample_id)
        display_sample = display_dataset.get_sample_by_id(sample_id)
        if display_sample is None:
            raise ValueError(f"Sample '{sample_id}' was not found in the display dataset")

        if cache_path.exists():
            sample_records_by_id[sample_id] = load_sample_cache(cache_path)
            compute_progress.update(1)
            continue

        map_results = compute_maps(
            sample_id,
            compute_dataset,
            reference_params_dir if reference_params_ready else None, args.pathlength_mode, args.pathlength_debug_dir
        )
        # save coef_list and scatter_params for potential debugging/ analysis use (not currently used in rendering)
        coef_path = reference_params_dir / f"{sample_id}_coef_list.npy"
        print(f"Saving reference spectrum parameters for sample {sample_id} to {coef_path}")
        scatter_path = reference_params_dir / f"{sample_id}_scatter_params.npy"
        np.save(coef_path, map_results["coef_list"])
        np.save(scatter_path, map_results["scatter_params"])

        sample_record = build_sample_record(sample_id, display_sample, map_results)
        save_sample_cache(cache_path, sample_id, sample_record)

        if not reference_params_ready:
            np.save(reference_params_dir / "a_t1.npy", np.asarray(map_results["a_t1"]))
            np.save(reference_params_dir / "b_t1.npy", np.asarray(map_results["b_t1"]))
            reference_params_ready = True
       
        sample_records_by_id[sample_id] = sample_record
        compute_progress.update(1)

    compute_progress.close()

    for figure_index, sample_group in enumerate(
        tqdm(grouped_sample_ids, desc="Rendering figures from cached results"), start=1
    ):
        sample_records = [sample_records_by_id[sample_id] for sample_id in sample_group]

        hbt_stack = np.concatenate([record["hbt_map"].ravel() for record in sample_records])
        diffcco_stack = np.concatenate([record["diffcco_map"].ravel() for record in sample_records])
        b_stack = np.concatenate([record["b_map"].ravel() for record in sample_records])

        diffcco_abs = max(float(np.nanmax(np.abs(diffcco_stack))), 1e-6)
        row_norms = {
            "hbt": Normalize(vmin=float(np.nanmin(hbt_stack)), vmax=float(np.nanmax(hbt_stack))),
            "diffcco": TwoSlopeNorm(vcenter=0.0, vmin=-diffcco_abs, vmax=diffcco_abs),
            "b": Normalize(vmin=float(np.nanmin(b_stack)), vmax=float(np.nanmax(b_stack))),
        }

        fig, axes = plt.subplots(4, 3, figsize=(18, 20))
        fig.subplots_adjust(top=0.92, wspace=0.04, hspace=0.08)
        fig.suptitle(
            f"HELICOID Molecular Maps | samples {sample_group[0]} to {sample_group[-1]}",
            fontsize=15,
            fontweight="bold",
        )

        mappables: dict[str, Any] = {}
        for column_index, sample_record in enumerate(sample_records):
            mappables = render_sample_column(
                axes[0, column_index],
                axes[1, column_index],
                axes[2, column_index],
                axes[3, column_index],
                sample_record["rgb"],
                sample_record["gt_map"],
                sample_record["reference_pixel"],
                sample_record["hbt_map"],
                sample_record["diffcco_map"],
                sample_record["b_map"],
                sample_record["sample_id"],
                row_norms,
            )

        axes[0, 0].legend(
            handles=legend_handles,
            loc="lower left",
            frameon=True,
            framealpha=0.92,
            fontsize=9,
        )

        for row_index, row_title in enumerate(row_titles):
            axes[row_index, 0].set_ylabel(row_title, fontsize=12, fontweight="bold")

        fig.colorbar(
            mappables["hbt"],
            ax=axes[1, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        fig.colorbar(
            mappables["diffcco"],
            ax=axes[2, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )
        fig.colorbar(
            mappables["b"],
            ax=axes[3, :].ravel().tolist(),
            fraction=0.025,
            pad=0.02,
        )

        output_path = args.output_dir / f"helicoid_molecular_maps_{figure_index:02d}.png"
        fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        all_figures.append(
            {
                "figure_index": figure_index,
                "output_file": str(output_path),
                "sample_ids": sample_group,
                "samples": [
                    {
                        "sample_id": record["sample_id"],
                        "reference_pixel": record["reference_pixel"],
                        "shape": record["shape"],
                        "cache_file": str(sample_cache_path(sample_cache_dir, record["sample_id"])),
                    }
                    for record in sample_records
                ],
            }
        )

    # After all groups rendered: create three separate 1x1 figures (one per biomarker)
    # Each figure shows all patients (args.sample_ids) with per-sample grouped boxes (one box per label).
    sample_list = [sample_records_by_id[sid] for sid in args.sample_ids]
    ordered_labels = [3, 1, 2]
    ordered_label_names = ["Blood", "Normal", "Tumor"]
    label_colors = [LABEL_COLORS[lab] for lab in ordered_labels]
    n_labels = len(ordered_labels)
    width = 0.22

    metrics = [
        ("HbT", "hbt_map", "diffHb"),
        ("diffCCO", "diffcco_map", "diffCCO"),
        ("delta_b", "b_map", "delta_b"),
    ]

    for display_name, key, fname_tag in metrics:
        fig = plt.figure(figsize=(max(8, len(sample_list) * 1.2), 6))
        ax = fig.add_subplot(1, 1, 1)

        data: list[np.ndarray] = []
        positions: list[float] = []
        means: list[float] = []
        stds: list[float] = []

        for i, rec in enumerate(sample_list):
            for j, lab in enumerate(ordered_labels):
                mask = rec["gt_map"] == lab
                vals = np.asarray(rec[key])[mask].ravel()
                if display_name == "delta_b":
                    ry, rx = rec["reference_pixel"]
                    ref = float(rec["b_map"][ry, rx])
                    vals = vals - ref
                data.append(vals)
                pos = i + (j - (n_labels - 1) / 2) * width
                positions.append(pos)
                means.append(float(np.nanmean(vals)))
                stds.append(float(np.nanstd(vals)))

        boxes = ax.boxplot(
            data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            showfliers=False,
        )
        for idx, box in enumerate(boxes["boxes"]):
            label_idx = idx % n_labels
            box.set(facecolor=label_colors[label_idx])

        ax.errorbar(positions, means, yerr=stds, fmt="o", color="k", ms=4, label="mean±std")
        ax.set_xticks(np.arange(len(sample_list)))
        ax.set_xticklabels([rec["sample_id"] for rec in sample_list], rotation=45, ha="right")
        ax.set_title(f"Mean {display_name} by patient and tissue class")
        ax.set_xlabel("Patient")
        ax.set_ylabel(display_name)
        legend_handles = [Patch(facecolor=LABEL_COLORS[lab], label=name) for lab, name in zip(ordered_labels, ordered_label_names)]
        ax.legend(handles=legend_handles + [Line2D([0], [0], marker="o", color="k", label="mean±std", linestyle="")], loc="upper right")

        outpath = args.output_dir / f"helicoid_molecular_maps_allpatients_{fname_tag}.png"
        fig.tight_layout()
        fig.savefig(outpath, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

        # record in all_figures manifest-level entry
        all_figures.append({"metric": display_name, "file": str(outpath)})

        # (per-group boxplots removed) -- aggregate, per-biomarker figures will be created after rendering all groups

    results_manifest = {
        "data_root": str(args.data_root),
        "seed": args.seed,
        "cache_dir": str(sample_cache_dir),
        "reference_params_dir": str(reference_params_dir),
        "sample_caches": [
            {
                "sample_id": record["sample_id"],
                "cache_file": str(sample_cache_path(sample_cache_dir, record["sample_id"])),
                "reference_pixel": record["reference_pixel"],
                "shape": record["shape"],
            }
            for record in sample_records_by_id.values()
        ],
    }
    with results_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(results_manifest), handle, indent=2)

    manifest = {
        "data_root": str(args.data_root),
        "seed": args.seed,
        "left_cut_nm": UNMIX_LEFT_CUT,
        "right_cut_nm": UNMIX_RIGHT_CUT,
        "results_manifest": str(results_manifest_path),
        "results_dir": str(args.results_dir),
        "sample_groups": all_figures,
    }
    manifest_path = args.output_dir / "helicoid_molecular_maps_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(manifest), handle, indent=2)

    print(f"Saved figures to {args.output_dir}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()