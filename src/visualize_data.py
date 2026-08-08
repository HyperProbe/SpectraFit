from typing import List, Union
from loguru import logger
import numpy as np
import torch
import torch.nn as nn

from src.dataset.concentrations_dataset import ConcentrationsDataset, Signal
from src.dataset.data_sample.fused_helicoid_sample import FusedHelicoidSample
from src.dataset.dataset_utils import collate_dataclass
from src.dataset.fused_helicoid_concentration_dataset import (
    FusedHelicoidConcentrationDataset,
)
from src.dataset.helicoid_dataset import HelicoidDataset
from src.models.pipeline.model_pipeline import ModelPipeline, load_model_pipeline
from .constants import WAVELENGTHS, HELICOID_WAVELENGTHS
from .wavelength_selection.wavelength_selector import SampleType
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from .molecules import MoleculeIndex


def create_rgb(
    hsi_data, r_band=650, g_band=550, b_band=450, sample_type=SampleType.BIOPSY2
):
    """
    Create an (H, W, 3) RGB image from a hyperspectral cube,
    performing per-channel min-max normalization while handling NaN and infinite values.
    """
    if sample_type == SampleType.HELICOID:
        # For helicoid data, use the helicoid wavelengths
        wavelength_array = HELICOID_WAVELENGTHS
    else:
        # For biopsy2 data, use the standard wavelengths
        wavelength_array = WAVELENGTHS
    # pick nearest bands
    r_idx = np.argmin(np.abs(wavelength_array - r_band))
    g_idx = np.argmin(np.abs(wavelength_array - g_band))
    b_idx = np.argmin(np.abs(wavelength_array - b_band))

    # stack into H×W×3 and cast to float32
    rgb = np.stack([hsi_data[r_idx], hsi_data[g_idx], hsi_data[b_idx]], axis=-1).astype(
        np.float32
    )

    return rgb


def get_diffCCO_map(concentrations: np.ndarray) -> np.ndarray:
    """
    Compute the diffCCO map from concentration data.

    diffCCO is defined as the difference between COXA and CREDA concentrations.

    Parameters
    ----------
    concentrations : np.ndarray
        A 3D numpy array of shape (H, W, num_molecules) containing concentration data.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (H, W) representing the diffCCO map.
    """
    diffCCO = (
        concentrations[..., MoleculeIndex.COXA]
        - concentrations[..., MoleculeIndex.CREDA]
    )
    return diffCCO


def get_HbT_map(concentrations: np.ndarray) -> np.ndarray:
    """
    Compute the HbT map from concentration data.

    HbT is defined as the sum of HBO2 and HB concentrations.

    Parameters
    ----------
    concentrations : np.ndarray
        A 3D numpy array of shape (H, W, num_molecules) containing concentration data.

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (H, W) representing the HbT map.
    """
    HbT = (
        concentrations[..., MoleculeIndex.HBO2] + concentrations[..., MoleculeIndex.HB]
    )
    return HbT


def show_interactive_image_with_spectrum(
    hsi_data,
    r_band=650,
    g_band=550,
    b_band=450,
    sample_type=SampleType.BIOPSY2,
    cut_wavelengths=None,
    rbg_image=None,
):
    """
    Show an interactive image with a spectrum plot.
    """
    rgb = (
        create_rgb(hsi_data, r_band, g_band, b_band, sample_type)
        if rbg_image is None
        else rbg_image
    )
    if cut_wavelengths is not None:
        # If cut_wavelengths is provided, use it to filter the wavelengths
        wavelength_array = cut_wavelengths
    elif sample_type == SampleType.HELICOID:
        # For helicoid data, use the helicoid wavelengths
        wavelength_array = HELICOID_WAVELENGTHS
    else:
        # For biopsy2 data, use the standard wavelengths
        wavelength_array = WAVELENGTHS

    # Create a figure with two subplots
    fig, (ax_img, ax_spec) = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [1, 2]},  # ax_img narrower, ax_spec wider
    )

    # Display the RGB image
    ax_img.imshow(rgb)
    ax_img.set_title("Synthetic RGB Image")
    ax_img.grid(False)
    scatter_plot = ax_img.scatter([], [], c=[], s=100, marker="+", edgecolors=[])

    selected_pixel = []
    lines = []
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    def on_clear_button_click(event):
        # Clear the scatter plot
        selected_pixel.clear()
        scatter_plot.set_offsets(np.empty((0, 2)))
        scatter_plot.set_edgecolors([])
        for line in lines:
            line.remove()
        lines.clear()
        legend = ax_spec.legend()
        if legend:
            legend.remove()
        fig.canvas.draw()

    clear_button = widgets.Button(description="Clear Selection")
    clear_button.on_click(on_clear_button_click)
    display(clear_button)

    def on_click(event):
        if event.inaxes == ax_img:
            # Get the pixel coordinates
            x, y = int(event.xdata), int(event.ydata)

            pixel_coords = (y, x)

            # Check if the pixel is already selected
            if pixel_coords in selected_pixel:
                selected_pixel.remove(pixel_coords)
            else:
                selected_pixel.append(pixel_coords)
            update_plot()

    def update_plot():
        offsets = np.array([[px[1], px[0]] for px in selected_pixel])
        scatter_plot.set_offsets(offsets)
        marker_colors = [colors[i % len(colors)] for i in range(len(selected_pixel))]
        scatter_plot.set_edgecolors(marker_colors)
        scatter_plot.set_facecolors("none")

        for line in lines:
            line.remove()
        lines.clear()

        for idx, (py, px) in enumerate(selected_pixel):
            spectrum = hsi_data[:, py, px]
            (line,) = ax_spec.plot(
                wavelength_array,
                spectrum,
                color=marker_colors[idx],
                label=f"Pixel {px},{py}",
                linestyle="-",
                # marker="o",
            )
            lines.append(line)
        update_legend()
        fig.canvas.draw_idle()

    def update_legend():
        # Clear the previous legend
        legend = ax_spec.legend()
        if legend:
            legend.remove()

        # Create a new legend with the current lines
        if lines:
            ax_spec.legend(
                lines, [line.get_label() for line in lines], loc="upper right"
            )

    ax_spec.set_title("Pixel Intensity Across Channels")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity")
    ax_spec.grid(axis="y", linestyle="--", alpha=0.7)

    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()


def display_with_metadata(biopsy2Datset, patient_id, save_to_path=None):
    """
    Display all HSI samples for a specific patient with their metadata.

    Parameters
    ----------
    biopsy2Dataset : Biopsy2Dataset
        The dataset containing HSI samples
    patient_id : str
        Patient ID to retrieve samples for (e.g., "S1.2")
    save_to_path : str, optional
        If provided, the figures will be saved to this directory path with the
        naming convention S1_2FOV{FOV_num}.png (patient ID without "S" prefix,
        underscores instead of dots, followed by FOV number)

    Returns
    -------
    None
        Displays the visualizations in the notebook
    """
    # Get all samples for the patient
    samples = biopsy2Datset.get_samples_by_patient_id(patient_id)

    if not samples:
        print(f"No samples found for patient ID '{patient_id}'")
        return

    # Display each sample with its metadata
    for i, sample in enumerate(samples):
        # Create RGB image
        rgb = create_rgb(sample["hsi_cube"])

        # Create figure with two columns: image and metadata
        fig, (ax_img, ax_meta) = plt.subplots(
            1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [1, 1]}
        )

        # Display RGB image
        ax_img.imshow(rgb)
        ax_img.set_title(f"Patient {sample['patient_id']}, FOV {sample['fov']}")
        ax_img.axis("on")

        # Turn off axis for metadata panel and use it for text
        ax_meta.axis("off")
        ax_meta.set_title("Metadata")

        # Display metadata as text
        metadata_text = "\n".join(
            [f"{key}: {value}" for key, value in sample["metadata"].items()]
        )
        ax_meta.text(
            0.05,
            0.95,
            metadata_text,
            transform=ax_meta.transAxes,
            fontsize=10,
            verticalalignment="top",
            family="monospace",
        )

        plt.tight_layout()

        # Save the figure if save_to_path is provided
        if save_to_path:
            import os

            # Create directory if it doesn't exist
            os.makedirs(save_to_path, exist_ok=True)

            # Extract patient number (without "S" prefix if present)
            patient_number = sample["patient_id"]
            if patient_number.startswith("S"):
                patient_number = patient_number[1:]

            # Replace dots with underscores for filename
            patient_number = patient_number.replace(".", "_")

            # Construct filename using the requested naming convention
            filename = f"S{patient_number}FOV{sample['fov']}.png"
            filepath = os.path.join(save_to_path, filename)

            # Save figure
            fig.savefig(filepath, dpi=300, bbox_inches="tight")
            print(f"Saved figure to {filepath}")

        plt.show()


def plot_inferred_concentrations(
    spectrum_pxl: tuple[int, int],
    delta_c: np.ndarray,
    delta_A: np.ndarray,
    modeled_delta_A: np.ndarray,
    cut_wavelengths: np.ndarray,
    coarseness: int = 1,
):

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    hbt_map = get_HbT_map(delta_c[::coarseness, ::coarseness])
    im_hbt = axes[0].imshow(
        hbt_map,
        cmap="Reds",
    )
    axes[0].scatter(*spectrum_pxl, color="blue", s=100, marker="x")
    axes[0].set_title("inferred HBT")
    fig.colorbar(im_hbt, ax=axes[0], orientation="vertical")

    diffCCO = get_diffCCO_map(delta_c[::coarseness, ::coarseness])
    im_diffcco = axes[1].imshow(diffCCO, cmap="terrain")
    axes[1].set_title("inferred diffCCO")
    fig.colorbar(im_diffcco, ax=axes[1], orientation="vertical")

    axes[2].plot(
        cut_wavelengths,
        delta_A[spectrum_pxl[0], spectrum_pxl[1], :],
        label="measured",
    )
    axes[2].plot(
        cut_wavelengths,
        modeled_delta_A[spectrum_pxl[0], spectrum_pxl[1], :],
        label="modeled",
    )
    plt.legend()


def get_mean_fat_diff_cco_hbo2(concentrations):
    mean_fat = np.mean(
        concentrations[:, :, MoleculeIndex.FAT].flatten(),
        axis=0,
    )
    mean_hbo2 = np.mean(
        concentrations[:, :, MoleculeIndex.HBO2].flatten(),
        axis=0,
    )
    diff_cco = (
        concentrations[:, :, MoleculeIndex.COXA]
        - concentrations[:, :, MoleculeIndex.CREDA]
    )
    mean_diff_cco = np.mean(
        diff_cco.flatten(),
        axis=0,
    )
    return mean_fat, mean_diff_cco, mean_hbo2


def get_mean_fat_diff_cco_diff_hb(
    concentrations: np.ndarray, crop: tuple[int, int] | None = None
) -> tuple[float, float, float]:
    """
    Return (mean_fat, mean_diff_cco, mean_diff_hb) over either:
      - the full H×W image (if crop is None), or
      - a centered crop of size crop[0]×crop[1].
    """
    # apply center‐crop if requested
    if crop is not None:
        H, W = concentrations.shape[:2]
        ch, cw = crop
        if ch > H or cw > W:
            raise ValueError(f"crop size {crop} exceeds image dims {(H,W)}")
        start_h = (H - ch) // 2
        start_w = (W - cw) // 2
        concentrations = concentrations[
            start_h : start_h + ch, start_w : start_w + cw, :
        ]

    # 1) fat
    fat = concentrations[..., MoleculeIndex.FAT]
    mean_fat = np.mean(fat)

    # 2) hemoglobin difference
    diff_hb = (
        concentrations[..., MoleculeIndex.HBO2] - concentrations[..., MoleculeIndex.HB]
    )
    mean_diff_hb = np.mean(diff_hb)

    # 3) cytochrome‐cco difference
    diff_cco = (
        concentrations[..., MoleculeIndex.COXA]
        - concentrations[..., MoleculeIndex.CREDA]
    )
    mean_diff_cco = np.mean(diff_cco)

    return mean_fat, mean_diff_cco, mean_diff_hb


def get_mean_data(concentrations):
    mean_data = []
    for i in range(len(MoleculeIndex)):
        mean_data.append(
            np.mean(
                concentrations[:, :, i].flatten(),
                axis=0,
            )
        )
    return mean_data


def plot_reconstruction(
    dataset: ConcentrationsDataset,
    model_pipeline: ModelPipeline,
    id: str,
    coarseness: int = 1,
):
    """
    Plots a 3×4 grid of concentration maps for three summaries:
    • HBT      = HbO2 + Hb
    • diffCCO  = COXA – CREDA
    • FAT      = FAT

    Columns are:
    1) Ground truth (full wavelengths)
    2) Reconstructed
    3) Input (reduced wavelengths)
    4) Absolute error |recon – truth|

    Args:
        dataset (ConcentrationsDataset): Dataset containing the samples with ground truth
            and reduced wavelength concentration data.
        model_pipeline (ModelPipeline): Trained model pipeline used to generate
            reconstructed concentrations from reduced wavelength input.
        id (str): Sample identifier to retrieve specific data from the dataset.
        coarseness (int, optional): Subsampling factor for spatial dimensions.
            If > 1, the maps will be downsampled by this factor. Defaults to 1.
    """

    # Prepare maps
    def sub(m):
        return m[::coarseness, ::coarseness]

    data = dataset.get_sample_by_id(id)
    original = data["gt"]["coef_list"]

    input_data = data["reduced_wl"]["coef_list"]
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)

    reconstructed = model_pipeline.predict(input_tensor).squeeze(0).cpu().numpy()

    # HBT
    orig_hbt = sub(get_HbT_map(original))
    recon_hbt = sub(get_HbT_map(reconstructed))
    input_hbt = sub(get_HbT_map(input_data))
    # diffCCO
    orig_cco = sub(get_diffCCO_map(original))
    recon_cco = sub(get_diffCCO_map(reconstructed))
    input_cco = sub(get_diffCCO_map(input_data))
    # FAT
    orig_fat = sub(original[..., MoleculeIndex.FAT])
    recon_fat = sub(reconstructed[..., MoleculeIndex.FAT])
    input_fat = sub(input_data[..., MoleculeIndex.FAT])

    summaries = [
        ("HBT", orig_hbt, recon_hbt, input_hbt, "Reds"),
        ("diffCCO", orig_cco, recon_cco, input_cco, "terrain"),
        ("FAT", orig_fat, recon_fat, input_fat, "viridis"),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    for row, (name, o, r, inp, cmap) in enumerate(summaries):
        err = np.abs(r - o)
        maps = [o, r, inp, err]
        titles = ["Truth", "Reconstructed", "Input (reduced)", "Absolute Error"]
        cmaps = [cmap, cmap, cmap, "bwr"]

        for col, (m, title, cm) in enumerate(zip(maps, titles, cmaps)):
            ax = axes[row, col]
            im = ax.imshow(m, cmap=cm)
            ax.set_title(f"{name} – {title}")
            plt.colorbar(im, ax=ax, orientation="vertical")

    plt.tight_layout()
    plt.show()


def plot_interactive_reconstruction(
    dataset: Union[ConcentrationsDataset, FusedHelicoidConcentrationDataset],
    model_pipeline: ModelPipeline,
    coarseness: int = 1,
    figsize: tuple = (16, 12),
):
    """
    Interactive version of plot_reconstruction with a dropdown menu to select sample IDs.

    Creates a dropdown widget populated with all available sample IDs from the dataset,
    and updates the reconstruction plot when a new ID is selected.

    Args:
        dataset (ConcentrationsDataset): Dataset containing the samples with ground truth
            and reduced wavelength concentration data.
        model_pipeline (ModelPipeline): Trained model pipeline used to generate
            reconstructed concentrations from reduced wavelength input.
        coarseness (int, optional): Subsampling factor for spatial dimensions.
            If > 1, the maps will be downsampled by this factor. Defaults to 1.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (16, 12).
    """
    # Get all available sample IDs from the dataset
    sample_ids = list(dataset.sample_map.keys())

    if not sample_ids:
        print("No samples found in the dataset.")
        return

    # Create dropdown widget
    id_dropdown = widgets.Dropdown(
        options=sample_ids,
        value=sample_ids[0],
        description="Sample ID:",
        style={"description_width": "initial"},
    )

    # Create the figure and axes once - reuse for smooth updates
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    plt.tight_layout()

    # Store references to image objects and colorbars for efficient updates
    image_objects = []
    colorbar_objects = []

    def sub(m):
        return m[::coarseness, ::coarseness]

    def get_data_for_id(sample_id):
        """Extract and process data for a given sample ID."""
        data = dataset.get_sample_by_id(sample_id)
        if isinstance(dataset, FusedHelicoidConcentrationDataset):
            original = data.concentration_data["gt"]["coef_list"]
            original = np.transpose(original, (1, 2, 0))  # permute to HWC
            input_concentration = data.concentration_data["reduced_wl"]["coef_list"]

            reconstructed = (
                (model_pipeline.predict_sample(data))
                .permute(0, 2, 3, 1)  # permute to BHWC
                .squeeze(0)
                .cpu()
                .numpy()
            )
            input_data = np.transpose(input_concentration, (1, 2, 0))  # permute to HWC

        else:
            original = data["gt"]["coef_list"]
            input_data = data["reduced_wl"]["coef_list"]
            reconstructed = model_pipeline.predict_sample(data).squeeze(0).cpu().numpy()
            if reconstructed.shape != original.shape:
                logger.warning(
                    f"Reconstructed shape {reconstructed.shape} does not match original shape {original.shape}"
                )
                reconstructed = crop_to_input_shape(reconstructed, input_data)

        if model_pipeline.config.merge_channels_to_signals:
            summaries = _get_signal_summaries(
                model_pipeline.config.merge_channels_to_signals,
                original,
                reconstructed,
                input_data,
                sub,
            )
        else:
            summaries = _get_default_summaries(
                model_pipeline,
                original,
                reconstructed,
                input_data,
                sub,
            )

        return summaries

    def initialize_plot(sample_id):
        """Initialize the plot with the first sample."""
        summaries = get_data_for_id(sample_id)
        titles = ["Truth", "Reconstructed", "Input (reduced)", "Absolute Error"]

        for row, (name, o, r, inp, cmap) in enumerate(summaries):
            err = np.abs(r - o)
            maps = [o, r, inp, err]
            cmaps = [cmap, cmap, cmap, "bwr"]

            row_images = []
            row_colorbars = []

            for col, (m, title, cm) in enumerate(zip(maps, titles, cmaps)):
                ax = axes[row, col]
                im = ax.imshow(m, cmap=cm)
                ax.set_title(f"{name} – {title}")
                cbar = plt.colorbar(im, ax=ax, orientation="vertical")

                row_images.append(im)
                row_colorbars.append(cbar)

            image_objects.append(row_images)
            colorbar_objects.append(row_colorbars)

    def update_plot(change):
        """Update the plot when dropdown selection changes."""
        selected_id = change["new"]
        summaries = get_data_for_id(selected_id)
        titles = ["Truth", "Reconstructed", "Input (reduced)", "Absolute Error"]

        for row, (name, o, r, inp, cmap) in enumerate(summaries):
            err = np.abs(r - o)
            maps = [o, r, inp, err]

            for col, m in enumerate(maps):
                # Update image data
                image_objects[row][col].set_array(m)
                image_objects[row][col].set_clim(vmin=m.min(), vmax=m.max())

                # Update colorbar
                colorbar_objects[row][col].update_normal(image_objects[row][col])

        fig.canvas.draw_idle()

    # Initialize plot with first sample
    initialize_plot(sample_ids[0])

    # Connect the dropdown to the update function
    id_dropdown.observe(update_plot, names="value")

    # Display the widgets
    display(widgets.VBox([id_dropdown]))
    plt.show()


def _get_signal_summaries(
    signals_to_log,
    original: np.ndarray,
    reconstructed: np.ndarray,
    input_data: np.ndarray,
    sub,
):
    """Get summaries for specific signals defined in merge_channels_to_signals."""
    summaries = []

    for i, signal in enumerate(signals_to_log):
        if signal == Signal.HbT:
            orig_hbt = sub(original[..., i])
            recon_hbt = sub(reconstructed[..., i])
            input_hbt = sub(input_data[..., i])
            summaries.append(("HBT", orig_hbt, recon_hbt, input_hbt, "Reds"))

        elif signal == Signal.diffCCO:
            orig_cco = sub(original[..., i])
            recon_cco = sub(reconstructed[..., i])
            input_cco = sub(input_data[..., i])
            summaries.append(("diffCCO", orig_cco, recon_cco, input_cco, "terrain"))

        elif signal == "b":
            # For signal "b", we might need to add specific logic here
            orig_b = sub(original[..., i])
            recon_b = sub(reconstructed[..., i])
            input_b = sub(input_data[..., i])
            summaries.append(("b", orig_b, recon_b, input_b, "seismic_r"))

    return summaries


def _get_default_summaries(
    model_pipeline: ModelPipeline,
    original: np.ndarray,
    reconstructed: np.ndarray,
    input_data: np.ndarray,
    sub,
):
    """Get the default summaries"""
    if model_pipeline.config.chosen_molecules is not None:
        molecule_to_filtered_idx = {
            mol_name: i
            for i, mol_name in enumerate(model_pipeline.config.chosen_molecules)
        }
    else:
        molecule_to_filtered_idx = {
            name: MoleculeIndex[name].value
            for name, _ in MoleculeIndex.__members__.items()
        }

    summaries = []

    # HBT = HBO2 + HB (only if both molecules are available)
    if "HBO2" in molecule_to_filtered_idx and "HB" in molecule_to_filtered_idx:
        hbo2_idx = molecule_to_filtered_idx["HBO2"]
        hb_idx = molecule_to_filtered_idx["HB"]

        orig_hbt = sub(original[..., hbo2_idx] + original[..., hb_idx])
        recon_hbt = sub(reconstructed[..., hbo2_idx] + reconstructed[..., hb_idx])
        input_hbt = sub(input_data[..., hbo2_idx] + input_data[..., hb_idx])
        summaries.append(("HBT", orig_hbt, recon_hbt, input_hbt, "Reds"))

    # diffCCO = COXA - CREDA (only if both molecules are available)
    if "COXA" in molecule_to_filtered_idx and "CREDA" in molecule_to_filtered_idx:
        coxa_idx = molecule_to_filtered_idx["COXA"]
        creda_idx = molecule_to_filtered_idx["CREDA"]

        orig_cco = sub(original[..., coxa_idx] - original[..., creda_idx])
        recon_cco = sub(reconstructed[..., coxa_idx] - reconstructed[..., creda_idx])
        input_cco = sub(input_data[..., coxa_idx] - input_data[..., creda_idx])
        summaries.append(("diffCCO", orig_cco, recon_cco, input_cco, "terrain"))

    # FAT (if available)
    if "FAT" in molecule_to_filtered_idx:
        fat_idx = molecule_to_filtered_idx["FAT"]

        orig_fat = sub(original[..., fat_idx])
        recon_fat = sub(reconstructed[..., fat_idx])
        input_fat = sub(input_data[..., fat_idx])
        summaries.append(("FAT", orig_fat, recon_fat, input_fat, "viridis"))

    # If no standard summaries are available, show first 3 individual molecules
    if not summaries and model_pipeline.config.chosen_molecules is not None:
        for i, mol_name in enumerate(model_pipeline.config.chosen_molecules[:3]):
            orig_mol = sub(original[..., i])
            recon_mol = sub(reconstructed[..., i])
            input_mol = sub(input_data[..., i])
            summaries.append((mol_name, orig_mol, recon_mol, input_mol, "viridis"))

    return summaries


def crop_to_input_shape(output: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Crop the output array so that it matches the shape of the reference array.
    Assumes arrays are HWC (height, width, channels).
    """
    H_ref, W_ref, _ = reference.shape
    H_out, W_out, _ = output.shape

    # Start/stop indices for cropping
    h_start = max((H_out - H_ref) // 2, 0)
    w_start = max((W_out - W_ref) // 2, 0)

    h_end = h_start + H_ref
    w_end = w_start + W_ref

    return output[h_start:h_end, w_start:w_end, :]


def plot_models_visualization(
    sample_id: str,
    model_paths: List[str],
    signal_or_molecule_idx: Union[Signal, int] = Signal.HbT,
    device: str = "cuda:0",
    model_names: List[str] | None = None,
    figsize: tuple = (15, 5),
    cmap: str = "viridis",
):
    """
    Compare the outputs of multiple models on a single sample from the dataset.

    Args:
        sample_id (str): The ID of the sample to analyze.
        model_paths (List[str]): List of file paths to the saved model checkpoints.
        signal (Signal): Signal type to visualize (default: HbT).
        device (str): Device to run inference on (default: "cuda:0").
        model_names (List[str] | None): Optional list of model names for display.
        molecule_idx (int): Index of molecule/channel to visualize (default: 0).
        figsize (tuple): Figure size (default: (15, 5)).
        cmap (str): Colormap for visualization (default: 'viridis').
    """
    num_models = len(model_paths)

    # Use model paths as names if no names provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]

    # Storage for results
    input_images = []
    reconstructions = []
    absolute_errors = []

    print(f"Processing {num_models} models for sample {sample_id}...")

    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        print(f"Loading model {i+1}/{num_models}: {model_name}")

        try:
            # Load model pipeline
            pipeline = load_model_pipeline(model_path, device=device)

            # Get sample data
            data = pipeline.data_manager.dataset.get_sample_by_id(sample_id)

            # Extract input and ground truth
            if isinstance(data, FusedHelicoidSample):
                input_data = data.concentration_data["reduced_wl"]["coef_list"]
                gt_data = data.concentration_data["gt"]["coef_list"]
            else:
                input_data = data["reduced_wl"]["coef_list"]
                gt_data = data["gt"]["coef_list"]

            # Make prediction
            with torch.no_grad():
                reconstruction = pipeline.predict_sample(data)
                reconstruction = reconstruction.squeeze(0).cpu().numpy()

            # Determine what to visualize based on signal parameter
            if signal_or_molecule_idx == Signal.HbT:
                # HbT = HBO2 + HB
                input_signal = get_HbT_map(input_data)
                recon_signal = get_HbT_map(reconstruction)
                gt_signal = get_HbT_map(gt_data)
                error_signal = np.abs(recon_signal - gt_signal)

            elif signal_or_molecule_idx == Signal.diffCCO:
                # diffCCO = COXA - CREDA
                input_signal = get_diffCCO_map(input_data)
                recon_signal = get_diffCCO_map(reconstruction)
                gt_signal = get_diffCCO_map(gt_data)
                error_signal = np.abs(recon_signal - gt_signal)

            else:
                # Default to individual molecule channel
                input_signal = input_data[:, :, signal_or_molecule_idx]
                recon_signal = reconstruction[:, :, signal_or_molecule_idx]
                gt_signal = gt_data[:, :, signal_or_molecule_idx]
                error_signal = np.abs(recon_signal - gt_signal)

            # Store results
            input_images.append(input_signal)
            reconstructions.append(recon_signal)
            absolute_errors.append(error_signal)

            # Clean up GPU memory
            del pipeline
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            # Add empty placeholders to maintain array structure
            input_images.append(np.zeros((100, 100)))
            reconstructions.append(np.zeros((100, 100)))
            absolute_errors.append(np.zeros((100, 100)))

    # Determine colormap for visualization
    if signal_or_molecule_idx == Signal.HbT:
        vis_cmap = "Reds"
        signal_display_name = "HbT"
    elif signal_or_molecule_idx == Signal.diffCCO:
        vis_cmap = "terrain"
        signal_display_name = "diffCCO"
    else:
        vis_cmap = cmap
        signal_display_name = f"Molecule {signal_or_molecule_idx}"

    # Create the comparison plot
    fig, axes = plt.subplots(
        num_models, 3, figsize=(figsize[0], figsize[1] * num_models)
    )

    # Handle single model case
    if num_models == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_models):
        # Input image
        im1 = axes[i, 0].imshow(input_images[i], cmap=vis_cmap)
        axes[i, 0].set_title(f"Input ({signal_display_name})" if i == 0 else "")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        # Add model name as ylabel
        axes[i, 0].set_ylabel(
            model_names[i], rotation=90, labelpad=20, fontsize=12, fontweight="bold"
        )

        plt.colorbar(im1, ax=axes[i, 0])

        # Reconstruction
        im2 = axes[i, 1].imshow(reconstructions[i], cmap=vis_cmap)
        axes[i, 1].set_title(
            f"Reconstruction ({signal_display_name})" if i == 0 else ""
        )
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])

        plt.colorbar(im2, ax=axes[i, 1])

        # Absolute error
        im3 = axes[i, 2].imshow(absolute_errors[i], cmap="bwr")
        axes[i, 2].set_title(
            f"Absolute Error ({signal_display_name})" if i == 0 else ""
        )
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

        # Calculate and display MSE on the error plot
        mse = np.mean(absolute_errors[i] ** 2)
        axes[i, 2].text(
            0.05,
            0.95,
            f"MSE: {mse:.6f}",
            transform=axes[i, 2].transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            fontsize=10,
            fontweight="bold",
            verticalalignment="top",
        )

        plt.colorbar(im3, ax=axes[i, 2])

    plt.suptitle(
        f"Model Comparison for Sample {sample_id} - {signal_display_name}",
        fontsize=16,
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"\nComparison Statistics for Sample {sample_id}, {signal_display_name}:")
    for i, model_name in enumerate(model_names):
        mse = np.mean(absolute_errors[i] ** 2)
        mae = np.mean(absolute_errors[i])
        print(f"  {model_name}: MSE={mse:.6f}, MAE={mae:.6f}")

    return {
        "input_images": input_images,
        "reconstructions": reconstructions,
        "absolute_errors": absolute_errors,
        "model_names": model_names,
    }


def plot_samples_models_comparison(
    sample_ids: List[str],
    model_paths: List[str],
    signal: Union[Signal, int] = Signal.HbT,
    device: str = "cuda:0",
    model_names: List[str] | None = None,
    figsize: tuple = (16, 3),
) -> None:
    """
    Compare multiple models across multiple samples for a specific signal.

    Creates a plot where:
    - Each row corresponds to a different sample_id (with rotated sample ID label)
    - Columns show: Input, Model1, Model2, ..., Ground Truth

    Parameters
    ----------
    sample_ids : List[str]
        List of sample IDs to visualize.
    model_paths : List[str]
        List of file paths to the saved model checkpoints (usually 2 models for comparison).
    signal : Union[Signal, int], optional
        Signal type to visualize (Signal.HbT or Signal.diffCCO). Defaults to Signal.HbT.
    device : str, optional
        Device to run inference on. Defaults to "cuda:0".
    model_names : List[str] | None, optional
        Optional list of model names for display. If None, uses "Model 1", "Model 2", etc.
    figsize : tuple, optional
        Base figure size per row. Defaults to (16, 3).
    """
    num_samples = len(sample_ids)
    num_models = len(model_paths)
    num_cols = 2 + num_models  # Input + Models + Ground Truth

    # Use model paths as names if no names provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]

    # Determine colormap and signal name for visualization
    if signal == Signal.HbT:
        vis_cmap = "Reds"
        signal_display_name = "HbT"
    elif signal == Signal.diffCCO:
        vis_cmap = "terrain"
        signal_display_name = "diffCCO"
    else:
        vis_cmap = "viridis"
        signal_display_name = f"Molecule {signal}"

    # Create the plot
    fig, axes = plt.subplots(
        num_samples, num_cols, figsize=(figsize[0], figsize[1] * num_samples)
    )

    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.1, wspace=0.3)

    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    # Load all model pipelines upfront
    print(f"Loading {num_models} models...")
    pipelines = []
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        try:
            print(f"Loading model {i+1}/{num_models}: {model_name}")
            pipeline = load_model_pipeline(model_path, device=device)
            pipelines.append(pipeline)
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            pipelines.append(None)  # Add None as placeholder for failed models

    print(f"Processing {num_samples} samples with {num_models} models...")

    for row, sample_id in enumerate(sample_ids):
        print(f"Processing sample {row+1}/{num_samples}: {sample_id}")

        # Storage for this sample's data
        sample_input = None
        sample_gt = None
        sample_reconstructions = []

        # Process each model for this sample
        for model_idx, (pipeline, model_name) in enumerate(zip(pipelines, model_names)):
            try:
                if pipeline is None:
                    # Handle failed model loading
                    raise Exception("Model failed to load")

                # Get sample data (use first successful pipeline to get the data)
                if sample_input is None:
                    data = pipeline.data_manager.dataset.get_sample_by_id(sample_id)

                    # Extract input and ground truth
                    if isinstance(data, FusedHelicoidSample):
                        input_data = data.concentration_data["reduced_wl"]["coef_list"]
                        gt_data = data.concentration_data["gt"]["coef_list"]
                    else:
                        input_data = data["reduced_wl"]["coef_list"]
                        gt_data = data["gt"]["coef_list"]

                    # Compute input and ground truth signals
                    if signal == Signal.HbT:
                        sample_input = get_HbT_map(input_data)
                        sample_gt = get_HbT_map(gt_data)
                    elif signal == Signal.diffCCO:
                        sample_input = get_diffCCO_map(input_data)
                        sample_gt = get_diffCCO_map(gt_data)
                    else:
                        sample_input = input_data[:, :, signal]
                        sample_gt = gt_data[:, :, signal]
                else:
                    # For subsequent models, just get the data (should be the same across models)
                    data = pipeline.data_manager.dataset.get_sample_by_id(sample_id)

                # Make prediction
                with torch.no_grad():
                    reconstruction = pipeline.predict_sample(data)
                    reconstruction = reconstruction.squeeze(0).cpu().numpy()

                # Compute reconstruction signal
                if signal == Signal.HbT:
                    recon_signal = get_HbT_map(reconstruction)
                elif signal == Signal.diffCCO:
                    recon_signal = get_diffCCO_map(reconstruction)
                else:
                    recon_signal = reconstruction[:, :, signal]

                sample_reconstructions.append(recon_signal)

            except Exception as e:
                print(
                    f"Error processing model {model_name} for sample {sample_id}: {str(e)}"
                )
                # Add empty placeholder
                if sample_input is None:
                    sample_input = np.zeros((100, 100))
                    sample_gt = np.zeros((100, 100))
                sample_reconstructions.append(np.zeros_like(sample_input))

        # Plot the row for this sample
        col = 0

        # Column 1: Input
        im_input = axes[row, col].imshow(sample_input, cmap=vis_cmap)
        if row == 0:  # Only add title to first row
            axes[row, col].set_title(f"Input")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        # Add sample ID as ylabel (rotated 90 degrees)
        axes[row, col].set_ylabel(
            sample_id, rotation=90, labelpad=20, fontsize=12, fontweight="bold"
        )
        plt.colorbar(im_input, ax=axes[row, col], shrink=0.7)
        col += 1

        # Columns 2 to N+1: Model reconstructions
        for model_idx, (recon_signal, model_name) in enumerate(
            zip(sample_reconstructions, model_names)
        ):
            im_recon = axes[row, col].imshow(recon_signal, cmap=vis_cmap)
            if row == 0:  # Only add title to first row
                axes[row, col].set_title(f"{model_name}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            plt.colorbar(im_recon, ax=axes[row, col], shrink=0.7)
            col += 1

        # Last column: Ground Truth
        im_gt = axes[row, col].imshow(sample_gt, cmap=vis_cmap)
        if row == 0:  # Only add title to first row
            axes[row, col].set_title(f"Ground Truth")
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        plt.colorbar(im_gt, ax=axes[row, col], shrink=0.7)

    # plt.suptitle(
    #     f"Model Comparison Across Samples - {signal_display_name}",
    #     fontsize=16,
    #     y=1.01
    # )
    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=1.0)
    plt.show()

    # Clean up GPU memory by deleting all pipelines
    print("Cleaning up GPU memory...")
    for pipeline in pipelines:
        if pipeline is not None:
            del pipeline
    torch.cuda.empty_cache()


def plot_single_sample_reconstruction_quality(
    sample_id: str,
    model_paths: List[str],
    signal: Union[Signal, int] = Signal.HbT,
    device: str = "cuda:0",
    model_names: List[str] | None = None,
    figsize: tuple = (8, 6),
) -> None:
    """
    Compare reconstruction quality of multiple models on a single sample.

    Creates a plot where:
    - Each row corresponds to a different model (with rotated model name label)
    - Columns show: Reconstruction, Absolute Error (vs Ground Truth)

    Parameters
    ----------
    sample_id : str
        Sample ID to analyze.
    model_paths : List[str]
        List of file paths to the saved model checkpoints.
    signal : Union[Signal, int], optional
        Signal type to visualize (Signal.HbT or Signal.diffCCO). Defaults to Signal.HbT.
    device : str, optional
        Device to run inference on. Defaults to "cuda:0".
    model_names : List[str] | None, optional
        Optional list of model names for display. If None, uses "Model 1", "Model 2", etc.
    figsize : tuple, optional
        Figure size. Defaults to (8, 6).
    """
    num_models = len(model_paths)
    num_cols = 2  # Reconstruction + Absolute Error

    # Use model paths as names if no names provided
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(num_models)]

    # Determine colormap and signal name for visualization
    if signal == Signal.HbT:
        vis_cmap = "Reds"
        signal_display_name = "HbT"
    elif signal == Signal.diffCCO:
        vis_cmap = "terrain"
        signal_display_name = "diffCCO"
    else:
        vis_cmap = "viridis"
        signal_display_name = f"Molecule {signal}"

    # Create the plot
    fig, axes = plt.subplots(num_models, num_cols, figsize=figsize)

    # Reduce spacing between subplots
    plt.subplots_adjust(hspace=0.15, wspace=0.3)

    # Handle single model case
    if num_models == 1:
        axes = axes.reshape(1, -1)

    # Load all model pipelines upfront
    print(f"Loading {num_models} models...")
    pipelines = []
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        try:
            print(f"Loading model {i+1}/{num_models}: {model_name}")
            pipeline = load_model_pipeline(model_path, device=device)
            pipelines.append(pipeline)
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            pipelines.append(None)  # Add None as placeholder for failed models

    print(f"Processing sample {sample_id} with {num_models} models...")

    # Get ground truth data (use first successful pipeline)
    sample_gt = None
    for pipeline in pipelines:
        if pipeline is not None:
            try:
                data = pipeline.data_manager.dataset.get_sample_by_id(sample_id)

                # Extract ground truth
                if isinstance(data, FusedHelicoidSample):
                    gt_data = data.concentration_data["gt"]["coef_list"]
                else:
                    gt_data = data["gt"]["coef_list"]

                # Compute ground truth signal
                if signal == Signal.HbT:
                    sample_gt = get_HbT_map(gt_data)
                elif signal == Signal.diffCCO:
                    sample_gt = get_diffCCO_map(gt_data)
                else:
                    sample_gt = gt_data[:, :, signal]
                break
            except Exception as e:
                print(f"Error getting ground truth data: {str(e)}")
                continue

    if sample_gt is None:
        print("Error: Could not load ground truth data")
        return

    # Process each model
    for row, (pipeline, model_name) in enumerate(zip(pipelines, model_names)):
        try:
            if pipeline is None:
                # Handle failed model loading
                raise Exception("Model failed to load")

            # Get sample data
            data = pipeline.data_manager.dataset.get_sample_by_id(sample_id)

            # Make prediction
            with torch.no_grad():
                reconstruction = pipeline.predict_sample(data)
                reconstruction = reconstruction.squeeze(0).cpu().numpy()

            # Compute reconstruction signal
            if signal == Signal.HbT:
                recon_signal = get_HbT_map(reconstruction)
            elif signal == Signal.diffCCO:
                recon_signal = get_diffCCO_map(reconstruction)
            else:
                recon_signal = reconstruction[:, :, signal]

            # Compute absolute error
            abs_error = np.abs(recon_signal - sample_gt)

        except Exception as e:
            print(f"Error processing model {model_name}: {str(e)}")
            # Create placeholder data
            recon_signal = np.zeros_like(sample_gt)
            abs_error = np.zeros_like(sample_gt)

        # Plot reconstruction (column 0)
        im_recon = axes[row, 0].imshow(recon_signal, cmap=vis_cmap)
        if row == 0:  # Only add title to first row
            axes[row, 0].set_title(f"Reconstruction\n({signal_display_name})")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        # Add model name as ylabel (rotated 90 degrees)
        axes[row, 0].set_ylabel(
            model_name, rotation=90, labelpad=20, fontsize=12, fontweight="bold"
        )
        plt.colorbar(im_recon, ax=axes[row, 0], shrink=0.7)

        # Plot absolute error (column 1)
        im_error = axes[row, 1].imshow(abs_error, cmap="hot")
        if row == 0:  # Only add title to first row
            axes[row, 1].set_title(f"Absolute Error\n({signal_display_name})")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])
        plt.colorbar(im_error, ax=axes[row, 1], shrink=0.7)

    plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=1.0)
    plt.show()

    # Print some statistics
    print(
        f"\nReconstruction Quality Statistics for Sample {sample_id}, {signal_display_name}:"
    )
    for i, model_name in enumerate(model_names):
        if pipelines[i] is not None:
            try:
                data = pipelines[i].data_manager.dataset.get_sample_by_id(sample_id)
                with torch.no_grad():
                    reconstruction = pipelines[i].predict_sample(data)
                    reconstruction = reconstruction.squeeze(0).cpu().numpy()

                if signal == Signal.HbT:
                    recon_signal = get_HbT_map(reconstruction)
                elif signal == Signal.diffCCO:
                    recon_signal = get_diffCCO_map(reconstruction)
                else:
                    recon_signal = reconstruction[:, :, signal]

                abs_error = np.abs(recon_signal - sample_gt)
                mse = np.mean(abs_error**2)
                mae = np.mean(abs_error)
                print(f"  {model_name}: MSE={mse:.6f}, MAE={mae:.6f}")
            except Exception as e:
                print(f"  {model_name}: Error computing statistics - {str(e)}")

    # Clean up GPU memory by deleting all pipelines
    print("Cleaning up GPU memory...")
    for pipeline in pipelines:
        if pipeline is not None:
            del pipeline
    torch.cuda.empty_cache()


def plot_helicoid_concentration_montage(
    dataset: FusedHelicoidConcentrationDataset,
    sample_ids: List[str],
    figsize: tuple = (20, 4),
    colormap_hbt: str = "Reds",
    colormap_diffcco: str = "terrain",
) -> None:
    """
    Create a montage of helicoid images with concentration maps for HbT and diffCCO.

    For each sample, displays:
    - Sample ID (rotated 90° on the left side of the first column)
    - HSI data as RGB (column 1)
    - HbT concentration map from reduced wavelength set (column 2)
    - HbT ground truth concentration map (column 3)
    - diffCCO concentration map from reduced wavelength set (column 4)
    - diffCCO ground truth concentration map (column 5)

    Parameters
    ----------
    dataset : FusedHelicoidConcentrationDataset
        The dataset containing HSI data and concentration maps.
    sample_ids : List[str]
        List of sample IDs to include in the montage.
    figsize : tuple, optional
        Figure size as (width, height). Defaults to (20, 4).
    colormap_hbt : str, optional
        Colormap for HbT concentration maps. Defaults to "Reds".
    colormap_diffcco : str, optional
        Colormap for diffCCO concentration maps. Defaults to "terrain".
    """
    n_samples = len(sample_ids)
    n_cols = 5  # RGB, HbT_reduced, HbT_gt, diffCCO_reduced, diffCCO_gt (no separate sample ID column)

    fig, axes = plt.subplots(
        n_samples,
        n_cols,
        figsize=(figsize[0], figsize[1] * n_samples),
    )

    # Handle case of single sample
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    column_titles = [
        "Synthetic RGB",
        "HbT (reduced)",
        "HbT (GT)",
        "diffCCO (reduced)",
        "diffCCO (GT)",
    ]

    for row, sample_id in enumerate(sample_ids):
        # Get sample data
        sample = dataset.get_sample_by_id(sample_id)
        if sample is None:
            logger.warning(f"Sample {sample_id} not found in dataset")
            continue

        # Extract concentration data
        reduced_wl_coef = sample.concentration_data["reduced_wl"]["coef_list"]
        gt_coef = sample.concentration_data["gt"]["coef_list"]

        # Calculate HbT (HBO2 + HB)
        hbt_reduced = get_HbT_map(reduced_wl_coef)
        hbt_gt = get_HbT_map(gt_coef)

        # Calculate diffCCO (COXA - CREDA)
        diffcco_reduced = get_diffCCO_map(reduced_wl_coef)
        diffcco_gt = get_diffCCO_map(gt_coef)

        rgb_image = create_rgb(
            sample.hsi_cube_original,
            sample_type=SampleType.HELICOID,
            r_band=708.97,
            g_band=542.03,
            b_band=479.06,
        )

        # Normalize RGB for display (handle NaN/inf values)
        rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)

        # Per-channel min-max normalization for RGB
        for channel in range(3):
            channel_data = rgb_image[..., channel]
            if channel_data.max() > channel_data.min():
                rgb_image[..., channel] = (channel_data - channel_data.min()) / (
                    channel_data.max() - channel_data.min()
                )

        # Plot data
        images = [rgb_image, hbt_reduced, hbt_gt, diffcco_reduced, diffcco_gt]
        cmaps = [
            None,
            colormap_hbt,
            colormap_hbt,
            colormap_diffcco,
            colormap_diffcco,
        ]

        for col in range(n_cols):
            ax = axes[row, col]

            if col == 0:  # RGB image (first column)
                im_rgb = ax.imshow(rgb_image)
                ax.set_xticks([])
                ax.set_yticks([])

                # Add a dummy colorbar to maintain consistent spacing with other columns
                # Make it invisible by setting alpha=0
                cbar_rgb = plt.colorbar(
                    im_rgb, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8
                )
                cbar_rgb.ax.set_visible(False)  # Hide the colorbar but keep the space

                # Add rotated sample ID on the left side of the first column
                ax.text(
                    -0.1,  # Position to the left of the axis
                    0.5,  # Vertical center
                    sample_id,
                    transform=ax.transAxes,
                    fontsize=12,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    rotation=90,  # Rotate 90 degrees
                )
            else:  # Concentration maps
                img_data = images[col]
                cmap = cmaps[col]

                # Handle NaN/inf values
                img_data = np.nan_to_num(
                    img_data,
                    nan=0.0,
                    posinf=(
                        img_data[np.isfinite(img_data)].max()
                        if np.any(np.isfinite(img_data))
                        else 1.0
                    ),
                    neginf=(
                        img_data[np.isfinite(img_data)].min()
                        if np.any(np.isfinite(img_data))
                        else 0.0
                    ),
                )

                im = ax.imshow(img_data, cmap=cmap)
                ax.set_xticks([])
                ax.set_yticks([])

                # Add colorbar
                plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8)

            # Add column titles only for the first row
            if row == 0:
                ax.set_title(column_titles[col], fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.show()


def plot_ssr_model_inference_with_comparison(
    model_pipeline: ModelPipeline,
    sample_id: str,
    r_band: float = 708.97,
    g_band: float = 542.03,
    b_band: float = 479.06,
    pixel: tuple = (50, 50),
    figsize: tuple = (15, 5),
) -> None:
    """
    Plot SSR model inference results with comparison to ground truth.

    Creates a two-panel plot:
    - Left: RGB representation of the hyperspectral cube with the selected pixel marked
    - Right: Spectrum comparison between ground truth and SSR enhanced spectrum at the selected pixel

    Parameters
    ----------
    model_pipeline : ModelPipeline
        The trained SSR model pipeline for inference
    sample_id : str
        ID of the sample to analyze
    r_band : float, optional
        Red band wavelength for RGB creation. Defaults to 708.97.
    g_band : float, optional
        Green band wavelength for RGB creation. Defaults to 542.03.
    b_band : float, optional
        Blue band wavelength for RGB creation. Defaults to 479.06.
    pixel : tuple, optional
        (y, x) coordinates of the pixel to analyze. Defaults to (50, 50).
    device : str, optional
        Device to run inference on. Defaults to "cuda:0".
    figsize : tuple, optional
        Figure size as (width, height). Defaults to (15, 5).
    """
    helicoid_sample = HelicoidDataset(
        left_cut=400, right_cut=1000, with_delta_A=True
    ).get_sample_by_id(sample_id)
    dataset = model_pipeline.data_manager.dataset
    dataset.set_inference_mode(True)
    sample = dataset.get_sample_by_id(sample_id)
    logger.debug(f"Reference Pixel: {helicoid_sample.reference_pixel}")

    # Get SSR enhanced output
    output = model_pipeline.predict_sample(sample)
    output = output.squeeze(0).cpu().numpy()

    # Get ground truth hyperspectral cube
    gt_spectrum = sample.hsi_cube_original

    # Create RGB image using the specified bands
    rgb_image = create_rgb(
        helicoid_sample.hsi_cube,
        r_band=r_band,
        g_band=g_band,
        b_band=b_band,
        sample_type=SampleType.HELICOID,
    )

    # Normalize RGB for display (handle NaN/inf values)
    rgb_image = np.nan_to_num(rgb_image, nan=0.0, posinf=1.0, neginf=0.0)

    # Per-channel min-max normalization for RGB
    for channel in range(3):
        channel_data = rgb_image[..., channel]
        if channel_data.max() > channel_data.min():
            rgb_image[..., channel] = (channel_data - channel_data.min()) / (
                channel_data.max() - channel_data.min()
            )

    # Extract pixel coordinates
    y_pixel, x_pixel = pixel

    # Extract spectra at the selected pixel
    gt_pixel_spectrum = gt_spectrum[:, y_pixel, x_pixel]
    ssr_pixel_spectrum = output[:, y_pixel, x_pixel]

    # Create the plot
    fig, (ax_img, ax_spec) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 1.5]}
    )

    # Left panel: RGB image with pixel marker
    ax_img.imshow(rgb_image)
    ax_img.scatter(x_pixel, y_pixel, c="red", s=100, marker="+", linewidths=3)
    ax_img.set_title(f"Sample {sample_id}\nRGB Image with Selected Pixel")
    ax_img.set_xlabel("X coordinate")
    ax_img.set_ylabel("Y coordinate")
    ax_img.grid(False)

    # Add text annotation for the pixel coordinates
    ax_img.text(
        x_pixel + 5,
        y_pixel - 5,
        f"({x_pixel}, {y_pixel})",
        color="red",
        fontweight="bold",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Right panel: Spectrum comparison
    ax_spec.plot(
        dataset.helicoid_dataset._cut_wavelengths,
        gt_pixel_spectrum,
        "b-",
        label="Ground Truth",
        linewidth=2,
    )
    ax_spec.plot(
        dataset.helicoid_dataset._cut_wavelengths,
        ssr_pixel_spectrum,
        "r-",
        label="SSR Enhanced",
        linewidth=2,
    )

    for wl in dataset.wavelength_array:
        ax_spec.axvline(x=wl, color="green", linestyle="--", alpha=0.7, linewidth=1)
    # Add a single legend entry for all selected wavelengths
    ax_spec.axvline(
        x=dataset.wavelength_array[0],
        color="green",
        linestyle="--",
        alpha=0.7,
        linewidth=1,
        label=f"Selected WL ({len(dataset.wavelength_array)} bands)",
    )

    ax_spec.set_title(f"Spectrum Comparison at Pixel ({x_pixel}, {y_pixel})")
    ax_spec.set_xlabel("Wavelength (nm)")
    ax_spec.set_ylabel("Intensity")
    ax_spec.legend()
    ax_spec.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
