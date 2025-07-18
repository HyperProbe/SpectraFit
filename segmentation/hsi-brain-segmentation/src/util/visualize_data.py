import matplotlib.pyplot as plt
import numpy as np
from src.dataset.dataset import build_hsi_testloader, get_wavelengths_from_metadata
import cv2
import ipywidgets as widgets
from IPython.display import display
import torch
from src.model.dimensionality_reduction.gaussian import build_gaussian_channel_reducer


def show_interactive_image_with_spectrum(image_id=4, gcr_path=None):
    # Precompute data outside of the onclick function
    testloader_HSI_all_channels = build_hsi_testloader()

    if gcr_path is not None:
        gcr = build_gaussian_channel_reducer(
            load_from_path=gcr_path, device=torch.device("cpu")
        )
        gcr.eval()
        with torch.no_grad():
            input_image = (
                gcr(testloader_HSI_all_channels.dataset[image_id][0].unsqueeze(0))
                .squeeze()
                .cpu()
                .numpy()
            )
            input_image = input_image.transpose(1, 2, 0)

    else:
        # Full 826 channels
        input_image_3d = (
            testloader_HSI_all_channels.dataset[image_id][0].cpu().numpy().squeeze()
        )
        # Example: picking 3 channels by index (replace with your real channel selection or Gaussian)
        input_image = np.stack(
            [
                input_image_3d[425, :, :],
                input_image_3d[192, :, :],
                input_image_3d[109, :, :],
            ],
            axis=-1,
        )  # Convert to 3-channel grayscale

    label = testloader_HSI_all_channels.dataset[image_id][1].cpu().numpy().squeeze()

    # Normalize to 0-255 for display
    image_min, image_max = input_image.min(), input_image.max()
    input_image = 255 * (input_image - image_min) / (image_max - image_min)
    input_image = input_image.astype(np.uint8)

    label_overlay = np.zeros_like(input_image)
    label_overlay[label == 1] = [0, 0, 255]  # Red overlay where label == 1
    combined = cv2.addWeighted(input_image, 0.7, label_overlay, 0.3, 0)

    # Full 826-ch data for spectrum plotting
    pixel_spectra = testloader_HSI_all_channels.dataset[image_id][0].cpu().numpy()
    labels = testloader_HSI_all_channels.dataset[image_id][1].cpu().numpy()

    # ----------------------------------------------------------------
    # Setup figure and axes
    # ----------------------------------------------------------------
    fig, (ax_img, ax_spectrum) = plt.subplots(1, 2, figsize=(12, 6))

    # Show the image
    im_display = ax_img.imshow(combined)
    ax_img.set_title("Click pixels to view spectra")

    # Track selected pixels
    selected_pixels = []
    scatter_plot = ax_img.scatter([], [], c=[], s=100, marker="+", edgecolors=[])

    # For the pixel spectra lines
    lines = []
    colors = ["b", "g", "r", "c", "m", "y", "k"]

    # ----------------------------------------------------------------
    # Gaussians toggling
    # ----------------------------------------------------------------
    show_gaussians = False  # Initially hidden
    gaussian_lines = []  # Store line objects for Gaussians

    def plot_gaussians_on_ax_spectrum():
        """
        Plots the learned Gaussians from `gcr` onto ax_spectrum.
        Returns the list of line objects created.
        """
        new_lines = []
        if gcr is None:
            return new_lines  # If there's no GCR, do nothing

        with torch.no_grad():
            # channel_indices: 0..825
            channel_indices = torch.arange(gcr.num_input_channels).float()
            mu = gcr.mu.detach()  # (num_output_channels,)
            sigma = gcr.log_sigma.exp().detach()  # (num_output_channels,)
            colors = ["r", "g", "b"]

            for i in range(gcr.num_output_channels):
                # Compute normalized Gaussian
                w = torch.exp(-0.5 * ((channel_indices - mu[i]) / sigma[i]) ** 2)
                w = w / w.sum()  # Normalize
                # Plot on the same x-axis as the pixel spectra
                (line,) = ax_spectrum.plot(
                    get_wavelengths_from_metadata(),  # or channel_indices if you prefer
                    w.numpy(),
                    label=f"G{i+1} (μ={mu[i].item():.2f}, σ={sigma[i].item():.2f})",
                    color=colors[i % len(colors)],
                )
                new_lines.append(line)
        return new_lines

    def toggle_gaussians_callback(b):
        """
        Button callback that toggles the 'show_gaussians' flag
        and updates the spectrum plot accordingly.
        """
        nonlocal show_gaussians, gaussian_lines
        show_gaussians = not show_gaussians

        # If turning gaussians ON, re-plot them
        if show_gaussians:
            gaussian_lines = plot_gaussians_on_ax_spectrum()
        else:
            # Remove existing lines
            for gl in gaussian_lines:
                gl.remove()
            gaussian_lines.clear()

        # Rebuild legend if needed
        update_legend()
        fig.canvas.draw_idle()

    # Button to toggle Gaussians
    toggle_gaussians_button = widgets.Button(description="Toggle Gaussians")
    toggle_gaussians_button.on_click(toggle_gaussians_callback)
    display(toggle_gaussians_button)

    # ----------------------------------------------------------------
    # Clearing pixel selections
    # ----------------------------------------------------------------
    def on_clear_button_clicked(b):
        """Clear all pixel selections."""
        selected_pixels.clear()
        scatter_plot.set_offsets(np.empty((0, 2)))
        scatter_plot.set_edgecolors([])
        # Clear pixel spectrum lines
        for line in lines:
            line.remove()
        lines.clear()
        # Also remove the legend if present
        legend = ax_spectrum.get_legend()
        if legend:
            legend.remove()
        fig.canvas.draw_idle()

    clear_button = widgets.Button(description="Clear Selections")
    clear_button.on_click(on_clear_button_clicked)
    display(clear_button)

    # ----------------------------------------------------------------
    # Handling mouse clicks on the image
    # ----------------------------------------------------------------
    def onclick(event):
        if event.inaxes == ax_img:
            x = int(event.xdata)
            y = int(event.ydata)
            pixel_coords = (y, x)

            # Toggle selection
            if pixel_coords in selected_pixels:
                selected_pixels.remove(pixel_coords)
            else:
                selected_pixels.append(pixel_coords)

            update_plots()

    # ----------------------------------------------------------------
    # Update plots (spectra) after each user action
    # ----------------------------------------------------------------
    def update_plots():
        # Update scatter of selected pixels
        offsets = np.array([[px[1], px[0]] for px in selected_pixels])
        scatter_plot.set_offsets(offsets)

        marker_colors = [
            colors[idx % len(colors)] for idx in range(len(selected_pixels))
        ]
        scatter_plot.set_edgecolors(marker_colors)
        scatter_plot.set_facecolors("none")

        # Clear existing pixel-spectrum lines
        for line in lines:
            line.remove()
        lines.clear()

        # Plot each selected pixel's spectrum
        for idx, (py, px) in enumerate(selected_pixels):
            pixel_values = pixel_spectra[:, py, px]
            label_value = labels[:, py, px].item()
            color = colors[idx % len(colors)]
            (line,) = ax_spectrum.plot(
                get_wavelengths_from_metadata(),
                pixel_values,
                marker="o",
                linestyle="-",
                color=color,
                label=f"Pixel({py},{px}) Label={label_value}",
            )
            lines.append(line)

        update_legend()
        fig.canvas.draw_idle()

    def update_legend():
        # Remove old legend
        old_legend = ax_spectrum.get_legend()
        if old_legend:
            old_legend.remove()

        # If at least one pixel line or Gaussian line is present, build new legend
        plotted_objects = [obj for obj in list(ax_spectrum.get_lines())]
        if plotted_objects:
            ax_spectrum.legend()

    ax_spectrum.set_title("Pixel Intensity Across Channels")
    ax_spectrum.set_xlabel("Wavelength (nm)")
    ax_spectrum.set_ylabel("Intensity")
    ax_spectrum.grid(axis="y", linestyle="--", alpha=0.7)

    # Connect the click event
    cid = fig.canvas.mpl_connect("button_press_event", onclick)

    # Show the initial figure
    plt.show()


def plot_intensity(dataloader, bins=50, title="Intensity Distribution of Dataset"):
    """
    Plots the intensity distribution of a PyTorch dataset.

    Args:
        dataset (Dataset): PyTorch dataset containing the data.
        batch_size (int): Batch size for DataLoader.
        bins (int): Number of bins for the histogram.
    """

    # Collect intensity values from the dataset
    intensities = []
    for batch in dataloader:
        image = batch[0]
        intensities.append(image.flatten())

    # Flatten all intensities into a single array
    intensities = torch.cat(intensities).numpy()

    # Plotting the intensity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(intensities, bins=bins, alpha=0.7, color="blue")
    plt.title(title)
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def plot_intensity_for_label(
    dataloader, label=1, bins=50, title="Intensity Distribution for Label"
):
    """
    Plots the intensity distribution of pixels with a specific label in a PyTorch dataset.

    Args:
        dataloader (DataLoader): PyTorch DataLoader containing the dataset.
        label (int): The label value to filter pixels (default is 1).
        bins (int): Number of bins for the histogram.
        title (str): Title for the plot.
    """
    # Collect intensity values for the specified label
    intensities = []
    for batch in dataloader:
        images = batch[0]  # Batch of images
        masks = batch[1]  # Corresponding segmentation masks

        # Filter intensities where the mask is equal to the specified label
        for image, mask in zip(images, masks):
            intensities.append(image[mask == label].flatten())

    # Flatten all intensities into a single array
    intensities = torch.cat(intensities).numpy()

    # Plotting the intensity distribution
    plt.figure(figsize=(8, 6))
    plt.hist(intensities, bins=bins, alpha=0.7, color="blue")
    plt.title(f"{title} (Label={label})")
    plt.xlabel("Intensity Value")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def compute_snr(image, label):
    """
    Compute the Signal-to-Noise Ratio (SNR) for a given image and segmentation label.

    Args:
        image (torch.Tensor): The image tensor of shape (1, H, W).
        label (torch.Tensor): The segmentation label tensor of shape (1, H, W).

    Returns:
        float: The SNR value.
    """
    # Ensure image and label are in numpy format for easier indexing
    image = image.squeeze().numpy()
    label = label.squeeze().numpy()

    # Signal: Pixels where the label > 0
    signal_region = image[label > 0]

    # Background: Pixels where the label == 0
    background_region = image[label == 0]

    # Compute signal statistics
    mu_signal = signal_region.mean()

    # Compute noise as the standard deviation of the background
    sigma_noise = background_region.std()

    # Compute SNR
    if sigma_noise == 0:
        return float("inf")  # Handle edge case where noise is zero
    snr = mu_signal / sigma_noise
    return snr


def plot_snr_distribution(dataloader, bins=50, title="SNR Distribution of Dataset"):
    """
    Compute and plot the SNR distribution for a dataset.

    Args:
        dataset (Dataset): PyTorch dataset containing images and labels.
        batch_size (int): Batch size for DataLoader.
        bins (int): Number of bins for the histogram.
    """

    # Collect SNR values
    snr_values = []
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        for i in range(images.size(0)):
            snr = compute_snr(images[i], labels[i])
            snr_values.append(snr)

    # Plotting the SNR distribution
    plt.figure(figsize=(8, 6))
    plt.hist(snr_values, bins=bins, alpha=0.7, color="green")
    plt.title(title)
    plt.xlabel("SNR")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)
    plt.show()
