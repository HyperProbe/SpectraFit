import torch
import torch.nn as nn
from src.dataset.dataset import get_wavelengths_from_metadata
import numpy as np


class WindowChannelReducer(nn.Module):
    def __init__(self, windows):
        """
        Initializes the WindowChannelReducer.

        Args:
            windows (list of lists): A list containing three lists, each specifying the channel indices
                                     to be averaged for the R, G, and B channels respectively.
                                     Example: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        """
        super(WindowChannelReducer, self).__init__()
        if len(windows) != 3:
            raise ValueError(
                "There must be exactly three windows for R, G, and B channels."
            )

        self.windows = windows

    def forward(self, x):
        """
        Forward pass to reduce channels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 826, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 3, H, W)
        """
        reduced_channels = []
        for window in self.windows:
            # Select the specified channels
            selected = x[:, window, :, :]  # Shape: (batch_size, len(window), H, W)
            # Compute the mean across the channel dimension
            mean = selected.mean(dim=1, keepdim=True)  # Shape: (batch_size, 1, H, W)
            reduced_channels.append(mean)

        # Concatenate the reduced channels to form the RGB output
        out = torch.cat(reduced_channels, dim=1)  # Shape: (batch_size, 3, H, W)
        return out


def build_window_reducer(windows_in_nm, device):
    """
    Builds a WindowChannelReducer based on specified wavelength windows.

    Args:
        windows_in_nm (list of tuples): A list of three tuples, each defining the (min_nm, max_nm) range
                                        for the R, G, and B channels respectively.
                                        Example: [(400, 500), (500, 600), (600, 700)]
        device (torch.device or str): The device to which the WindowChannelReducer will be moved.

    Returns:
        WindowChannelReducer: An instance of WindowChannelReducer configured with the specified wavelength windows.
    """
    # Validate input
    if not isinstance(windows_in_nm, list) or len(windows_in_nm) != 3:
        raise ValueError(
            "windows_in_nm must be a list of three (min_nm, max_nm) tuples."
        )

    for i, window in enumerate(windows_in_nm):
        if (not isinstance(window, tuple) and not isinstance(window, list)) or len(
            window
        ) != 2:
            raise ValueError(f"Window {i} is not a valid (min_nm, max_nm) tuple.")

    # Retrieve the wavelength array
    wavelength_array = (
        get_wavelengths_from_metadata()
    )  # Should return array-like with length=826

    if len(wavelength_array) != 826:
        raise ValueError(
            "Wavelength array must have exactly 826 elements corresponding to each channel."
        )

    windows = []
    for i, (min_nm, max_nm) in enumerate(windows_in_nm):
        # Find channel indices where wavelength is within the window [min_nm, max_nm]
        indices = np.where((wavelength_array >= min_nm) & (wavelength_array <= max_nm))[
            0
        ].tolist()

        if len(indices) == 0:
            raise ValueError(
                f"No channels found in the wavelength window {min_nm}-{max_nm} nm for channel {i}."
            )

        windows.append(indices)

    # Instantiate the ChannelReducer with the identified windows
    channel_reducer = WindowChannelReducer(windows=windows).to(device)

    return channel_reducer


class SingleWindowReducer(nn.Module):
    def __init__(self, window):
        """
        Initializes the SingleWindowReducer.

        Args:
            window (list): A list specifying the channel indices to be averaged to create the single-channel output.
                           Example: [0, 1, 2]
        """
        super(SingleWindowReducer, self).__init__()
        if not isinstance(window, list) or len(window) == 0:
            raise ValueError("Window must be a non-empty list of channel indices.")

        self.window = window

    def forward(self, x):
        """
        Forward pass to reduce channels.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, C, H, W), where C is the number of channels.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, H, W)
        """
        # Select the channel range
        selected = x[:, self.window[0] : self.window[1], :, :] # Shape: (batch_size, selected_channels, H, W)

        # Compute the median across the channel dimension
        median = selected.median(
            dim=1, keepdim=True
        ).values  # Shape: (batch_size, 1, H, W)

        # Normalize to [0, 1]
        min_val = median.amin(
            dim=(2, 3), keepdim=True
        )  # Min value in spatial dimensions
        max_val = median.amax(
            dim=(2, 3), keepdim=True
        )  # Max value in spatial dimensions
        normalized = (median - min_val) / (max_val - min_val)

        return normalized


def build_single_window_reducer(window_in_nm, device):
    """
    Builds a SingleWindowReducer based on a specified wavelength window.

    Args:
        window_in_nm (tuple): A tuple defining the (min_nm, max_nm) range for the single-channel output.
                              Example: (400, 500)
        device (torch.device or str): The device to which the SingleWindowReducer will be moved.

    Returns:
        SingleWindowReducer: An instance of SingleWindowReducer configured with the specified wavelength window.
    """
    if (
        not isinstance(window_in_nm, tuple) and not isinstance(window_in_nm, list)
    ) or len(window_in_nm) != 2:
        raise ValueError("Window must be a tuple defining (min_nm, max_nm).")

    # Retrieve the wavelength array
    wavelength_array = (
        get_wavelengths_from_metadata()
    )  # Should return array-like with length=C

    if len(wavelength_array) != 826:
        raise ValueError(
            "Wavelength array must have exactly 826 elements corresponding to each channel."
        )

    # Find channel indices where wavelength is within the window [min_nm, max_nm]
    indices = np.where(
        (wavelength_array >= window_in_nm[0]) & (wavelength_array <= window_in_nm[1])
    )[0]

    if len(indices) == 0:
        raise ValueError(
            f"No channels found in the wavelength window {window_in_nm[0]}-{window_in_nm[1]} nm."
        )

    start_channel, end_channel = indices[0], indices[-1]
    # Instantiate the SingleWindowReducer with the identified window
    single_window_reducer = SingleWindowReducer(window=[start_channel, end_channel]).to(
        device
    )

    return single_window_reducer
