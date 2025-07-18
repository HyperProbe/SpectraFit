import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvReducer(nn.Module):
    def __init__(self, in_channels=826, out_channels=3, mean_list=None, log_sigma_list=None):
        """
        Initializes the ConvReducer module with per-output-channel Gaussian initialization.

        Args:
            in_channels (int): Number of input channels (default: 826).
            out_channels (int): Number of output channels (default: 3 for RGB).
            mean_list (list or tuple): List of means for each output channel. Length must equal out_channels.
            log_sigma_list (list or tuple): List of log standard deviations for each output channel. Length must equal out_channels.
        """
        super(ConvReducer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Define the 1x1 convolutional layer
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)

        # Initialize weights with custom Gaussian distributions if provided
        if mean_list is not None and log_sigma_list is not None:
            if len(mean_list) != out_channels or len(log_sigma_list) != out_channels:
                raise ValueError("Length of mean_list and log_sigma_list must match out_channels.")
            self.custom_gaussian_init(mean_list, log_sigma_list)
        else:
            # Default initialization (e.g., Xavier Uniform)
            nn.init.xavier_uniform_(self.conv1x1.weight)
            if self.conv1x1.bias is not None:
                nn.init.zeros_(self.conv1x1.bias)

    def custom_gaussian_init(self, mean_list, log_sigma_list):
        """
        Initializes each output channel's weights with a Gaussian distribution.

        Args:
            mean_list (list or tuple): Means for each output channel.
            log_sigma_list (list or tuple): Log standard deviations for each output channel.
        """
        with torch.no_grad():
            for out_ch in range(self.out_channels):
                mean = mean_list[out_ch]
                sigma = torch.exp(torch.tensor(log_sigma_list[out_ch])).item()  # Convert log_sigma to sigma
                # Initialize weights for this output channel
                self.conv1x1.weight[out_ch] = torch.normal(mean=mean, std=sigma, size=(self.in_channels, 1, 1))
            # Initialize biases to zero or another specified value
            if self.conv1x1.bias is not None:
                nn.init.constant_(self.conv1x1.bias, 0.0)

    def forward(self, x):
        """
        Forward pass to convert hyperspectral image to RGB.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        rgb = self.conv1x1(x)
        return rgb
