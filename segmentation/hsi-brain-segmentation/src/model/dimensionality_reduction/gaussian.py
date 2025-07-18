import torch
import torch.nn as nn

from src.util.segmentation_util import load_model

class GaussianChannelReduction(nn.Module):
    def __init__(self, num_input_channels=826, num_output_channels=3, mu=None, sigma=None):
        super(GaussianChannelReduction, self).__init__()
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        
        if mu is None:
            mu = torch.linspace(0, num_input_channels - 1, num_output_channels)
        if sigma is None:
            sigma = torch.ones(num_output_channels)
            
        self.mu = nn.Parameter(mu.clone())
        # Initialize mu and log_sigma as learnable parameters
        self.log_sigma = nn.Parameter(torch.log(sigma))  # Use log_sigma for stability

        # Register a buffer for channel indices
        self.register_buffer('channel_indices', torch.arange(num_input_channels).float())

    def forward(self, x):
        # x shape: (batch_size, num_input_channels, H, W)
        batch_size, _, H, W = x.size()

        # Reshape mu and sigma for broadcasting
        mu = self.mu.view(self.num_output_channels, 1)
        sigma = self.log_sigma.exp().view(self.num_output_channels, 1)  # Ensure sigma > 0

        # Compute Gaussian weights
        channel_indices = self.channel_indices.view(1, self.num_input_channels)
        weights = torch.exp(-0.5 * ((channel_indices - mu) / sigma) ** 2)
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize weights

        # Reshape weights for multiplication
        weights = weights.view(1, self.num_output_channels, self.num_input_channels, 1, 1)

        # Expand x for multiplication
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, num_input_channels, H, W)

        # Compute the weighted sum over input channels
        output = (x * weights).sum(dim=2)  # Shape: (batch_size, num_output_channels, H, W)

        return output


class GaussianAutoEncoder(nn.Module):
    def __init__(
        self, num_input_channels=826, num_reduced_channels=3, mu=None, sigma=None
    ):
        super(GaussianAutoEncoder, self).__init__()
        self.encoder = GaussianChannelReduction(
            num_input_channels, num_reduced_channels, mu, sigma
        )
        self.decoder = nn.Sequential(
            # Optional: Additional convolutional layers
            nn.Conv2d(num_reduced_channels, num_input_channels, kernel_size=1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed


def build_gaussian_channel_reducer(
    num_input_channels=826, num_reduced_channels=3, load_from_path=None, device="cuda"
):
    model = GaussianAutoEncoder(num_input_channels, num_reduced_channels).to(device)
    if load_from_path:
        model = load_model(model, load_from_path, device)
    return model.encoder
