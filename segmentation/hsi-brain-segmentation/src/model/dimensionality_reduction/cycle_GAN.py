import torch
import torch.nn as nn

class GeneratorF(nn.Module):
    def __init__(self, in_channels, out_channels, num_filters=32):
        """
        Args:
            in_channels (int): Number of channels in the input FIVES-like images (e.g., 1 or 3).
            out_channels (int): Number of channels in the output hyperspectral images (e.g., 826).
            num_filters (int): Number of filters used in the intermediate layers.
        """
        super(GeneratorF, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        
        # Final layer maps the features to the hyperspectral domain.
        self.layer3 = nn.Conv2d(num_filters, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
