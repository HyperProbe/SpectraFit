import torch
from torch import nn

class FeatureExtractor(nn.Module):
    def __init__(self, base_model):
        super(FeatureExtractor, self).__init__()
        self.encoder = base_model.encoder

    def forward(self, x):
        features = self.encoder(x)
        return features
    
class BaseFeatureExtractorWithDimReduction(nn.Module):
    """
    Base class for feature extraction that includes the 1x1 convolution (for dimensionality reduction)
    and the pretrained encoder. Subclasses can add additional processing (e.g. CNN transformation).
    """
    def __init__(self, base_model, hyperspectral_channels=826, freeze_encoder=False, encoder_in_channels=1):
        """
        Args:
            base_model (nn.Module): Pretrained segmentation model containing:
                                    - base_model.encoder: the encoder module.
            hyperspectral_channels (int): Number of channels in the hyperspectral input.
            freeze_encoder (bool): If True, freeze the encoder's parameters.
            encoder_in_channels (int): Number of channels expected by the encoder.
        """
        super(BaseFeatureExtractorWithDimReduction, self).__init__()
        
        self.expected_channels = encoder_in_channels
        self.hyperspectral_channels = hyperspectral_channels
        
        # 1x1 convolution for dimensionality reduction.
        self.dim_reduction = nn.Conv2d(
            in_channels=self.hyperspectral_channels,
            out_channels=self.expected_channels,
            kernel_size=1
        )
        
        # The pretrained encoder from the base model.
        self.encoder = base_model.encoder
        
        # Optionally freeze the encoder parameters.
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

class FeatureExtractorWithCNN(BaseFeatureExtractorWithDimReduction):
    """
    Feature extractor that applies the 1x1 conv reducer followed by a CNN transformation
    before passing the result into the encoder.
    """
    def __init__(self, base_model, hyperspectral_channels=826, freeze_encoder=False, encoder_in_channels=1, kernel_size=3):
        super(FeatureExtractorWithCNN, self).__init__(
            base_model, hyperspectral_channels, freeze_encoder, encoder_in_channels
        )
        
        padding = kernel_size // 2
        
        # CNN transformation block that preserves spatial dimensions.
        self.cnn_transform = nn.Sequential(
            nn.Conv2d(self.expected_channels, self.expected_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(self.expected_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.expected_channels, self.expected_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(self.expected_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # If the input channel count doesn't match the expected channels, reduce and transform.
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
            x = self.cnn_transform(x)
        # Extract features using the encoder.
        features = self.encoder(x)
        return features
    
    def forward_transform(self, x):
        """
        Applies the dimensionality reduction and CNN transformation without passing
        the data through the encoder. This is useful for visualization or computing cycle loss.
        """
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
            x = self.cnn_transform(x)
        return x

class FeatureExtractorWith1x1ConvReducer(BaseFeatureExtractorWithDimReduction):
    """
    Feature extractor that applies only the 1x1 convolution (dimensionality reduction)
    before passing the result into the encoder.
    """
    def __init__(self, base_model, hyperspectral_channels=826, freeze_encoder=False, encoder_in_channels=1):
        super(FeatureExtractorWith1x1ConvReducer, self).__init__(
            base_model, hyperspectral_channels, freeze_encoder, encoder_in_channels
        )
        # No additional CNN transformation block.
    
    def forward(self, x):
        # If the input channel count doesn't match the expected channels, reduce only.
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
        features = self.encoder(x)
        return features
    
    def forward_transform(self, x):
        """
        Applies only the dimensionality reduction (1x1 conv) without passing the result
        through the encoder. Useful for visualization or auxiliary losses.
        """
        if x.shape[1] != self.expected_channels:
            x = self.dim_reduction(x)
        return x