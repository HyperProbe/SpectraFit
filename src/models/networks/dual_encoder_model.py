import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from .autoencoder import UnetAutoencoder


class DualEncoder(nn.Module):
    """
    UNet-based dual encoder that processes both molecule concentration maps and HSI cubes.

    This encoder creates a shared latent representation from two different input modalities
    using UNet architecture. It combines features from both modalities at each encoder level
    to create shared skip connections for the decoder.

    Args:
        molecule_channels (int): Number of channels in molecule concentration maps (default: 10)
        hsi_channels (int): Number of channels in HSI cubes (default: 127)
        base_channels (int): Base number of channels for the encoder (default: 64)
        depth (int): Number of encoding layers (default: 4)
        latent_dim (int): Dimension of the shared latent space (default: 512)
        activation (callable): Activation function (default: nn.LeakyReLU)
        use_batchnorm (bool): Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        molecule_channels: int = 10,
        hsi_channels: int = 127,
        base_channels: int = 64,
        depth: int = 4,
        activation: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = True,
    ):
        super(DualEncoder, self).__init__()

        self.molecule_channels = molecule_channels
        self.hsi_channels = hsi_channels
        self.depth = depth

        # UNet-based molecule concentration encoder
        self.molecule_encoder = UNetEncoder(
            molecule_channels, base_channels, depth, activation, use_batchnorm
        )

        # UNet-based HSI cube encoder
        self.hsi_encoder = UNetEncoder(
            hsi_channels, base_channels, depth, activation, use_batchnorm
        )

        # Feature fusion layers for combining skip connections at each level
        self.skip_fusion_layers = nn.ModuleList()
        for i in range(depth):
            channels = base_channels * (2**i)
            # Simple fusion: concatenate + 1x1 conv to reduce back to original channels
            fusion_layer = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 1),  # 2x because we concatenate
                nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity(),
                activation(inplace=True),
            )
            self.skip_fusion_layers.append(fusion_layer)

        # Bottleneck fusion
        bottleneck_channels = base_channels * (2 ** (depth - 1))
        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(bottleneck_channels * 2, bottleneck_channels, 1),
            nn.BatchNorm2d(bottleneck_channels) if use_batchnorm else nn.Identity(),
            activation(inplace=True),
        )

    def forward(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Forward pass combining both inputs into shared features with skip connections.

        Args:
            molecule_input: Molecule concentration maps (B, C, H, W)
            hsi_input: HSI cubes (B, C, H, W)

        Returns:
            Tuple of (shared_bottleneck, shared_skip_features)
            - shared_bottleneck: Combined bottleneck features (B, C, H, W)
            - shared_skip_features: List of combined skip features for decoder
        """
        # Get features from both encoders
        molecule_bottleneck, molecule_skips = self.molecule_encoder(molecule_input)
        hsi_bottleneck, hsi_skips = self.hsi_encoder(hsi_input)

        # Combine skip connections at each level
        shared_skip_features = []
        for i, (mol_skip, hsi_skip, fusion_layer) in enumerate(
            zip(molecule_skips, hsi_skips, self.skip_fusion_layers)
        ):
            # Ensure spatial dimensions match
            if mol_skip.shape[-2:] != hsi_skip.shape[-2:]:
                hsi_skip = F.interpolate(
                    hsi_skip, size=mol_skip.shape[-2:], mode="bilinear", align_corners=False
                )
            
            # Concatenate and fuse
            combined_skip = torch.cat([mol_skip, hsi_skip], dim=1)
            fused_skip = fusion_layer(combined_skip)
            shared_skip_features.append(fused_skip)

        # Combine bottleneck features
        if molecule_bottleneck.shape[-2:] != hsi_bottleneck.shape[-2:]:
            hsi_bottleneck = F.interpolate(
                hsi_bottleneck, size=molecule_bottleneck.shape[-2:], 
                mode="bilinear", align_corners=False
            )
        
        combined_bottleneck = torch.cat([molecule_bottleneck, hsi_bottleneck], dim=1)
        shared_bottleneck = self.bottleneck_fusion(combined_bottleneck)

        return shared_bottleneck, shared_skip_features


class DualDecoder(nn.Module):
    """
    UNet-based dual decoder that uses shared skip connections from the DualEncoder.

    This decoder takes the shared features (bottleneck + skip connections) from the
    DualEncoder and reconstructs both molecule maps and HSI cubes using proper
    UNet architecture with skip connections.

    Args:
        molecule_channels (int): Number of channels in molecule concentration maps (default: 10)
        hsi_channels (int): Number of channels in HSI cubes (default: 127)
        base_channels (int): Base number of channels for the decoder (default: 64)
        depth (int): Number of decoding layers (default: 4)
        target_size (Tuple[int, int]): Target spatial size for outputs (default: (224, 224))
        activation (callable): Activation function (default: nn.LeakyReLU)
        use_batchnorm (bool): Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        molecule_channels: int = 10,
        hsi_channels: int = 127,
        base_channels: int = 64,
        depth: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        activation: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = True,
    ):
        super(DualDecoder, self).__init__()

        self.molecule_channels = molecule_channels
        self.hsi_channels = hsi_channels
        self.target_size = target_size

        # Use UNet decoder components
        self.shared_decoder = UNetDecoder(
            out_channels=base_channels,  # Output base_channels, then use heads for final channels
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        # Molecule reconstruction head
        self.molecule_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2) if use_batchnorm else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(base_channels // 2, molecule_channels, 1),
        )

        # HSI reconstruction head
        self.hsi_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2) if use_batchnorm else nn.Identity(),
            activation(inplace=True),
            nn.Conv2d(base_channels // 2, hsi_channels, 1),
        )

    def forward(self, shared_features: Tuple[torch.Tensor, list]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through UNet-based decoder using shared skip connections.

        Args:
            shared_features: Tuple of (shared_bottleneck, shared_skip_features) from DualEncoder

        Returns:
            Tuple of (molecule_reconstruction, hsi_reconstruction)
            - molecule_reconstruction: (B, C, H, W)
            - hsi_reconstruction: (B, C, H, W)
        """
        shared_bottleneck, shared_skip_features = shared_features

        # Use UNet decoder with shared skip connections
        x = self.shared_decoder(shared_bottleneck, shared_skip_features)

        # Ensure correct spatial size
        if x.shape[-2:] != self.target_size:
            x = F.interpolate(
                x, size=self.target_size, mode="bilinear", align_corners=False
            )

        # Dual heads for different outputs
        molecule_output = self.molecule_head(x)
        hsi_output = self.hsi_head(x)

        return molecule_output, hsi_output


class DualEncoderModel(nn.Module):
    """
    Complete dual encoder-decoder model for joint processing of molecule concentration
    maps and HSI cubes.

    This model takes both reduced 10-channel molecule concentration maps and reduced
    HSI cubes as input, encodes them into a shared latent space, and then reconstructs
    both the original 10-channel molecule maps and full HSI cubes.

    For inference, the model can be used to predict molecule concentration maps
    from both reduced concentration maps and reduced HSI cubes.

    Args:
        molecule_channels (int): Number of channels in molecule concentration maps (default: 10)
        hsi_in_channels (int): Number of channels in input HSI cubes (default: 127)
        hsi_out_channels (int): Number of channels in output HSI cubes (default: 127)
        base_channels (int): Base number of channels for encoder/decoder (default: 64)
        depth (int): Number of encoder/decoder layers (default: 4)
        target_size (Tuple[int, int]): Target spatial size for outputs (default: (64, 64))
        activation (callable): Activation function (default: nn.LeakyReLU)
        use_batchnorm (bool): Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        molecule_channels: int = 10,
        hsi_in_channels: int = 15,
        hsi_out_channels: int = 302,  # That's the number of channels of a helicoid HSI Cube with 530 to 750 nm
        base_channels: int = 64,
        depth: int = 4,
        target_size: Tuple[int, int] = (224, 224),
        activation: nn.Module = nn.ELU,
        use_batchnorm: bool = True,
    ):
        super(DualEncoderModel, self).__init__()

        self.molecule_channels = molecule_channels
        self.hsi_in_channels = hsi_in_channels
        self.hsi_out_channels = hsi_out_channels

        # Dual encoder
        self.encoder = DualEncoder(
            molecule_channels=molecule_channels,
            hsi_channels=hsi_in_channels,
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        # Dual decoder
        self.decoder = DualDecoder(
            molecule_channels=molecule_channels,
            hsi_channels=hsi_out_channels,
            base_channels=base_channels,
            depth=depth,
            target_size=target_size,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the complete UNet-based model with skip connections.

        Args:
            molecule_input: Reduced molecule concentration maps (B, C, H, W)
            hsi_input: Reduced HSI cubes (B, C, H, W)

        Returns:
            - In training mode: Tuple of (molecule_reconstruction, hsi_reconstruction)
                - molecule_reconstruction: Reconstructed molecule maps (B, C, H, W)
                - hsi_reconstruction: Reconstructed HSI cubes (B, C, H, W)
            - In eval mode: molecule_reconstruction only (B, C, H, W)
        """
        # Encode both inputs into shared features with skip connections
        shared_features = self.encoder(molecule_input, hsi_input)

        # Decode using shared features and skip connections
        molecule_output, hsi_output = self.decoder(shared_features)

        # Return only molecule output during inference (eval mode)
        if not self.training:
            return molecule_output

        # Return both outputs during training
        return molecule_output, hsi_output

    def encode(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> Tuple[torch.Tensor, list]:
        """
        Encode inputs to shared features with skip connections.

        Args:
            molecule_input: Reduced molecule concentration maps (B, C, H, W)
            hsi_input: Reduced HSI cubes (B, C, H, W)

        Returns:
            Tuple of (shared_bottleneck, shared_skip_features)
        """
        return self.encoder(molecule_input, hsi_input)

    def decode(self, shared_features: Tuple[torch.Tensor, list]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode from shared features with skip connections to outputs.

        Args:
            shared_features: Tuple of (shared_bottleneck, shared_skip_features)

        Returns:
            Tuple of (molecule_reconstruction, hsi_reconstruction)
            - molecule_reconstruction: (B, C, H, W)
            - hsi_reconstruction: (B, C, H, W)
        """
        return self.decoder(shared_features)


class UNetEncoder(nn.Module):
    """
    UNet-style encoder that reuses components from UnetAutoencoder.
    Returns both the latent representation and skip connection features.
    """

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        activation: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = True,
    ):
        super(UNetEncoder, self).__init__()

        # Create a UnetAutoencoder and reuse its encoder components
        # We set out_channels to match in_channels since we only need the encoder
        temp_unet = UnetAutoencoder(
            in_channels=in_channels,
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        # Extract and reuse the encoder components
        self.enc_convs = temp_unet.enc_convs
        self.enc_pools = temp_unet.enc_pools
        self.bottleneck = temp_unet.bottleneck
        self.depth = depth

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, list]:
        """
        Forward pass through encoder, handling channels-first input.

        Args:
            x: Input tensor (B, C, H, W)

        Returns:
            bottleneck: Deepest feature representation
            skip_features: List of features for skip connections
        """
        skip_features = []

        # Encoder path - reuse UnetAutoencoder's encoder logic
        for conv, pool in zip(self.enc_convs, self.enc_pools):
            x = conv(x)
            skip_features.append(x)
            x = pool(x)

        # Bottleneck
        bottleneck = self.bottleneck(x)

        return bottleneck, skip_features


class UNetDecoder(nn.Module):
    """
    UNet-style decoder that reuses components from UnetAutoencoder.
    Uses skip connections to reconstruct spatial information.
    """

    def __init__(
        self,
        out_channels: int,
        base_channels: int = 64,
        depth: int = 4,
        activation: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = True,
        final_activation=None,
    ):
        super(UNetDecoder, self).__init__()

        # Create a UnetAutoencoder and reuse its decoder components
        temp_unet = UnetAutoencoder(
            in_channels=out_channels,  # Use out_channels as in_channels for the temp UNet
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            final_activation=final_activation,
            use_batchnorm=use_batchnorm,
        )

        # Extract and reuse the decoder components
        self.dec_ups = temp_unet.dec_ups
        self.dec_convs = temp_unet.dec_convs
        self.final_conv = temp_unet.final_conv
        self.final_activation = temp_unet.final_activation

    def forward(self, bottleneck: torch.Tensor, skip_features: list) -> torch.Tensor:
        """
        Forward pass through decoder using skip connections, handling channels-first.

        Args:
            bottleneck: Deepest feature representation from encoder (B, C, H, W)
            skip_features: List of features from encoder for skip connections

        Returns:
            Reconstructed output (B, C, H, W)
        """
        x = bottleneck

        # Decoder path - reuse UnetAutoencoder's decoder logic
        for up, conv, skip in zip(self.dec_ups, self.dec_convs, reversed(skip_features)):
            x = up(x)
            # if needed, pad x to match skip's spatial dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # final projection
        x = self.final_conv(x)
        x = self.final_activation(x)
        return x


class UNetDualEncoder(nn.Module):
    """
    UNet-based dual encoder that processes both molecule concentration maps and HSI cubes
    into separate feature spaces while preserving spatial information through skip connections.
    """

    def __init__(
        self,
        molecule_channels: int = 10,
        hsi_channels: int = 15,
        base_channels: int = 64,
        depth: int = 4,
        activation: nn.Module = nn.LeakyReLU,
        use_batchnorm: bool = True,
    ):
        super(UNetDualEncoder, self).__init__()

        self.molecule_channels = molecule_channels
        self.hsi_channels = hsi_channels
        self.depth = depth

        # Molecule UNet encoder
        self.molecule_encoder = UNetEncoder(
            molecule_channels, base_channels, depth, activation, use_batchnorm
        )

        # HSI UNet encoder
        self.hsi_encoder = UNetEncoder(
            hsi_channels, base_channels, depth, activation, use_batchnorm
        )

    def forward(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, list], Tuple[torch.Tensor, list]]:
        """
        Forward pass producing separate UNet features with skip connections.

        Args:
            molecule_input: Molecule concentration maps (B, C, H, W)
            hsi_input: HSI cubes (B, C, H, W)

        Returns:
            Tuple of ((molecule_bottleneck, molecule_skips), (hsi_bottleneck, hsi_skips))
        """
        molecule_features = self.molecule_encoder(molecule_input)
        hsi_features = self.hsi_encoder(hsi_input)

        return molecule_features, hsi_features


class CrossModalAttention(nn.Module):
    """
    Memory-efficient cross-modal attention mechanism that allows molecule features
    to attend to HSI features. Uses spatial downsampling and chunked processing
    to reduce memory usage.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 8,
        attention_downsample: int = 4,
        chunk_size: int = 2048,
    ):
        super(CrossModalAttention, self).__init__()

        self.channels = channels
        self.attention_downsample = attention_downsample
        self.chunk_size = chunk_size
        reduced_channels = max(channels // reduction, 1)

        # Query projection for molecule features
        self.query_conv = nn.Conv2d(channels, reduced_channels, 1)

        # Key and Value projections for HSI features
        self.key_conv = nn.Conv2d(channels, reduced_channels, 1)
        self.value_conv = nn.Conv2d(channels, channels, 1)

        # Output projection
        self.output_conv = nn.Conv2d(channels, channels, 1)

        # Scaling factor for attention scores
        self.scale = reduced_channels**-0.5

        # Gate to control how much HSI information to use
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def _chunked_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute attention in chunks to reduce memory usage.

        Args:
            q: (B, HW, C') - queries
            k: (B, C', HW_down) - keys
            v: (B, HW_down, C) - values

        Returns:
            attended_features: (B, HW, C)
        """
        B, HW, C_q = q.shape
        _, C_k, HW_down = k.shape
        _, _, C_v = v.shape

        # Process queries in chunks
        attended_chunks = []
        for i in range(0, HW, self.chunk_size):
            end_idx = min(i + self.chunk_size, HW)
            q_chunk = q[:, i:end_idx, :]  # (B, chunk_size, C')

            # Compute attention scores for this chunk
            attention_scores = (
                torch.bmm(q_chunk, k) * self.scale
            )  # (B, chunk_size, HW_down)
            attention_weights = F.softmax(attention_scores, dim=-1)

            # Apply attention to values
            attended_chunk = torch.bmm(attention_weights, v)  # (B, chunk_size, C)
            attended_chunks.append(attended_chunk)

        # Concatenate chunks
        attended_features = torch.cat(attended_chunks, dim=1)  # (B, HW, C)
        return attended_features

    def forward(
        self, molecule_features: torch.Tensor, hsi_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            molecule_features: (B, C, H, W) - features from molecule decoder
            hsi_features: (B, C, H, W) - features from HSI decoder at same resolution

        Returns:
            Enhanced molecule features with HSI information
        """
        B, C, H, W = molecule_features.shape

        # Generate query from molecule features (full resolution)
        q = (
            self.query_conv(molecule_features).view(B, -1, H * W).permute(0, 2, 1)
        )  # (B, HW, C')

        # Downsample HSI features for attention computation to reduce memory
        if self.attention_downsample > 1:
            hsi_downsampled = F.avg_pool2d(
                hsi_features,
                kernel_size=self.attention_downsample,
                stride=self.attention_downsample,
            )
            H_down, W_down = hsi_downsampled.shape[-2:]
            HW_down = H_down * W_down
        else:
            hsi_downsampled = hsi_features
            H_down, W_down = H, W
            HW_down = H * W

        # Generate key and value from downsampled HSI features
        k = self.key_conv(hsi_downsampled).view(B, -1, HW_down)  # (B, C', HW_down)
        v = (
            self.value_conv(hsi_downsampled).view(B, -1, HW_down).permute(0, 2, 1)
        )  # (B, HW_down, C)

        # Use chunked attention to reduce memory usage
        attended_features = self._chunked_attention(q, k, v)  # (B, HW, C)
        attended_features = attended_features.permute(0, 2, 1).view(B, C, H, W)

        # Apply output projection
        attended_features = self.output_conv(attended_features)

        # Compute gating weight based on both feature types
        combined_features = torch.cat([molecule_features, hsi_features], dim=1)
        gate_weight = self.gate(combined_features)

        # Combine original molecule features with attended HSI features
        output = molecule_features + gate_weight * attended_features

        return output


class CrossModalDualEncoderModel(nn.Module):
    """
    UNet-based dual encoder-decoder model with cross-modal attention for better molecule reconstruction.

    This model uses UNet architecture with separate feature spaces for molecules and HSI, 
    but allows HSI information to help with molecule reconstruction through controlled 
    cross-modal attention while preserving spatial information through skip connections.

    Args:
        molecule_channels (int): Number of channels in molecule concentration maps (default: 10)
        hsi_in_channels (int): Number of channels in input HSI cubes (default: 15)
        hsi_out_channels (int): Number of channels in output HSI cubes (default: 302)
        base_channels (int): Base number of channels for encoder/decoder (default: 64)
        depth (int): Number of encoder/decoder layers (default: 4)
        activation (callable): Activation function (default: nn.ELU)
        use_batchnorm (bool): Whether to use batch normalization (default: True)
    """

    def __init__(
        self,
        molecule_channels: int = 10,
        hsi_in_channels: int = 15,
        hsi_out_channels: int = 302,
        base_channels: int = 64,
        depth: int = 4,
        activation: nn.Module = nn.ELU,
        use_batchnorm: bool = True,
    ):
        super(CrossModalDualEncoderModel, self).__init__()

        self.molecule_channels = molecule_channels
        self.hsi_in_channels = hsi_in_channels
        self.hsi_out_channels = hsi_out_channels

        # UNet-based dual encoder for separate feature spaces with skip connections
        self.encoder = UNetDualEncoder(
            molecule_channels=molecule_channels,
            hsi_channels=hsi_in_channels,
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        # Cross-modal attention for molecule decoder
        bottleneck_channels = base_channels * (2 ** (depth - 1))
        self.cross_modal_attention = CrossModalAttention(
            channels=bottleneck_channels,
            reduction=8,
            attention_downsample=4,
            chunk_size=2048,
        )

        # UNet decoders for both modalities
        self.molecule_decoder = UNetDecoder(
            out_channels=molecule_channels,
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

        self.hsi_decoder = UNetDecoder(
            out_channels=hsi_out_channels,
            base_channels=base_channels,
            depth=depth,
            activation=activation,
            use_batchnorm=use_batchnorm,
        )

    def forward(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with UNet architecture and HSI information helping molecule reconstruction.

        Args:
            molecule_input: Molecule concentration maps (B, C, H, W)
            hsi_input: HSI cubes (B, C, H, W)

        Returns:
            - In training mode: Tuple of (molecule_reconstruction, hsi_reconstruction)
            - In eval mode: molecule_reconstruction only (B, C, H, W)
        """
        # Encode both inputs with UNet encoders (get bottleneck + skip connections)
        molecule_features, hsi_features = self.encoder(molecule_input, hsi_input)
        
        molecule_bottleneck, molecule_skips = molecule_features
        hsi_bottleneck, hsi_skips = hsi_features

        # Apply cross-modal attention to enhance molecule features with HSI information
        enhanced_molecule_bottleneck = self.cross_modal_attention(
            molecule_bottleneck, hsi_bottleneck
        )

        # Decode using UNet decoders with skip connections
        molecule_output = self.molecule_decoder(enhanced_molecule_bottleneck, molecule_skips)
        hsi_output = self.hsi_decoder(hsi_bottleneck, hsi_skips)

        # Return only molecule output during inference (eval mode)
        if not self.training:
            return molecule_output

        # Return both outputs during training
        return molecule_output, hsi_output

    def encode(
        self, molecule_input: torch.Tensor, hsi_input: torch.Tensor
    ) -> Tuple[Tuple[torch.Tensor, list], Tuple[torch.Tensor, list]]:
        """
        Encode inputs using UNet encoders with skip connections.

        Args:
            molecule_input: Molecule concentration maps (B, C, H, W)
            hsi_input: HSI cubes (B, C, H, W)

        Returns:
            Tuple of ((molecule_bottleneck, molecule_skips), (hsi_bottleneck, hsi_skips))
        """
        return self.encoder(molecule_input, hsi_input)

    def decode(
        self,
        molecule_features: Tuple[torch.Tensor, list],
        hsi_features: Tuple[torch.Tensor, list],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode from UNet features with optional cross-modal enhancement.

        Args:
            molecule_features: (bottleneck, skip_features) from molecule encoder
            hsi_features: (bottleneck, skip_features) from HSI encoder

        Returns:
            Tuple of (molecule_reconstruction, hsi_reconstruction)
        """
        molecule_bottleneck, molecule_skips = molecule_features
        hsi_bottleneck, hsi_skips = hsi_features

        enhanced_molecule_bottleneck = self.cross_modal_attention(
            molecule_bottleneck, hsi_bottleneck
        )


        # Decode using UNet decoders
        molecule_output = self.molecule_decoder(enhanced_molecule_bottleneck, molecule_skips)
        hsi_output = self.hsi_decoder(hsi_bottleneck, hsi_skips)

        return molecule_output, hsi_output
