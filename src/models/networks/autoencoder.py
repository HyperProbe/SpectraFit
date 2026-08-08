import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from monai.networks.nets.swin_unetr import SwinUNETR
from monai.networks.nets.autoencoder import AutoEncoder


class ConvAutoencoder(nn.Module):
    """
    A customizable convolutional autoencoder with channel-last input/outputs.

    Args:
        in_channels (int): Number of channels in the input images (last dimension).
        encoder_channels (list of int): Output channels for each encoder layer.
        decoder_channels (list of int, optional): Output channels for each decoder layer.
            If None, it will be set to reversed encoder_channels without the last element,
            ending with in_channels.
        kernel_size (int or tuple): Size of the convolutional kernels (default: 3).
        stride (int or tuple): Stride for convolutions (default: 2).
        padding (int or tuple): Padding for convolutions (default: 1).
        activation (callable): Activation function class (default: nn.ReLU).
        final_activation (callable, optional): Activation for final layer (default: None).
    """

    def __init__(
        self,
        in_channels=10,
        encoder_channels=[64, 128, 256],
        decoder_channels=None,
        kernel_size=3,
        stride=2,
        padding=1,
        activation=nn.ReLU,
        final_activation=None,
    ):
        super(ConvAutoencoder, self).__init__()

        # Build encoder (channels-first)
        enc_layers = []
        prev_channels = in_channels
        for ch in encoder_channels:
            enc_layers.append(
                nn.Conv2d(prev_channels, ch, kernel_size, stride, padding)
            )
            enc_layers.append(activation(inplace=True))
            prev_channels = ch
        self.encoder = nn.Sequential(*enc_layers)

        # Build decoder (channels-first)
        if decoder_channels is None:
            decoder_channels = encoder_channels[::-1][1:] + [in_channels]

        dec_layers = []
        prev_channels = encoder_channels[-1]
        for ch in decoder_channels:
            dec_layers.append(
                nn.ConvTranspose2d(
                    prev_channels, ch, kernel_size, stride, padding, output_padding=1
                )
            )
            if ch != in_channels:
                dec_layers.append(activation(inplace=True))
            prev_channels = ch

        if final_activation is not None:
            dec_layers.append(final_activation())

        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns reconstructed tensor of same shape.
        """
        # Move to channels-first for conv layers
        x_cf = x.permute(0, 3, 1, 2)
        # Encode
        z = self.encoder(x_cf)
        # Decode
        x_rec_cf = self.decoder(z)
        # Move back to channels-last
        x_rec = x_rec_cf.permute(0, 2, 3, 1)
        return x_rec


class UnetAutoencoder(nn.Module):
    def __init__(
        self,
        in_channels=10,
        base_channels=64,
        depth=4,
        kernel_size=3,
        activation=nn.LeakyReLU,
        final_activation=None,
        use_batchnorm=True,
    ):
        """
        U-Net style autoencoder.
        Args:
            in_channels (int): number of input channels (last dim)
            base_channels (int): channels in the first conv layer; doubles each level
            depth (int): number of down/up‐sampling levels
            kernel_size (int): conv kernel size
            activation (callable): nonlinearity to use after each conv
            final_activation (callable or None): e.g. nn.Identity to allow neg values
            use_batchnorm (bool): whether to add BatchNorm after each conv
        """
        super().__init__()
        Act = activation

        # build encoder
        self.enc_convs = nn.ModuleList()
        self.enc_pools = nn.ModuleList()
        chs = in_channels
        for d in range(depth):
            out_ch = base_channels * (2**d)
            block = [
                nn.Conv2d(chs, out_ch, kernel_size, stride=1, padding=kernel_size // 2),
                Act(inplace=True),
            ]
            if use_batchnorm:
                block.insert(1, nn.BatchNorm2d(out_ch))
            block += [
                nn.Conv2d(
                    out_ch, out_ch, kernel_size, stride=1, padding=kernel_size // 2
                ),
                Act(inplace=True),
            ]
            if use_batchnorm:
                block.insert(-1, nn.BatchNorm2d(out_ch))
            self.enc_convs.append(nn.Sequential(*block))
            self.enc_pools.append(nn.MaxPool2d(2))
            chs = out_ch

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(chs, chs * 2, kernel_size, padding=kernel_size // 2),
            Act(inplace=True),
            nn.Conv2d(chs * 2, chs, kernel_size, padding=kernel_size // 2),
            Act(inplace=True),
        )

        # build decoder
        self.dec_ups = nn.ModuleList()
        self.dec_convs = nn.ModuleList()
        for d in reversed(range(depth)):
            in_ch = chs
            out_ch = base_channels * (2**d)
            self.dec_ups.append(nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2))
            # two convs after concatenation
            block = [
                nn.Conv2d(out_ch * 2, out_ch, kernel_size, padding=kernel_size // 2),
                Act(inplace=True),
            ]
            if use_batchnorm:
                block.insert(1, nn.BatchNorm2d(out_ch))
            block += [
                nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2),
                Act(inplace=True),
            ]
            if use_batchnorm:
                block.insert(-1, nn.BatchNorm2d(out_ch))
            self.dec_convs.append(nn.Sequential(*block))
            chs = out_ch

        # final conv back to in_channels
        self.final_conv = nn.Conv2d(chs, in_channels, 1)
        self.final_activation = (
            final_activation() if final_activation else nn.Identity()
        )

    def forward(self, x):
        # x: (B, H, W, C) → (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        # encoder
        skips = []
        for conv, pool in zip(self.enc_convs, self.enc_pools):
            x = conv(x)
            skips.append(x)
            x = pool(x)

        # bottleneck
        x = self.bottleneck(x)

        # decoder
        for up, conv, skip in zip(self.dec_ups, self.dec_convs, reversed(skips)):
            x = up(x)
            # if needed, pad x to match skip’s spatial dims
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = conv(x)

        # final projection and back to (B, H, W, C)
        x = self.final_conv(x)
        x = self.final_activation(x)
        x = x.permute(0, 2, 3, 1)
        return x


class UnetWithEncoderWrapper(nn.Module):
    """
    Wrapper around smp.Unet to handle channel-last input/output.

    Expects inputs of shape (B, H, W, C) and outputs the same.
    """

    def __init__(
        self, encoder_name, encoder_depth, in_channels, final_activation_fn=None
    ):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=in_channels,
            activation=final_activation_fn,
        )

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns output of same shape.
        """
        # Permute to channels-first: (B, C, H, W)
        x_cf = x.permute(0, 3, 1, 2)
        # Forward through the Unet
        y_cf = self.model(x_cf)
        # Permute back to channels-last: (B, H, W, C)
        y = y_cf.permute(0, 2, 3, 1)
        return y


class LinknetWithEncoderWrapper(nn.Module):
    """
    Wrapper around smp.Linknet to handle channel-last input/output.

    Expects inputs of shape (B, H, W, C) and outputs the same.
    """

    def __init__(
        self, encoder_name, encoder_depth, in_channels, final_activation_fn=None
    ):
        super().__init__()
        self.model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=None,
            encoder_depth=encoder_depth,
            in_channels=in_channels,
            classes=in_channels,
            activation=final_activation_fn,
        )

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns output of same shape.
        """
        # Permute to channels-first: (B, C, H, W)
        x_cf = x.permute(0, 3, 1, 2)
        # Forward through the Unet
        y_cf = self.model(x_cf)
        # Permute back to channels-last: (B, H, W, C)
        y = y_cf.permute(0, 2, 3, 1)
        return y


class UnetEnsemble(nn.Module):
    """
    Ensemble of U-Net autoencoders, one for each molecule channel.

    This architecture trains a separate U-Net for each molecule, allowing each
    network to specialize in reconstructing a specific molecule's concentration map.
    The hypothesis is that different molecules have different semantic characteristics
    and spatial distributions that may confuse a single shared network.

    Args:
        num_molecules (int): Number of molecule channels (typically 10).
        input_channels (int): Number of input channels per molecule network.
                             Can be 1 (single channel) or num_molecules (all channels).
        base_channels (int): Base channels for each U-Net.
        depth (int): Depth of each U-Net.
        kernel_size (int): Kernel size for convolutions.
        activation (callable): Activation function.
        final_activation (callable, optional): Final activation.
        use_batchnorm (bool): Whether to use batch normalization.
    """

    def __init__(
        self,
        num_molecules=10,
        base_channels=32,  # Smaller since we have multiple networks
        depth=4,
        kernel_size=3,
        activation=nn.LeakyReLU,
        final_activation=None,
        use_batchnorm=True,
    ):
        super().__init__()

        self.num_molecules = num_molecules

        # Create one U-Net for each molecule
        self.molecule_nets = nn.ModuleList()

        for i in range(num_molecules):
            net = UnetAutoencoder(
                in_channels=1,
                base_channels=base_channels,
                depth=depth,
                kernel_size=kernel_size,
                activation=activation,
                final_activation=final_activation,
                use_batchnorm=use_batchnorm,
            )
            self.molecule_nets.append(net)

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns reconstructed tensor of same shape.

        Args:
            x (torch.Tensor): Input tensor of shape (B, H, W, num_molecules)

        Returns:
            torch.Tensor: Reconstructed tensor of shape (B, H, W, num_molecules)
        """
        outputs = []

        for i, net in enumerate(self.molecule_nets):
            # Pass only the corresponding input channel
            net_input = x[..., i : i + 1]  # Shape: (B, H, W, 1)

            # Each network outputs a single channel for its molecule
            net_output = net(
                net_input
            )  # Shape: (B, H, W, 1) or (B, H, W, num_molecules)

            outputs.append(net_output)

        # Concatenate all molecule outputs along channel dimension
        result = torch.cat(outputs, dim=-1)  # Shape: (B, H, W, num_molecules)
        return result


class ResidualAutoencoder(nn.Module):
    def __init__(self, base_ae: nn.Module):
        """
        base_ae: your “old” autoencoder that predicts the residual
                 delta = f(input), same shape as input
        """
        super().__init__()
        self.base = base_ae

    def forward(self, x):
        # x: (B, H, W, C) or (B, C, H, W) depending on your convention—
        # make sure it's whatever format your base expects.
        delta = self.base(x)
        return x + delta


class SmoothResidualAutoencoder(nn.Module):
    def __init__(self, base_ae: nn.Module, C: int):
        super().__init__()
        self.base = base_ae
        # depthwise 3×3 smoothing, initialized to identity
        self.smooth = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=3,
            padding=1,
            groups=C,
            bias=False,
        )
        nn.init.dirac_(self.smooth.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.base(x)  # (B,H,W,C)
        recon = x + delta  # (B,H,W,C)
        # permute to (B,C,H,W)
        r = recon.permute(0, 3, 1, 2).contiguous()
        r = self.smooth(r)  # smooths both input noise+delta
        # back to (B,H,W,C)
        return r.permute(0, 2, 3, 1)


class MonaiAutoencoderWrapper(nn.Module):
    """
    Wrapper around MONAI AutoEncoder to handle channel-last input/output.

    Expects inputs of shape (B, H, W, C) and outputs the same.
    """

    def __init__(
        self,
        spatial_dims=2,
        in_channels=10,
        out_channels=10,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        kernel_size=3,
        up_kernel_size=3,
        num_res_units=0,
        inter_channels=None,
        inter_dilations=None,
        num_inter_units=2,
        act="PRELU",
        norm="INSTANCE",
        dropout=0.0,
        bias=True,
    ):
        """
        Initialize MONAI AutoEncoder wrapper.

        Args:
            spatial_dims (int): Number of spatial dimensions (2 for 2D, 3 for 3D).
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            channels (Sequence[int]): Sequence of channels for each level.
            strides (Sequence[int]): Sequence of strides for each level.
            kernel_size (Union[Sequence[int], int]): Size of the convolving kernel.
            up_kernel_size (Union[Sequence[int], int]): Size of the upsampling kernel.
            num_res_units (int): Number of residual units in each level.
            inter_channels (Sequence[int], optional): Sequence of channels for intermediate layers.
            inter_dilations (Sequence[int], optional): Sequence of dilations for intermediate layers.
            num_inter_units (int): Number of intermediate units.
            act (Union[Tuple, str]): Activation type and arguments.
            norm (Union[Tuple, str]): Feature normalization type and arguments.
            dropout (float): Dropout ratio.
            bias (bool): Whether to have bias in convolution layers.
        """
        super().__init__()
        self.model = AutoEncoder(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            kernel_size=kernel_size,
            up_kernel_size=up_kernel_size,
            num_res_units=num_res_units,
            inter_channels=inter_channels,
            inter_dilations=inter_dilations,
            num_inter_units=num_inter_units,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns output of same shape.
        """
        # Permute to channels-first: (B, C, H, W)
        x_cf = x.permute(0, 3, 1, 2)
        # Forward through the MONAI AutoEncoder
        y_cf = self.model(x_cf)
        # Permute back to channels-last: (B, H, W, C)
        y = y_cf.permute(0, 2, 3, 1)
        return y


class SwinUNETRWrapper(nn.Module):
    """
    Wrapper around SwinUNETR to handle channel-last input/output.

    Expects inputs of shape (B, H, W, C) and outputs the same.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        patch_size=2,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        window_size=7,
        qkv_bias=True,
        mlp_ratio=4.0,
        feature_size=24,
        norm_name="instance",
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        norm_layer=nn.LayerNorm,
        patch_norm=False,
        use_checkpoint=False,
        spatial_dims=2,
        downsample="merging",
        use_v2=False,
    ):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            mlp_ratio=mlp_ratio,
            feature_size=feature_size,
            norm_name=norm_name,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            dropout_path_rate=dropout_path_rate,
            normalize=normalize,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=downsample,
            use_v2=use_v2,
        )

    def forward(self, x):
        """
        Forward pass expecting x of shape (B, H, W, C).
        Returns output of same shape.
        """
        # Permute to channels-first: (B, C, H, W)
        x_cf = x.permute(0, 3, 1, 2)
        # Forward through the SwinUnet
        y_cf = self.model(x_cf)
        # Permute back to channels-last: (B, H, W, C)
        y = y_cf.permute(0, 2, 3, 1)
        return y
