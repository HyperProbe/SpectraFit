from typing import Sequence
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.molecules import MoleculeIndex
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torchvision.models as models
from monai.losses import PerceptualLoss as PL


class WeightedReconstructionLoss(nn.Module):
    """
    Weighted sum of MSE and MAE:
      L = alpha * MSE(output, target) + (1 - alpha) * MAE(output, target)
    """

    def __init__(self, alpha: float = 0.5):
        """
        Args:
            alpha (float): weight on the MSE term.
                           If alpha=1.0 → pure MSE; alpha=0.0 → pure MAE.
        """
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.mae = nn.L1Loss()

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(output, target)
        mae_loss = self.mae(output, target)
        return self.alpha * mse_loss + (1.0 - self.alpha) * mae_loss


class WeightedMSELossWithSignalPenalty(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        lambda_weighted_mse: float = 1.0,
        lambda_hbt: float = 1.0,
        lambda_diffCCO: float = 1.0,
    ):
        """
        Args:
            alpha (float): weight on the MSE term.
                           If alpha=1.0 → pure MSE; alpha=0.0 → pure MAE.
            lambda_weighted_mse (float): weight for the weighted MSE term.
            lambda_hbt (float): weight for the HBT term.
            lambda_diffCCO (float): weight for the diffCCO term.
        """
        super().__init__()
        self.weighted_mse = WeightedReconstructionLoss(alpha)
        self.lambda_weighted_mse = lambda_weighted_mse
        self.lambda_hbt = lambda_hbt
        self.lambda_diffCCO = lambda_diffCCO

    def forward(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            output (torch.Tensor): model output.
            target (torch.Tensor): ground truth target.

        Returns:
            torch.Tensor: computed loss.
        """
        # Compute the weighted MSE loss
        weighted_mse_loss = self.weighted_mse(output, target)

        # Compute the HBT term
        hbt_pred = (
            output[:, :, :, MoleculeIndex.HB] + target[:, :, :, MoleculeIndex.HBO2]
        )
        hbt_gt = target[:, :, :, MoleculeIndex.HB] + target[:, :, :, MoleculeIndex.HBO2]
        hbt_loss = F.mse_loss(hbt_pred, hbt_gt)

        # Compute the diffCCO term
        diffcco_pred = (
            output[:, :, :, MoleculeIndex.COXA] - output[:, :, :, MoleculeIndex.CREDA]
        )
        diffcco_gt = (
            target[:, :, :, MoleculeIndex.COXA] - target[:, :, :, MoleculeIndex.CREDA]
        )
        diffcco_loss = F.mse_loss(diffcco_pred, diffcco_gt)

        return (
            self.lambda_weighted_mse * weighted_mse_loss
            + self.lambda_hbt * hbt_loss
            + self.lambda_diffCCO * diffcco_loss
        )


class SSIMLoss(nn.Module):
    def __init__(
        self,
        gaussian_kernel: bool = True,
        sigma: float | Sequence[float] = 1.5,
        kernel_size: int = 11,
        reduction: str = "elementwise_mean",
        device: torch.device | str = "cpu",
    ):
        """
        SSIM-based loss that takes inputs in BxHxWxC and targets in the same shape.
        Returns 1 - SSIM so that lower is better.
        """
        super().__init__()
        self.ssim = StructuralSimilarityIndexMeasure(
            data_range=None,
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
        ).to(device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred:  BxHxWxC
        target: BxHxWxC
        """
        # permute to B×C×H×W
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

        ssim_val = self.ssim(pred, target)

        # return as a loss
        return 1.0 - ssim_val


class WeightedSSIMWithMSELoss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gaussian_kernel: bool = True,
        sigma: float | Sequence[float] = 1.5,
        kernel_size: int = 11,
        reduction: str = "elementwise_mean",
        device: torch.device | str = "cpu",
    ):
        """
        Weighted SSIM loss that combines SSIM with a weighted MSE loss.
        """
        super().__init__()
        self.ssim_loss = SSIMLoss(
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
            device=device,
        )
        self.alpha = alpha
        self.mse_loss = nn.MSELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = self.ssim_loss(pred, target)
        mse_loss = self.mse_loss(pred, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * mse_loss

class WeightedSSIMWithL1Loss(nn.Module):
    def __init__(
        self,
        alpha: float = 0.5,
        gaussian_kernel: bool = True,
        sigma: float | Sequence[float] = 1.5,
        kernel_size: int = 11,
        reduction: str = "elementwise_mean",
        device: torch.device | str = "cpu",
    ):
        """
        Weighted SSIM loss that combines SSIM with a weighted MSE loss.
        """
        super().__init__()
        self.ssim_loss = SSIMLoss(
            gaussian_kernel=gaussian_kernel,
            sigma=sigma,
            kernel_size=kernel_size,
            reduction=reduction,
            device=device,
        )
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_loss = self.ssim_loss(pred, target)
        l1_loss = self.l1_loss(pred, target)
        return self.alpha * ssim_loss + (1 - self.alpha) * l1_loss


# Project 10→3 channels with a 1×1 conv
class ChannelProjector(nn.Module):
    def __init__(self, in_ch: int = 10, out_ch: int = 3):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,10,H,W) → (B,3,H,W)
        return self.proj(x)


# Wrap VGG-16’s conv‐layers as a fixed feature extractor
class VGG16Features(nn.Module):
    def __init__(self, layers: tuple[str, ...] = ("3", "8", "17")):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True).features
        for p in vgg16.parameters():
            p.requires_grad = False
        self.vgg_feats = vgg16
        self.layers = layers

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        feats = []
        for idx, layer in enumerate(self.vgg_feats):
            x = layer(x)
            if str(idx) in self.layers:
                feats.append(x)
        return feats


# 3) Combine them into a single PerceptualLoss module
class PerceptualLoss(nn.Module):
    def __init__(
        self,
        projector: ChannelProjector,
        feature_extractor: VGG16Features,
        layer_weights: list[float] | None = None,
        lambda_mse: float = 1.0,
        lambda_perceptual: float = 0.1,
    ):
        """
        - projector: maps (B,10,H,W) → (B,3,H,W)
        - feature_extractor: frozen VGG16Features
        - layer_weights: one weight per extracted VGG layer
        - lambda_mse / lambda_perceptual: λ_mse, λ_perc
        """
        super().__init__()
        self.proj = projector
        self.feature_extractor = feature_extractor
        nw = len(self.feature_extractor.layers)
        self.layer_weights = layer_weights if layer_weights is not None else [1.0] * nw
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, H, W, C)
        returns: weighted sum of MSE + perceptual feature loss
        """
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)
        # Pixel‐wise MSE
        loss_mse = F.mse_loss(pred, target)

        # Project to 3 channels
        p_pred = self.proj(pred)  # (B,3,H,W)
        p_target = self.proj(target)  # share same projector

        # Extract VGG features
        f_pred = self.feature_extractor(p_pred)
        f_target = self.feature_extractor(p_target)

        # Feature‐reconstruction loss
        loss_perc = 0.0
        for w, fp, ft in zip(self.layer_weights, f_pred, f_target):
            loss_perc += w * F.mse_loss(fp, ft)

        return self.lambda_mse * loss_mse + self.lambda_perceptual * loss_perc


class TVLoss(nn.Module):
    """
    Total Variation Loss for images in BHWC format, with weight built in.

    Computes the sum of absolute differences between neighboring pixels
    in the horizontal and vertical directions, multiplied by lambda_tv.
    """

    def __init__(self, lambda_tv: float = 0.001, reduction: str = "mean"):
        """
        Parameters:
        -----------
        lambda_tv: float
            Weight for the TV regularization term.
        reduction: str, either 'mean' or 'sum'
            - 'mean': average over all pixels and channels
            - 'sum' : sum over all pixels and channels
        """
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be 'mean' or 'sum'")
        self.lambda_tv = lambda_tv
        self.reduction = reduction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters:
        -----------
        x: torch.Tensor of shape (B, H, W, C)

        Returns:
        --------
        torch.Tensor: scalar weighted TV loss
        """
        # Permute to (B, C, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Vertical and horizontal differences
        dh = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        dw = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])

        if self.reduction == "mean":
            tv = dh.mean() + dw.mean()
        else:  # 'sum'
            tv = dh.sum() + dw.sum()

        return self.lambda_tv * tv


class CompositeLossWithTV(nn.Module):
    """
    Wraps a base reconstruction loss and adds TV regularization.
    """

    def __init__(
        self, base_loss: nn.Module, lambda_tv: float = 0.0, tv_reduction: str = "mean"
    ):
        """
        Parameters:
        -----------
        base_loss: nn.Module
            The primary loss (e.g., MSELoss, L1Loss, WeightedSSIMWithMSELoss, etc.).
        lambda_tv: float
            Weight for the TV regularization term.
        tv_reduction: str
            'mean' or 'sum' reduction for the TV loss.
        """
        super().__init__()
        self.base_loss = base_loss
        self.tv_loss = TVLoss(lambda_tv=lambda_tv, reduction=tv_reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: torch.Tensor of shape (B, H, W, C)
        """
        # Compute base reconstruction loss
        loss = self.base_loss(pred, target)
        # Add TV regularization
        loss = loss + self.tv_loss(pred)
        return loss


class PerceptualLossWrapper(nn.Module):
    """
    Wrapper for MONAI's PerceptualLoss that handles BHWC → BCHW conversion and channel projection.

    This wrapper converts input tensors from (B, H, W, C) format to (B, C, H, W) format
    as required by MONAI's PerceptualLoss, and includes a trainable ChannelProjector to
    project input channels down to 3 channels for compatibility with pretrained models.

    Example:
        >>> # Initialize with specific parameters for 10-channel input
        >>> loss_fn = PerceptualLossWrapper(
        ...     spatial_dims=2,
        ...     network_type="alex",
        ...     input_channels=10,  # Number of input channels
        ...     is_fake_3d=True,
        ...     fake_3d_ratio=0.5,
        ...     pretrained=True,
        ...     channel_wise=False
        ... )
        >>>
        >>> # Use with BHWC tensors
        >>> pred = torch.randn(4, 128, 128, 10)  # B=4, H=128, W=128, C=10
        >>> target = torch.randn(4, 128, 128, 10)
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        spatial_dims: int,
        channel_projector: ChannelProjector,
        network_type: str = "alex",
        is_fake_3d: bool = True,
        fake_3d_ratio: float = 0.5,
        cache_dir: str | None = None,
        pretrained: bool = True,
        pretrained_path: str | None = None,
        pretrained_state_dict_key: str | None = None,
        channel_wise: bool = False,
        base_loss: nn.Module = F.mse_loss,
        lambda_base: float = 1.0,
        lambda_perceptual: float = 0.1,
    ):
        """
        Args:
            spatial_dims (int): Number of spatial dimensions (2 or 3).
            input_channels (int): Number of input channels in your data (default: 3).
                If > 3, a trainable ChannelProjector will be used to project to 3 channels.
            network_type (str): Network architecture to use. Options:
                - "alex": AlexNet (default)
                - "vgg": VGG
                - "squeeze": SqueezeNet
                - "radimagenet_resnet50": RadImageNet ResNet50
                - "medicalnet_resnet10_23datasets": MedicalNet ResNet10
                - "medicalnet_resnet50_23datasets": MedicalNet ResNet50
                - "resnet50": Torchvision ResNet50
            is_fake_3d (bool): If True, use 2.5D approach for 3D perceptual loss.
            fake_3d_ratio (float): Ratio of slices per axis used in 2.5D approach.
            cache_dir (str | None): Path to cache directory for pretrained weights.
            pretrained (bool): Whether to load pretrained weights.
            pretrained_path (str | None): Path to custom pretrained weights.
            pretrained_state_dict_key (str | None): Key for extracting state dict.
            channel_wise (bool): If True, return loss per channel.
        """
        super().__init__()

        # Add trainable channel projector if input channels > 3
        self.network_type = network_type
        self.base_loss = base_loss
        self.lambda_base = lambda_base
        self.lambda_perceptual = lambda_perceptual
        self.channel_projector = channel_projector

        self.perceptual_loss = PL(
            spatial_dims=spatial_dims,
            network_type=network_type,
            is_fake_3d=is_fake_3d,
            fake_3d_ratio=fake_3d_ratio,
            cache_dir=cache_dir,
            pretrained=pretrained,
            pretrained_path=pretrained_path,
            pretrained_state_dict_key=pretrained_state_dict_key,
            channel_wise=channel_wise,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with automatic BHWC → BCHW conversion and optional channel projection.

        Args:
            pred (torch.Tensor): Predicted tensor of shape (B, H, W, C) or (B, H, W, D, C).
            target (torch.Tensor): Target tensor of shape (B, H, W, C) or (B, H, W, D, C).

        Returns:
            torch.Tensor: The computed perceptual loss.
        """
        # Convert from BHWC to BCHW format (or BHWDC to BCDHW for 3D)
        if pred.dim() == 4:  # 2D case: BHWC -> BCHW
            pred_bchw = pred.permute(0, 3, 1, 2).contiguous()
            target_bchw = target.permute(0, 3, 1, 2).contiguous()
        elif pred.dim() == 5:  # 3D case: BHWDC -> BCDHW
            pred_bchw = pred.permute(0, 4, 1, 2, 3).contiguous()
            target_bchw = target.permute(0, 4, 1, 2, 3).contiguous()
        else:
            raise ValueError(
                f"Unsupported tensor dimension: {pred.dim()}. Expected 4D or 5D tensors."
            )

        # Apply channel projection if needed (from input_channels to 3 channels)
        if self.channel_projector is not None:
            pred_bchw = self.channel_projector(pred_bchw)
            target_bchw = self.channel_projector(target_bchw)

        # Normalize to [0,1] range for RadImageNet models
        if self.network_type == "radimagenet_resnet50":
            # Clamp to ensure values are in valid range, then normalize
            pred_bchw = torch.clamp(pred_bchw, min=0.0)
            target_bchw = torch.clamp(target_bchw, min=0.0)

            # Normalize to [0,1] by dividing by the maximum value in each tensor
            pred_max = pred_bchw.view(pred_bchw.size(0), -1).max(dim=1, keepdim=True)[0]
            target_max = target_bchw.view(target_bchw.size(0), -1).max(
                dim=1, keepdim=True
            )[0]

            # Reshape max values for broadcasting
            pred_max = pred_max.view(-1, 1, 1, 1)
            target_max = target_max.view(-1, 1, 1, 1)

            # Avoid division by zero
            pred_max = torch.clamp(pred_max, min=1e-8)
            target_max = torch.clamp(target_max, min=1e-8)

            pred_bchw = pred_bchw / pred_max
            target_bchw = target_bchw / target_max

        # Apply the MONAI PerceptualLoss
        return self.base_loss(
            pred, target
        ) * self.lambda_base + self.lambda_perceptual * self.perceptual_loss(
            pred_bchw, target_bchw
        )
