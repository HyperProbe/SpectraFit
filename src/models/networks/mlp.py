import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchMLP(nn.Module):
    def __init__(
        self,
        in_channels: int = 10,
        patch_size: int = 8,
        stride: int = None,
        mlp_hidden_dims=(1024, 512),
        mlp_dropout=0.1,
    ):
        """
        in_channels:        number of channels C (e.g. 10)
        patch_size:         size of each square patch (8)
        stride:             how far to slide (default = patch_size for non-overlap)
        mlp_hidden_dims:    tuple of hidden-layer widths for the patch-MLP
        mlp_dropout:        dropout probability in the MLP
        """
        super().__init__()
        self.C = in_channels
        self.ps = patch_size
        self.stride = stride or patch_size

        # build the MLP: in_dim = C * ps * ps, out_dim = same
        in_dim = in_channels * patch_size * patch_size
        out_dim = in_dim
        layers = []
        prev = in_dim
        for h in mlp_hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.ReLU(inplace=True),
                nn.Dropout(mlp_dropout),
            ]
            prev = h
        # final layer back to patch-sized vector
        layers += [nn.Linear(prev, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C]
        returns: [B, H, W, C] (same shape)
        """
        B, H, W, C = x.shape
        assert C == self.C, f"Expected {self.C} channels, got {C}"

        # 1) move to [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # 2) extract patches: → [B, C*ps*ps, L]
        patches = F.unfold(
            x, kernel_size=self.ps, stride=self.stride
        )  # shape [B, D_in, L]
        B, D_in, L = patches.shape

        # 3) reshape to [B*L, D_in]
        patches = patches.permute(0, 2, 1).reshape(-1, D_in)

        # 4) run through MLP → [B*L, D_in]
        out_p = self.mlp(patches)

        # 5) reshape back to [B, D_in, L]
        out_p = out_p.reshape(B, L, D_in).permute(0, 2, 1)

        # 6) fold to full image → [B, C, H, W]
        recon = F.fold(
            out_p, output_size=(H, W), kernel_size=self.ps, stride=self.stride
        )

        # 7) back to [B, H, W, C]
        recon = recon.permute(0, 2, 3, 1)
        return recon

class SmoothPatchMLP(nn.Module):
    def __init__(self,
                 in_channels: int = 10,
                 patch_size: int = 8,
                 stride: int = None,
                 mlp_hidden_dims=(1024, 512),
                 mlp_dropout=0.1):
        super().__init__()
        self.C   = in_channels
        self.ps  = patch_size
        # default to 50% overlap
        self.stride = stride or patch_size // 2

        # build a 2D Hann window of size ps×ps
        win1d = torch.hann_window(self.ps, periodic=False)  # shape [ps]
        win2d = win1d[:, None] * win1d[None, :]             # shape [ps,ps]
        self.register_buffer('win2d', win2d.flatten())      # shape [ps*ps]

        # build the MLP: in_dim = C * ps * ps, out_dim = same
        in_dim  = in_channels * patch_size * patch_size
        out_dim = in_dim
        layers = []
        prev   = in_dim
        for h in mlp_hidden_dims:
            layers += [nn.Linear(prev, h),
                       nn.ReLU(inplace=True),
                       nn.Dropout(mlp_dropout)]
            prev = h
        layers += [nn.Linear(prev, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, H, W, C]
        returns: [B, H, W, C] with smoothed patch seams
        """
        B, H, W, C = x.shape
        assert C == self.C

        # [B, C, H, W]
        x = x.permute(0, 3, 1, 2)

        # extract patches: [B, D, L]
        patches = F.unfold(x,
                           kernel_size=self.ps,
                           stride=self.stride)  # D = C*ps*ps
        B, D, L = patches.shape

        # apply window: multiply each patch-vector channel-wise
        # win2d is [ps*ps], so repeat per channel
        win = self.win2d.unsqueeze(0).repeat(self.C, 1).reshape(-1)  # [C*ps*ps]
        patches = patches * win.view(1, D, 1)

        # flatten to [B*L, D]
        patches = patches.permute(0, 2, 1).reshape(-1, D)

        # MLP → [B*L, D]
        out = self.mlp(patches)

        # reapply window for smooth blending
        out = out * win.unsqueeze(0)

        # back to [B, D, L]
        out = out.reshape(B, L, D).permute(0, 2, 1)

        # fold into [B, C, H, W]
        recon_num = F.fold(out,
                           output_size=(H, W),
                           kernel_size=self.ps,
                           stride=self.stride)

        # build a “weight” map by folding ones*window
        ones = torch.ones_like(x[:, :1, :, :])  # [B,1,H,W]
        weight_patches = F.unfold(ones,
                                  kernel_size=self.ps,
                                  stride=self.stride)  # [B,1*ps*ps,L]
        weight_patches = weight_patches * self.win2d.view(1, -1, 1)  # apply same window
        recon_denom = F.fold(weight_patches,
                             output_size=(H, W),
                             kernel_size=self.ps,
                             stride=self.stride)
        # avoid division by zero
        recon = recon_num / (recon_denom + 1e-8)

        # back to [B, H, W, C]
        return recon.permute(0, 2, 3, 1)
