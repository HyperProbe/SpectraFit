import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch: int, ndf: int = 64, n_layers: int = 3):
        """
        in_ch: number of input channels (10 for hyperspectral, or 3 if projected)
        ndf: base number of filters
        n_layers: how many downsampling blocks before final conv
        """
        super().__init__()
        layers = []
        # initial conv, no norm
        layers += [
            nn.Conv2d(in_ch, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        # down-sampling blocks
        nf_mult = 1
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**i, 8)
            layers += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        # one more conv with stride=1
        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        layers += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        # final 1-channel “real/fake” map
        layers += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, H, W, C) tensor
        x = x.permute(0, 3, 1, 2)
        return self.model(x)  # → (B,1,H′,W′) patch scores