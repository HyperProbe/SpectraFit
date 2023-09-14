import torch.nn as nn


class SpectraCNN2D(nn.Module):
    def __init__(self, n_wavelengths, n_params, layer_width=1024, act_fc=nn.ELU):
        super(SpectraCNN2D, self).__init__()

        self.layers = nn.Sequential(nn.Conv2d(n_wavelengths, layer_width, 1), act_fc(),
                                    nn.Conv2d(layer_width, layer_width, 1), act_fc(),
                                    nn.Conv2d(layer_width, layer_width, 1), act_fc(),
                                    nn.Conv2d(layer_width, layer_width, 1), act_fc(),
                                    nn.Conv2d(layer_width, n_params, 1))

    def forward(self, x):
        x = self.layers(x)
        return x
