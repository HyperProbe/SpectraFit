import torch.nn as nn


# Create model
class SpectraMLP(nn.Module):
    '''
    generic Feed Forward NN
    '''

    def __init__(self, n_params, n_layers=4, layer_width=1024, act_fc=nn.ELU):
        super(SpectraMLP, self).__init__()

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            self.layers.append(nn.LazyLinear(layer_width))
            self.layers.append(act_fc())
        self.layers.append(nn.LazyLinear(n_params))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
