import torch
import torch.nn as nn
import numpy as np
from torch.nn import Module
from torch import Tensor
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Sigmoid

# Create model
class SpectraMLP(Module):
    def __init__(self, n_inputs, n_outputs, layer_width=512):
        super(SpectraMLP, self).__init__()
        
        self.hidden1 = Linear(n_inputs, layer_width)
        self.act1 = ReLU()
        self.hidden2 = Linear(layer_width, layer_width)
        self.act2 = ReLU()
        self.hidden3 = Linear(layer_width, layer_width)
        self.act3 = ReLU()
        self.hidden4 = Linear(layer_width, layer_width)
        self.act4 = ReLU()
        self.hidden5 = Linear(layer_width,n_outputs)
        
        #self.act5 = Sigmoid()

    def forward(self, x):
        
        x = self.hidden1(x)
        x = self.act1(x)
        x = self.hidden2(x)
        x = self.act2(x)
        x = self.hidden3(x)
        x = self.act3(x)
        x = self.hidden4(x)
        x = self.act4(x)
        x = self.hidden5(x)
        
        return x