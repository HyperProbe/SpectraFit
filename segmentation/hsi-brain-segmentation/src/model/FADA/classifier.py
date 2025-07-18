import torch
from torch import nn

class Classifier(nn.Module):
    def __init__(self, base_model):
        super(Classifier, self).__init__()
        self.decoder = base_model.decoder
        self.segmentation_head = base_model.segmentation_head

    def forward(self, x):
        decoder_output = self.decoder(*x)
        segmentation_output = self.segmentation_head(decoder_output)
    
        return segmentation_output