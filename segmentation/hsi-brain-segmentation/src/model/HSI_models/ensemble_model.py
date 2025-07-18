import torch.nn as nn
import torch.nn.functional as F
import torch

class EnsembleModelAverage(nn.Module):
    def __init__(self, models):
        super(EnsembleModelAverage, self).__init__()
        self.models = models

    def forward(self, x):
        outputs = [model(x) for model in self.models]
        avg_output = torch.mean(torch.stack(outputs), dim=0)
        return avg_output

class EnsembleModelMajorityVoting(nn.Module):
    def __init__(self, models):
        super(EnsembleModelMajorityVoting, self).__init__()
        self.models = models

    def forward(self, x):
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        # Stack the outputs to get a tensor of shape (num_models, batch_size, num_classes, height, width)
        stacked_outputs = torch.stack(outputs)  # Shape: (num_models, batch_size, num_classes, height, width)
        # Apply softmax to get probabilities along the class dimension
        probabilities = torch.sigmoid(stacked_outputs)
        # Convert probabilities to binary predictions (assuming binary segmentation)
        binary_predictions = probabilities > 0.5
        # Perform majority voting along the num_models dimension
        majority_vote = torch.sum(binary_predictions.int(), dim=0) > (len(self.models) // 2)  # Shape: (batch_size, num_classes, height, width)
        return majority_vote.float()
    
class EnsembleModelAbsoluteVoting(nn.Module):
    def __init__(self, models):
        super(EnsembleModelAbsoluteVoting, self).__init__()
        self.models = models

    def forward(self, x):
        # Get predictions from all models
        outputs = [model(x) for model in self.models]
        # Stack the outputs to get a tensor of shape (num_models, batch_size, num_classes, height, width)
        stacked_outputs = torch.stack(outputs)  # Shape: (num_models, batch_size, num_classes, height, width)
        # Apply softmax to get probabilities along the class dimension
        probabilities = torch.sigmoid(stacked_outputs)
        # Convert probabilities to binary predictions (assuming binary segmentation)
        binary_predictions = probabilities > 0.5
        # Perform majority voting along the num_models dimension
        majority_vote = torch.sum(binary_predictions.int(), dim=0) == (len(self.models))  # Shape: (batch_size, num_classes, height, width)
        return majority_vote.float()