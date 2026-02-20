import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.model(x)