import torch
import torch.nn as nn

torch.manual_seed(42)

class Net(nn.Module):
    """
    A model representing a single layer perceptron
    """
    def __init__(self, in_feat=8, out_feat=1):
        super().__init__()
        # create a linear layer of size in_feat, out_feat
        self.linear = nn.Linear(in_feat, out_feat)

    def forward(self, x):
        y = self.linear(x)
        return y