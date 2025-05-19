import torch
import torch.nn as nn


class PEGate(nn.Module):
    def __init__(self, init_value, sigmoid=True):
        super(PEGate, self).__init__()
        self.gate = nn.Parameter(torch.tensor(init_value))
        if sigmoid:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Identity()

    def forward(self, x):
        return self.sigmoid(x * self.gate)