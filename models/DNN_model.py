import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN_net(torch.nn.Module):
    def __init__(self):
        super(DNN_net, self).__init__()
        self.l1 = torch.nn.Linear(1024 * 3, 1024)
        self.l2 = torch.nn.Linear(1024, 256)
        self.l3 = torch.nn.Linear(256, 64)
        self.l4 = torch.nn.Linear(64, 2)
    def forward(self, x):
        x = x.view(-1, 1024 * 3)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)