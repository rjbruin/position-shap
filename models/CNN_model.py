import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    '''
    Create network with 4 Conv layers
    '''

    def __init__(self, padding_size, pad_type):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)

        self.fc1 = nn.Linear(64 * 1 * 1, 2)
        self.adap_max = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.adap_max(x)
        x = x.view(-1, 64 * 1 * 1)
        x = self.fc1(x)
        return x