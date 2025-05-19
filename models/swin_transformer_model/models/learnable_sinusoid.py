import numpy as np
import torch
import torch.nn as nn
from .pos_embeddings import get_raw_coordinate, get_sinusoidal_2d


def cross_cat(x1, x2):
    num, d = x1.shape
    res = torch.zeros((num, 2 * d))
    res[:, 0: 2 * d:2] = x1
    res[:, 1: 2 * d:2] = x2
    res = res.to(x1.device)
    return res

class LearnableSinusoidPositionalEncoding(nn.Module):
    def __init__(self, in_dim, s_dim, pos_dim, h_dim):
        super().__init__()
        self.in_dim = in_dim
        self.s_dim = s_dim
        self.pos_dim = pos_dim
        self.h_dim = h_dim
        self.Wr = nn.Linear(self.in_dim, self.s_dim, bias=False)
        self.init_weights()

    def init_weights(self):
        self.Wr.weight = nn.Parameter(torch.load('./models/pretrain_pos/wr_96.pt').cpu().t())

    def forward(self, x):
        projected = self.Wr(x)
        sines = torch.sin(projected)
        cosines = torch.cos(projected)
        sin_pos = cross_cat(sines, cosines)
        sin_pos = sin_pos.unsqueeze(0)
        return sin_pos


if __name__ == '__main__':
    x = get_raw_coordinate(196)
    model = LearnableSinusoidPositionalEncoding(2, 384, 768, 3072)
    sin_pos = model(x)
    sin = get_sinusoidal_2d(768, 14, 14, sigma=torch.Tensor([10000]))[0]
    print(sin_pos)
    print(torch.dist(sin, sin_pos))