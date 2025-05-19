import numpy as np
import torch
import torch.nn as nn


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, G: int, M: int, F_dim: int, H_dim: int, D: int, gamma: float, mode: str = 'gaussian'):
        """
        Learnable Fourier Features from https://arxiv.org/pdf/2106.02795.pdf (Algorithm 1)
        Implementation of Algorithm 1: Compute the Fourier feature positional encoding of a multi-dimensional position
        Computes the positional encoding of a tensor of shape [N, G, M]
        :param G: positional groups (positions in different groups are independent)
        :param M: each point has a M-dimensional positional values
        :param F_dim: depth of the Fourier feature dimension
        :param H_dim: hidden layer dimension
        :param D: positional encoding dimension
        :param gamma: parameter to initialize Wr
        """
        super().__init__()
        self.G = G
        self.M = M
        self.F_dim = F_dim
        self.H_dim = H_dim
        self.D = D
        self.gamma = gamma
        self.mode = mode

        # Projection matrix on learned lines (used in eq. 2)
        self.Wr = nn.Linear(self.M, self.F_dim // 2, bias=False)
        # MLP (GeLU(F @ W1 + B1) @ W2 + B2 (eq. 6)
        self.mlp = nn.Sequential(
            # nn.Linear(self.F_dim, self.D // self.G, bias=True)
            nn.Linear(self.F_dim, self.H_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.H_dim, self.D // self.G)
        )
        # self.layer_norm = nn.LayerNorm(self.D // self.G, eps=1e-6)
        self.init_weights(self.mode)

    def init_weights(self, mode):
        if mode == 'gaussian':
            print('Gussian initialization')
            nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma ** -2)
        elif mode == 'pretrain':
            print('Loading pretrained weight')
            self.Wr.weight = nn.Parameter(torch.load('../models/pretrain_pos/wr.pt').cpu())
            # self.Wr.requires_grad_(False)
            self.mlp[0].weight = nn.Parameter(torch.load('../models/pretrain_pos/mlp_weight1.pt').cpu())
            self.mlp[2].weight = nn.Parameter(torch.load('../models/pretrain_pos/mlp_weight2.pt').cpu())
            self.mlp[0].bias = nn.Parameter(torch.load('../models/pretrain_pos/mlp_bias1.pt').cpu())
            self.mlp[2].bias = nn.Parameter(torch.load('../models/pretrain_pos/mlp_bias2.pt').cpu())

    def forward(self, x):
        """
        Produce positional encodings from x
        :param x: tensor of shape [N, G, M] that represents N positions where each position is in the shape of [G, M],
                  where G is the positional group and each group has M-dimensional positional values.
                  Positions in different positional groups are independent
        :return: positional encoding for X
        """
        N, G, M = x.shape
        # Step 1. Compute Fourier features (eq. 2)
        projected = self.Wr(x)
        cosines = torch.cos(projected)
        sines = torch.sin(projected)
        F = 1 / np.sqrt(self.F_dim) * torch.cat([cosines, sines], dim=-1)
        # F = torch.cat([cosines, sines], dim=-1)
        # Step 2. Compute projected Fourier features (eq. 6)
        Y = self.mlp(F)
        # Step 3. Reshape to x's shape
        PEx = Y.reshape((N, self.D))
        PEx = PEx.unsqueeze(0)
        # PEx = self.layer_norm(PEx)
        # F = F.reshape((N, self.D))
        # F = F.unsqueeze(0)
        return PEx


if __name__ == '__main__':
    G = 1
    M = 2
    x = torch.randn((196, G, M))
    enc = LearnableFourierPositionalEncoding(G, M, 768, 3072, 768, 10, 'pretrain')
    pex = enc(x)
    print(pex.shape)
    print(enc)