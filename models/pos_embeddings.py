import torch
import torch.nn as nn
from torch import einsum
import numpy as np
import math
from einops import rearrange


def transfer_back(idx, length):
    return idx // length, idx % length


def get_raw_coordinate(num):
    pos = torch.zeros((num, 2))
    for i in range(num):
        y, x = transfer_back(i, np.sqrt(num))
        pos[i][0] = x
        pos[i][1] = y
    return pos


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    # return torch.FloatTensor(sinusoid_table).unsqueeze(0).to(device) if device else torch.FloatTensor(sinusoid_table).unsqueeze(0)
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


def get_sinusoidal_2d(d_model, height, width, sigma=torch.Tensor([10000])):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(torch.log(sigma) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.flatten(1).transpose(0,1).unsqueeze(0)
    return pe
    # return pe.cuda() if device else pe

def get_sinusoidal_2d_learnable(d_model, height, width, sigma, div_d_model, indices):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    div_term = torch.exp(indices *
                         -(torch.log(sigma) / div_d_model))
    d_model = int(d_model / 2)
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe = pe.flatten(1).transpose(0,1).unsqueeze(0)
    return pe


def get_gaussian(x, y, mean=0.0, sigma=5.0):
    dst = np.sqrt(x * x + y * y)
    return np.exp(-((dst - mean) ** 2 / (2.0 * sigma ** 2)))


def get_gaussian_pos(n_position):
    l = int(np.sqrt(n_position))
    pos = np.zeros((n_position, n_position))
    for i in range(n_position):
        x = i % l
        y = i // l
        for j in range(n_position):
            center_x = j % l
            center_y = j // l
            pos[i][j] = get_gaussian(x - center_x, y - center_y)
    pos = torch.FloatTensor(pos)
    # pos = pos.unsqueeze(0)
    # pos = nn.functional.interpolate(pos, dim)
    # return pos.to(device) if device else pos
    return pos


def relative_to_absolute(q):
    """
    Converts the dimension that is specified from the axis
    from relative distances (with length 2*tokens-1) to absolute distance (length tokens)
      Input: [bs, heads, length, 2*length - 1]
      Output: [bs, heads, length, length]
    """
    b, h, l, _, device, dtype = *q.shape, q.device, q.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((q, col_pad), dim=3)  # zero pad 2l-1 to 2l
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim=2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l - 1):]
    return final_x


class AbsPosEmb1D(nn.Module):
    """
    Given query q of shape [batch heads tokens dim] we multiply
    q by all the flattened absolute differences between tokens.
    Learned embedding representations are shared across heads
    """

    def __init__(self, tokens, dim_head):
        """
        Output: [batch head tokens tokens]
        Args:
            tokens: elements of the sequence
            dim_head: the size of the last dimension of q
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.abs_pos_emb = nn.Parameter(torch.randn(tokens, dim_head) * scale)

    def forward(self, q):
        return einsum('b h i d, j d -> b h i j', q, self.abs_pos_emb)


def rel_pos_emb_1d(q, rel_emb, shared_heads, per_sample=False):
    """
    Same functionality as RelPosEmb1D

    Args:
       q: a 4d tensor of shape [batch, heads, tokens, dim]
       rel_emb: a 2D or 3D tensor
       of shape [ 2*tokens-1 , dim] or [ heads, 2*tokens-1 , dim]
    """
    per_sample = (rel_emb.dim() == 4 and not shared_heads) or \
                 (rel_emb.dim() == 3 and shared_heads)
    if per_sample:
        if shared_heads:
            emb = torch.einsum('b h t d, b r d -> b h t r', q, rel_emb)
        else:
            emb = torch.einsum('b h t d, b h r d -> b h t r', q, rel_emb)
    else:
        if shared_heads:
            emb = torch.einsum('b h t d, r d -> b h t r', q, rel_emb)
        else:
            emb = torch.einsum('b h t d, h r d -> b h t r', q, rel_emb)
    return relative_to_absolute(emb)


class RelPosEmb1D(nn.Module):
    def __init__(self, tokens, dim_head, heads=None):
        """
        Output: [batch head tokens tokens]
        Args:
           tokens: the number of the tokens of the seq
           dim_head: the size of the last dimension of q

           heads: if None representation is shared across heads.
           else the number of heads must be provided
        """
        super().__init__()
        scale = dim_head ** -0.5
        self.shared_heads = False if heads is not None else True
        if self.shared_heads:
           self.rel_pos_emb = nn.Parameter(torch.randn(2 * tokens - 1, dim_head) * scale)
        else:
           self.rel_pos_emb = nn.Parameter(torch.randn(heads, 2 * tokens - 1, dim_head) * scale)


    def forward(self, q):
        return rel_pos_emb_1d(q, self.rel_pos_emb, self.shared_heads)


class RelPosEmb2D(nn.Module):
    def __init__(self, feat_map_size, dim_head, heads=None):
        """
        Based on Bottleneck transformer paper
        paper: https://arxiv.org/abs/2101.11605 . Figure 4
        Output: qr^T [batch head tokens tokens]
        Args:
           tokens: the number of the tokens of the seq
           dim_head: the size of the last dimension of q

           heads: if None representation is shared across heads.
           else the number of heads must be provided
        expand_emb() is used to expand the x-axis (y-axis) pos embedding along the y-axis (x-axis).
        """
        super().__init__()
        self.h, self.w = feat_map_size  # height , width
        self.total_tokens = self.h * self.w
        self.shared_heads = heads if heads is not None else True

        self.emb_w = RelPosEmb1D(self.h, dim_head, heads)
        self.emb_h = RelPosEmb1D(self.w, dim_head, heads)

    def expand_emb(self, r, dim_size):
        # Decompose and unsqueeze dimension
        r = rearrange(r, 'b (h x) i j -> b h x () i j', x=dim_size)
        expand_index = [-1, -1, -1, dim_size, -1, -1]  # -1 indicates no expansion
        r = r.expand(expand_index)
        return rearrange(r, 'b h x1 x2 y1 y2 -> b h (x1 y1) (x2 y2)')

    def forward(self, q):
        """
        Args:
           q: [batch, heads, tokens, dim_head]
        Returns: [ batch, heads, tokens, tokens]
        """
        assert self.total_tokens == q.shape[2], f'Tokens {q.shape[2]} of q must \
        be equal to the product of the feat map size {self.total_tokens} '

        # out: [batch head*w h h]
        r_h = self.emb_w(rearrange(q, 'b h (x y) d -> b (h x) y d', x=self.h, y=self.w))
        r_w = self.emb_h(rearrange(q, 'b h (x y) d -> b (h y) x d', x=self.h, y=self.w))
        q_r = self.expand_emb(r_h, self.h) + self.expand_emb(r_w, self.h)
        return q_r

def get_simple_position(n_position):
    l = int(np.sqrt(n_position))
    pos = np.zeros((n_position, 2))
    for i in range(n_position):
        x = i % l
        y = i // l
        pos[i][0] = x / (l - 1)
        pos[i][1] = y / (l - 1)
    pos = torch.FloatTensor(pos)
    return pos


def gaussian_1d(x, mean=0.0, sigma=5.0):
    return np.exp(-((x - mean) ** 2 / (2.0 * sigma ** 2)))


def get_gaussian_pos_v2(n_position):
    l = int(np.sqrt(n_position))
    pos = np.zeros((n_position,  2 * n_position))
    for i in range(n_position):
        x = i % l
        y = i // l
        for j in range(n_position):
            center_x = j % l
            center_y = j // l
            pos[i][2 * j] = gaussian_1d(x - center_x) if x >= center_x else -gaussian_1d(x - center_x)
            pos[i][2 * j + 1] = gaussian_1d(y - center_y) if y >= center_y else -gaussian_1d(y - center_y)
    pos = torch.FloatTensor(pos)
    # pos = pos.unsqueeze(0)
    # pos = nn.functional.interpolate(pos, dim)
    # return pos.to(device) if device else pos
    return pos


def get_simple_v2(n_position):
    l = int(np.sqrt(n_position))
    pos = np.zeros((n_position, 2 * n_position))
    for i in range(n_position):
        x = i % l
        y = i // l
        for j in range(n_position):
            center_x = j % l
            center_y = j // l
            pos[i][2 * j] = (x - center_x) / (l - 1)
            pos[i][2 * j + 1] = (y - center_y) / (l - 1)
    pos = torch.FloatTensor(pos)
    # pos = pos.unsqueeze(0)
    # pos = nn.functional.interpolate(pos, dim)
    # return pos.to(device) if device else pos
    return pos

def get_gaussian_pos_v3(n_position):
    l = int(np.sqrt(n_position))
    pos = np.zeros((n_position,  2 * n_position))
    for i in range(n_position):
        x = i % l
        y = i // l
        for j in range(n_position):
            center_x = j % l
            center_y = j // l
            pos[i][2 * j] = gaussian_1d(x - center_x)
            pos[i][2 * j + 1] = gaussian_1d(y - center_y)
    pos = torch.FloatTensor(pos)
    # pos = pos.unsqueeze(0)
    # pos = nn.functional.interpolate(pos, dim)
    # return pos.to(device) if device else pos
    return pos
