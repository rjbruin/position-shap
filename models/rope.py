"""
This code was originally obtained from:
https://github.com/meta-llama/codellama/blob/main/llama/model.py
"""

import torch
import torch.nn as nn
import math
from functools import partial

from models.pe_gate import PEGate


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat([mag * torch.cos(angles), mag * torch.cos(torch.pi/2 + angles)], dim=-1)
        fy = torch.cat([mag * torch.sin(angles), mag * torch.sin(torch.pi/2 + angles)], dim=-1)
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs

def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode='floor').float()
    return t_x, t_y

def compute_mixed_cis(freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_y = (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2)).view(N, num_heads, -1).permute(1, 0, 2)
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis

def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor, gate: torch.nn.Module = None):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    if freqs_cis.ndim < 4:
        freqs_cis = reshape_for_broadcast(freqs_cis, xq_)

    # PE Gating: freqs_cis to polar, multiply angles by gate, back to complex
    if gate:
        freqs_cis_real = torch.view_as_real(freqs_cis)
        freqs_cis_angles = torch.atan2(freqs_cis_real[..., 1], freqs_cis_real[..., 0])
        gated = gate(freqs_cis_angles)
        freqs_cis_gated = torch.polar(torch.ones_like(freqs_cis_real[..., 0]), gated)
    else:
        freqs_cis_gated = freqs_cis

    xq_out = torch.view_as_real(xq_ * freqs_cis_gated).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_gated).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class RoPEAttentionBase(nn.Module):
    # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    def __init__(self, config, img_size, qkv_bias=False, qk_scale=None, **kwargs):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.transformer["num_heads"]
        head_dim = self.dim // self.num_heads
        self.attention_head_size = int(config.hidden_size / self.num_heads)
        self.all_head_size = self.num_heads * self.attention_head_size
        self.num_patches = (img_size // config.patches['size'][0], img_size // config.patches['size'][1])

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(self.dim, self.all_head_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(config.transformer["attention_dropout_rate"])
        self.proj = nn.Linear(self.all_head_size, self.dim)
        self.proj_drop = nn.Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale

        attn = (q @ k.transpose(-2, -1))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(RoPEAttentionBase):
    """Multi-head Attention block with rotary position embeddings."""
    def __init__(self, config, *args, rope_theta=10.0, rope_mixed=True, pos_emb_gate=False, pos_emb_gate_params=None, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.rope_mixed = rope_mixed
        self.uses_cls_token = config.classifier == 'token'
        self.cls_discounter = 1 if self.uses_cls_token else 0

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = init_2d_freqs(
                dim=self.dim // self.num_heads, num_heads=self.num_heads, theta=rope_theta,
                rotate=True
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            t_x, t_y = init_t_xy(end_x=self.num_patches[0], end_y=self.num_patches[1])
            self.register_buffer('freqs_t_x', t_x)
            self.register_buffer('freqs_t_y', t_y)
        else:
            self.compute_cis = partial(compute_axial_cis, dim=self.dim // self.num_heads, theta=rope_theta)
            freqs_cis = self.compute_cis(end_x=14, end_y=14)
            self.freqs_cis = freqs_cis

        self.gate_shared = False
        if pos_emb_gate:
            if 'shared' in pos_emb_gate_params and pos_emb_gate_params['shared']:
                # self.gate = pos_emb_gate_params['gate']
                self.gate = None
                self.gate_shared = True
            else:
                self.gate = PEGate(init_value=pos_emb_gate_params['init_value'], sigmoid=pos_emb_gate_params['sigmoid'])
        else:
            self.gate = None

    def forward(self, x, gate=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        w = h = math.sqrt(x.shape[1] - self.cls_discounter)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if self.freqs_t_x.shape[0] != x.shape[1] - self.cls_discounter:
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            self.freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if self.freqs_cis.shape[0] != x.shape[1] - self.cls_discounter:
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
            self.freqs_cis = freqs_cis.to(x.device)

        _gate = self.gate if gate is None else gate
        # If gate is shared, it is passed to forward() but not available in
        # forward_fixed_pos_emb() in SHAP analysis. Therefore, we store the
        # values here to recreate the PEGate in forward_fixed_pos_emb().
        self.gate_val = None if gate is None else gate.gate.detach()
        self.gate_sigmoid = None if gate is None else isinstance(gate.sigmoid, nn.Sigmoid)

        if self.uses_cls_token:
            q_rope, k_rope = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=self.freqs_cis, gate=_gate)
            q = torch.cat((q[:, :, :1], q_rope), dim=2)
            k = torch.cat((k[:, :, :1], k_rope), dim=2)
        else:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis, gate=_gate)
        #########

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x, None, None

    def forward_fixed_pos_emb(self, x, pos_emb):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        # w = h = math.sqrt(x.shape[1] - self.cls_discounter)
        # if self.rope_mixed:
        #     t_x, t_y = self.freqs_t_x, self.freqs_t_y
        #     if self.freqs_t_x.shape[0] != x.shape[1] - self.cls_discounter:
        #         t_x, t_y = init_t_xy(end_x=w, end_y=h)
        #         t_x, t_y = t_x.to(x.device), t_y.to(x.device)
        #     self.freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        # else:
        #     freqs_cis = self.freqs_cis
        #     if self.freqs_cis.shape[0] != x.shape[1] - self.cls_discounter:
        #         freqs_cis = self.compute_cis(end_x=w, end_y=h)
        #     self.freqs_cis = freqs_cis.to(x.device)

        # Store old values
        old_freqs = self.freqs_cis.data.clone()
        self.freqs_cis.data = pos_emb.to(x.device)

        # If gate is shared, it is passed to forward() but not available in
        # forward_fixed_pos_emb() in SHAP analysis. Therefore, we retrieve the
        # stored values for PEGate here and recreate it.
        _gate = self.gate if self.gate_val is None else PEGate(self.gate_val, sigmoid=self.gate_sigmoid)
        if self.uses_cls_token:
            q_rope, k_rope = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=self.freqs_cis, gate=_gate)
            q = torch.cat((q[:, :, :1], q_rope), dim=2)
            k = torch.cat((k[:, :, :1], k_rope), dim=2)
        else:
            q, k = apply_rotary_emb(q, k, freqs_cis=self.freqs_cis, gate=_gate)
        #########

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # Restore old values
        self.freqs_cis.data = old_freqs

        return x