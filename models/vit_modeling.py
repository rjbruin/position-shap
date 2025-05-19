# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np
from torch import einsum
# from torchsummary import summary
from torchinfo import summary
from torchvision import transforms
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from einops import rearrange
from analysis_shap.shap_analysis import model_agnostic_interface
from models.cape import CAPE2d
from models.pos_embeddings import RelPosEmb2D, get_sinusoid_encoding_table, get_sinusoidal_2d, get_gaussian_pos, get_sinusoidal_2d_learnable, get_simple_position, get_gaussian_pos_v2, get_simple_v2, get_gaussian_pos_v3, get_raw_coordinate
from models.fourier import LearnableFourierPositionalEncoding
from models.learnable_sinusoid import LearnableSinusoidPositionalEncoding
from models.peg import PosCNN
from models.pe_gate import PEGate

import models.configs as configs
import torch.nn.functional as F

from models.rope import RoPEAttention
from models.vit_modeling_resnet import ResNetV2
from timm.models.layers import trunc_normal_

from models.drloc import DenseRelativeLoc
from munch import Munch

import analysis.sources

from models.irpe import get_rpe_config, build_rpe
rpe_config = get_rpe_config(
    ratio=1.9,
    method="product",
    mode='ctx',
    shared_head=False,
    skip=1,
    rpe_on='k',
)

logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query/"
ATTENTION_K = "MultiHeadDotProductAttention_1/key/"
ATTENTION_V = "MultiHeadDotProductAttention_1/value/"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out/"
FC_0 = "MlpBlock_3/Dense_0/"
FC_1 = "MlpBlock_3/Dense_1/"
ATTENTION_NORM = "LayerNorm_0/"
MLP_NORM = "LayerNorm_2/"


def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


def shared_dropout(dropout_module, x):
    dropout_dummy = torch.ones_like(x, requires_grad=False)
    dropout_mask = dropout_module(dropout_dummy)
    return dropout_mask


def decomposed_relu_zeros(a, b, count_linear_errors=False, measure_positive_errors=False):
    """
    ReLU function for x = a + b. For x > 0, ReLU is identity, so propagate
    original values. For x <= 0, propagate zeros, assuming a model of ReLU that
    intervenes on negative values to "turn them off", so any components of x
    should also be "off".
    """
    mask_x_positive = (a + b >= 0).float()

    # If x > 0, ReLU is identity, so propagate original values
    out_a = mask_x_positive * a
    out_b = mask_x_positive * b

    # Otherwise, propagate zeros for a and b. out_a and out_b are already zero
    # where this is the case.

    if count_linear_errors:
        mask_x_neg_a_pos = (1. - mask_x_positive) * (a >= 0).float()
        mask_x_neg_b_pos = (1. - mask_x_positive) * (b >= 0).float()
        linear_error_count_a = mask_x_neg_a_pos.sum()
        linear_error_count_b = mask_x_neg_b_pos.sum()
        return out_a, out_b, linear_error_count_a, linear_error_count_b

    if measure_positive_errors:
        mask_x_neg_a_pos = (1. - mask_x_positive) * (a >= 0).float()
        a_pos_term = (mask_x_neg_a_pos * a).sum()
        mask_x_neg_b_pos = (1. - mask_x_positive) * (b >= 0).float()
        b_pos_term = (mask_x_neg_b_pos * b).sum()
        return out_a, out_b, a_pos_term, b_pos_term

    return out_a, out_b



ACT2FN = {
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU,
    "swish": swish # Not compatible with SHAP, because it needs to be a module
}
ACT2FN_decomposition_compatible = {
    "relu": decomposed_relu_zeros
}


class Attention(nn.Module):
    def __init__(self, config, img_size, use_rel_pos=False, rel_pos_heads=None, use_irpe=False, softmax=True, override_attn_out_dim=None, compute_decomposed_attn=False, compute_gradbased_attr=False):
        super(Attention, self).__init__()
        self.compute_decomposed_attn = compute_decomposed_attn
        self.compute_gradbased_attr = compute_gradbased_attr
        if self.compute_decomposed_attn and self.compute_gradbased_attr:
            raise ValueError("Cannot compute both decomposed attention and grad-based attribution")

        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        if override_attn_out_dim is not None:
            self.attention_head_size = override_attn_out_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        # self.out = Linear(config.hidden_size, config.hidden_size)
        self.out = Linear(self.all_head_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)
        if not softmax:
            self.softmax = torch.nn.Identity()

        self.use_rel_pos = use_rel_pos
        self.use_irpe = use_irpe

        if self.use_rel_pos:
            img_size = _pair(img_size)
            patch_size = _pair(config.patches["size"])
            w, h = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
            self.rel_pos = RelPosEmb2D((w, h), self.attention_head_size, rel_pos_heads)

        if self.use_irpe:
            self.rpe_q, self.rpe_k, self.rpe_v = \
                build_rpe(rpe_config,
                          head_dim=self.attention_head_size,
                          num_heads=self.num_attention_heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, hidden_sem=None, hidden_pos=None):
        """
        This forward pass implements the attention mechanism for x, as well as
        for x=s+p if s and p are passed. The attention mechanism for s and p is
        intentionally interspersed with the code for x, so that the discrete
        parts of the algorithm are kept together.
        """
        # x: projections + scores
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if self.use_rel_pos:
            attention_scores += self.rel_pos(query_layer)

        if self.use_irpe:
            if self.rpe_k is not None:
                attention_scores += self.rpe_k(query_layer)

                # image relative position on queries
            if self.rpe_q is not None:
                attention_scores += self.rpe_q(key_layer * self.scale).transpose(2, 3)

        attention_probs = self.softmax(attention_scores)

        attention_probs_sliced = None
        if self.compute_gradbased_attr:
            # To make attention probabilities differentiable *per sample* we
            # need to include the slicing per sample in the graph. Otherwise
            # autograd.grad() complains that the sliced tensors are not in the
            # graph when we later compute the image/position attributions wrt
            # the attention probabilities.
            attention_probs_sliced = []
            # Shape of probabilities: [batch_size, num_heads, ...]
            N, K, H, W = attention_probs.shape
            for i in range(N):
                # What we want: a list of [num_heads * H * W] probabilities, so
                # autograd.grad() can compute the "batched" vector-Jacobian for
                # each unique probability in each head, as we will use the
                # "batched" mode of grad() where the first dimension needs to
                # contain the "batches".
                grad_compat_slice = attention_probs[i].reshape(-1)
                if grad_compat_slice.requires_grad:
                    grad_compat_slice.retain_grad()
                attention_probs_sliced.append(grad_compat_slice)
            attention_probs = torch.stack(attention_probs_sliced).reshape(N, K, H, W)

        weights_to_vis = attention_probs.detach()

        # x: dropout (on weights, no sync necessary)
        attention_probs_dropout = self.attn_dropout(attention_probs)

        # x: context layer (gathering value tokens), output layer
        context_layer = torch.matmul(attention_probs_dropout, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)

        # x: dropout (synchronized with s+p)
        # NOTE: Dropout is stochastic, even in a single forward pass or same
        # layer. To make sure x=s+p we need to compute one dropout mask, and
        # apply it identically to x, s and p. We do this by applying dropout to
        # a "dummy" tensor which then functions as a mask.
        dropout_mask_out = shared_dropout(self.proj_dropout, attention_output)
        attention_output = attention_output * dropout_mask_out

        if self.compute_gradbased_attr:
            return attention_output, weights_to_vis, attention_probs_sliced

        return attention_output, weights_to_vis, None




class Mlp(nn.Module):
    def __init__(self, config, decomposed=False, act_fn="gelu"):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN[act_fn]()
        if decomposed:
            print("Overriding activation function to ReLU for decomposition")
            # Not compatible with SHAP, because it needs to be a module
            self.act_fn = ACT2FN["relu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        # DEBUG: store activation, for measuring "zero detectors"
        self.post_act_out = self.act_fn(x)
        dropout_1_mask = shared_dropout(self.dropout, self.post_act_out)
        x = self.post_act_out * dropout_1_mask

        # DEBUG: store activation, for measuring "zero detectors"
        self.fc_out = self.fc2(x)
        dropout_2_mask = shared_dropout(self.dropout, self.fc_out)
        x = self.fc_out * dropout_2_mask

        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3, pos_emb=None, fourier_gamma=2.5, pos_emb_gate=False, pos_emb_gate_params=None):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.pos_emb = pos_emb
        img_size = _pair(img_size)
        self.img_size = img_size
        self.pooling = config.classifier

        if config.patches.get("grid") is not None:
            if self.pooling == 'token':
                raise NotImplementedError("Pooling 'token' is not supported with grid patches")
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            if self.pooling == 'token':
                n_patches += 1
                if self.pos_emb == 'absolute_learnable_concat_equald':
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size // 2))
                else:
                    self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            self.hybrid = False

        self.cached_position_embeddings = None

        self.hidden_size = config.hidden_size
        self.width = self.height = img_size[0] // patch_size[0]

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16

        if not self.pos_emb == 'absolute_learnable_concat_equald':
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size,
                                           kernel_size=patch_size,
                                           stride=patch_size)
        else:
            assert config.hidden_size % 2 == 0, "Hidden size must be divisible by 2 for pos_emb=absolute_learnable_concat_equald"
            self.patch_embeddings = Conv2d(in_channels=in_channels,
                                           out_channels=config.hidden_size // 2,
                                           kernel_size=patch_size,
                                           stride=patch_size)

        # Track the parameters used to create the position embeddings, for PE weight decay
        pos_emb_modules = []
        if self.pos_emb in ['absolute_learnable', 'absolute_learnable_skipconnect']:
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
            pos_emb_modules = [self.position_embeddings]
            trunc_normal_(self.position_embeddings, std=.02)
        elif self.pos_emb == 'absolute_learnable_concat_equald':
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size // 2))
            pos_emb_modules = [self.position_embeddings]
            trunc_normal_(self.position_embeddings, std=.02)
        elif self.pos_emb == 'absolute_freeze':
            self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size), requires_grad=False)
            pos_emb_modules = []
            trunc_normal_(self.position_embeddings, std=.02)
        elif self.pos_emb == 'sinusoid_1d':
            self.position_embeddings = nn.Parameter(get_sinusoid_encoding_table(n_patches, config.hidden_size), requires_grad=False)
            pos_emb_modules = []
        elif self.pos_emb == 'sinusoid_2d':
            self.position_embeddings = nn.Parameter(get_sinusoidal_2d(config.hidden_size, img_size[0] // patch_size[0], img_size[0] // patch_size[0], sigma=torch.Tensor([10000])), requires_grad=False)
            pos_emb_modules = []
        elif self.pos_emb == 'sinusoid_2d_learnable':
            sin_pos = get_sinusoidal_2d(config.hidden_size, img_size[0] // patch_size[0], img_size[0] // patch_size[0], sigma=torch.Tensor([10000]))
            self.position_embeddings = nn.Parameter(sin_pos, requires_grad=True)
            pos_emb_modules = [self.position_embeddings]
        elif self.pos_emb == 'sinusoid_2d_learnable_scale':
            sin_pos = get_sinusoidal_2d(config.hidden_size, img_size[0] // patch_size[0], img_size[0] // patch_size[0], sigma=torch.Tensor([10000]))
            sin_pos /= (torch.std(sin_pos) / 0.02)

            # sin_pos = transforms.Normalize(0.5, 20)(sin_pos)[0]
            self.position_embeddings = nn.Parameter(sin_pos, requires_grad=True)
            pos_emb_modules = [self.position_embeddings]
        elif self.pos_emb == 'gaussian_2d':
            # self.position_embeddings = nn.Parameter(get_gaussian_pos(n_patches, config.hidden_size), requires_grad=False)
            self.pos = nn.Parameter(get_gaussian_pos(n_patches), requires_grad=False)
            self.mlp = nn.Sequential(
                nn.Linear(n_patches, config.hidden_size)
                # nn.GELU(),
                # nn.Linear(4 * config.hidden_size, config.hidden_size)
                # nn.LayerNorm(config.hidden_size, eps=1e-6)
            )
            #self.layer_norm = nn.LayerNorm(config.hidden_size, eps=1e-6)
            #self.batch_norm = nn.BatchNorm1d(n_patches, eps=1e-6)
            self.position_embeddings = None
            pos_emb_modules = [self.mlp]
        elif self.pos_emb == 'pretrain':
            pretrain_pos = torch.load('../models/pretrain_pos/vit-b-16-224-pos.pt')
            pretrain_pos = pretrain_pos[:, 1:, :]
            print(pretrain_pos.shape)
            self.position_embeddings = nn.Parameter(pretrain_pos, requires_grad=True)
            pos_emb_modules = [self.position_embeddings]
        elif self.pos_emb == 'sinusoid_learnable_small':
            self.pos = nn.Parameter(get_sinusoidal_2d(n_patches, img_size[0] // patch_size[0], img_size[0] // patch_size[0])[0], requires_grad=False)
            self.pos_weight = nn.Linear(n_patches, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'sinusoid_learnable_big':
            self.pos = nn.Parameter(get_sinusoidal_2d(config.hidden_size, img_size[0] // patch_size[0], img_size[0] // patch_size[0])[0], requires_grad=False)
            self.pos_weight = nn.Linear(config.hidden_size, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'sinusoid_para_learnable':
            self.sigma = nn.Parameter(torch.Tensor([20]), requires_grad=True)
            self.div_d_model = nn.Parameter(torch.Tensor([int(config.hidden_size / 2)]), requires_grad=True)
            self.indices = nn.Parameter(torch.arange(0., int(config.hidden_size / 2), 2), requires_grad=True)
            self.position_embeddings = None
            pos_emb_modules = [self.sigma, self.div_d_model, self.indices]
        elif self.pos_emb == 'simple_position':
            self.pos = nn.Parameter(get_simple_position(n_patches), requires_grad=False)
            self.pos_weight = nn.Linear(2, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'gaussian_2dv2':
            # self.position_embeddings = nn.Parameter(get_gaussian_pos(n_patches, config.hidden_size), requires_grad=False)
            self.pos = nn.Parameter(get_gaussian_pos_v2(n_patches), requires_grad=False)
            self.pos_weight = nn.Linear(2 * n_patches, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'gaussian_2dv3':
            # self.position_embeddings = nn.Parameter(get_gaussian_pos(n_patches, config.hidden_size), requires_grad=False)
            self.pos = nn.Parameter(get_gaussian_pos_v3(n_patches), requires_grad=False)
            self.pos_weight = nn.Linear(2 * n_patches, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'simple_v2':
            self.pos = nn.Parameter(get_simple_v2(n_patches), requires_grad=False)
            self.pos_weight = nn.Linear(2 * n_patches, config.hidden_size)
            self.position_embeddings = None
            pos_emb_modules = [self.pos_weight]
        elif self.pos_emb == 'abs_sin':
            self.sin_pos = nn.Parameter(
                get_sinusoidal_2d(config.hidden_size, img_size[0] // patch_size[0], img_size[0] // patch_size[0],
                                  sigma=torch.Tensor([10000])), requires_grad=False)
            self.abs_pos = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))
            trunc_normal_(self.abs_pos, std=.02)
            self.lambda_sin = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
            self.lambda_abs = nn.Parameter(torch.Tensor([0.5]), requires_grad=True)
            pos_emb_modules = [self.abs_pos, self.lambda_sin, self.lambda_abs]
        elif self.pos_emb == 'fourier':
            self.pos = nn.Parameter(get_raw_coordinate(n_patches).unsqueeze(1), requires_grad=False)
            # self.pos = nn.Parameter(get_gaussian_pos(n_patches).unsqueeze(1), requires_grad=False)
            self.fourier_layer = LearnableFourierPositionalEncoding(1, 2, config.hidden_size, 4*config.hidden_size, config.hidden_size, fourier_gamma)
            self.position_embeddings = None
            pos_emb_modules = [self.fourier_layer]
        elif self.pos_emb == 'fourier_pretrain':
            self.pos = nn.Parameter(get_raw_coordinate(n_patches).unsqueeze(1), requires_grad=False)
            self.fourier_layer = LearnableFourierPositionalEncoding(1, 2, config.hidden_size, 4 * config.hidden_size,
                                                                    config.hidden_size, fourier_gamma, 'pretrain')
            self.position_embeddings = None
            pos_emb_modules = [self.fourier_layer]

        elif self.pos_emb == 'sinusoid_learnable_new_pregenerated':
            self.pos = nn.Parameter(get_raw_coordinate(n_patches), requires_grad=False)
            self.sin_model = LearnableSinusoidPositionalEncoding(2, config.hidden_size // 2, config.hidden_size, 4 * config.hidden_size, initialize='load')
            self.position_embeddings = None
            pos_emb_modules = [self.sin_model]
        elif self.pos_emb == 'sinusoid_learnable_new':
            self.pos = nn.Parameter(get_raw_coordinate(n_patches), requires_grad=False)
            self.sin_model = LearnableSinusoidPositionalEncoding(2, config.hidden_size // 2, config.hidden_size, 4 * config.hidden_size, initialize='sinusoid')
            self.position_embeddings = None
            pos_emb_modules = [self.sin_model]
        elif self.pos_emb == 'sinusoid_learnable_new_random':
            self.pos = nn.Parameter(get_raw_coordinate(n_patches), requires_grad=False)
            self.sin_model = LearnableSinusoidPositionalEncoding(2, config.hidden_size // 2, config.hidden_size, 4 * config.hidden_size, initialize='random')
            self.position_embeddings = None
            pos_emb_modules = [self.sin_model]

        elif self.pos_emb == 'cape':
            pe_gate_kwargs = {}
            if pos_emb_gate:
                pe_gate_kwargs = {'pos_emb_gate': pos_emb_gate, 'pos_emb_gate_params': pos_emb_gate_params}
            self.cape = CAPE2d(d_model=config.hidden_size, max_global_shift=0.5, max_local_shift=0.5, max_global_scaling=1.4, batch_first=True, **pe_gate_kwargs)
            self.position_embeddings = None
            pos_emb_modules = [self.cape]

        self.gate = nn.Identity()
        if pos_emb_gate and self.pos_emb != 'cape':
            self.gate = PEGate(init_value=pos_emb_gate_params['init_value'], sigmoid=pos_emb_gate_params['sigmoid'])

        # Flatten and get parameters from modules
        self.pos_emb_params = []
        for module in pos_emb_modules:
            if isinstance(module, nn.Module):
                self.pos_emb_params.extend(module.parameters())
            else:
                self.pos_emb_params.append(module)

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def gate_params(self):
        if self.pos_emb == 'cape':
            return list(self.cape.gate.parameters())
        elif isinstance(self.gate, nn.Identity):
            return []
        else:
            return list(self.gate.parameters())

    def forward(self, x, update_pos_emb=True):
        # B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        cape_x = x
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        self.cached_patch_embeddings = x.data
        # x = torch.cat((cls_tokens, x), dim=1)

        if self.pooling == 'token':
            # Add a "CLS" token to the input
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        if self.pos_emb == None or self.pos_emb == 'relative_learnable' or self.pos_emb == 'conditional_pos' or self.pos_emb == 'conditional_pos_fixed' or self.pos_emb == 'irpe' or self.pos_emb == 'rope':
            embeddings = x
        elif self.pos_emb == 'gaussian_2d' or self.pos_emb == 'sinusoid_learnable_small' or self.pos_emb == 'sinusoid_learnable_big' or self.pos_emb == 'simple_position' or self.pos_emb == 'gaussian_2dv2' or self.pos_emb == 'gaussian_2dv3' or self.pos_emb == 'simple_v2':
            if update_pos_emb:
                self.position_embeddings = self.mlp(self.pos).unsqueeze(0)
            embeddings = x + self.gate(self.position_embeddings)
        elif self.pos_emb == 'sinusoid_para_learnable':
            if update_pos_emb:
                sinusoid_position_embeddings = get_sinusoidal_2d_learnable(self.hidden_size, self.width, self.height, self.sigma.cpu(), self.div_d_model.cpu(), self.indices.cpu())
                self.position_embeddings = sinusoid_position_embeddings.cuda()
            embeddings = x + self.gate(self.position_embeddings)
        elif self.pos_emb == 'abs_sin':
            if update_pos_emb:
                self.position_embeddings = self.sin_pos + self.abs_pos
            embeddings = x + self.gate(self.position_embeddings)
        elif self.pos_emb == 'fourier' or self.pos_emb == 'fourier_pretrain':
            if update_pos_emb:
                self.position_embeddings = self.fourier_layer(self.pos)
                # DEBUG: temp fix for analysis
                self.position_embeddings.requires_grad_(True)
            embeddings = x + self.gate(self.position_embeddings)
        elif self.pos_emb in ['sinusoid_learnable_new', 'sinusoid_learnable_new_pregenerated', 'sinusoid_learnable_new_random']:
            if update_pos_emb:
                self.position_embeddings = self.sin_model(self.pos)
                # DEBUG: temp fix for analysis
                self.position_embeddings.requires_grad_(True)
            embeddings = x + self.gate(self.position_embeddings)
        elif self.pos_emb == 'cape':
            if update_pos_emb:
                cape_x = cape_x.permute(0, 2, 3, 1)
                self.cape.to(cape_x.device)
                self.position_embeddings = self.cape(cape_x)
            embeddings = self.position_embeddings
            embeddings = embeddings.flatten(1, 2)
        elif self.pos_emb == 'absolute_learnable_concat_equald':
            embeddings = torch.cat([x, self.gate(self.position_embeddings).expand(x.shape[0], -1, -1)], dim=2)
        else:
            embeddings = x + self.gate(self.position_embeddings)

        if self.pos_emb is not None and self.pos_emb not in ['relative_learnable', 'conditional_pos', 'conditional_pos_fixed', 'irpe', 'rope']:
            self.cached_position_embeddings = self.position_embeddings.data

        embeddings = self.dropout(embeddings)

        if self.pos_emb == 'absolute_learnable_skipconnect':
            return embeddings, self.position_embeddings
        else:
            return embeddings

    def forward_fixed_pos_emb(self, x, pos_emb):
        self.position_embeddings.data = pos_emb
        return self.forward(x, update_pos_emb=False)


def layer_norm_stats(data, normalized_shape):
    unnormed_dims = data.dim() - len(normalized_shape)
    unnormed_shape = data.shape[:unnormed_dims]
    flat_data = data.view(*unnormed_shape, -1)
    mean = flat_data.mean(dim=-1)
    # Use biased variance estimator, as in original LayerNorm
    var = flat_data.var(unbiased=False, dim=-1)
    return mean, var

def layer_norm_from_stats(data, mean, var, eps=1e-6):
    normed_dims = data.dim() - mean.dim()
    mean = mean.view(*mean.shape, *([1] * normed_dims))
    var = var.view(*var.shape, *([1] * normed_dims))
    return (data - mean) / (torch.sqrt(var + eps))

class PermuteLayer(torch.nn.Module):
    """
    https://github.com/pytorch/pytorch/issues/128412
    """
    dims: tuple[int, ...]

    def __init__(self, dims: tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(*self.dims)

class Block(nn.Module):
    def __init__(self, config, img_size, use_rel_pos=False, rel_pos_heads=None, use_irpe=False, use_rope=False, softmax=True, override_attn_out_dim=None, pe_skip_connect=False, compute_decomposed_attn=False, compute_gradbased_attr=False, norm=True, residual=True, act_fn="gelu", norm_fn="layernorm", pos_emb_gate=False, pos_emb_gate_params=None):
        super().__init__()
        self.compute_decomposed_attn = compute_decomposed_attn
        self.compute_gradbased_attr = compute_gradbased_attr
        self.pe_skip_connect = pe_skip_connect

        self.hidden_size = config.hidden_size
        if norm:
            if norm_fn == "layernorm":
                self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
                self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
            elif norm_fn == "batchnorm":
                self.attention_norm = nn.Sequential(
                    PermuteLayer((0, 2, 1)),
                    nn.BatchNorm1d(config.hidden_size, eps=1e-6),
                    PermuteLayer((0, 2, 1)),
                )
                self.ffn_norm = nn.Sequential(
                    PermuteLayer((0, 2, 1)),
                    nn.BatchNorm1d(config.hidden_size, eps=1e-6),
                    PermuteLayer((0, 2, 1)),
                )
            else:
                raise ValueError(f"Unknown normalization function: {norm_fn}")
        else:
            self.attention_norm = nn.Identity()
            self.ffn_norm = nn.Identity()
        self.ffn = Mlp(config, decomposed=compute_decomposed_attn, act_fn=act_fn)
        attn_class = RoPEAttention if use_rope else Attention
        pe_gating_kwargs = {'pos_emb_gate': pos_emb_gate, 'pos_emb_gate_params': pos_emb_gate_params} if use_rope else {}
        self.attn = attn_class(config, img_size, use_rel_pos=use_rel_pos, rel_pos_heads=rel_pos_heads, use_irpe=use_irpe, softmax=softmax, override_attn_out_dim=override_attn_out_dim, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr, **pe_gating_kwargs)
        self.residual = residual

    def forward(self, *inputs, **kwargs):
        if self.pe_skip_connect:
            # Apply skip-connection from PEs to here
            x, pos_embs = inputs
            x = x + pos_embs
        else:
            x = inputs[0]

        gate = {}
        if 'gate' in kwargs:
            gate = {'gate': kwargs['gate']}

        h = x
        x = self.attention_norm(x)
        x, weights_to_vis, attn_probs_sliced = self.attn(x, **gate)
        if self.residual:
            x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)

        if self.residual:
            x = x + h

        return x, weights_to_vis, attn_probs_sliced

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}/"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, img_size, pos_emb, compute_decomposed_attn=False, compute_gradbased_attr=False, pos_emb_gate=False, pos_emb_gate_params=None, pos_emb_gate_shared=False):
        super(Encoder, self).__init__()
        self.compute_decomposed_attn = compute_decomposed_attn
        self.compute_gradbased_attr = compute_gradbased_attr

        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.pos_emb = pos_emb
        self.img_size = img_size
        self.patch_size = config.patches.size

        self.pe_skip_connect = pos_emb == 'absolute_learnable_skipconnect'

        self.gate_shared = False
        if pos_emb_gate and pos_emb_gate_shared:
            self.gate = PEGate(init_value=pos_emb_gate_params['init_value'], sigmoid=pos_emb_gate_params['sigmoid'])
            # pos_emb_gate_params['gate'] = gate
            pos_emb_gate_params['shared'] = True
            self.gate_shared = True

        for i in range(config.transformer["num_layers"]):

            if i == 0 and pos_emb == 'relative_learnable':
                # RPE is used only in the first layer
                layer = Block(config, img_size, use_rel_pos=True, pe_skip_connect=self.pe_skip_connect, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr)
            elif i == 0 and pos_emb == 'irpe':
                # iRPE is used only in the first layer
                layer = Block(config, img_size, use_irpe=True, pe_skip_connect=self.pe_skip_connect, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr)
            elif pos_emb == 'rope':
                # RoPE is used in every layer
                layer = Block(config, img_size, use_rope=True, pe_skip_connect=self.pe_skip_connect, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr, pos_emb_gate=pos_emb_gate, pos_emb_gate_params=pos_emb_gate_params)
            else:
                layer = Block(config, img_size, pe_skip_connect=self.pe_skip_connect, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr)

            self.layer.append(copy.deepcopy(layer))

        if pos_emb in ['conditional_pos', 'conditional_pos_fixed']:
            self.conditional_pos = PosCNN(in_chans=config.hidden_size, embed_dim=config.hidden_size)

    def gate_params(self):
        if self.pos_emb == 'rope':
            # If we are using a single shared gate, return it
            if self.gate_shared:
                return list(self.gate.parameters())

            # Else, find all gates in the layers
            params = []
            for block in self.layer:
                params.extend(list(block.attn.gate.parameters()))

            # if params[0] == params[1]:
            #     # If all gates are the same, return only one
            #     return [params[0]]

            return params
        else:
            return []

    def forward(self, *inputs):
        if self.pe_skip_connect:
            hidden_states, pos_embs = inputs
        else:
            hidden_states = inputs[0]

        attn_weights = []
        for i, layer_block in enumerate(self.layer):
            # Apply CPE to *input* of first block
            if i == 0 and self.pos_emb == 'conditional_pos_fixed':
                hidden_states = self.conditional_pos(hidden_states, self.img_size // self.patch_size[0], self.img_size // self.patch_size[1])

            gate = {'gate': self.gate} if self.gate_shared else {}
            if self.pe_skip_connect:
                hidden_states, weights, _ = layer_block(hidden_states, pos_embs, **gate)
            else:
                hidden_states, weights, _ = layer_block(hidden_states, **gate)

            if i == 0 and self.pos_emb == 'conditional_pos':
                hidden_states = self.conditional_pos(hidden_states, self.img_size // self.patch_size[0], self.img_size // self.patch_size[1])

            attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, pos_emb, fourier_gamma=2.5, compute_decomposed_attn=False, compute_gradbased_attr=False, pos_emb_gate=False, pos_emb_gate_params=None, pos_emb_gate_shared=False):
        super(Transformer, self).__init__()
        self.compute_decomposed_attn = compute_decomposed_attn
        self.compute_gradbased_attr = compute_gradbased_attr

        embeddings_pe_gate = False
        encoder_pe_gate = False
        if pos_emb_gate:
            if pos_emb == 'rope':
                encoder_pe_gate = True
            else:
                embeddings_pe_gate = True
                pos_emb_gate_shared = False

        self.embeddings = Embeddings(config, img_size=img_size, pos_emb=pos_emb, fourier_gamma=fourier_gamma, pos_emb_gate=embeddings_pe_gate, pos_emb_gate_params=pos_emb_gate_params)
        self.encoder = Encoder(config, img_size=img_size, pos_emb=pos_emb, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr, pos_emb_gate=encoder_pe_gate, pos_emb_gate_params=pos_emb_gate_params, pos_emb_gate_shared=pos_emb_gate_shared)

    def gate_params(self):
        return self.embeddings.gate_params() + self.encoder.gate_params()

    def forward(self, input_ids):
        if self.embeddings.pos_emb == 'absolute_learnable_skipconnect':
            embedding_output, pos_embs = self.embeddings(input_ids)
            encoded, attn_weights = self.encoder(embedding_output, pos_embs)
        else:
            embedding_output = self.embeddings(input_ids)
            encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, pos_emb=None, use_drloc=False, drloc_mode="l1", sample_size=32, use_abs=False, fourier_gamma=2.5, compute_decomposed_attn=False, compute_gradbased_attr=False, pos_emb_gate=False, pos_emb_gate_params=None, pos_emb_gate_shared=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, pos_emb, fourier_gamma=fourier_gamma, compute_decomposed_attn=compute_decomposed_attn, compute_gradbased_attr=compute_gradbased_attr, pos_emb_gate=pos_emb_gate, pos_emb_gate_params=pos_emb_gate_params, pos_emb_gate_shared=pos_emb_gate_shared)
        self.head = Linear(config.hidden_size, num_classes)
        self.use_drloc = use_drloc
        self.drloc_mode = drloc_mode
        self.fourier_gamma = fourier_gamma
        self.config = config
        self.pooling = config.classifier

        if self.use_drloc:
            if self.pooling == 'token':
                raise ValueError("DRLOC not compatible with cls_token")

            self.drloc = DenseRelativeLoc(
                in_dim=config.hidden_size,
                out_dim=2 if drloc_mode == "l1" else 14,
                sample_size=sample_size,
                drloc_mode=drloc_mode,
                use_abs=use_abs
            )
            self.to_latent = nn.Identity()
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(config.hidden_size),
                nn.Linear(config.hidden_size, num_classes)
            )

    def gate_params(self):
        return self.transformer.gate_params()

    def patch_size(self):
        return self.config.patches['size']

    def forward(self, x, labels=None):
        self.image_in = x

        if self.use_drloc:
            x, attn_weights = self.transformer(x)
            outs = Munch()

            x_last = x[:, 0:]  # B, L, C
            x_last = x_last.transpose(1, 2)  # [B, C, L]
            B, C, HW = x_last.size()
            H = W = int(math.sqrt(HW))
            x_last = x_last.view(B, C, H, W)  # [B, C, H, W]

            drloc_feats, deltaxy = self.drloc(x_last)
            outs.drloc = [drloc_feats]
            outs.deltaxy = [deltaxy]
            outs.plz = [H]  # plane size

            x = torch.mean(x, dim=1)
            x = self.to_latent(x)
            sup = self.mlp_head(x)
            outs.sup = sup
            return outs, attn_weights

        else:
            x, attn_weights = self.transformer(x)

            if self.pooling == 'token':
                logits = self.head(x[:, 0])
            elif self.pooling == 'avg':
                logits = self.head(torch.mean(x, dim=1))
            else:
                raise ValueError(f"Unknown pooling mode: {self.pooling}")

            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
                return loss
            else:
                return logits, attn_weights

    def load_from(self, weights):
        with torch.no_grad():
            if self.zero_head:
                nn.init.zeros_(self.head.weight)
                nn.init.zeros_(self.head.bias)
            else:
                self.head.weight.copy_(np2th(weights["head/kernel"]).t())
                self.head.bias.copy_(np2th(weights["head/bias"]).t())

            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            self.transformer.embeddings.cls_token.copy_(np2th(weights["cls"]))
            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])
            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)

                if self.classifier == "token":
                    posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
                    ntok_new -= 1
                else:
                    posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = np.concatenate([posemb_tok, posemb_grid], axis=1)
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(weights["conv_root/kernel"], conv=True))
                gn_weight = np2th(weights["gn_root/scale"]).view(-1)
                gn_bias = np2th(weights["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(weights, n_block=bname, n_unit=uname)

    def get_attribution_sources(self):
        """
        Returns model attributes pertaining to bias/information sources for
        gradient-based attribution analysis.
        """
        return {
            'image': 'image_in',
            'pos_emb': analysis.sources.nested_attribute('transformer.embeddings.cached_position_embeddings'),
            # 'pos_emb': analysis.sources.nested_attribute('transformer.embeddings.position_embeddings'),
            'bias': analysis.sources.collect_by_substring('bias'),
            # 'relpos': None, # TODO
        }

    def get_logits(self, outputs):
        """
        Given all model outputs, returns the logits. Used as a wrapper function
        for abstracting away model details in gradient-based analysis.
        """
        return outputs[0]

    def get_shap_interface(self, args, shap_method, input_images, **kwargs):
        def unpack_model_outs(model_outs):
            return model_outs[0]

        if shap_method == 'kernel':
            # Determine attribute of the model that has the PE
            pos_emb_attr = {
                'absolute_learnable': 'transformer.embeddings.position_embeddings',
                'absolute_learnable_concat_equald': 'transformer.embeddings.position_embeddings',
                'absolute_learnable_skipconnect': 'transformer.embeddings.position_embeddings',
                'cape': 'transformer.embeddings.position_embeddings',
                'sinusoid_2d': 'transformer.embeddings.position_embeddings',
                'fourier': 'transformer.embeddings.position_embeddings',
                'irpe': 'transformer.encoder.layer[0].attn.rpe_k.lookup_table_weight',
                'rope': 'transformer.encoder.layer[x].attn.freqs_cis',
                'relative_learnable': [
                    'transformer.encoder.layer[0].attn.rel_pos.emb_h.rel_pos_emb',
                    'transformer.encoder.layer[0].attn.rel_pos.emb_w.rel_pos_emb',
                ],
            }[args.pos_emb]

            # Determine PE and input shape settings
            pe_format_flags = {
                "input_format_spatial": True,
                "input_format_channels_first": True,
                "pos_emb_format": 'B P D',
            }
            if args.pos_emb in ['absolute_learnable',
                                'absolute_learnable_concat_equald',
                                'absolute_learnable_skipconnect',
                                'sinusoid_2d']:
                pass
            elif args.pos_emb == 'cape':
                pe_format_flags["use_hooks"] = True
                pe_format_flags["pos_emb_format"] = 'B H W D'
            elif args.pos_emb in ['relative_learnable']:
                pe_format_flags["pos_emb_format"] = 'D P'
            elif args.pos_emb in ['irpe']:
                pe_format_flags["pos_emb_format"] = 'D1 P D2'
            elif args.pos_emb in ['rope']:
                pe_format_flags["pos_emb_format"] = 'D1 P D2'
                pe_format_flags["use_hooks"] = True
            elif args.pos_emb in ['fourier']:
                pe_format_flags["use_hooks"] = True
            else:
                raise ValueError(f"SHAP not implemented for pos_emb={args.pos_emb}")

            return model_agnostic_interface(self, pos_emb_attr, input_images,
                                            **pe_format_flags,
                                            model_out_unpack_fn=unpack_model_outs,
                                            **kwargs)
        else:
            raise ValueError(f"Unimplemented SHAP method: {shap_method}")


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'testing': configs.get_testing(),
    'red-green': configs.get_red_green_config(),
    'pos': configs.get_pos_config(),
    'cifar': configs.get_cifar_config(),
    'cifar_gani': configs.get_cifar_gani_config(),
    'cifar_ganiv2': configs.get_cifar_ganiv2_config(),
    'cifar_ganiv2_thin': configs.get_cifar_ganiv2_thin_config(),
    'cifar_ganiv2_p2': configs.get_cifar_ganiv2_p2_config(),
    'cifar_ganiv2_p8': configs.get_cifar_ganiv2_p8_config(),
    'cifar_ganiv2_cls': configs.get_cifar_ganiv2_cls_config(),
    'cifar_ganiv2_dropout': configs.get_cifar_ganiv2_dropout_config(),
    'cifar_ganiv2_dropout_cls': configs.get_cifar_ganiv2_dropout_cls_config(),
    'cifar_v1': configs.get_cifar_v1_config(),
    'cifar_v2': configs.get_cifar_v2_config(),
    'cifar_v3': configs.get_cifar_v3_config(),
    'cifar_ganiv2_224px': configs.get_cifar_ganiv2_224px_config(),
    'imagenet-tiny': configs.get_imagenet_tiny_config(),
    'eurosat': configs.get_eurosat_config(),
    'eurosat_p4': configs.get_eurosat_p4_config(),
    'eurosat_dropout': configs.get_eurosat_dropout_config(),
    'eurosat_p4_dropout': configs.get_eurosat_p4_dropout_config(),
    'eurosat_p4_dropout_cls': configs.get_eurosat_p4_dropout_cls_config(),
    'nih': configs.get_nih_config(),
    'nih_dropout': configs.get_nih_dropout_config(),
    'nih_dropout_cls': configs.get_nih_dropout_cls_config(),
    # TODO: original configs using token classification
    'toy': configs.get_toy_config(),
    'toy_cls': configs.get_toy_cls_config(),
}


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = CONFIGS['cifar']
    model = VisionTransformer(config, img_size=224, num_classes=10, pos_emb='irpe', use_drloc=False)
    x = torch.randn((32, 3, 224, 224)).to(device)
    model = model.to(device)
    #summary(model, input_size=(1, 3, 224, 224))
    y = model(x)
    print(y)