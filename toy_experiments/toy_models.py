#
# APPEARANCE
#

import numpy as np
import torch
from torch import nn
import ml_collections
from timm.models.layers import trunc_normal_
import shap

import sys
sys.path.append('..')

from models.vit_modeling import Block
from analysis_shap import model_agnostic_interface

def appearance_block_config(d=4):
    """Returns the vit configuration for the cifar dataset"""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    config.hidden_size = d
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = d
    config.transformer.num_heads = 1
    # config.transformer.num_heads = 2
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    return config

class TriViTalAppearance(nn.Module):
    def __init__(self, d=4, handcrafted_weights=None, pool='avg', norm=True, residual=True, pos_emb='absolute', use_rel_pos=True, pos_add='add', pos_emb_factor=0.1, pos_emb_init='uniform'):
        super().__init__()
        self.pos_add = pos_add
        self.pos_emb_init = pos_emb_init

        self.patch_layer = nn.Conv2d(3, d, kernel_size=1, padding=0)
        if pos_emb == 'absolute':
            if self.pos_emb_init == 'uniform':
                self.pos_embedding = nn.Parameter(torch.randn(1, d, 5, 5) * pos_emb_factor)
            elif self.pos_emb_init == 'trunc_normal':
                self.pos_embedding = nn.Parameter(torch.zeros(1, d, 5, 5))
                trunc_normal_(self.pos_embedding, std=.02)
        elif pos_emb == 'none':
            self.pos_embedding = torch.zeros(1, d, 5, 5)
            # self.pos_embedding = None

        if self.pos_add == 'concat':
            d *= 2

        # self.pos_embedding = nn.Parameter(torch.zeros(1, d, 5, 5))
        # self.pos_embedding.requires_grad_(False)
        # self.block1 = Block(appearance_block_config(d), True, None, False, False, override_attn_out_dim=2)
        self.block1 = Block(config=appearance_block_config(d), img_size=(5, 5), use_rel_pos=use_rel_pos, compute_gradbased_attr=True, norm=norm, residual=residual)
        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool1d([1])
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool1d([1])
        else:
            raise ValueError(f"Unknown pooling type: {pool}")
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(d, 2)

        self.image_in = None
        self.pos_emb_in = None
        self.attn_probs = None
        self.attn_out = None
        self.attn_probs_sliced = []

        if handcrafted_weights is not None:
            params = dict(self.named_parameters())
            training = {n: True for n in params.keys()}
            for name, param in handcrafted_weights.items():
                params[name].data = handcrafted_weights[name]
                training[name] = False

            # print('\nTraining:')
            # for name in training:
            #     print(f"{name}: {training[name]}")

            if all([not training[name] for name in params]):
                print('\nNo parameters to train!')

    def forward(self, x):
        self.image_in = x
        self.image_in.requires_grad = True
        self.image_in.retain_grad()

        def to_attn_format(data):
            return data.permute((0, 2, 3, 1)).view(data.shape[0], -1, data.shape[1])
        def from_attn_format(data):
            return data.permute([0, 2, 1])

        out_sem = self.patch_layer(x)
        self.patch_activations = out_sem
        out_pos = self.pos_embedding
        self.pos_emb_in = out_pos
        if self.pos_embedding is not None:
            if self.pos_add == 'add':
                out = out_sem + out_pos
            elif self.pos_add == 'concat':
                out_pos = out_pos.expand(out_sem.shape[0], -1, -1, -1)
                out = torch.cat([out_sem, out_pos], dim=1)
        else:
            out = out_sem
        self.token_activations = out

        out = to_attn_format(out)
        out, self.attn_probs, self.attn_probs_sliced = self.block1(out)

        # To make feature maps differentiable *per sample* we
        # need to include the slicing per sample in the graph. Otherwise
        # autograd.grad() complains that the sliced tensors are not in the
        # graph when we later compute the image/position attributions wrt
        # the feature map.
        self.attn_out_sliced = []
        # Shape of feature map: [batch_size, tokens, channels]
        B, D, C = out.shape
        for i in range(B):
            # What we want: first dimension should be [num_heads], so
            # autograd.grad() can compute the "batched" vector-Jacobian for each
            # head, as we will use the "batched" mode of grad() where the first
            # dimension needs to contain the "batches".
            num_heads = self.block1.attn.num_attention_heads
            grad_compat_slice = out[i].permute((1, 0)).reshape(num_heads, -1, D)
            grad_compat_slice.retain_grad()
            self.attn_out_sliced.append(grad_compat_slice)
        out = torch.stack(self.attn_out_sliced).reshape(B, C, D).permute((0, 2, 1))

        out = from_attn_format(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out

#
# ABSOLUTE POSITION & MIXED POSITION
#

from models.pos_embeddings import get_raw_coordinate
from models.learnable_sinusoid import LearnableSinusoidPositionalEncoding

def abspos_block_config(d=4, mlp_d=None, n_heads=1):
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (1, 1)})
    config.hidden_size = d
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = mlp_d if mlp_d is not None else d
    config.transformer.num_heads = n_heads
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.0
    return config

class TriViTalAbsolutePosition(nn.Module):
    def __init__(self, handcrafted_weights=None, n_classes=2, d=4, n_blocks=1, n_heads=1,
                 size=6, patch_size=1, pos_emb='absolute', use_rel_pos=True,
                 pos_add='add', pool='avg', norm=True, residual=True,
                 pos_emb_factor=0.1, pos_emb_init='uniform', input_d=3,
                 compute_gradbased_attr=True, act_fn='gelu', norm_fn='layernorm',
                 mlp_d=None):
        super().__init__()
        self.pos_emb = pos_emb
        self.pos_add = pos_add
        self.pos_emb_init = pos_emb_init
        self.compute_gradbased_attr = compute_gradbased_attr

        if pool == 'cls':
            raise NotImplementedError("CLS token pooling not implemented for this model")

        # Flag to mark whether the PE has been set by some external method to
        # the parameter self.pos_emb_expanded (if True) or not (if False), in
        # which case self.pos_embedding needs to be expanded to all samples as
        # per the default ViT method.
        self.pe_per_sample = False

        # Flag to mark whether we are to store the image & PE tokens for later
        # analysis
        self.store_tokens = False
        self.image_tokens = None
        self.pe_tokens = None

        if self.pos_add == 'add':
            self.patch_layer = nn.Conv2d(input_d, d, kernel_size=patch_size, padding=0, stride=patch_size)
        elif self.pos_add == 'concat':
            self.patch_layer = nn.Conv2d(input_d, d - 1, kernel_size=patch_size, padding=0, stride=patch_size)
        elif self.pos_add == 'concat_equald':
            if d % 2 != 0:
                raise ValueError(f"d must be even when using pos_add=concat_equald, got d={d}")
            self.patch_layer = nn.Conv2d(input_d, d // 2, kernel_size=patch_size, padding=0, stride=patch_size)

        if patch_size > 1:
            size = (size - patch_size) // patch_size + 1

        if self.pos_emb == 'absolute':
            if self.pos_add == 'add':
                shape = (1, d, size, size)
            elif self.pos_add == 'concat':
                shape = (1, 1, size, size)
            elif self.pos_add == 'concat_equald':
                shape = (1, d // 2, size, size)
            if self.pos_emb_init == 'uniform':
                self.pos_embedding = nn.Parameter(torch.randn(*shape) * pos_emb_factor)
            elif self.pos_emb_init == 'trunc_normal':
                self.pos_embedding = nn.Parameter(torch.zeros(*shape))
                trunc_normal_(self.pos_embedding, std=pos_emb_factor)
        elif self.pos_emb == 'sinusoid_learnable_new':
            self.pos = nn.Parameter(get_raw_coordinate(size * size), requires_grad=False)
            self.sin_model = LearnableSinusoidPositionalEncoding(2, d // 2, d, 4 * d, initialize='sinusoid')
            self.pos_embedding = None
        elif self.pos_emb == 'none':
            self.pos_embedding = None
        else:
            raise NotImplementedError(f"Unknown pos_emb: {pos_emb}")
        self.size = size

        self.blocks = []
        for i in range(n_blocks):
            block = Block(abspos_block_config(d, n_heads=n_heads, mlp_d=mlp_d), img_size=(size,size), use_rel_pos=use_rel_pos, compute_gradbased_attr=compute_gradbased_attr, norm=norm, residual=residual, act_fn=act_fn, norm_fn=norm_fn)
            self.blocks.append(block)
            self.add_module(f'block{i}', block)

        if pool == 'avg':
            self.pool = nn.AdaptiveAvgPool1d([1])
        elif pool == 'max':
            self.pool = nn.AdaptiveMaxPool1d([1])
        else:
            raise ValueError(f"Unknown pooling type: {pool}")
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(d, n_classes)
        self.patch_activations = None
        self.token_activations = None
        self.attn_activations = None

        if handcrafted_weights is not None:
            params = dict(self.named_parameters())
            training = {n: True for n in params.keys()}
            for name, param in handcrafted_weights.items():
                if params[name].data.shape != handcrafted_weights[name].shape:
                    print(f"Shape mismatch on {name}: init as {params[name].data.shape}, set to {handcrafted_weights[name].shape}")
                params[name].data = handcrafted_weights[name]
                training[name] = False

            # print('\nTraining:')
            # for name in training:
            #     print(f"{name}: {training[name]}")

            if all([not training[name] for name in params]):
                print('\nNo parameters to train!')

        self.pos_emb_in = None

    def forward(self, x, pos_emb=None):
        self.image_in = x
        if self.compute_gradbased_attr:
            self.image_in.requires_grad = True
            self.image_in.retain_grad()

        def to_attn_format(data):
            return data.permute((0, 2, 3, 1)).view(data.shape[0], -1, data.shape[1])
        def from_attn_format(data):
            return data.permute([0, 2, 1])
        def from_attn_format_2d(data):
            return data.permute([0, 2, 1]).view(data.shape[0], -1, self.size, self.size)

        out_sem = self.patch_layer(x)
        self.patch_activations = out_sem

        # If PE has not been set by some external method, compute the PE tensor
        # as per default ViT methodology, by expanding from self.pos_embedding
        if not self.pe_per_sample:
            # Learnable sinusoid
            if self.pos_emb == 'sinusoid_learnable_new':
                self.pos_embedding = from_attn_format_2d(self.sin_model(self.pos))
                if self.compute_gradbased_attr:
                    self.pos_embedding.retain_grad()

            if pos_emb is not None:
                out_pos = pos_emb
            else:
                out_pos = self.pos_embedding

            self.pos_emb_in = out_pos.data
            self.pos_emb_expanded = out_pos.expand(out_sem.shape[0], -1, -1, -1)

        if self.store_tokens:
            self.image_tokens = out_sem
            self.pe_tokens = out_pos

        if self.pos_emb == 'none':
            out = out_sem
        elif self.pos_add == 'add':
            out = out_sem + self.pos_emb_expanded
        elif self.pos_add == 'concat' or self.pos_add == 'concat_equald':
            # out_pos = out_pos.expand(out_sem.shape[0], -1, -1, -1)
            out = torch.cat([out_sem, self.pos_emb_expanded], dim=1)
        else:
            raise NotImplementedError(f"Unknown pos_add: {self.pos_add}")
        self.token_activations = out

        out = to_attn_format(out)

        for i in range(len(self.blocks)):
            out, self.attn_probs, self.attn_probs_sliced = self.blocks[i](out)

        # To make feature maps differentiable *per sample* we
        # need to include the slicing per sample in the graph. Otherwise
        # autograd.grad() complains that the sliced tensors are not in the
        # graph when we later compute the image/position attributions wrt
        # the feature map.
        self.attn_out_sliced = []
        # Shape of feature map: [batch_size, tokens, channels]
        B, D, C = out.shape
        for i in range(B):
            # What we want: first dimension should be [num_heads], so
            # autograd.grad() can compute the "batched" vector-Jacobian for each
            # head, as we will use the "batched" mode of grad() where the first
            # dimension needs to contain the "batches".
            num_heads = self.blocks[-1].attn.num_attention_heads
            grad_compat_slice = out[i].permute((1, 0)).reshape(num_heads, -1, D)
            # Build in this check for compatibility with KernelSHAP which is run
            # with torch.no_grad()
            if self.compute_gradbased_attr:
                if grad_compat_slice.requires_grad:
                    grad_compat_slice.retain_grad()
            self.attn_out_sliced.append(grad_compat_slice)
        out = torch.stack(self.attn_out_sliced).reshape(B, C, D).permute((0, 2, 1))

        out = from_attn_format(out)

        out = self.pool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out

    def set_pe_per_sample(self, value):
        self.pe_per_sample = True
        self.pos_emb_expanded.data = value

    def unset_pe_per_sample(self):
        self.pe_per_sample = False

    def get_shap_interface(self, shap_method, input_images, **kwargs):
        if shap_method == 'kernel':
            return model_agnostic_interface(self, 'pos_embedding', input_images, **kwargs)
        elif shap_method == 'deep':
            assert 'on_cpu' not in kwargs or not kwargs['on_cpu'], "DeepSHAP does not support CPU mode"
            assert 'batch_size' not in kwargs or kwargs['batch_size'] is None, "DeepSHAP does not support batch_size"
            assert 'spatial_features' not in kwargs or not kwargs['spatial_features'], "DeepSHAP does not support spatial_features"
            return self.deep_shap_interface(input_images, **kwargs)
        else:
            raise ValueError(f"Unknown SHAP method: {shap_method}")

    def kernel_shap_interface(self, input_images):
        """Returns various methods to implement an interface of this model to be
        compatible with the KernelSHAP implementation in the shap library.

        KernelExplainer takes a model that takes NumPy inputs, takes background
        data in NumPy format, and outputs NumPy SHAP values.

        This interface also takes a position embedding as input,
        and this function returns the interface as well as the position
        embeddings to use as input, representing the model state at the time
        this function was called.

        This interface makes the model to run in "PE per sample" mode, where
        the PE tensor is set to the provided value for each sample in the input,
        instead of expanding a single value over all samples at inference time.
        This is necessary because KernelSHAP requires input features for each
        sample.
        """
        x_shape = input_images.shape
        # NOTE: though input_images are given to define the interface, use of
        # the interface may be done with varying number of samples, so we cannot
        # use the predetermined shapes to set the number of samples in the
        # interface.
        _, C, H, W = x_shape
        pe_shape = self.pos_embedding.data.shape
        Np, D, Hp, Wp = pe_shape
        # The interface implemented here assumes the model's PE parameter does
        # not have a sample dimension.
        assert Np == 1, f"This interface assumes PE parameter does not have a sample dimension, but got {Np} != 1"

        # Move to CUDA if available
        cuda = False
        if torch.cuda.is_available():
            cuda = True
            self.cuda()
            print("Using CUDA in KernelSHAP inference.")

        def shap_model(x):
            """
            Wrapper around model for the model to be compatible with
            `shap.KernelExplainer`s internal usage of the model. Takes 2D NumPy
            tensor of input features, converts to 4D Torch tensor and
            "per-sample" PE for them to be inputtable to the Torch model, and
            runs model in "PE per-sample" mode, returning the result of the
            inference as a NumPy tensor representing class logits.
            """
            # (N, C*H*W + D*H*W) -> (N, C, H, W) and (N, D, Hp, Wp)
            x = torch.tensor(x)
            if cuda:
                x = x.cuda()
            image = x[:, :C*H*W].reshape(x.shape[0], C, H, W)
            pos_emb = x[:, C*H*W:].reshape(x.shape[0], D, Hp, Wp)
            # We make the model skip the expansion of the true PE parameter, and
            # set the "per sample" PE directly instead
            self.set_pe_per_sample(pos_emb)
            result = self.forward(image).detach().cpu().numpy()
            self.unset_pe_per_sample()
            return result

        def explainer_input(x, pos_embedding, background=False):
            """
            Takes 4D Torch tensor and original PE tensor (shape (1, D, Hp, Wp),
            not "per sample"), converts to 2D NumPy tensor representing all
            model input features, in a format compatible with
            `shap.KernelExplainer()`.

            If background=True, we shuffle the PE tokens along H*W dimensions
            (not D!), so that each D-dimensional PE token is intact but randomly
            displaced.
            """
            # image = (N, C, H, W) -> (N, C*H*W)
            image = x.numpy().reshape(x.shape[0], -1)
            # pos_embedding = (1, D, Hp, Wp) -> (N, D*Hp*Wp)
            if cuda:
                pos_embedding = pos_embedding.cpu()
            pos_emb = pos_embedding.expand(x.shape[0], -1, -1, -1).clone().reshape(x.shape[0], -1).numpy()

            if background:
                _, D, _, _ = pos_embedding.shape
                pos_emb = pos_emb.reshape((x.shape[0], D, -1)).transpose((0, 2, 1))
                for i in range(x.shape[0]):
                    pos_emb_i = pos_emb[i]
                    np.random.shuffle(pos_emb_i)
                    pos_emb[i] = pos_emb_i
                pos_emb = pos_emb.transpose((0, 2, 1)).reshape((x.shape[0], -1))

            return np.concatenate((image, pos_emb), axis=1)

        def shap_output(shap_values):
            """
            Reshape NumPy array outputs of `shap.KernelExplainer:shap_values()`
            to (N, Cin, H, W, Cout) for image and (N, D, H, W, Cout) for PE, for
            visualization.

            TODO: PE shap values should be identical (up to estimation error)
            for all values; build in a check for this.
            """
            N, HWCD, Cout = shap_values.shape
            assert HWCD == C * H * W + D * Hp * Wp, \
                f'Error in SHAP values shape: {HWCD} != C={C} * H={H} * W={W} + D={D} * Hp={Hp} * Wp={Wp}'
            cutoff = C * H * W
            image = shap_values[:,:cutoff].reshape(N, C, H, W, Cout)
            pos_emb = shap_values[:,cutoff:].reshape(N, D, Hp, Wp, Cout)
            return image, pos_emb

        explainer_kwargs = {"silent": True}

        # For PE, we return only the value of the expanded PE tensor, instead of
        # the reference to the tensor itself. This is sufficient because
        # KernelSHAP only needs to work with the values, and the other interface
        # methods we provide know how to set the provided values to the right
        # tensor in the model.
        return shap.KernelExplainer, shap_model, explainer_input, explainer_kwargs, shap_output, self.pos_embedding.data

    def deep_shap_interface(self, input_images):
        """Returns various methods to implement an interface of this model to be
        compatible with the DeepSHAP implementation in the shap library.

        DeepExplainer takes a model that takes Torch inputs, takes background
        data in Torch format, and outputs NumPy SHAP values.

        This interface also takes a position embedding as input,
        and this function returns the interface as well as the position
        embeddings to use as input, representing the model state at the time
        this function was called.

        This interface makes the model to run in "PE per sample" mode, where
        the PE tensor is set to the provided value for each sample in the input,
        instead of expanding a single value over all samples at inference time.
        This is necessary because DeepSHAP requires input features for each
        sample.
        """
        x_shape = input_images.shape
        # NOTE: though input_images are given to define the interface, use of
        # the interface may be done with varying number of samples, so we cannot
        # use the predetermined shapes to set the number of samples in the
        # interface.
        _, C, H, W = x_shape
        pe_shape = self.pos_embedding.data.shape
        Np, D, Hp, Wp = pe_shape
        # The interface implemented here assumes the model's PE parameter does
        # not have a sample dimension.
        assert Np == 1, f"This interface assumes PE parameter does not have a sample dimension, but got {Np} != 1"

        # Move to CUDA if available
        cuda = False
        if torch.cuda.is_available():
            cuda = True
            self.cuda()
            print("Using CUDA in DeepSHAP inference.")

        # The suppied model must support `model.named_parameters()` to be
        # detected as a PyTorch model, so wrapping with a function (as done for
        # KernelSHAP) is not possible.
        shap_model = ShapModel(self, cuda, x_shape, pe_shape)

        def explainer_input(x, pos_embedding, background=False):
            """
            Takes 4D Torch tensor and original PE tensor (shape (1, D, Hp, Wp),
            not "per sample"), converts to 2D Torch tensor representing all
            model input features, in a format compatible with
            `shap.DeepExplainer()`.

            Note that the position embeddings are passed to this method instead
            of being inferred from the data `x`. This is possible because
            position embeddings are model biases, not dependent on input data.

            If background=True, we shuffle the PE tokens along H*W dimensions
            (not D!), so that each D-dimensional PE token is intact but randomly
            displaced.
            """
            # image = (N, C, H, W) -> (N, C*H*W)
            image = x.reshape(x.shape[0], -1)
            # pos_embedding = (1, D, Hp, Wp) -> (N, D*Hp*Wp)
            if cuda:
                pos_embedding = pos_embedding.cpu()
            pos_emb = pos_embedding.expand(x.shape[0], -1, -1, -1).clone().reshape(x.shape[0], -1)

            if background:
                _, D, _, _ = pos_embedding.shape
                pos_emb = pos_emb.reshape(x.shape[0], D, -1).permute(0, 2, 1)
                for i in range(x.shape[0]):
                    pos_emb[i] = pos_emb[i][torch.randperm(pos_emb[i].shape[0])]
                pos_emb = pos_emb.permute(0, 2, 1).reshape(x.shape[0], -1)

            return torch.cat((image, pos_emb), dim=1)

        def shap_output(shap_values):
            """
            Reshape NumPy array outputs of `shap.DeepExplainer:shap_values()`
            to (N, Cin, H, W, Cout) for image and (N, D, H, W, Cout) for PE, for
            visualization.

            TODO: PE shap values should be identical (up to estimation error)
            for all values; build in a check for this.
            """
            N, HWCD, Cout = shap_values.shape
            assert HWCD == C * H * W + D * Hp * Wp, \
                f'Error in SHAP values shape: {HWCD} != C={C} * H={H} * W={W} + D={D} * Hp={Hp} * Wp={Wp}'
            cutoff = C * H * W
            image = shap_values[:,:cutoff].reshape(N, C, H, W, Cout)
            pos_emb = shap_values[:,cutoff:].reshape(N, D, Hp, Wp, Cout)
            return image, pos_emb

        explainer_kwargs = {}

        # For PE, we return only the value of the expanded PE tensor, instead of
        # the reference to the tensor itself. This is sufficient because
        # KernelSHAP only needs to work with the values, and the other interface
        # methods we provide know how to set the provided values to the right
        # tensor in the model.
        return shap.DeepExplainer, shap_model, explainer_input, explainer_kwargs, shap_output, self.pos_embedding.data


class ShapModel(nn.Module):
    def __init__(self, model, cuda, x_shape, pe_shape):
        """Wrapper around model for the model to be compatible with
        `shap.DeepExplainer`s internal usage of the model. Takes 2D Torch tensor
        of input features, converts to 4D Torch tensor and "per-sample" PE for
        them to be inputtable to the Torch model, and runs model in "PE
        per-sample" mode, returning the result of the inference as a Torch
        tensor representing class logits.

        The suppied model must support `model.named_parameters()` to be detected
        as a PyTorch model, so wrapping with a function is not possible.
        """
        super().__init__()
        self.model = model
        self.use_cuda = cuda
        self.x_shape = x_shape
        self.pe_shape = pe_shape

    def parameters(self):
        return self.model.parameters()

    def named_parameters(self):
        return self.model.named_parameters()

    def forward(self, x):
        # Turn off gradbased attribution if set
        gradbased_attr = self.model.compute_gradbased_attr
        self.model.compute_gradbased_attr = False

        _, C, H, W = self.x_shape
        Np, D, Hp, Wp = self.pe_shape
        # (N, C*H*W + D*H*W) -> (N, C, H, W) and (N, D, Hp, Wp)
        if self.use_cuda:
            x = x.cuda()
        image = x[:, :C*H*W].reshape(x.shape[0], C, H, W)
        pos_emb = x[:, C*H*W:].reshape(x.shape[0], D, Hp, Wp)
        # We pass PE as input, so that SHAP values are computed for it
        result = self.model.forward(image, pos_emb)

        # Restore gradbased attribution setting
        self.model.compute_gradbased_attr = gradbased_attr

        return result