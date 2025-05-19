"""
Google Research ViT
"""
import math
import os
import timm
import tensorflow as tf
import torch
from torch import nn
import torchvision
import torchxrayvision as xrv

from models.cape import CAPE1d
from models.fourier import LearnableFourierPositionalEncoding
from models.pos_embeddings import get_raw_coordinate, get_sinusoidal_2d
from published_models import BasePublishedModel
import analysis
from published_models.model_implementations.google_research_vit.weights import WEIGHTS
import datasets
from analysis_shap import model_agnostic_interface
from models.vit_modeling import PEGate

from published_models.model_implementations.google_research_vit.vision_transformer_relpos import \
    vit_relpos_small_patch16_224, vit_relpos_medium_patch16_224, vit_relpos_base_patch16_224


VERSIONS = {
    'Ti/16': 'vit_tiny_patch16_224',
    'Ti/16-384': 'vit_tiny_patch16_384',
    'B/16': 'vit_base_patch16_224',
    'B/16-384': 'vit_base_patch16_384',
    'B/32': 'vit_base_patch32_224',
    'B/32-384': 'vit_base_patch32_384',
    'S/16': 'vit_small_patch16_224',
    'S/16-384': 'vit_small_patch16_384',
    'S/32-384': 'vit_small_patch32_384',
}

class GoogleResearchViT(BasePublishedModel):
    # NOTE: provide a list of testable configurations
    CONFIGURATIONS = [
        # CIFAR-100
        {'model_version': 'Ti/16', 'model_weights': 'Ti/16-224-imagenet21k-cifar100', 'num_classes': 100, 'reported_val_acc': 0.8801},
        {'model_version': 'B/32', 'model_weights': 'B/32-224-imagenet21k-cifar100', 'num_classes': 100, 'reported_val_acc': 0.8749},
        {'model_version': 'S/32-384', 'model_weights': 'S/32-384-imagenet21k-cifar100', 'num_classes': 100, 'reported_val_acc': 0.8749},
        # Oxford-IIIT Pets
        {'model_version': 'Ti/16-384', 'model_weights': 'Ti/16-384-imagenet21k-oxfordpets', 'num_classes': 37, 'reported_val_acc': 0.911420},
        {'model_version': 'B/32-384', 'model_weights': 'B/32-384-imagenet21k-oxfordpets', 'num_classes': 37, 'reported_val_acc': 0.929662},
        {'model_version': 'S/32-384', 'model_weights': 'S/32-384-imagenet21k-oxfordpets', 'num_classes': 37, 'reported_val_acc': 0.923412},
    ]

    def __init__(self, version, weights=None, num_classes=1000, pos_emb='default', reset_pe=False, pos_emb_gate=False, pos_emb_gate_params=None):
        super().__init__()

        # For available model names, see here:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer_hybrid.py
        avg_pool = False
        if 'avg' in version:
            version = version.replace('-avg', '')
            avg_pool = True

        timm_name = VERSIONS[version]
        if 'rpe' in pos_emb:
            if not avg_pool:
                raise NotImplementedError()
            timm_name = timm_name.replace('vit_', 'vit_relpos_')
            model_args = dict(num_classes=num_classes, rel_pos_type='bias', global_pool='avg', class_token=False)
            if timm_name == 'vit_relpos_small_patch16_224':
                model = vit_relpos_small_patch16_224(**model_args)
            elif timm_name == 'vit_relpos_medium_patch16_224':
                model = vit_relpos_medium_patch16_224(**model_args)
            elif timm_name == 'vit_relpos_base_patch16_224':
                model = vit_relpos_base_patch16_224(**model_args)
            # model = timm.create_model(timm_name, num_classes=num_classes, rel_pos_type='bias')
        else:
            if avg_pool:
                model = timm.create_model(timm_name, num_classes=num_classes, global_pool='avg', class_token=False)
            else:
                model = timm.create_model(timm_name, num_classes=num_classes)

        self.abs_embed = False
        self.rel_embed = False
        if pos_emb in ['default', 'absolute_learnable', 'sinusoid_2d', 'cape', 'fourier']:
            self.abs_embed = True
            self.rel_embed = False

        # Trick to initialize a subclass without having to reinitialize
        if self.rel_embed:
            self.model = object.__new__(TimmVisionTransformerRelPos)
        else:
            self.model = object.__new__(TimmVisionTransformer)
        self.model.__dict__ = model.__dict__
        self.model.setup(abs_embed=self.abs_embed, rel_embed=self.rel_embed,
                         ape_method=pos_emb, reset_ape=reset_pe,
                         pos_emb_gate=pos_emb_gate, pos_emb_gate_params=pos_emb_gate_params)

        # Select a value from "adapt_filename" above that is a fine-tuned checkpoint.
        self.weights_metadata = None
        if weights is not None:
            self.weights_metadata = WEIGHTS[weights]
            filename = self.weights_metadata['filename']

            # Non-default checkpoints need to be loaded from local files.
            ckpt_path = os.path.join(os.path.dirname(__file__), f'{filename}.npz')
            if not tf.io.gfile.exists(ckpt_path):
                # TensorFlow GS certificate fix. See known issues.
                os.environ['CURL_CA_BUNDLE'] = "/etc/ssl/certs/ca-certificates.crt"
                tf.io.gfile.copy(f'gs://vit_models/augreg/{filename}.npz', ckpt_path)
            timm.models.load_checkpoint(self.model, ckpt_path)

        if reset_pe:
            self.model.reset_pos_emb(device=None, experiment=None)

    def forward(self, x):
        return self.model(x)

    def patch_size(self):
        """
        Returns:
            tuple(int): The patch size of the model, in (y, x).
        """
        return self.model.patch_embed.patch_size

    def get_attribution_sources(self):
        """
        Returns model attributes pertaining to bias/information sources for
        gradient-based attribution analysis.

        Returns:
            dict: A dictionary mapping source names `image`, `bias` and
                optionally `pos_emb` and `relpos` to model attributes.
        """
        sources = {
            'image': analysis.sources.nested_attribute('model.image_in'),
            'bias': analysis.sources.collect_by_substring('bias', exclude=['relative_position_bias_table']),
        }
        if self.abs_embed:
            sources['pos_emb'] = analysis.sources.nested_attribute('model.pos_embed')
        if self.rel_embed:
            sources['relpos'] = analysis.sources.collect_by_substring('relative_position_bias_table')
        return sources

    def get_logits(self, outputs):
        """
        Given all model outputs, returns the logits. Used as a wrapper function
        for abstracting away model details in gradient-based analysis.
        """
        return outputs

    @staticmethod
    def dataset_policy(experiment, args):
        """
        Returns a dataset policy for the model. If this method is implemented,
        these dataset definitions will be used by train.py. Otherwise, one needs
        to specify a --dataset_policy that is implemented in
        datasets/datasets.py.

        NOTE: a `@staticmethod` is just a method implemented for the class
        rather than the class instance. The only difference with a regular class
        method is that there is no `self` attribute. This is because nothing can
        (or should) be saved in `self`, but rather this is code just executed
        from `train.py`. You *can* assign things to `experiment`, which is the
        instance of the `Experiment` class in `train.py`.
        """
        # Original train transforms, in JAX, for reference.
        # Source: https://github.com/google-research/vision_transformer/blob/main/vit_jax/input_pipeline.py#L198
        """
        channels = im.shape[-1]
        begin, size, _ = tf.image.sample_distorted_bounding_box(
            tf.shape(im),
            tf.zeros([0, 0, 4], tf.float32),
            area_range=(0.05, 1.0),
            min_object_covered=0,  # Don't enforce a minimum area.
            use_image_if_no_bounding_boxes=True)
        im = tf.slice(im, begin, size)
        # Unfortunately, the above operation loses the depth-dimension. So we
        # need to restore it the manual way.
        im.set_shape([None, None, channels])
        im = tf.image.resize(im, [image_size, image_size])
        if tf.random.uniform(shape=[]) > 0.5:
            im = tf.image.flip_left_right(im)
        """
        if args.dataset_policy in ['own', 'GoogleResearchViT']:
            args.dataset_policy = 'GoogleResearchViT'
        elif args.dataset_policy == 'own-vit':
            return datasets.own_vit.own_vit_policy(experiment, args)
        else:
            raise NotImplementedError()

        # Same training transforms, in PyTorch
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(
                (args.internal_img_size, args.internal_img_size),
                scale=(0.05, 1.0),
            ),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            # Original implementation normalizes to [-1, 1] instead of [0,
            # 1] as is default in PyTorch
            torchvision.transforms.Normalize(0.5, 0.5),
        ])

        transform_val = torchvision.transforms.Compose([
            torchvision.transforms.Resize((args.internal_img_size,args.internal_img_size)),
            torchvision.transforms.ToTensor(),
            # Original implementation normalizes to [-1, 1] instead of [0,
            # 1] as is default in PyTorch
            torchvision.transforms.Normalize(0.5, 0.5),
        ])

        if args.dataset == 'eurosat':
            transform_train = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                datasets.datasets.EuroSATTrainTransform(),
                torchvision.transforms.Resize((args.internal_img_size, args.internal_img_size)),
                # Original implementation normalizes to [-1, 1] instead of [0,
                # 1] as is default in PyTorch
                torchvision.transforms.Normalize(0.5, 0.5),
            ])
            transform_val = torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Resize((args.internal_img_size, args.internal_img_size)),
                # Original implementation normalizes to [-1, 1] instead of [0,
                # 1] as is default in PyTorch
                torchvision.transforms.Normalize(0.5, 0.5),
            ])

        if args.dataset in ['nih', 'nih_google']:
            # Remove resizing and random horizontal flip
            transform_train = [
                xrv.datasets.ToPILImage(),
                # NOTE: we choose not to augment with geometric transformations
                # as we are doing this task to emphasize absolute location bias
                # torchvision.transforms.RandomAffine(45, translate=(0.15, 0.15),
                #                                     scale=(0.85, 1.15)),
                torchvision.transforms.ToTensor(),
                # Inflate torch Tensor to 3 channels, for compatibility with
                # pretrained models
                datasets.grayscale_to_rgb_transform.GrayscaleToRGBTensor(),
            ]
            transform_val = [
                xrv.datasets.ToPILImage(),
                torchvision.transforms.ToTensor(),
                # Inflate torch Tensor to 3 channels, for compatibility with
                # pretrained models
                datasets.grayscale_to_rgb_transform.GrayscaleToRGBTensor(),
            ]

            if args.internal_img_size != 224:
                transform_train.insert(1, torchvision.transforms.Resize((args.internal_img_size, args.internal_img_size)))
                transform_val.insert(1, torchvision.transforms.Resize((args.internal_img_size, args.internal_img_size)))
            transform_train = torchvision.transforms.Compose(transform_train)
            transform_val = torchvision.transforms.Compose(transform_val)

        datasets.setup_loaders(experiment, args, transform_train, transform_val)

    def gate_params(self):
        return self.model.gate_params()

    @staticmethod
    def finetune_policy(experiment, args):
        """
        Sets the arguments related to finetuning to the values used in the
        original paper.

        For this model, these are the sources:
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/argss/common.py#L20
        https://github.com/google-research/vision_transformer/blob/main/vit_jax/argss/augreg.py#L42
        """
        if args.finetune_policy == 'own':
            args.finetune_policy = 'GoogleResearchViT'
        if args.finetune_policy == 'none':
            # No changes to optimization settings
            return
        elif args.finetune_policy not in ['GoogleResearchViT', 'GoogleResearchViT-DeiT']:
            raise NotImplementedError()

        args.clip = 1.0
        # NOTE: Our GPUs don't support bf16
        # args.precision = 'bf16'
        args.precision = '16'
        args.opt = 'sgd-momentum'
        args.accumulate_gradients = 32
        if args.training:
            args.batch_size = 128
        args.lr = 0.03
        args.label_smoothing = 0.0

        # NOTE: we did not run with these values, but according to the DeiT paper they should be used
        if args.finetune_policy == 'GoogleResearchViT-DeiT':
            args.weight_decay = 0.3

        if args.model_version in ['B/16-384', 'S/16-384']:
            args.accumulate_gradients = 128
            args.batch_size = 32

        if args.dataset in ['cifar10', 'cifar100']:
            dataset_size = 50000
            # Custom, shorter policy
            steps = 2000
            steps_per_epoch = dataset_size / (args.batch_size * args.accumulate_gradients)
            args.n_epochs = int(steps // steps_per_epoch)
            args.warmup_epochs = int(max(100 // steps_per_epoch, 1))
        elif args.dataset == 'flowers102':
            dataset_size = 1020
            # Reduce gradient accumulation because it breaks when the
            # accumulation is spread over multiple epochs
            args.accumulate_gradients = args.accumulate_gradients // 4
            args.lr = args.lr / 4.
            args.check_val_every_n_epoch = 10
            steps = 500
            steps_per_epoch = dataset_size / (args.batch_size * args.accumulate_gradients)
            args.n_epochs = int(steps // steps_per_epoch)
            args.warmup_epochs = int(max(100 // steps_per_epoch, 1))
        elif args.dataset == 'imagenet' or args.dataset == 'imagenette':
            # TODO: we make no changes to the default policy
            pass
        elif args.dataset == 'eurosat':
            # TODO: there is no established policy for this dataset, AFAIK
            pass
        elif args.dataset == 'oxfordpets':
            # TODO: there is no established policy for this dataset, AFAIK
            pass
        elif args.dataset == 'nih':
            # TODO: there is no established policy for this dataset, AFAIK
            pass
        elif args.dataset == 'toy':
            pass
        else:
            raise NotImplementedError(f'Unknown dataset {args.dataset}')

    def get_shap_interface(self, args, shap_method, input_images, **kwargs):
        if shap_method == 'kernel':
            return model_agnostic_interface(self, 'model.pos_embed', input_images,
                                            pos_emb_format='B P D',
                                            input_format_spatial=True,
                                            input_format_channels_first=True,
                                            **kwargs)
        else:
            raise ValueError(f"Unimplemented SHAP method: {shap_method}")


class TimmVisionTransformerRelPos(timm.models.vision_transformer_relpos.VisionTransformerRelPos):
    """
    Wrapper around VisionTransformerRelPos to expose the attribution sources.

    Can do rpe and ape+rpe.
    """
    image_in = None

    def setup(self, abs_embed, rel_embed, pos_emb_gate=False, pos_emb_gate_params=None):
        self.abs_embed = abs_embed
        self.rel_embed = rel_embed

        assert self.rel_embed, "TimmVisionTransformerRelPos does not support not using relative position embeddings"

        if self.abs_embed:
            # Add absolute embedding
            num_patches = self.patch_embed.num_patches
            embed_len = num_patches + self.num_prefix_tokens
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.embed_dim) * .02)

        self.gate = nn.Identity()
        if pos_emb_gate:
            assert not self.rel_embed, "pos_emb_gate is not compatible with relative position embeddings"
            assert self.abs_embed, "pos_emb_gate is only compatible with absolute position embeddings"
            self.gate = PEGate(init_value=pos_emb_gate_params['init_value'], sigmoid=pos_emb_gate_params['sigmoid'])

    # Overwriting forward pass to save input as attribute and apply
    # requires_grad to input image
    def forward_features(self, x):
        self.image_in = x
        x.requires_grad_(True)

        # if self.rel_embed:
        x = self.patch_embed(x)
        # NOTE: added for compat with abs_token
        if self.abs_embed:
            x = x + self.gate(self.pos_embed)
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        shared_rel_pos = self.shared_rel_pos.get_bias() if self.shared_rel_pos is not None else None
        for blk in self.blocks:
            # if self.grad_checkpointing and not torch.jit.is_scripting():
            #     x = checkpoint(blk, x, shared_rel_pos=shared_rel_pos)
            # else:
            x = blk(x, shared_rel_pos=shared_rel_pos)
        x = self.norm(x)
        return x

class TimmVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Wrapper around VisionTransformer to expose the attribution sources.

    Can do none and ape.
    """
    image_in = None

    def setup(self, abs_embed, rel_embed, ape_method='absolute_learnable', reset_ape=False, pos_emb_gate=False, pos_emb_gate_params=None, experiment=None):
        self.abs_embed = abs_embed
        self.rel_embed = rel_embed
        self.ape_method = ape_method
        self.reset_ape = reset_ape
        self.pos_emb_gate = pos_emb_gate
        self.pos_emb_gate_params = pos_emb_gate_params

        assert not self.rel_embed, "TimmVisionTransformer does not support relative position embeddings"

        if not self.reset_ape and self.abs_embed and self.ape_method != 'absolute_learnable':
            raise NotImplementedError(f"PE must be reset when using {self.ape_method} method")

        self.gate = nn.Identity()
        if self.pos_emb_gate:
            assert not self.rel_embed, "pos_emb_gate is not compatible with relative position embeddings"
            assert self.abs_embed, "pos_emb_gate is only compatible with absolute position embeddings"
            self.gate = PEGate(init_value=pos_emb_gate_params['init_value'], sigmoid=pos_emb_gate_params['sigmoid'])

        # Set APE methods
        if self.abs_embed:
            num_patches = self.patch_embed.num_patches
            embed_len = num_patches + self.num_prefix_tokens

            if self.ape_method == 'sinusoid_2d':
                # raise NotImplementedError()
                num_patches_x = int(math.sqrt(embed_len))
                self.sin_pos_embed = nn.Parameter(get_sinusoidal_2d(self.embed_dim, num_patches_x, num_patches_x, sigma=torch.Tensor([10000])), requires_grad=False)
            elif self.ape_method == 'fourier':
                self.pos = nn.Parameter(get_raw_coordinate(embed_len).unsqueeze(1), requires_grad=False)
                self.fourier_layer = LearnableFourierPositionalEncoding(1, 2, self.embed_dim, 4*self.embed_dim, self.embed_dim, 2.5)
            elif self.ape_method == 'cape':
                pe_gate_kwargs = {}
                if self.pos_emb_gate:
                    pe_gate_kwargs = {'pos_emb_gate': self.pos_emb_gate, 'pos_emb_gate_params': self.pos_emb_gate_params}
                self.cape = CAPE1d(d_model=self.embed_dim, max_global_shift=0.5, max_local_shift=0.5, max_global_scaling=1.4, batch_first=True, **pe_gate_kwargs)

    def reset_pos_emb(self, device='cpu', experiment=None):
        if self.abs_embed:
            num_patches = self.patch_embed.num_patches
            embed_len = num_patches + self.num_prefix_tokens

            if self.ape_method == 'absolute_learnable' and self.reset_ape:
                # Reset the position embedding to a new random value, overwriting any pre-trained weights
                # Make sure not to assign a new Parameter, but instead modify the value
                self.pos_embed.data = torch.randn(1, embed_len, self.embed_dim, device=device) * .02
        else:
            # # NOTE: this is a hack to remove the pos_embed parameter
            # opt = experiment.optimizer
            # assert len(opt.param_groups) == 1, "Only one param group supported"
            # index = list(zip(*list(self.named_parameters())))[0].index('pos_embed')
            # if index != -1:
            #     del opt.param_groups[0]['params'][index]
            del self.pos_embed

    def gate_params(self):
        if self.ape_method == 'cape':
            return list(self.cape.gate.parameters())
        elif isinstance(self.gate, nn.Identity):
            return []
        else:
            return list(self.gate.parameters())

    def _pos_embed_addition(self, x):
        if self.ape_method == 'absolute_learnable':
            x = x + self.gate(self.pos_embed)
        elif self.ape_method == 'sinusoid_2d':
            pos_embed = self.gate(self.sin_pos_embed)
            if self.num_prefix_tokens > 0:
                cls_token_embed = torch.zeros_like(pos_embed[:, :self.num_prefix_tokens, :])
                pos_embed = torch.cat((cls_token_embed, pos_embed), dim=1)
            x = x + pos_embed
        elif self.ape_method == 'fourier':
            x = x + self.gate(self.fourier_layer(self.pos))
        elif self.ape_method == 'cape':
            x = self.cape(x)
        else:
            raise NotImplementedError(f"ape_method = {self.ape_method}")
        return x

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if self.abs_embed:
                x = self._pos_embed_addition(x)
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            if self.abs_embed:
                x = self._pos_embed_addition(x)
        return self.pos_drop(x)

    # Overwriting forward pass to save input as attribute and apply
    # requires_grad to input image
    def forward_features(self, x):
        self.image_in = x
        x.requires_grad_(True)

        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def get_shap_interface(self, args, shap_method, input_images, on_cpu=False, batch_size=None, spatial_features=False):
        if shap_method == 'kernel':
            return model_agnostic_interface(self, 'pos_embed', input_images,
                                            pos_emb_format_has_sample=True,
                                            pos_emb_format_spatial=False,
                                            pos_emb_format_channels_first=False,
                                            input_format_spatial=True,
                                            input_format_channels_first=True,
                                            on_cpu=on_cpu, batch_size=batch_size, spatial_features=spatial_features)
        else:
            raise ValueError(f"Unimplemented SHAP method: {shap_method}")

