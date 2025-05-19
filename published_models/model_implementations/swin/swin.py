"""
Swin Transformer
"""
import math
import timm
import torch
from torch import nn

from published_models import BasePublishedModel, GoogleResearchViT, FacebookDeiT
import analysis
import datasets
from analysis_shap import model_agnostic_interface


VERSIONS = {
    'B/4/7-224': 'swin_base_patch4_window7_224',
    'B/4/12-384': 'swin_base_patch4_window12_384',
    'L/4/7-224': 'swin_large_patch4_window7_224',
    'L/4/12-384': 'swin_large_patch4_window12_384',
    'S/4/7-224': 'swin_small_patch4_window7_224',
    'Ti/4/7-224': 'swin_tiny_patch4_window7_224',
}

WEIGHTS = {
    'B/4/7-224-imagenet': {'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_base_patch4_window7_224.ms_in1k', 'reported_val_acc': 0.83606},
    'B/4/7-224-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet22k', 'timm_name': 'swin_base_patch4_window7_224.ms_in22k'},
    'B/4/7-224-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_base_patch4_window7_224.ms_in22k_ft_in1k', 'reported_val_acc': 0.85272},
    'B/4/12-384-imagenet': {'model_pretrained': 'imagenet', 'internal_img_size': 384, 'dataset': 'imagenet', 'timm_name': 'swin_base_patch4_window12_384.ms_in1k', 'reported_val_acc': 0.84476},
    'B/4/12-384-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 384, 'dataset': 'imagenet22k', 'timm_name': 'swin_base_patch4_window12_384.ms_in22k'},
    'B/4/12-384-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 384, 'dataset': 'imagenet', 'timm_name': 'swin_base_patch4_window12_384.ms_in22k_ft_in1k', 'reported_val_acc': 0.86438},

    'L/4/7-224-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet22k', 'timm_name': 'swin_large_patch4_window7_224.ms_in22k'},
    'L/4/7-224-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_large_patch4_window7_224.ms_in22k_ft_in1k', 'reported_val_acc': 0.86312},
    'L/4/12-384-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 384, 'dataset': 'imagenet22k', 'timm_name': 'swin_large_patch4_window12_384.ms_in22k'},
    'L/4/12-384-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 384, 'dataset': 'imagenet', 'timm_name': 'swin_large_patch4_window12_384.ms_in22k_ft_in1k', 'reported_val_acc': 0.87132},

    # 'S3/4-224-imagenet': {'model_pretrained': None, 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_s3_base_224.ms_in1k'},
    # 'S3/4-224-imagenet': {'model_pretrained': None, 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_s3_small_224.ms_in1k'},
    # 'S3/4-224-imagenet': {'model_pretrained': None, 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_s3_tiny_224.ms_in1k'},

    'S/4/7-224-imagenet': {'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_small_patch4_window7_224.ms_in1k', 'reported_val_acc': 0.83208},
    'S/4/7-224-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet22k', 'timm_name': 'swin_small_patch4_window7_224.ms_in22k'},
    'S/4/7-224-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_small_patch4_window7_224.ms_in22k_ft_in1k', 'reported_val_acc': 0.83298},
    'Ti/4/7-224-imagenet': {'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_tiny_patch4_window7_224.ms_in1k', 'reported_val_acc': 0.81376},
    'Ti/4/7-224-imagenet22k': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet22k', 'timm_name': 'swin_tiny_patch4_window7_224.ms_in22k'},
    'Ti/4/7-224-imagenet22k-imagenet': {'model_pretrained': 'imagenet22k', 'internal_img_size': 224, 'dataset': 'imagenet', 'timm_name': 'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k', 'reported_val_acc': 0.80968},
}

# WEIGHTS = {
#     'Ti/16-imagenet-imagenet': {'reported_val_acc': 0.745, 'timm_name': 'Ti/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
#     'B/16-imagenet-imagenet': {'reported_val_acc': 0.812, 'timm_name': 'B/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
#     'B/16-384-imagenet-imagenet': {'reported_val_acc': 0.834, 'timm_name': 'B/16-384', 'model_pretrained': 'imagenet', 'internal_img_size': 384, 'dataset': 'imagenet'},
#     'S/16-imagenet-imagenet': {'reported_val_acc': 0.852, 'timm_name': 'S/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
# }

class Swin(BasePublishedModel):
    # NOTE: provide a list of all possible configurations, so that they can be
    # tested automatically
    # Use WEIGHTS; append num_classes
    CONFIGURATIONS = []
    for weights in WEIGHTS:
        num_classes = 1000 if WEIGHTS[weights]['dataset'] == 'imagenet' in weights else 21843
        CONFIGURATIONS.append(WEIGHTS[weights] | {'model_weights': weights, 'num_classes': num_classes})

    def __init__(self, version, weights, num_classes, pos_emb='rel'):
        super().__init__()

        # If weights are used, we need to use the timm name with the right
        # suffix, but we still want to ensure that this version matches the
        # version provided by the user in argsweights_version
        if weights is not None:
            self.weights_metadata = WEIGHTS[weights]
            weights_version = self.weights_metadata['timm_name']
            if weights_version.split('.')[0] != VERSIONS[version]:
                raise ValueError(f'Version mismatch: args.model_version is \"{version}\" while model_version of weights is \"{weights_version}\"')
            self.timm_name = weights_version
        else:
            self.timm_name = VERSIONS[version]

        # For available model names, see here:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer_hybrid.py
        model = timm.create_model(self.timm_name, pretrained=weights is not None, num_classes=num_classes)

        # initialize a subclass without having to reinitialize
        self.model = object.__new__(TimmSwin)
        self.model.__dict__ = model.__dict__
        self.model.setup(pos_emb=pos_emb)

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
            'bias': analysis.sources.collect_by_substring('.bias'),
        }
        if self.model.abs_embed:
            sources['pos_emb'] = analysis.sources.nested_attribute('model.pos_embed')
        if self.model.rel_embed:
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
        if args.dataset_policy == 'GoogleResearchViT':
            return GoogleResearchViT.dataset_policy(experiment, args)

        elif args.dataset_policy == 'FacebookDeiT':
            return FacebookDeiT.dataset_policy(experiment, args)

        elif args.dataset_policy in ['own', 'Swin']:
            if args.dataset == 'nih':
                raise ValueError(f"Use --dataset_policy GoogleResearchViT for NIH dataset, not {args.dataset_policy}")

            args.dataset_policy = 'Swin'
            # get model specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(experiment.net.timm_name)
            data_config['input_size'] = (args.internal_img_size, args.internal_img_size)
            # AutoAugment
            data_config['auto_augment'] = 'rand-m9-mstd0.5-inc1'
            # Random Erasing
            data_config['re_prob'] = 0.25
            data_config['re_mode'] = 'pixel'
            data_config['re_count'] = 1
            transforms_train = timm.data.create_transform(**data_config, is_training=True)
            transform_val = timm.data.create_transform(**data_config, is_training=False)
            # Swin does not use RASampler, where DeiT does. Otherwise, the
            # setting is the same as for FacebookDeiT.
            datasets.setup_loaders(experiment, args, transforms_train, transform_val)

        else:
            raise NotImplementedError()

    @staticmethod
    def finetune_policy(experiment, args):
        """
        Sets the arguments related to finetuning to the values used in the
        original paper.
        """
        if args.finetune_policy.startswith('GoogleResearchViT'):
            return GoogleResearchViT.finetune_policy(experiment, args)

        elif args.finetune_policy in ['own', 'Swin']:
            # The only difference is Repeated Augmentation, but that is set in the dataset policy.
            print(f"Swin's finetuning policy is FacebookDeiT's policy. Using that policy...")
            args.finetune_policy = 'FacebookDeiT'

        else:
            raise NotImplementedError()

        return FacebookDeiT.finetune_policy(experiment, args)

    def get_shap_interface(self, args, shap_method, input_images, **kwargs):
        if shap_method == 'kernel':
            return model_agnostic_interface(self, 'model.pos_embed', input_images,
                                            pos_emb_format_has_sample=True,
                                            pos_emb_format_spatial=False,
                                            pos_emb_format_channels_first=False,
                                            input_format_spatial=True,
                                            input_format_channels_first=True,
                                            **kwargs)
        else:
            raise ValueError(f"Unimplemented SHAP method: {shap_method}")


class TimmSwin(timm.models.swin_transformer.SwinTransformer):
    """
    Wrapper around VisionTransformer to expose the attribution sources.

    Base: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer.py
    """
    image_in = None

    def setup(self, pos_emb='default'):
        if pos_emb == 'none':
            pos_emb = ''
        if pos_emb == 'default':
            pos_emb = 'rpe'
        self.abs_embed = 'ape' in pos_emb
        self.rel_embed = 'rpe' in pos_emb

        if self.abs_embed:
            num_patches = self.patch_embed.num_patches
            embed_len = int(math.sqrt(num_patches))
            self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_len, self.embed_dim) * .02)

        if not self.rel_embed:
            for layer in self.layers:
                for block in layer.blocks:
                    attn = block.attn
                    # attn.relative_position_bias_table.data
                    data = attn.relative_position_bias_table.data.detach()
                    del attn.relative_position_bias_table
                    attn.register_buffer("relative_position_bias_table", data)
                    # attn.relative_position_bias_table = attn.relative_position_bias_table.data
                    attn.relative_position_bias_table.zero_()

    # Overwriting forward pass to save input as attribute and apply
    # requires_grad to input image
    def forward_features(self, x):
        # NOTE: added code for attribution
        self.image_in = x
        x.requires_grad_(True)

        x = self.patch_embed(x)

        # Add positional embedding
        if self.abs_embed:
            x = x + self.pos_embed

        x = self.layers(x)
        x = self.norm(x)
        return x

