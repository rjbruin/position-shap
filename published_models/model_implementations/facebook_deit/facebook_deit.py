"""
Facebook DeiT ViT
"""
from functools import partial
import os
import torch
import timm
import tensorflow as tf
import torchvision
from torchvision import transforms

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import datasets
from published_models import BasePublishedModel, GoogleResearchViT
import analysis



VERSIONS = {
    'Ti/16': 'deit_tiny_distilled_patch16_224.fb_in1k',
    'B/16': 'deit_base_distilled_patch16_224.fb_in1k',
    'B/16-384': 'deit_base_distilled_patch16_384.fb_in1k',
    # 'B/16': 'deit_base_patch16_224.fb_in1k',
    # 'B/16': 'deit_base_patch16_384.fb_in1k',
    'S/16': 'deit_small_distilled_patch16_224.fb_in1k',
    # 'B/16': 'deit_small_patch16_224.fb_in1k',
    # 'B/16': 'deit_tiny_patch16_224.fb_in1k',
}

WEIGHTS = {
    'Ti/16-imagenet-imagenet': {'reported_val_acc': 0.745, 'model_version': 'Ti/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
    'B/16-imagenet-imagenet': {'reported_val_acc': 0.812, 'model_version': 'B/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
    'B/16-384-imagenet-imagenet': {'reported_val_acc': 0.834, 'model_version': 'B/16-384', 'model_pretrained': 'imagenet', 'internal_img_size': 384, 'dataset': 'imagenet'},
    'S/16-imagenet-imagenet': {'reported_val_acc': 0.852, 'model_version': 'S/16', 'model_pretrained': 'imagenet', 'internal_img_size': 224, 'dataset': 'imagenet'},
}

class FacebookDeiT(BasePublishedModel):
    # NOTE: provide a list of all possible configurations, so that they can be
    # tested automatically
    # Use WEIGHTS; append num_classes for imagenet
    CONFIGURATIONS = [WEIGHTS[weights] | {'model_weights': weights, 'num_classes': 1000} for weights in WEIGHTS]

    def __init__(self, version, weights, num_classes):
        super().__init__()

        # Select a value from "adapt_filename" above that is a fine-tuned checkpoint.
        self.weights_metadata = None
        if weights is not None:
            self.weights_metadata = WEIGHTS[weights]

        # For available model names, see here:
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer_hybrid.py
        self.version = version
        model = timm.create_model(VERSIONS[version], pretrained=weights is not None, num_classes=num_classes)

        # initialize a subclass without having to reinitialize
        self.model = object.__new__(TimmFacebookDeit)
        self.model.__dict__ = model.__dict__

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
        return {
            'image': analysis.sources.nested_attribute('model.image_in'),
            'pos_emb': analysis.sources.nested_attribute('model.pos_embed'),
            'bias': analysis.sources.collect_by_substring('bias'),
            # This model does not use RPE
        }

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

        elif args.dataset_policy in ['own', 'FacebookDeiT']:
            if args.dataset == 'nih':
                raise ValueError(f"Use --dataset_policy GoogleResearchViT for NIH dataset, not {args.dataset_policy}")

            args.dataset_policy = 'FacebookDeiT'
            # get model specific transforms (normalization, resize)
            data_config = timm.data.resolve_model_data_config(VERSIONS[experiment.net.version])
            data_config['input_size'] = (args.internal_img_size, args.internal_img_size)
            # AutoAugment
            data_config['auto_augment'] = 'rand-m9-mstd0.5-inc1'
            # Random Erasing
            data_config['re_prob'] = 0.25
            data_config['re_mode'] = 'pixel'
            data_config['re_count'] = 1
            transforms_train = timm.data.create_transform(**data_config, is_training=True)
            transform_val = timm.data.create_transform(**data_config, is_training=False)
            sampler = datasets.repeatedaugmentation.RASampler
            datasets.setup_loaders(experiment, args, transforms_train, transform_val, sampler=sampler)

        else:
            raise NotImplementedError()

    @staticmethod
    def finetune_policy(experiment, args):
        """
        Sets the arguments related to finetuning to the values used in the
        original paper.

        Reference: Table 9 of https://arxiv.org/abs/2012.12877
        """
        if args.finetune_policy.startswith('GoogleResearchViT'):
            return GoogleResearchViT.finetune_policy(experiment, args)

        elif args.finetune_policy == 'own':
            args.finetune_policy = 'FacebookDeiT'

        elif args.finetune_policy not in ['FacebookDeiT']:
            raise NotImplementedError()

        args.clip = 0.0
        # NOTE: Our GPUs don't support bf16
        # args.precision = 'bf16'
        args.precision = '16'
        args.opt = 'adamw'
        args.accumulate_gradients = 8
        args.batch_size = 128
        args.lr = 0.0005 * (args.batch_size * args.accumulate_gradients) / 512.
        args.weight_decay = 0.05
        args.warmup_epochs = 5
        args.label_smoothing = 0.1

        args.mixup_alpha = 0.8
        args.cutmix_alpha = 1.0
        args.label_smoothing = 0.1

        # NOTE: Stochastic Depth / DropPath (0.1) is a model feature, not a
        # training policy, so will be included in the model construction

        if args.model_version.endswith('-384'):
            args.accumulate_gradients = args.accumulate_gradients * 2
            args.batch_size = args.batch_size // 2

        if args.dataset in ['cifar10', 'cifar100']:
            dataset_size = 50000
            # Custom, shorter training policy
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
        elif args.dataset == 'imagenet':
            # TODO: we make no changes to the default policy
            pass
        elif args.dataset == 'eurosat':
            # TODO: there is no established policy for this dataset, AFAIK
            pass
        elif args.dataset == 'oxfordpets':
            # TODO: there is no established policy for this dataset, AFAIK
            pass
        elif args.dataset in ['nih', 'nih_google']:
            # TODO: there is no established policy for this dataset, AFAIK
            # Remove label smoothing, because we have no implementation for it
            args.label_smoothing = 0.0
            pass
        else:
            raise NotImplementedError(f'Unknown dataset {args.dataset}')


class TimmFacebookDeit(timm.models.deit.VisionTransformerDistilled):
    """
    Wrapper around VisionTransformerDistilled to expose the attribution sources.
    """
    image_in = None

    # Overwriting forward pass to save input as attribute and apply
    # requires_grad to input image
    def forward_features(self, x) -> torch.Tensor:
        # NOTE: added code
        self.image_in = x
        x.requires_grad_(True)

        x = self.patch_embed(x)
        x = torch.cat((
            self.cls_token.expand(x.shape[0], -1, -1),
            self.dist_token.expand(x.shape[0], -1, -1),
            x),
            dim=1)
        x = self.pos_drop(x + self.pos_embed)
        # if self.grad_checkpointing and not torch.jit.is_scripting():
        #     x = checkpoint_seq(self.blocks, x)
        # else:
        x = self.blocks(x)
        x = self.norm(x)
        return x