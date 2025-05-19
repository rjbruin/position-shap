import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import time
import wandb
import pandas as pd
import numpy as np

import torchvision
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, CatMetric

from analysis.saving import save_raw_attributions
from datasets.position_datasets import CIFAR10Position, CIFAR10Shortcut
from models.swin_transformer_model.models.swin_transformer import SwinTransformer
from models.tnt_model.tnt import tnt_s_patch16_224
from models.t2t_model.models.t2t_vit import t2t_vit_7

from models.vit_modeling import VisionTransformer, CONFIGS

from models.drloc import cal_selfsupervised_loss
from timm.loss import LabelSmoothingCrossEntropy
from timm.data import Mixup

from published_models.model_implementations.google_research_vit.google_research_vit import GoogleResearchViT
from toy_experiments import TriViTalAbsolutePosition

import published_models
import analysis_shap
import datasets
from utils import get_git_revision

# Fix issue with expired SSL certificates
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# import logging
# logging.basicConfig(level=logging.DEBUG)


# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# NOTE: default is --training, but this exact order of add_argument() calls is necessary for this default... somehow.
parser.add_argument('--no-training', dest='training', action='store_false', help='Do not run training.')
parser.add_argument('--training', action='store_true', help='Run training.')
parser.add_argument('--test', action='store_true', help='Run testing (on validation) after training.')
parser.add_argument('--only_print_val_batches', action='store_true', help='Only print number of validation batches, no training or anything else.')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')  # resnets.. 1e-3, Vit..1e-4?
parser.add_argument('--min_lr', default=0.0, type=float, help='minimum learning rate for cosine decay')
parser.add_argument('--warmup_epochs', type=float, default=None, help='LR warmup iterations')
parser.add_argument('--opt', default="adam", help="Optimizer")
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--pos_emb_weight_decay', type=float, default=0.0, help='PE weight decay')
parser.add_argument('--clip', type=float, default=0.0, help='Gradient clipping')
parser.add_argument('--accumulate_gradients', type=int, default=1, help='Accumulate gradients, to simulate larger batch size')
parser.add_argument('--check_val_every_n_epoch', type=int, default=1, help='Validate every n epochs')
parser.add_argument('--resume_id', type=str, default=None, help='Resume from W&B run ID')
parser.add_argument('--resume_in_new_run', action='store_true', help='Resume, but don\'t override the existing run.')
parser.add_argument('--resume_from_artifact', type=str, default=None, help='Resume from artifact')
parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='Resume from checkpoint saved to disk.')
parser.add_argument('--save_checkpoint', action='store_true', help='Manually save checkpoint after training')
parser.add_argument('--deit_augmentations', action='store_true', help='Use DeiT augmentations: RA, Random Erasing.')
# parser.add_argument('--auto_augment', action='store_true', help='Use AutoAugment, prespecified policy.')
parser.add_argument('--mixup-alpha', type=float, default=0.0, help='MixUp alpha. Enables MixUp if > 0.0.')
parser.add_argument('--cutmix-alpha', type=float, default=0.0, help='CutMix alpha. Enables CutMix if > 0.0.')
parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing value')
parser.add_argument('--multi-label', action='store_true', help='Multi-label classification')
parser.add_argument('--random-erasing', type=float, default=0.0, help='Random Erasing probability, 0 disables.')
parser.add_argument('--net', default='vit', help='Network definition')
parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
parser.add_argument('--val_batch_size', type=int, default=None, help='Validation natch size. Same as batch size if not set.')
parser.add_argument('--n_epochs', type=int, default='50')
parser.add_argument('--patch_size', default=16, type=int)
parser.add_argument('--internal_img_size', default=224, type=int, help="Image size to run through model.")
parser.add_argument('--vit_config', type=str, default='ViT-B_16', help='Config for ViT models.')
parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
parser.add_argument("--pos_emb", type=str, default='absolute_learnable', help="Type of position embedding used.")
parser.add_argument("--fourier_gamma", type=float, default=2.5, help="Learnable Fourier: gamma")
parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset. Can be many, including `toy`.")
parser.add_argument("--toy_dataset", type=str, default='appearance_absolute_position_three_colors', help="If --dataset is `toy`, provides name of toy dataset to use.")
parser.add_argument('--toy_size', type=int, default=6, help='Toy dataset: size')
parser.add_argument("--num_per_class", type=str, default='all', help="Number of samples per class, for computing learning curves.")
parser.add_argument("--num_classes", type=int, default=None, help="Number of classes in the dataset. Flag can be used to override default settings.")
parser.add_argument('--adv_augmentations', action='store_true', help='Apply advanced augmentations.')
parser.add_argument('--pos_augmentations', action='store_true', help='Apply position-related augmentations.')
parser.add_argument('--imagenet_normalization_huggingface', action='store_true', help='Use huggingface imagenet normalization stats (all 0.5).')
parser.add_argument("--seed", type=int, default='42')
parser.add_argument("--data_seed", type=int, default=0, help="Seed for dataset setup.")
parser.add_argument("--deterministic", action='store_true', help="Set deterministic mode")
parser.add_argument("--use_drloc", type=bool, default=False, help="Use self-supervision \"DRLoc\"")
parser.add_argument("--exp_name", type=str, default=None, help="W&B experiment name. Autogenerated if not set.")
parser.add_argument('--debug', action='store_true', help='Debug mode: don\'t track on W&B')
parser.add_argument('--dryrun', action='store_true', help='Dummy mode: run one epoch of training with five batches, then just two validation batches.')
parser.add_argument('--wandb_dir', type=str, default='/tmp/wandb_robertjanbruin', help='Storage location for W&B temp files.')
parser.add_argument('--wandb_project', type=str, default='vit-visualization', help='W&B project to log to')
parser.add_argument('--tags', type=str, default='', help='Comma-separated tags')
# Data
parser.add_argument("--data_root", type=str, default='~/data', help="Base directory of dataset")
parser.add_argument("--num_samples", type=int, default=None, help="For ImageNet: number of training samples.")
parser.add_argument("--num_workers", type=int, default=2, help="Number of threads used in data loading.")
parser.add_argument("--precision", type=str, default='32', help="Precision for training.")
# Debugging tools
parser.add_argument("--log_first_batch", action='store_true', help="Log images of first batch to W&B.")
# Bug fixes
parser.add_argument("--dont-fix-schedulers", action='store_true', help="Undo fix of learning rate schedulers with gradient accumulation.")
# SHAP analysis
parser.add_argument('--shap', action='store_true', help='Do SHAP analysis after training finishes.')
parser.add_argument('--shap_debug', action='store_true', help='Debug SHAP analysis.')
parser.add_argument("--shap_seed", type=int, default=0, help="Seed for SHAP analysis.")
parser.add_argument('--shap_fast', action='store_true', help='Set nsamples argument for KernelSHAP to the number of features.')
parser.add_argument('--shap_on_cpu', action='store_true', help='Run SHAP analysis on CPU.')
parser.add_argument('--shap_spatial_features', action='store_true', help='Use spatial features for SHAP analysis. Will be much slower!')
parser.add_argument('--shap_image_channels_features', action='store_true', help='Use image channels as features for SHAP analysis.')
# parser.add_argument('--shap_save', action='store_true', help='Save SHAP values to disk.')
parser.add_argument('--shap_batched', action='store_true', help='Save SHAP analysis for each batch.')
parser.add_argument('--shap_start_at_batch', type=int, default=None, help='Start SHAP analysis at batch.')
parser.add_argument('--shap_stop_at_batch', type=int, default=None, help='Stop SHAP analysis at batch, exclusive.')
parser.add_argument('--shap_single_batch_bg', action='store_true', help='Use first batch of validation set as background samples for all samples in validation set.')
parser.add_argument('--shap_fold_size', type=int, default=None, help='Size of a fold in a single batch of SHAP analysis. If `None`, is set to half the batch size.')
# parser.add_argument('--shap_bg_folds', type=str, default='all', help='Background samples for SHAP analysis. Either `all` or an integer.')
parser.add_argument('--shap_verbosity', type=int, default=1, help='Verbosity of SHAP. 0 = print nothing; 1 = print only stats after computation; 2 = print stats and background sample processing; 3 = also print processing time of each batch.')
parser.add_argument('--shap_continue', action='store_true', help='Do not overwrite SHAP analysis if it already exists.')
parser.add_argument('--shuffle_val', action='store_true', help='Shuffle samples in validation.')
# Published models
parser.add_argument('--model', type=str, default='own-vit', help='Model identifier. Use `own-vit` to use own-built definition.')
parser.add_argument('--model_version', type=str, default=None, help='Version of published model.')
parser.add_argument('--model_weights', type=str, default=None, help='Path to published model weights.')
parser.add_argument('--model_pretrained', type=str, default=None, help='Track on which dataset the given weights are pretrained.')
parser.add_argument('--reported_val_acc', type=float, default=None, help='Reported accuracy of model, used to report difference at validation time.')
parser.add_argument('--dataset_policy', type=str, default='own-vit', help='Dataset policy. If `own`, will be replaced with the name for the model.')
parser.add_argument('--finetune_policy', type=str, default='own', help='Use a specific policy to finetune the published model.  If `own`, will be replaced with the name for the model.')
parser.add_argument('--pos_emb_reset', action='store_true', help='Reset position embedding weights to random values.')
# Toy model
parser.add_argument('--toy_d', type=int, default=16, help='Toy model: d')
parser.add_argument('--toy_mlp_d', type=int, default=None, help='Toy model: mlp_d')
parser.add_argument('--toy_n_blocks', type=int, default=1, help='Toy model: n_blocks')
parser.add_argument('--toy_n_heads', type=int, default=2, help='Toy model: n_heads')
parser.add_argument('--toy_pos_add', type=str, default='add', help='Toy model: pos_add')
parser.add_argument('--toy_pos_init', type=str, default='uniform:0.1', help='Toy model: pos_init')
parser.add_argument('--toy_pooling', type=str, default='avg', help='Toy model: pooling')
# CIFAR-10 Position dataset
parser.add_argument('--c10pos_scale', type=int, default=-1, help='CIFAR-10 Position: scale')
parser.add_argument('--c10pos_pos_classes', type=int, default=4, help='CIFAR-10 Position: number of position classes.')
parser.add_argument('--c10pos_pos_per_class', type=int, default=2, help='CIFAR-10 Position: number of positions per class.')
parser.add_argument('--c10pos_shuffle_classes', action='store_false', help='CIFAR-10 Position: shuffle classes.')
# CIFAR-10 Shortcut dataset
parser.add_argument('--c10cut_cut_classes', type=int, default=4, help='CIFAR-10 Shortcut: number of shortcut classes.')
parser.add_argument('--c10cut_shuffle_classes', action='store_false', help='CIFAR-10 Shortcut: shuffle classes.')
parser.add_argument('--c10cut_test_ood', action='store_true', help='CIFAR-10 Shortcut: test on OOD data.')
# PE gating
parser.add_argument('--pos_emb_gate', action='store_true', help='PE gating: use gating mechanism for position embedding.')
parser.add_argument('--pos_emb_gate_lr', type=float, default=1e-1, help='PE gating: learning rate for the gate. If None, identical to overall LR.')
parser.add_argument('--pos_emb_gate_init_value', type=float, default=0.5, help='PE gating: initial value of the gate.')
parser.add_argument('--pos_emb_gate_sigmoid', action='store_true', help='PE gating: use sigmoid activation for the gate.')
parser.add_argument('--pos_emb_gate_shared', action='store_true', help='PE gating: for RoPE, share a single weight between all gates.')
# To prevent you from making stupid mistakes
parser.add_argument('--git_commit', type=str, default=None, help='Git commit hash of the code used to run this experiment.')


class Experiment(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.num_devices = torch.cuda.device_count()
        self.use_drloc = self.args.use_drloc
        self.mode = "l1"
        self.lambda_drloc = 0.05
        self.cutmix = None

        self.print_val_batches = False
        self.c10cut_test_ood = False

        # Create analysis flags that we only set after training
        self.do_shap = False
        self.shap_batched = args.shap_batched
        self.shap_batched = self.shap_batched or args.shap_start_at_batch is not None or args.shap_stop_at_batch is not None
        self.shap = None

        # Create flags for logging first batch of first epoch
        self.first_batch_train = False
        self.first_batch_val = False
        if args.log_first_batch:
            self.first_batch_train = True
            self.first_batch_val = True

        # Dataset settings
        self.img_size = self.args.internal_img_size
        self.patch_size = self.args.patch_size
        if self.args.num_classes is None:
            if self.args.dataset == 'cifar10':
                self.args.num_classes = 10
            elif self.args.dataset == 'cifar10_debug':
                self.args.num_classes = 10
            elif self.args.dataset == 'cifar10_position':
                self.args.num_classes = CIFAR10Position.get_nr_classes(self.args.c10pos_pos_classes, self.args.c10pos_pos_per_class)
            elif self.args.dataset == 'cifar10_shortcut':
                self.args.num_classes = CIFAR10Shortcut.get_nr_classes()
            elif self.args.dataset == 'cifar100':
                self.args.num_classes = 100
            elif self.args.dataset == 'flowers102':
                self.args.num_classes = 102
            elif self.args.dataset == 'svhn':
                self.args.num_classes = 10
            elif self.args.dataset == 'stanfordcars':
                self.args.num_classes = 196
            elif self.args.dataset == 'oxfordpets':
                self.args.num_classes = 37
            elif self.args.dataset == 'imagenet-tiny':
                self.args.num_classes = 200
            elif self.args.dataset == 'imagenet':
                self.args.num_classes = 1000
            elif self.args.dataset == 'imagenette':
                self.args.num_classes = 10
            elif self.args.dataset == 'eurosat':
                self.args.num_classes = 10
            elif self.args.dataset == 'nih':
                self.args.num_classes = 14
            elif self.args.dataset == 'nih_google':
                self.args.num_classes = 14
            elif self.args.dataset == 'toy':
                if self.args.toy_dataset == 'appearance_absolute_position_three_colors':
                    self.args.num_classes = 4
                else:
                    raise NotImplementedError(f"Please manually set --num_classes.")
            else:
                raise NotImplementedError(f"Please manually set --num_classes.")

        # Model
        pos_emb_gate_params = {
            'init_value': args.pos_emb_gate_init_value,
            'sigmoid': args.pos_emb_gate_sigmoid,
        }
        if args.model == 'own-vit':
            print('==> Building model..')
            if args.net == "vit":
                pos_emb = None if args.pos_emb in ['none', 'None'] else args.pos_emb
                config = CONFIGS[args.vit_config]
                self.net = VisionTransformer(
                    config=config, img_size=self.img_size, num_classes=self.args.num_classes,
                    pos_emb=pos_emb, use_drloc=args.use_drloc, fourier_gamma=args.fourier_gamma,
                    pos_emb_gate=args.pos_emb_gate,
                    pos_emb_gate_params=pos_emb_gate_params,
                    pos_emb_gate_shared=args.pos_emb_gate_shared,
                )
            elif args.net == 'swin_transformer':
                self.net = SwinTransformer(
                    ape=True, pos_emb=args.pos_emb, num_classes=self.args.num_classes
                )
            elif args.net == "resnet-50":
                self.net = torchvision.models.resnet50(pretrained=False)
                num_in_features = self.net.fc.in_features
                self.net.fc = nn.Linear(num_in_features, self.args.num_classes)
            elif args.net == 'tnt':
                self.net = tnt_s_patch16_224(
                    pretrained=False, num_classes=self.args.num_classes,
                    pos_emb=args.pos_emb
                )
                # net = torch.load('tnt_cifar100_gaussian_2d_end.pth')
            elif args.net == 't2t':
                self.net = t2t_vit_7(
                    pretrained=False, num_classes=self.args.num_classes,
                    pos_emb=args.pos_emb
                )
            elif args.net == 'toy':
                pos_emb = None if self.args.pos_emb in ['none', 'None'] else self.args.pos_emb
                pos_emb = 'absolute' if self.args.pos_emb in ['default','absolute_learnable'] else pos_emb
                try:
                    pos_emb_init, pos_emb_factor = self.args.toy_pos_init.split(':')
                    pos_emb_factor = float(pos_emb_factor)
                except ValueError:
                    raise ValueError("Invalid format for --toy_pos_init, should be pos_emb_init:pos_emb_factor.")
                self.net = TriViTalAbsolutePosition(
                    n_classes=self.args.num_classes,
                    d=self.args.toy_d,
                    size=self.img_size,
                    patch_size=self.patch_size,
                    n_blocks=self.args.toy_n_blocks,
                    n_heads=self.args.toy_n_heads,
                    use_rel_pos=False,
                    pos_emb=pos_emb,
                    input_d=3,
                    pos_add=self.args.toy_pos_add,
                    pos_emb_init=pos_emb_init,
                    pos_emb_factor=pos_emb_factor,
                    pool=self.args.toy_pooling,
                    mlp_d=self.args.toy_mlp_d,
                    compute_gradbased_attr=False
                )
            else:
                raise NotImplementedError()

        else:
            print('==> Using published model configuration..')
            self.net = published_models.get(args.model, args)

            # Override arguments based on metadata about loaded weights
            if hasattr(self.net, "args_from_weights_metadata"):
                self.net.args_from_weights_metadata(args)

            # Override arguments based on finetuning policy
            if hasattr(self.net, "finetune_policy"):
                self.net.finetune_policy(self, self.args)

            # Set stuff based on dataset policy & finetune policy
            if args.mixup_alpha > 0. or args.cutmix_alpha > 0.:
                self.cutmix = Mixup(args.mixup_alpha, args.cutmix_alpha, num_classes=args.num_classes, label_smoothing=0.0)

        if self.args.label_smoothing > 0.0:
            if self.args.multi_label:
                raise NotImplementedError()
            self.criterion = LabelSmoothingCrossEntropy(smoothing=self.args.label_smoothing)
        else:
            self.criterion = nn.CrossEntropyLoss()
            if self.args.multi_label:
                self.criterion = nn.BCEWithLogitsLoss()

        # Metrics
        if self.args.multi_label:
            self.train_acc = Accuracy(task="multilabel", num_labels=self.args.num_classes)
            self.val_acc = Accuracy(task="multilabel", num_labels=self.args.num_classes)
        if not self.args.multi_label:
            self.train_acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)
            self.val_acc = Accuracy(task="multiclass", num_classes=self.args.num_classes)
            top_k = min(5, self.args.num_classes - 1)
            self.val_acc_topk = Accuracy(task="multiclass", num_classes=self.args.num_classes, top_k=top_k)
            self.val_targets = CatMetric()
        self.val_preds = CatMetric()

        self.epoch_start = time.time()

    def setup(self, stage=None):
        # Data
        print('==> Preparing data..')
        datasets.setup_datasets(self, self.args)

        # Add metrics for pos classes
        if hasattr(self, 'pos_labels'):
            self.train_pos_accuracy = Accuracy(task="multiclass", num_classes=experiment.args.num_classes)
            self.train_nonpos_accuracy = Accuracy(task="multiclass", num_classes=experiment.args.num_classes)
            self.val_pos_accuracy = Accuracy(task="multiclass", num_classes=experiment.args.num_classes)
            self.val_nonpos_accuracy = Accuracy(task="multiclass", num_classes=experiment.args.num_classes)

        print("Arguments (after any changes to configuration):")
        for k, v in sorted(vars(self.args).items(), key=lambda x: x[0]):
            print(f"{k}: {v}")

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def configure_optimizers(self):
        params = self.net.parameters()
        if self.args.pos_emb_weight_decay > 0.0:
            if not hasattr(self.net, 'transformer'):
                raise NotImplementedError("Position embedding weight decay not implemented for this model.")
            decay_params = self.net.transformer.embeddings.pos_emb_params
            no_decay_params = list(set(self.net.parameters()) - set(decay_params))
            params = [
                {'params': decay_params, 'weight_decay': self.args.pos_emb_weight_decay},
                {'params': no_decay_params, 'weight_decay': self.args.weight_decay},
            ]
            if self.args.pos_emb_gate_lr is not None:
                raise NotImplementedError("Position embedding weight decay and learning rate for gate not implemented together.")
        elif self.args.pos_emb_gate and self.args.pos_emb_gate_lr is not None:
            # Use different learning rate for the gate
            if isinstance(self.net, VisionTransformer) or isinstance(self.net, GoogleResearchViT):
                gate_params = self.net.gate_params()
                no_gate_params = list(set(self.net.parameters()) - set(gate_params))
                params = [
                    {'params': gate_params, 'lr': self.args.pos_emb_gate_lr},
                    {'params': no_gate_params, 'weight_decay': self.args.weight_decay},
                ]
            else:
                print("WARNING! Position embedding gate not used in this model; --pos_emb_gate_lr is not used.")

        if self.args.opt == "adam":
            self.optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.opt == "adamw":
            self.optimizer = optim.AdamW(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.opt == "sgd":
            self.optimizer = optim.SGD(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.opt == "sgd-momentum":
            self.optimizer = optim.SGD(params, lr=self.args.lr, weight_decay=self.args.weight_decay, momentum=0.9)
        else:
            raise NotImplementedError()

        its_per_epoch = self.n_samples / (self.args.batch_size * self.args.accumulate_gradients)
        # With gradient accumulation we need to manually adjust the estimate of
        # the iterations per epoch
        if self.args.dont_fix_schedulers:
            its_per_epoch = self.n_samples / self.args.batch_size

        # use cosine or reduce LR on Plateau scheduling
        if not self.args.cos:
            # scheduler = lr_scheduler.ReduceLROnPlateau(
            #     self.optimizer, 'min', patience=int(3 * its_per_epoch),
            #     verbose=True, min_lr=1e-3 * 1e-5, factor=0.1
            # )
            # scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[int(0.5 * args.n_epochs), int(0.75 * args.n_epochs)], gamma=0.1)
            scheduler = None
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, args.n_epochs * its_per_epoch, eta_min=args.min_lr,
            )

        if self.args.warmup_epochs is not None:
            if scheduler is None:
                raise NotImplementedError(f"Warmup without scheduler not implemented")

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=1e-8, end_factor=1.0,
                total_iters=self.args.warmup_epochs * its_per_epoch
            )
            scheduler = torch.optim.lr_scheduler.ChainedScheduler([
                scheduler, warmup_scheduler,
            ])

        if scheduler is not None:
            return {'optimizer': self.optimizer, 'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
            }}
        else:
            return {'optimizer': self.optimizer}

    def forward(self, x):
        return self.net(x)

    def _step(self, inputs, targets):
        outputs = self(inputs)
        if self.args.model == 'own-vit':
            if self.args.net != 'toy' and self.args.net != 'resnet-50':
                outputs = outputs[0]
        else:
            outputs = self.net.get_logits(outputs)

        if self.use_drloc:
            loss = self.criterion(outputs.sup, targets)
            loss_ssup, _ = cal_selfsupervised_loss(outputs, self.mode, self.lambda_drloc)
            loss += loss_ssup
        else:
            loss = self.criterion(outputs, targets)

        return outputs, loss

    def unpack_batch(self, batch, force_single_label=False):
        if self.args.dataset in ['nih', 'nih_google']:
            if self.args.multi_label and not force_single_label:
                return {
                    'inputs': batch['img'],
                    'targets': batch['lab'],
                    'inds': None,
                }
            else:
                return {
                    'inputs': batch['img'],
                    'targets': torch.argmax(batch['lab'], dim=1),
                    'inds': None,
                }
        data = {
            'inputs': batch[0],
            'targets': batch[1],
            'inds': None,
        }
        if len(batch) > 2:
            data['inds'] = batch[2]
        return data

    def training_step(self, batch, batch_idx):
        batch = self.unpack_batch(batch)
        inputs, targets = batch['inputs'], batch['targets']

        # Augment: DeiT augmentations: RA, CutMix, MixUp, random erasing, label smoothing
        original_targets = targets
        if self.cutmix is not None:
            # Augment: MixUp & CutMix
            inputs, _ = self.cutmix(inputs, targets)

        if self.first_batch_train:
            image = wandb.Image(torchvision.utils.make_grid(inputs))
            self.trainer.logger.log_image(
                key='train_sample', images=[image]
            )
            self.first_batch_train = False

        outputs, loss = self._step(inputs, targets)
        original_targets = original_targets.type(torch.long)
        predictions = torch.argmax(outputs, 1)
        self.train_acc(outputs, original_targets)
        self.log('train/acc_step', self.train_acc)
        self.log('train/loss', loss)

        if hasattr(self, 'pos_labels'):
            # Compute accuracy for position classes
            pos_labels = self.pos_labels
            # targets and predictions are of shape (N)
            pos_targets = []
            pos_preds = []
            nonpos_targets = []
            nonpos_preds = []
            for i in range(targets.size(0)):
                if targets[i] in pos_labels:
                    pos_targets.append(original_targets[i])
                    pos_preds.append(predictions[i])
                else:
                    nonpos_targets.append(original_targets[i])
                    nonpos_preds.append(predictions[i])
            if len(pos_targets) > 0:
                self.train_pos_accuracy(torch.tensor(pos_preds), torch.tensor(pos_targets))
            if len(nonpos_targets) > 0:
                self.train_nonpos_accuracy(torch.tensor(nonpos_preds), torch.tensor(nonpos_targets))

        return {"loss": loss, "logits": outputs.detach()}

    def validation_step(self, batch, batch_idx):
        if self.print_val_batches:
            return None

        # Apply batch skipping if shap_start/end_at_batch are set
        if self.do_shap and self.shap_batched:
            # Do not skip first batch, because we need to set the background
            if self.shap is not None and self.shap.background_set:
                if self.args.shap_start_at_batch is not None and batch_idx < self.args.shap_start_at_batch:
                    return None
                if self.args.shap_stop_at_batch is not None and batch_idx >= self.args.shap_stop_at_batch:
                    return None

        batch = self.unpack_batch(batch)
        inputs, targets, inds = batch['inputs'], batch['targets'], batch['inds']

        if self.first_batch_val and self.trainer.logger is not None:
            key = 'val_sample_ood' if self.c10cut_test_ood else 'val_sample'
            image = wandb.Image(torchvision.utils.make_grid(inputs))
            self.trainer.logger.log_image(
                key=key, images=[image]
            )
            self.first_batch_val = False

        # If we are doing SHAP analysis, and we are running on CPU, we cannot
        # also run the model regularly on the GPU.
        outputs, _ = self._step(inputs, targets)
        predictions = torch.argmax(outputs, 1)
        if not self.do_shap or not self.args.shap_on_cpu:
            targets = targets.type(torch.long)
            self.val_acc(outputs, targets)
            self.val_preds.update(predictions)
            if not self.args.multi_label:
                self.val_acc_topk(outputs, targets)
                self.val_targets.update(targets)

            if hasattr(self, 'pos_labels'):
                # Compute accuracy for position classes
                pos_labels = self.pos_labels
                # targets and predictions are of shape (N)
                pos_targets = []
                pos_preds = []
                nonpos_targets = []
                nonpos_preds = []
                for i in range(targets.size(0)):
                    if targets[i] in pos_labels:
                        pos_targets.append(targets[i])
                        pos_preds.append(predictions[i])
                    else:
                        nonpos_targets.append(targets[i])
                        nonpos_preds.append(predictions[i])
                if len(pos_targets) > 0:
                    self.val_pos_accuracy(torch.tensor(pos_preds), torch.tensor(pos_targets))
                if len(nonpos_targets) > 0:
                    self.val_nonpos_accuracy(torch.tensor(nonpos_preds), torch.tensor(nonpos_targets))

        if self.do_shap:
            # If SHAP is not set up, initialize SHAP. We have to do this in the
            # first step instead of the start of validation because we need a
            # batch of data for shape inference.
            if self.shap is None:
                pos_labels = None
                if hasattr(self, 'pos_labels'):
                    pos_labels = self.pos_labels

                # Compute number of batches for PositionSHAP.progress()
                stop_at = self.trainer.num_val_batches[0]
                if self.args.shap_stop_at_batch is not None:
                    stop_at = self.args.shap_stop_at_batch
                start_at = 0
                if self.args.shap_start_at_batch is not None:
                    start_at = self.args.shap_start_at_batch
                n_batches = stop_at - start_at

                self.shap = analysis_shap.PositionSHAP(self, 'kernel', self.net,
                                                       inputs, self.args.num_classes,
                                                       pos_labels=pos_labels,
                                                       save_batches_to_pickles=self.shap_batched,
                                                       reduce_shap_samples=self.args.shap_fast,
                                                       on_cpu=self.args.shap_on_cpu,
                                                       batch_size=self.args.batch_size,
                                                       spatial_features=self.args.shap_spatial_features,
                                                       image_channels_features=self.args.shap_image_channels_features,
                                                       print_timings=self.args.shap_verbosity,
                                                       n_batches=n_batches,
                                                       debug_mode=self.args.shap_debug)

            if self.args.shap_single_batch_bg:
                if not self.shap.background_set:
                    # Set the first validation batch as SHAP background data
                    self.shap.set_background(inputs)

                # Apply SHAP analysis to whole batch
                self.shap.batch_shap(inputs, targets,
                                     batch_idx=batch_idx,
                                     indices=inds,
                                     predictions=predictions,
                                     force=not self.args.shap_continue)
            else:
                # Apply SHAP analysis by making folds out of batch and
                # alternately using either as foreground and background samples
                self.shap.batch_shap_and_bg(inputs, targets, batch_idx=batch_idx,
                                            indices=inds,
                                            predictions=predictions,
                                            fold_size=self.args.shap_fold_size,
                                            force=not self.args.shap_continue)

            progress, hours_left = self.shap.progress()
            print(f"SHAP progress: {progress:.2f}%, hours left: {hours_left:.2f}")
            self.log('p-shap/progress', progress)
            self.log('p-shap/hours_left', hours_left)

        return outputs

    def on_train_epoch_end(self):
        self.log('train/acc', self.train_acc.compute())
        self.train_acc.reset()

        if hasattr(self.net, 'transformer') and not isinstance(self.net.transformer.embeddings.gate, nn.Identity):
            # Log the gate value
            gate = self.net.transformer.embeddings.gate.gate
            self.log('train/gate/value', gate.item())
            self.log('train/gate/sigmoid_value', torch.sigmoid(gate).item())

        if hasattr(self.net, 'transformer') and self.args.pos_emb == 'rope':
            if self.args.pos_emb_gate:
                if hasattr(self.net.transformer.encoder, 'gate'):
                    # Log the shared gate value
                    val = self.net.transformer.encoder.gate.gate.item()
                    self.log(f'train/gate_mean/value', val)
                else:
                    # Log the gate values
                    vals = []
                    for i, block in enumerate(self.net.transformer.encoder.layer):
                        gate = block.attn.gate.gate
                        vals.append(gate.item())
                        self.log(f'train/gate_{i}/value', gate.item())
                    self.log('train/gate_mean/value', np.mean(vals))

            # Log the RoPE angles
            vals = []
            for i, block in enumerate(self.net.transformer.encoder.layer):
                freqs_cis = block.attn.freqs_cis
                freqs_cis_real = torch.view_as_real(freqs_cis)
                freqs_cis_angles = torch.atan2(freqs_cis_real[..., 1], freqs_cis_real[..., 0])
                vals.append(freqs_cis_angles.mean().item())
                self.log(f'train/rope/angles_{i}/value', vals[-1])
            self.log('train/rope/angles_mean/value', np.mean(vals))

        if hasattr(self, 'pos_labels'):
            self.log('train/pos_acc', self.train_pos_accuracy.compute())
            self.log('train/nonpos_acc', self.train_nonpos_accuracy.compute())
            self.train_pos_accuracy.reset()
            self.train_nonpos_accuracy.reset()

        if self.print_val_batches:
            print(f"Number of validation batches: {self.trainer.num_val_batches[0]}")

        if self.do_shap:
            # Set seed for SHAP analysis (needs to be done here because if it's
            # in the main body of train.py it somehow gets overwritten
            # somewhere...)
            pl.seed_everything(self.args.shap_seed)

            # Clear any previous SHAP analysis
            self.shap = None

        if args.log_first_batch:
            self.first_batch_val = True

        if self.c10cut_test_ood:
            experiment.val_loader.dataset.ood_mode = True

    def on_validation_epoch_end(self):
        if self.print_val_batches:
            return

        prefix = 'ood/' if self.c10cut_test_ood else ''

        if not self.do_shap or not self.args.shap_on_cpu:
            val_acc = self.val_acc.compute()
            self.val_acc.reset()
            self.log(prefix + 'val/acc', val_acc)
            if args.reported_val_acc is not None:
                self.log('diff_with_reported_val_acc', val_acc - args.reported_val_acc)

            if not self.args.multi_label:
                self.log(prefix + 'val/acc_topk', self.val_acc_topk.compute())
                self.val_acc_topk.reset()

            if hasattr(self, 'pos_labels'):
                self.log(prefix + 'val/pos_acc', self.val_pos_accuracy.compute())
                self.log(prefix + 'val/nonpos_acc', self.val_nonpos_accuracy.compute())
                self.val_pos_accuracy.reset()
                self.val_nonpos_accuracy.reset()

        self.log('epoch_time', time.time() - self.epoch_start)
        self.epoch_start = time.time()
        # self.first_batch_val = True

        if not self.args.multi_label:
            if not self.do_shap or not self.args.shap_on_cpu:
                preds = self.val_preds.compute().cpu().numpy()
                targets = self.val_targets.compute().cpu().numpy()
                classes = None
                if hasattr(self.val_loader.dataset, 'classes'):
                    classes = self.val_loader.dataset.classes
                # cm = wandb.plot.confusion_matrix(
                #     preds=preds, y_true=targets, class_names=classes,
                # )
                # self.trainer.logger.log_metrics({'val/confmat': cm})
                self.val_preds.reset()
                self.val_targets.reset()
        else:
            if not self.do_shap or not self.args.shap_on_cpu:
                preds = self.val_preds.compute().cpu().numpy()
                classes = None
                if hasattr(self.val_loader.dataset, 'classes'):
                    classes = self.val_loader.dataset.classes
                preds = wandb.Table(data=[[int(p)] for p in preds], columns=['predictions'])
                hist = wandb.plot.histogram(preds, "predictions")
                self.trainer.logger.log_metrics({'val/pred_histogram': hist})
                self.val_preds.reset()

        if self.do_shap:
            self.shap.finalize_shap()


if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pl.seed_everything(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.only_print_val_batches:
        args.training = False
        args.debug = True
        args.resume_id = None
        args.resume_from_artifact = None

    if args.val_batch_size is None:
        args.val_batch_size = args.batch_size

    if args.git_commit is None:
        args.git_commit = get_git_revision()

    if args.pos_emb_reset and (args.resume_id is not None or args.resume_from_checkpoint is not None or args.resume_from_artifact is not None):
        raise ValueError("Cannot reset position embedding weights when resuming from a checkpoint.")

    experiment = Experiment(args)

    # Autogenerate experiment name
    if args.exp_name is None:
        args.exp_name = f"pm-{args.dataset}-{args.model}-{args.model_version}-{args.internal_img_size}-{args.model_pretrained}"

    if args.accumulate_gradients != 1:
        print(f"WARNING! Gradient accumulation doesn't lead to identical results.")

    if not os.path.exists(args.wandb_dir):
        os.makedirs(args.wandb_dir, exist_ok=True)

    # Callback to print model summary
    modelsummary_callback = pl.callbacks.ModelSummary(
        max_depth=-1,
    )

    # Callback for model checkpointing:
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val/acc",
        mode="max",  # Save on best validation accuracy
        save_last=True,  # Keep track of the model at the last epoch
        verbose=True,
        dirpath=os.path.join(args.wandb_dir, 'checkpoints', args.exp_name + str(int(time.time()))),
    )

    # Callback for learning rate monitoring
    lrmonitor_callback = pl.callbacks.LearningRateMonitor()

    # Set cache dir to W&B logging directory
    os.environ["WANDB_DIR"] = args.wandb_dir
    os.environ["WANDB_CACHE_DIR"] = os.path.join(args.wandb_dir, 'cache')
    os.environ["WANDB_DATA_DIR"] = os.path.join(args.wandb_dir, 'data')
    wandb_logger = WandbLogger(
        save_dir=args.wandb_dir,
        project=args.wandb_project,
        name=args.exp_name,
        id=args.resume_id,
        resume="allow",
        log_model=True if not args.debug else None,  # used to save models to wandb during training
        # offline=args.debug,
        # Keyword args passed to wandb.init()
        entity='tudcv',
        config=args,
        tags=None if args.tags == '' else args.tags.split(','),
        mode='disabled' if args.debug else 'online',
    )

    # Automatically resume from last checkpoint
    if args.resume_id is not None and args.resume_from_artifact is None:
        args.resume_from_artifact = f"model-{args.resume_id}:latest"

    checkpoint_path = None
    if args.resume_from_artifact:
        artifact_path = os.path.join(args.wandb_dir, 'artifacts', args.resume_from_artifact)
        if 'WORLD_SIZE' not in os.environ:
            # CPU thread, spinning up training, should download artifact
            artifact_name = f"tudcv/{args.wandb_project}/{args.resume_from_artifact}"
            if not args.debug and wandb_logger is not None:
                artifact = wandb_logger.experiment.use_artifact(artifact_name)
            else:
                api = wandb.Api()
                artifact = api.artifact(artifact_name)
            directory = artifact.download(root=artifact_path)
        else:
            # GPU thread, we can assume artifact is already downloaded
            directory = artifact_path

        # Construct artifact path.
        checkpoint_path = os.path.join(directory, 'model.ckpt')
        print(f"Resuming training from {checkpoint_path}")

    if args.resume_from_checkpoint:
        checkpoint_path = args.resume_from_checkpoint
        print(f"Resuming training from {checkpoint_path}")

    train_settings = {
        'max_epochs': args.n_epochs,
        'precision': args.precision if device == "cuda" else "32",
        'gradient_clip_val': args.clip,
        'accumulate_grad_batches': args.accumulate_gradients,
    }
    if args.dryrun:
        train_settings['limit_train_batches'] = 5
        train_settings['max_epochs'] = 1
        train_settings['limit_val_batches'] = 2
    if not args.training:
        train_settings['max_epochs'] = 0
        train_settings['limit_train_batches'] = 0

    trainer = pl.Trainer(
        # gpus=torch.cuda.device_count(),
        devices="auto",
        accelerator="auto",
        strategy="ddp_find_unused_parameters_false",
        logger=wandb_logger,
        callbacks=[
            modelsummary_callback,
            checkpoint_callback,
            lrmonitor_callback,
        ],
        # Necessary for gradient-based analysis, to have gradients in validation
        inference_mode=False,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        **train_settings,
    )

    # if args.training:
    trainer.fit(experiment, ckpt_path=checkpoint_path)
    # else:
    #     if checkpoint_path is not None:
    #         raise ValueError("Resuming with --no-training causes slight issues with loading the model. "\
    #                          "Remove --no-training. If the original model was fully trained, it will not be trained further.")
    #         # NOTE: somehow, results are not identical using this method
    #         # experiment = Experiment.load_from_checkpoint(checkpoint_path, args=args, strict=True)
    #     trainer.fit(experiment, ckpt_path=checkpoint_path)

    if args.only_print_val_batches:
        experiment.print_val_batches = True
        trainer.validate(experiment)
        exit()

    if args.shap:
        experiment.do_shap = True
        args.test = True

    if args.test:
        trainer.validate(experiment)

    if args.shap:
        # Reset the global seed
        pl.seed_everything(args.seed)

    if args.c10cut_test_ood:
        experiment.do_shap = False
        # Test on OOD data
        experiment.c10cut_test_ood = True
        trainer.validate(experiment)

    if args.save_checkpoint:
        checkpoint_path = os.path.join(args.wandb_dir, 'checkpoints', args.exp_name + str(int(time.time())), 'model.ckpt')
        trainer.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
