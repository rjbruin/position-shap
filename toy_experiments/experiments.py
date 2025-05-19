#
# APPEARANCE
#

import os
import time
import torch
import tqdm
import numpy as np

import training
import utils
from toy_models import TriViTalAppearance, TriViTalAbsolutePosition
from attribution_analysis import run_and_get_tokens
from analysis_shap import shap


def run(setting, seeds, n_epochs, lr, d, n_heads, n_classes, pos_emb, use_rel_pos,
        train_images, train_labels,
        test_images, test_labels,
        analysis_images, analysis_labels,
        # Training settings
        models=None, weight_decay=0., report_every_n=250,
        size=6, patch_size=1,
        # Model settings
        n_blocks=1, act_fn='gelu', norm=True, norm_fn='layernorm',
        # PE settings
        pos_add='add', pos_emb_factor=0.1, pos_emb_init='uniform', pos_emb_weight_decay=None,
        # Sensitivity settings
        sensitivity_analysis=True, save_raw=False,
        attribution_method='input_gradient_withnegative',
        aggregate_fn='sum', target='pred_class',
        sort_groups=None,
        # SHAP settings
        do_shap=None, shap_bg_folds='all',
        # Token retrieval settings
        get_tokens=False,
        # Other
        notebook=True, cleanup_models=False):
    """
    Train and test a single setting of a toy experiment, then run sensitivity
    and SHAP analysis and retrieve the image and PE tokens over the analysis
    samples.

    Arguments:
        - setting: str, one of ['appearance', 'absolute_position', 'relative_position', 'mixed_position'];
        - seeds: list of int, random seeds to use for each run;
        - n_epochs: int, number of epochs to train for;
        - lr: float, learning rate;
        - d: int, model hidden dimension;
        - n_heads: int, number of attention heads;
        - n_classes: int, number of classes in the dataset;
        - pos_emb: str, must be 'absolute'. Kept as argument for backwards compatibility, though it will raise an error;
        - use_rel_pos: bool, must be False. Kept as argument for backwards compatibility, though it will raise an error;
        - train_images: torch.Tensor, (N, Cin, H, W) tensor of training images;
        - train_labels: torch.Tensor, (N,) tensor of training labels;
        - test_images: torch.Tensor, (N, Cin, H, W) tensor of test images;
        - test_labels: torch.Tensor, (N,) tensor of test labels;
        - analysis_images: torch.Tensor, (N, Cin, H, W) tensor of analysis images;
        - analysis_labels: torch.Tensor, (N,) tensor of analysis labels;
        - Training settings (keyword arguments):
            - models: list of TriViTal models to analyse instead of training new models;
            - weight_decay: float, weight decay for the model;
            - report_every_n: int, number of epochs between reports;
            - size: int, size in pixels of the square input images;
            - patch_size: int, size of the patches in the model;
        - Model settings (keyword arguments):
            - n_blocks: int, number of layers in the model;
            - act_fn: str, activation function to use in the model;
            - norm: bool, whether to use normalization in the model;
            - norm_fn: str, normalization function to use in the model;
        - PE settings (keyword arguments):
            - pos_add: str, how to add positional embeddings. One of ['add', 'concat', 'concat_equald'];
            - pos_emb_factor: float, initialization factor for the positional embeddings;
            - pos_emb_init: str, initialization method for the positional embeddings;
            - pos_emb_weight_decay: float, weight decay for the positional embeddings;
        - Sensitivity settings (keyword arguments):
            - sensitivity_analysis: bool, whether to run sensitivity analysis;
            - save_raw: str or None, where to save raw saliency maps. Will not save if None;
            - attribution_method: str, method to use for sensitivity analysis;
            - aggregate_fn: str, function to use for aggregating attributions;
            - target: str, target for sensitivity analysis;
            - sort_groups: list of str, groups to sort the biases by in sensitivity post-processing;
        - SHAP settings (keyword arguments):
            - do_shap: str or None, whether to run SHAP analysis. One of ['deep', 'kernel'];
            - shap_bg_folds: str or int, background samples for SHAP analysis. One of ['all', int];
        - Token retrieval settings (keyword arguments):
            - get_tokens: bool, whether to retrieve the image and PE tokens and class labels over the analysis samples.
        - Other (keyword arguments):
            - notebook: bool, whether to use notebook-friendly progress bars;
            - cleanup_models: bool, whether to delete the models after training;

    Returns: dict, including:
        - 'accs': list of test accuracies;
        - if `sensitivity_analysis` and `save_raw` is None:
            - 'mean_biases': dict with keys 'bias', 'appearance', 'position', 'relative_position', each containing a list of mean biases over the seeds;
            - 'std_biases': dict with keys 'bias', 'appearance', 'position', 'relative_position', each containing a list of stddev biases over the seeds;
        - if `do_shap` is not None:
            - 'image_shap_values': torch.Tensor, (seeds, N, Cin, H, W, Cout) tensor of SHAP values for the image tokens;
            - 'pe_shap_values': torch.Tensor, (seeds, C, H, W) tensor of SHAP values for the PE tokens;
        - if `get_tokens`:
            - 'image_tokens': list of torch.Tensor, (N, Cin, H, W) tensors of image tokens over the analysis samples;
            - 'pe_tokens': list of torch.Tensor, (C, H, W) tensors of PE tokens over the analysis samples;
            - 'token_labels': list of torch.Tensor, (N,) tensors of class labels over the analysis samples
        - if not `cleanup_models`:
            - 'models': list of trained models.
    """
    assert pos_emb == 'absolute', f"pos_emb must be 'absolute', got {pos_emb}. Note that previously this method supported other PE methods, but the analysis added to this method doesn't."
    assert use_rel_pos == False, "Relative position embeddings are not supported in this method. Note that previously this method supported other PE methods, but the analysis added to this method doesn't."

    if save_raw and not sensitivity_analysis:
        raise ValueError("Cannot save raw saliency maps without sensitivity analysis")

    if notebook:
        import tqdm.notebook
        progress_bar = tqdm.notebook.tqdm
    else:
        import tqdm
        progress_bar = tqdm.tqdm

    compute_gradbased_attributions = True

    biases = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
    biases_withbias = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
    cls_biases = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
    cls_biases_withbias = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
    if models is None:
        models = []
    accs = []
    image_shap_values = []
    pe_shap_values = []
    image_tokens = []
    pe_tokens = []
    token_labels = []
    for seed in seeds:
        if seed < len(models):
            model = models[seed]
        else:
            torch.manual_seed(seed)
            # SHAP uses NumPy, so seed NumPy too
            np.random.seed(seed)

            if setting == 'appearance':
                model = TriViTalAbsolutePosition(n_classes=n_classes, d=d, size=size, patch_size=patch_size, n_blocks=n_blocks, n_heads=n_heads, use_rel_pos=use_rel_pos, pos_emb=pos_emb, pos_emb_factor=pos_emb_factor, pos_emb_init=pos_emb_init, input_d=train_images.shape[1], pos_add=pos_add, compute_gradbased_attr=compute_gradbased_attributions, act_fn=act_fn, norm=norm, norm_fn=norm_fn)
            elif setting in ['absolute_position', 'mixed_position']:
                model = TriViTalAbsolutePosition(n_classes=n_classes, d=d, size=size, patch_size=patch_size, n_blocks=n_blocks, n_heads=n_heads, use_rel_pos=use_rel_pos, pos_emb=pos_emb, pos_emb_factor=pos_emb_factor, pos_emb_init=pos_emb_init, input_d=train_images.shape[1], pos_add=pos_add, compute_gradbased_attr=compute_gradbased_attributions, act_fn=act_fn, norm=norm, norm_fn=norm_fn)
            elif setting == 'relative_position':
                model = TriViTalAbsolutePosition(n_classes=n_classes, d=d, size=size, patch_size=patch_size, n_blocks=n_blocks, n_heads=n_heads, use_rel_pos=use_rel_pos, pos_emb=pos_emb, pos_emb_factor=pos_emb_factor, pos_emb_init=pos_emb_init, input_d=train_images.shape[1], pos_add=pos_add, compute_gradbased_attr=compute_gradbased_attributions, act_fn=act_fn, norm=norm, norm_fn=norm_fn)
            acc = training.train_toy(model, train_images, train_labels, test_images, test_labels, epochs=n_epochs, report_every_n=report_every_n, batch_size=128, lr=lr, weight_decay=weight_decay, pos_emb_weight_decay=pos_emb_weight_decay)
            accs.append(acc)

        if get_tokens:
            seed_image_tokens, seed_pe_tokens, seed_labels = run_and_get_tokens(model, analysis_images, analysis_labels, batch_size=128)
            image_tokens.append(seed_image_tokens)
            pe_tokens.append(seed_pe_tokens)
            token_labels.append(seed_labels)

        if sensitivity_analysis:
            sources_available = ['image', 'bias']
            if pos_emb != 'none':
                sources_available.append('pos_emb')
            if use_rel_pos:
                sources_available.append('relpos')

            # Analysis to obtain measures, as per toy experiments
            seed_biases, seed_biases_withbias, seed_cls_biases, seed_cls_biases_withbias = \
                utils.toy_all_analyses(model, analysis_images, analysis_labels, n_classes, seed, sources_available=sources_available, attribution_method=attribution_method, aggregate_fn=aggregate_fn, target=target)

            for key in seed_biases:
                biases[key].append(seed_biases[key])
                for c in range(n_classes):
                    cls_biases[c][key].append(seed_cls_biases[c][key])

            for key in seed_biases_withbias:
                biases_withbias[key].append(seed_biases_withbias[key])
                for c in range(n_classes):
                    cls_biases_withbias[c][key].append(seed_cls_biases_withbias[c][key])

            if save_raw:
                # Analysis to save raw saliency maps
                filepath = f'./toy_saliency_maps/{save_raw}_{seed}.pt'
                if not os.path.exists(f'./toy_saliency_maps'):
                    os.makedirs(f'./toy_saliency_maps')
                saved = utils.toy_save_raw(
                    filepath, target, seed, model, analysis_images, analysis_labels,
                    sources_available=sources_available,
                )
                if not saved:
                    raise ValueError(f"Failed to save raw saliency maps for seed {seed}")

        if do_shap is not None:
            model_image_shaps, model_pe_shaps = shap(do_shap, model, analysis_images, shap_bg_folds, notebook=notebook)

            image_shap_values.append(model_image_shaps)
            pe_shap_values.append(model_pe_shaps)

        if cleanup_models:
            del model
        else:
            models.append(model)

    returns = {'accs': accs}

    if not cleanup_models:
        returns.update({'models': models})

    if get_tokens:
        # image_tokens = (seeds, N, Cin, H, W)
        # pe_tokens = (seeds, C, H, W)
        # token_labels = (seeds, N)
        returns.update({
            'image_tokens': image_tokens,
            'pe_tokens': pe_tokens,
            'token_labels': token_labels,
        })

    if sensitivity_analysis and save_raw is None:
        mean_biases, std_biases = utils.toy_postprocess_analysis(biases, biases_withbias, cls_biases, cls_biases_withbias, seeds, n_classes, sort_by_appearance=True, sort_groups=sort_groups)
        returns.update({
            'mean_biases': mean_biases,
            'std_biases': std_biases,
        })

    if do_shap:
        image_shap_values = torch.stack(image_shap_values)
        pe_shap_values = torch.stack(pe_shap_values)
        # shap_values = (seeds, N, Cin, H, W, Cout)
        returns.update({
            'image_shap_values': image_shap_values,
            'pe_shap_values': pe_shap_values,
        })

    return returns

# def run_appearance(seeds, n_epochs, lr, d, n_heads, n_classes, pos_emb, use_rel_pos, train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels, report_every_n=250, attribution_method='input_gradient_withnegative', aggregate_fn='sum', target='pred_class', models=None, pos_add='add', pos_emb_factor=0.1, weight_decay=0.):
#     biases = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
#     biases_withbias = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
#     cls_biases = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
#     cls_biases_withbias = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
#     if models is None:
#         models = []
#     accs = []
#     for seed in seeds:
#         if seed < len(models):
#             model = models[seed]
#         else:
#             torch.manual_seed(seed)
#             model = TriViTalAppearance(d=d, use_rel_pos=use_rel_pos, pos_emb=pos_emb, pos_add=pos_add, pos_emb_factor=pos_emb_factor)
#             acc = training.train_toy(model, train_images, train_labels, test_images, test_labels, epochs=n_epochs, report_every_n=report_every_n, batch_size=128, lr=lr, weight_decay=weight_decay)
#             accs.append(acc)

#         sources_available = ['image', 'bias']
#         if pos_emb != 'none':
#             sources_available.append('pos_emb')
#         if use_rel_pos:
#             sources_available.append('relpos')
#         seed_biases, seed_biases_withbias, seed_cls_biases, seed_cls_biases_withbias = \
#             utils.toy_all_analyses(model, analysis_images, analysis_labels, n_classes, seed, sources_available=sources_available, attribution_method=attribution_method, aggregate_fn=aggregate_fn, target=target)

#         for key in seed_biases:
#             biases[key].append(seed_biases[key])
#             for c in range(n_classes):
#                 cls_biases[c][key].append(seed_cls_biases[c][key])

#         for key in seed_biases_withbias:
#             biases_withbias[key].append(seed_biases_withbias[key])
#             for c in range(n_classes):
#                 cls_biases_withbias[c][key].append(seed_cls_biases_withbias[c][key])

#         models.append(model)

#     mean_biases = utils.toy_postprocess_analysis(biases, biases_withbias, cls_biases, cls_biases_withbias, seeds, n_classes, sort_by_appearance=True)

#     return models, accs, mean_biases

# #
# # ABSOLUTE POSITION
# #

# def run_absolute_position(seeds, n_epochs, lr, d, n_heads, n_classes, pos_emb, use_rel_pos, train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels, report_every_n=250, attribution_method='input_gradient_withnegative', aggregate_fn='sum', target='pred_class', models=None, pos_add='add', pos_emb_factor=0.1, weight_decay=0.):
#     biases = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
#     biases_withbias = {'bias': [], 'appearance': [], 'position': [], 'relative_position': []}
#     cls_biases = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
#     cls_biases_withbias = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
#     accs = []
#     for seed in seeds:
#         torch.manual_seed(seed)
#         model = TriViTalAbsolutePosition(n_classes=n_classes, d=d, size=6, n_heads=n_heads, use_rel_pos=use_rel_pos, pos_emb=pos_emb)
#         acc = training.train_toy(model, train_images, train_labels, test_images, test_labels, epochs=n_epochs, report_every_n=report_every_n, batch_size=128, lr=lr, weight_decay=weight_decay)
#         accs.append(acc)

#         sources_available = ['image', 'bias']
#         if pos_emb != 'none':
#             sources_available.append('pos_emb')
#         if use_rel_pos:
#             sources_available.append('relpos')
#         seed_biases, seed_biases_withbias, seed_cls_biases, seed_cls_biases_withbias = \
#             utils.toy_all_analyses(model, analysis_images, analysis_labels, n_classes, seed, sources_available=sources_available, attribution_method=attribution_method)

#         for key in seed_biases:
#             biases[key].append(seed_biases[key])
#             for c in range(n_classes):
#                 cls_biases[c][key].append(seed_cls_biases[c][key])

#         for key in seed_biases_withbias:
#             biases_withbias[key].append(seed_biases_withbias[key])
#             for c in range(n_classes):
#                 cls_biases_withbias[c][key].append(seed_cls_biases_withbias[c][key])

#     mean_biases = utils.toy_postprocess_analysis(biases, biases_withbias, cls_biases, cls_biases_withbias, seeds, n_classes, sort_by_appearance=True)

#     return models, accs, mean_biases