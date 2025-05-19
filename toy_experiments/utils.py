import os
import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use(['seaborn'])
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Palatino"]})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

import sys
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
from analysis import inference_to_gradbased_analysis
from analysis import position_biases
from analysis import plot_position_biases, plot_position_biases_legend
from analysis import save_raw_attributions

import training


def collect_biases(model):
    # Collect all bias parameters from the model's layers
    biases = []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases.append(param)
            # print('bias', name, param)
    return biases

def collect_rel_pos_emb(model):
    # block1.attn.rel_pos.emb_w.relpos
    embs = []
    for name, param in model.named_parameters():
        if 'rel_pos_emb' in name:
            embs.append(param)
            # print('rpe', name)
    return embs

TOY_SOURCES = {
    'image': 'image_in',
    'pos_emb': 'pos_emb_in',
    'relpos': collect_rel_pos_emb,
    'bias': collect_biases,
}


def toy_all_analyses(model, analysis_images, analysis_labels, n_classes, seed, sources_available=['image', 'pos_emb', 'relpos', 'bias'], attribution_method='input_gradient_withnegative', aggregate_fn='sum', target='pred_class'):
    # Analysis
    n_samples = None
    targets = [target]
    shape = 'scalar'
    # targets = ['attn_out_sliced']
    # shape = 'head'
    patch_size = 1

    seed_biases = toy_analysis(targets, shape, patch_size, seed, n_samples,
                            model, analysis_images, analysis_labels,
                            sources_available=sources_available,
                            exclude_bias=True,
                            attribution_method=attribution_method,
                            aggregate_fn=aggregate_fn)

    # for key in seed_biases:
    #     biases[key].append(seed_biases[key])

    seed_biases_withbias = toy_analysis(targets, shape, patch_size, seed, n_samples,
                                    model, analysis_images, analysis_labels,
                                    sources_available=sources_available,
                                    exclude_bias=False,
                                    attribution_method=attribution_method,
                                    aggregate_fn=aggregate_fn)

    # for key in seed_biases_withbias:
    #     biases_withbias[key].append(seed_biases_withbias[key])

    # Per class
    cls_seed_biases = {}
    for c in range(n_classes):
        indices = analysis_labels == c
        indices = np.where(indices)[0]
        if len(indices) > 0:
            cls_seed_biases[c] = toy_analysis(targets, shape, patch_size, seed, n_samples,
                                        model, analysis_images[indices], analysis_labels[indices],
                                        sources_available=sources_available,
                                        exclude_bias=True,
                                        attribution_method=attribution_method,
                                        aggregate_fn=aggregate_fn)

        # for key in cls_seed_biases:
        #     cls_biases[c][key].append(cls_seed_biases[key])

    cls_seed_biases_withbias = {}
    for c in range(n_classes):
        indices = analysis_labels == c
        indices = np.where(indices)[0]
        if len(indices) > 0:
            cls_seed_biases_withbias[c] = toy_analysis(targets, shape, patch_size, seed, n_samples,
                                        model, analysis_images[indices], analysis_labels[indices],
                                        sources_available=sources_available,
                                        exclude_bias=False,
                                        attribution_method=attribution_method,
                                        aggregate_fn=aggregate_fn)

        # for key in cls_seed_biases_withbias:
        #     cls_biases_withbias[c][key].append(cls_seed_biases_withbias[key])

    return seed_biases, seed_biases_withbias, cls_seed_biases, cls_seed_biases_withbias

def sort_by_groups(sensitivities, sort_groups):
    order = []
    for group in sort_groups:
        if len(group) == 0:
            raise ValueError("Empty group")
        if len(group) == 1:
            order.append(group[0])
            continue
        group_p_sens = [sensitivities[c] for c in group]
        reorder = sorted(group_p_sens, key=lambda x: x[1])
        order.extend([int(c) for c, _ in reorder])
    return order

def toy_postprocess_analysis(biases, biases_withbias, cls_biases, cls_biases_withbias, seeds, n_classes, sort_by_appearance=True, sort_groups=None, report=True):
    if sort_by_appearance and sort_groups is None:
        # Sort all classes as a single group
        sort_groups = [range(n_classes)]

    if sort_groups is not None:
        if report:
            print("Sorting by appearance for these groups of classes: ", sort_groups)

        # Reorder classwise sensitivities
        new_cls_biases = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
        for run in range(len(seeds)):
            class_p_sens = [(c, cls_biases[c]['appearance'][run]) for c in range(n_classes)]
            order = sort_by_groups(class_p_sens, sort_groups)

            # Sanity checks
            if len(order) != n_classes:
                raise ValueError("Not all classes were provided in the sort_groups")

            # Apply new order
            for new_c, old_c in zip(range(n_classes), order):
                for key in ['bias', 'appearance', 'position', 'relative_position']:
                    if len(cls_biases[old_c][key]) <= run:
                        continue
                    new_cls_biases[new_c][key].append(cls_biases[old_c][key][run])

        new_cls_biases_withbias = {c: {'bias': [], 'appearance': [], 'position': [], 'relative_position': []} for c in range(n_classes)}
        for run in range(len(seeds)):
            class_p_sens = [(c, cls_biases_withbias[c]['appearance'][run]) for c in range(n_classes)]
            order = sort_by_groups(class_p_sens, sort_groups)

            # Sanity checks
            if len(order) != n_classes:
                raise ValueError("Not all classes were provided in the sort_groups")

            # Apply new order
            for new_c, old_c in zip(range(n_classes), order):
                for key in ['bias', 'appearance', 'position', 'relative_position']:
                    if len(cls_biases_withbias[old_c][key]) <= run:
                        continue
                    new_cls_biases_withbias[new_c][key].append(cls_biases_withbias[old_c][key][run])
    else:
        new_cls_biases = cls_biases
        new_cls_biases_withbias = cls_biases_withbias

    # Report mean, std over all biases
    if report:
        print('')
        print('Without bias:')
        for key in ['bias', 'appearance', 'position', 'relative_position']:
            mean = np.mean(biases[key])
            std = np.std(biases[key])
            if len(biases[key]) == 0:
                continue
            print(f"{key} (all): {mean*100:.2f} +- {std*100:.2f} ({', '.join(list(map(lambda x: f'{x*100:.2f}', np.array(biases[key]))))})")

            for c in range(n_classes):
                mean = np.mean(new_cls_biases[c][key])
                std = np.std(new_cls_biases[c][key])
                if len(new_cls_biases[c][key]) == 0:
                    continue
                print(f"{key} (c{c}) : {mean*100:.2f} +- {std*100:.2f} ({', '.join(list(map(lambda x: f'{x*100:.2f}', np.array(new_cls_biases[c][key]))))})")
        # results.append(entry)
        print('')

        print('With bias:')
        for key in ['bias', 'appearance', 'position', 'relative_position']:
            mean = np.mean(biases_withbias[key])
            std = np.std(biases_withbias[key])
            if len(biases_withbias[key]) == 0:
                continue
            print(f"{key} (all): {mean*100:.2f} +- {std*100:.2f} ({', '.join(list(map(lambda x: f'{x*100:.2f}', np.array(biases_withbias[key]))))})")

            for c in range(n_classes):
                mean = np.mean(new_cls_biases_withbias[c][key])
                std = np.std(new_cls_biases_withbias[c][key])
                if len(new_cls_biases_withbias[c][key]) == 0:
                    continue
                print(f"{key} (c{c}) : {mean*100:.2f} +- {std*100:.2f} ({', '.join(list(map(lambda x: f'{x*100:.2f}', np.array(new_cls_biases_withbias[c][key]))))})")
        print('')

    mean_biases = {key: np.mean(biases[key]) for key in ['bias', 'appearance', 'position', 'relative_position'] if len(biases[key]) > 0}
    std_biases = {key: np.std(biases[key]) for key in ['bias', 'appearance', 'position', 'relative_position'] if len(biases[key]) > 0}

    return mean_biases, std_biases


def toy_analysis(targets, shape, patch_size, seed, n_samples, model, test_images, test_labels, sources_available=['image', 'pos_emb', 'relpos', 'bias'], exclude_bias=False, attribution_method='input_gradient', aggregate_fn='sum'):
    sources = {}
    for source in sources_available:
        sources[source] = TOY_SOURCES[source]

    if torch.cuda.is_available():
        model.to('cpu')

    # Inference: all code related to specific dataset/models (in this case: toy).
    # Outputs all the necessary components for the gradient-based attribution to be
    # setting-agnostic.
    activations, gradients = \
        inference_to_gradbased_analysis(
            model, test_images, test_labels, sources,
            targets=targets,
            patch_size_y=patch_size,
            patch_size_x=patch_size,
            seed=seed, n_samples=n_samples,
            progress=False,
            lrp=False,
        )

    # ---
    # The code below this line should be reusable for any model and dataset. It
    # should use methods from the `analysis` package, so that they can be reused.

    relpos_dispersion_method = 'variance_ratio'

    for target in targets:
        # overall_biases, class_biases = \
        overall_biases = \
            position_biases(activations[target], gradients[target],
                            shape, test_labels, patch_size, 2,
                            attribution_method=attribution_method,
                            aggregate_fn=aggregate_fn,
                            relpos_dispersion_method=relpos_dispersion_method,
                            normalize_method='sum',
                            exclude_bias=exclude_bias,
                            lrp=False)

        # DEBUG
        if 'learned_relative_position' in overall_biases:
            del overall_biases['learned_relative_position']

        return overall_biases

        # # Print the measures
        # target_name = 'predicted class' if target == 'pred_class' else target
        # method_name = 'input-gradient' if attribution_method == 'input_gradient' else '<not specified>'
        # print(f"--- Position biases w.r.t. {target_name} using {method_name} attribution ---")

        # if shape == 'scalar':
        #     print("Mean over classes:")
        #     print(f"\tBias: {overall_biases['bias']:.2f}")
        #     print(f"\tAppearance: {overall_biases['appearance']:.2f}")
        #     if 'position' in overall_biases:
        #         print(f"\tPosition: {overall_biases['position']:.2f}")
        #         # print(f"\tLearned relative position: {overall_biases['learned_relative_position']:.2f}")
        #     if 'relative_position' in overall_biases:
        #         print(f"\tRelative position: {overall_biases['relative_position']:.2f}")
        #     # for c in range(2):
        #     #     print(f"Class {c}:")
        #     #     print(f"\tBias: {class_biases['bias'][c]:.2f}")
        #     #     print(f"\tAppearance: {class_biases['appearance'][c]:.2f}")
        #     #     if 'position' in class_biases:
        #     #         print(f"\tPosition: {class_biases['position'][c]:.2f}")
        #     #         print(f"\tLearned relative position: {class_biases['learned_relative_position'][c]:.2f}")
        #     #     if 'relative_position' in class_biases:
        #     #         print(f"\tRelative position: {class_biases['relative_position'][c]:.2f}")

        # elif shape == 'head':
        #     print("Mean over classes:")
        #     print(f"\tBias: " + ", ".join([f"{a:.2f}" for a in overall_biases['bias']]))
        #     print(f"\tAppearance: " + ", ".join([f"{a:.2f}" for a in overall_biases['appearance']]))
        #     if 'position' in overall_biases:
        #         print(f"\tPosition: " + ", ".join([f"{a:.2f}" for a in overall_biases['position']]))
        #         # print(f"\tLearned relative position: " + ", ".join([f"{a:.2f}" for a in overall_biases['learned_relative_position']]))
        #     if 'relative_position' in overall_biases:
        #         print(f"\tRelative position: " + ", ".join([f"{a:.2f}" for a in overall_biases['relative_position']]))
        #     # for c in range(2):
        #     #     print(f"Class {c}:")
        #     #     print(f"\tBias: " + ", ".join([f"{a:.2f}" for a in class_biases['bias'][c]]))
        #     #     print(f"\tAppearance: " + ", ".join([f"{a:.2f}" for a in class_biases['appearance'][c]]))
        #     #     if 'position' in class_biases:
        #     #         print(f"\tPosition: " + ", ".join([f"{a:.2f}" for a in class_biases['position'][c]]))
        #     #         print(f"\tLearned relative position: " + ", ".join([f"{a:.2f}" for a in class_biases['learned_relative_position'][c]]))
        #     #     if 'relative_position' in class_biases:
        #     #         print(f"\tRelative position: " + ", ".join([f"{a:.2f}" for a in class_biases['relative_position'][c]]))

        # else:
        #     raise NotImplementedError()

        # fig, axs = plt.subplots(4, 4, figsize=(6, 6), dpi=150)
        # gs = axs[0,0].get_gridspec()
        # for ax in axs:
        #     for a in ax:
        #         a.remove()
        # axbig = fig.add_subplot(gs[:, 0:2])
        # rc_context = plot_position_biases(axbig, overall_biases, size=0.4)
        # axbig.set_title("Overall")

        # # axclass0 = fig.add_subplot(gs[0:3, 2])
        # # rc_context = plot_position_biases(axclass0, {s: class_biases[s][0] for s in class_biases}, size=0.4)
        # # axclass0.set_title("Class 0")

        # # axclass1 = fig.add_subplot(gs[0:3, 3])
        # # rc_context = plot_position_biases(axclass1, {s: class_biases[s][1] for s in class_biases}, size=0.4)
        # # axclass1.set_title("Class 1")

        # axlegend = fig.add_subplot(gs[2, 2])
        # plot_position_biases_legend(axlegend)
        # axlegend.axis('off')

        # with plt.rc_context(rc_context):
        #     plt.show()


def toy_save_raw(filepath, target, seed, model, test_images, test_labels, sources_available=['image', 'pos_emb', 'relpos', 'bias']):
    patch_size = 1

    sources = {}
    for source in sources_available:
        sources[source] = TOY_SOURCES[source]

    if torch.cuda.is_available():
        model.to('cpu')

    # Inference: all code related to specific dataset/models (in this case: toy).
    # Outputs all the necessary components for the gradient-based attribution to be
    # setting-agnostic.
    activations, gradients = \
        inference_to_gradbased_analysis(
            model, test_images, test_labels, sources,
            targets=[target],
            patch_size_y=patch_size,
            patch_size_x=patch_size,
            seed=seed, n_samples=None,
            progress=False,
            lrp=False,
        )

    return save_raw_attributions(filepath, activations[target], gradients[target], images=test_images, labels=test_labels)


def visualize_sensitivities(model, seed, images, labels, ape=True, rpe=True, sort='relative_position', manual_sort=None, n=10000, plot_abs=False, snr_signal=None, only_snr=False, aggregate_fn='sum', plot_codomain='all'):
    if not only_snr:
        fig, axs = plt.subplots(n, 7, figsize=(14, n * 2))
        if n == 1:
            axs = np.array([axs])

    rpe = collect_rel_pos_emb(model)
    bias = collect_biases(model)
    sources_available = ['image', 'bias']
    if ape:
        sources_available.append('pos_emb')
    if rpe:
        sources_available.append('relpos')
    aggregate = lambda x: np.sum(x) if aggregate_fn == 'sum' else np.max(x)

    biases = []
    for i in range(len(images)):
        sensitivities = toy_analysis(
            ['pred_class'], 'scalar', 1, seed, None, model,
            images[i:i+1], labels[i:i+1],
            sources_available=sources_available,
            exclude_bias=False,
            attribution_method='input_gradient_withnegative',
            aggregate_fn=aggregate_fn,
        )
        image_saliency = (model.image_in[0] * model.image_in.grad[0]).detach().numpy()
        bias_saliency = [(term * term.grad).detach().numpy() for term in bias]
        ape_saliency = None
        rpe_saliency = None
        if ape:
            ape_saliency = (model.pos_emb_in * model.pos_emb_in.grad).detach().numpy()
        if rpe:
            rpe_saliency = [(term * term.grad).detach().numpy() for term in rpe]
        if plot_abs:
            image_saliency = np.absolute(image_saliency)
            bias_saliency = [np.absolute(s) for s in bias_saliency]
            if ape:
                ape_saliency = np.absolute(ape_saliency)
            if rpe:
                rpe_saliency = [np.absolute(s) for s in rpe_saliency]
        biases.append((i, sensitivities, image_saliency, bias_saliency, ape_saliency, rpe_saliency))

    if manual_sort is not None:
        biases = [biases[i] for i in manual_sort]
    elif sort is not None:
        biases = sorted(biases, key=lambda x: x[1][sort], reverse=True)

    if plot_codomain == 'all':
        i, sample_biases, image_saliency, bias_saliency, ape_saliency, rpe_saliency = biases[0]
        if rpe_saliency is None:
            rpe_saliency = []
        vmin = min([s.min() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])
        vmax = max([s.max() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])
        for i, sample_biases, image_saliency, bias_saliency, ape_saliency, rpe_saliency in biases[1:n]:
            if rpe_saliency is None:
                rpe_saliency = []
            vmin = min([vmin] + [s.min() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])
            vmax = max([vmax] + [s.max() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])

    signal = 0.
    signal_terms = 0
    noise = 0.
    noise_terms = 0
    for j, (i, sample_biases, image_saliency, bias_saliency, ape_saliency, rpe_saliency) in enumerate(biases[:n]):
        # Set all axis to off
        for a in axs[j]:
            a.axis('off')

        k = 0

        if plot_codomain == 'sample':
            if rpe_saliency is None:
                rpe_saliency = []
            vmin = min([s.min() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])
            vmax = max([s.max() for s in [image_saliency] + bias_saliency + [ape_saliency] + rpe_saliency if s is not None])
        elif plot_codomain == 'none':
            vmin = None
            vmax = None

        # print(image_saliency.shape, len(bias_saliency), [b.shape for b in bias_saliency], len(rpe_saliency), rpe_saliency[0].shape)

        if not only_snr:
            axs[j, k].imshow(images[i].permute(1, 2, 0))
            axs[j, k].set_title(f"Sample {i} / class {labels[i]}")
            axs[j, k].axis('off')
            k += 1

        # Image saliency SNR
        if snr_signal == 'appearance':
            signal_index_y = np.argmax(images[i,labels[i]].sum(axis=1))
            signal_index_x = np.argmax(images[i,labels[i]].sum(axis=0))
            image_signal = image_saliency[labels[i], signal_index_y, signal_index_x] ** 2
            signal += image_signal
            signal_terms += 1
            noise += (image_saliency ** 2).sum() - image_signal
            noise_terms += image_saliency.shape[1] * image_saliency.shape[2] - 1
            # print(i, signal_index_y, signal_index_x, signal, noise)

        if not only_snr:
            # Image saliency
            # im = axs[j, k].imshow(image_saliency.transpose(1, 2, 0), cmap='viridis', vmin=vmin, vmax=vmax)
            im = axs[j, k].imshow(image_saliency[0], cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[j, k])
            unnorm_sens = aggregate(np.absolute(image_saliency[0]))
            axs[j, k].set_title(f"A[0]: {sample_biases['appearance']:.2f} ({unnorm_sens:.4f})")
            axs[j, k].axis('off')
            k += 1
            im = axs[j, k].imshow(image_saliency[1], cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[j, k])
            unnorm_sens = aggregate(np.absolute(image_saliency[1]))
            axs[j, k].set_title(f"A[1]: {sample_biases['appearance']:.2f} ({unnorm_sens:.4f})")
            axs[j, k].axis('off')
            k += 1
            im = axs[j, k].imshow(image_saliency[2], cmap='viridis', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=axs[j, k])
            unnorm_sens = aggregate(np.absolute(image_saliency[2]))
            axs[j, k].set_title(f"A[2]: {sample_biases['appearance']:.2f} ({unnorm_sens:.4f})")
            axs[j, k].axis('off')
            k += 1

            # Bias saliency
            max_length = max([s.shape[0] for s in bias_saliency])
            bias_saliency = [np.pad(s, (0, max_length - s.shape[0]), 'constant', constant_values=0) for s in bias_saliency]
            bias_saliency = np.stack(bias_saliency, axis=0)
            im = axs[j, k].imshow(bias_saliency, cmap='viridis', vmin=vmin, vmax=vmax)
            unnorm_sens = aggregate(np.absolute(bias_saliency))
            axs[j, k].set_title(f"B: {sample_biases['bias']:.2f} ({unnorm_sens:.4f})")
            axs[j, k].axis('off')
            # Add colorbar
            cbar = fig.colorbar(im, ax=axs[j, k])
            k += 1

            # APE saliency
            if ape_saliency is not None:
                im = axs[j, k].imshow(ape_saliency[0,0], cmap='viridis', vmin=vmin, vmax=vmax)
                unnorm_sens = aggregate(np.absolute(ape_saliency))
                axs[j, k].set_title(f"AP: {sample_biases['position']:.2f} ({unnorm_sens:.4f})")
                axs[j, k].axis('off')
                # Add colorbar
                cbar = fig.colorbar(im, ax=axs[j, k])
                k += 1

            # RPE saliency
            if rpe_saliency is not None and len(rpe_saliency) > 0:
                rpe_as_image = np.concatenate([s.transpose(1, 0) for s in rpe_saliency], axis=0)
                im = axs[j, k].imshow(rpe_as_image, cmap='viridis', vmin=vmin, vmax=vmax)
                unnorm_sens = aggregate(np.absolute(rpe_as_image))
                axs[j, k].set_title(f"RP: {sample_biases['relative_position']:.2f} ({unnorm_sens:.4f})")
                axs[j, k].axis('off')
                # Add colorbar
                cbar = fig.colorbar(im, ax=axs[j, k])
                k += 1

    # Report SNR
    if snr_signal == 'appearance':
        # print(signal, signal_terms, noise, noise_terms)
        signal_mean = signal / float(signal_terms)
        noise_mean = noise / float(noise_terms)
        snr = 10. * np.log10(signal_mean / noise_mean)
        print(f"Image saliency SNR: {snr:.4f}dB")
        return snr