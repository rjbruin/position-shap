import torch
import numpy as np
import pandas as pd

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
import os
sys.path.append(os.path.abspath('..'))
sys.path.append(os.path.abspath('.'))
from analysis import load_raw_attributions, discover_batched_attributions


def smap_postprocessing(smap, mode='abs'):
    """Supports multiple post-processing methods.

    Arguments:
        smap (torch.Tensor): Saliency map;
        mode (str): Post-processing method. Supports:
            - "abs": `abs(smap)`; default post-processing for FG;
            - "none": no post-processing.

    Returns:
        torch.Tensor: Post-processed saliency map.
    """
    if mode == 'abs':
        return smap.abs()
    elif mode == 'none':
        return smap
    elif mode == 'max':
        return torch.maximum(smap, torch.zeros_like(smap))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

def attribution_stats(filepath, batched, rank_by='pos_emb', postprocessing='abs', notebook=True):
    if batched:
        attr_files = discover_batched_attributions(filepath)
    else:
        attr_files = [filepath]

    if notebook:
        import tqdm.notebook
        progress_bar = tqdm.notebook.tqdm
    else:
        import tqdm
        progress_bar = tqdm.tqdm

    index = []
    i = 0
    for bi in progress_bar(range(len(attr_files)), desc='Pre-processing'):
        raw_attributions = load_raw_attributions(attr_files[bi])
        n_batch_samples = len(raw_attributions['saliency']['image'])

        for ib in range(n_batch_samples):
            # Compute saliency maps to compute normalization factor
            im_smap = smap_postprocessing(raw_attributions['saliency']['image'][ib], mode=postprocessing)
            bias_smap = smap_postprocessing(raw_attributions['saliency']['bias'][ib], mode=postprocessing)
            pos_emb_smap = smap_postprocessing(raw_attributions['saliency']['pos_emb'][ib], mode=postprocessing)
            # Compute magnitude of each source
            ims_mag = im_smap.sum()
            bias_mag = bias_smap.sum()
            pos_emb_mag = pos_emb_smap.sum()

            del im_smap
            del bias_smap
            del pos_emb_smap

            # Overall index, batch index, index within batch, magnitudes...
            index.append((i, bi, ib, ims_mag, bias_mag, pos_emb_mag, pos_emb_mag / (ims_mag + bias_mag + pos_emb_mag), raw_attributions['labels'][ib]))
            i += 1


    if rank_by == 'pos_emb':
        index = sorted(index, key=lambda x: x[5], reverse=True)
    elif rank_by == 'pos_emb_all':
        index = sorted(index, key=lambda x: x[6], reverse=True)
    index = [(r+1, i, bi, ib, ims_mag, bias_mag, pos_emb_mag, pe_all_mag, label) for r, (i, bi, ib, ims_mag, bias_mag, pos_emb_mag, pe_all_mag, label) in enumerate(index)]
    return index

def plot_attribution_samples(attr_files, index, image_source, dataset_images, top_n, bot_n, normalize_global=False, pos_emb_emb_dim=0, postprocessing='abs', notebook=True):
    if notebook:
        import tqdm.notebook
        progress_bar = tqdm.notebook.tqdm
    else:
        import tqdm
        progress_bar = tqdm.tqdm

    fig, axs = plt.subplots(top_n + bot_n, 4, figsize=(2 * 4, 1 + (top_n + bot_n) * 2), dpi=120)

    n_samples = len(index)
    for i, (r, ind, bi, ib, im_mag, bias_mag, pos_emb_mag, pe_all_mag, label) in progress_bar(enumerate(index[:top_n] + index[n_samples - bot_n:]), desc='Plotting', total=top_n + bot_n):
        # Load data for corresponding batch
        raw_attributions = load_raw_attributions(attr_files[bi])
        # Post-process saliency maps
        im_smap = smap_postprocessing(raw_attributions['saliency']['image'][ib], mode=postprocessing)
        bias_smap = smap_postprocessing(raw_attributions['saliency']['bias'][ib], mode=postprocessing)
        pos_emb_smap = smap_postprocessing(raw_attributions['saliency']['pos_emb'][ib], mode=postprocessing)
        # Remove channel/embedding dimension from image and pos_emb saliency map
        im_smap = im_smap.sum(dim=0)
        pos_emb_smap = pos_emb_smap.sum(dim=pos_emb_emb_dim)
        # Compute norm (after computing magnitude): maximum element
        norm = torch.maximum(im_smap.max(), torch.maximum(bias_smap.max(), pos_emb_smap.max()))
        # Apply norm
        if normalize_global:
            im_smap = im_smap / norm
            bias_smap = bias_smap / norm
            pos_emb_smap = pos_emb_smap / norm
        else:
            im_smap = im_smap / im_smap.max()
            bias_smap = bias_smap / bias_smap.max()
            pos_emb_smap = pos_emb_smap / pos_emb_smap.max()

        j = 0
        if image_source == 'saved':
            image = raw_attributions['images'][ib]
            image = image.permute((1, 2, 0))
        elif image_source == 'dataset':
            image = dataset_images[ind]
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image)
            else:
                image = image.permute((1, 2, 0))
        axs[i,j].imshow(image, vmin=0., vmax=1.)
        axs[i,j].set_title(f"Rank {r} sample (class {label})")
        axs[i,j].axis('off')

        j += 1
        axs[i,j].imshow(im_smap, cmap='hot', vmin=0., vmax=1.)
        axs[i,j].set_title(f"{im_mag:.3f} ({im_mag * 100. / (im_mag + bias_mag + pos_emb_mag):.1f}\%)")
        axs[i,j].axis('off')

        j += 1
        # axs[i,j].imshow(im_smap, cmap='hot')
        axs[i,j].set_title(f"(bias) {bias_mag:.3f} ({bias_mag * 100. / (im_mag + bias_mag + pos_emb_mag):.1f}\%)")
        axs[i,j].axis('off')

        j += 1
        axs[i,j].imshow(pos_emb_smap, cmap='hot', vmin=0., vmax=1.)
        axs[i,j].set_title(f"{pos_emb_mag:.3f} ({pos_emb_mag * 100. / (im_mag + bias_mag + pos_emb_mag):.1f}\%)")
        axs[i,j].axis('off')

    plt.tight_layout()

    return fig

def sens_scatterplots(results_container, indexes, seeds, accs, axs, label_names=['R/L', 'R/R', 'G', 'B'], pos_labels=[0, 1]):
    results = results_container

    for i, seed in enumerate(seeds):
        index = indexes[i]
        acc = accs[i]

        # Convert list of tuples "index" to dataframe
        df = pd.DataFrame(index, columns=['rank', 'index', 'batch', 'index_in_batch', 'image', 'bias', 'pos_emb', 'pos_emb_all', 'label'])

        # Convert columns from tensor to float
        df['image'] = df['image'].apply(lambda x: x.item())
        df['bias'] = df['bias'].apply(lambda x: x.item())
        df['pos_emb'] = df['pos_emb'].apply(lambda x: x.item())
        df['pos_emb_frac'] = df['pos_emb_all'].apply(lambda x: x.item())
        df['label'] = df['label'].apply(lambda x: x.item())
        df['uses-PE'] = df['label'].apply(lambda x: x in pos_labels)

        df['image + bias'] = df['image'] + df['bias']
        df['all'] = df['image'] + df['bias'] + df['pos_emb']

        # Label markup: make markers represent color and position of classes
        df['label'] = df['label'].apply(lambda x: int(x))
        df['PE-label'] = df['label'].apply(lambda x: "left" if x == 0 else ("right" if x == 1 else "no"))
        # Palette: colors to match colors in the task (red, red, green, blue)
        palette = sns.color_palette("tab10")
        palette = [palette[3], palette[3], palette[2], palette[0]]
        # Detect "bias" class: class with lowest total attribution
        class_all_max = df.groupby('label')['all'].mean()
        bias_class = class_all_max.idxmin()
        # Mark bias_class with gray
        palette[bias_class] = (0.5, 0.5, 0.5)
        # Label names
        df['label'] = df['label'].apply(lambda x: label_names[x])

        # pos_emb percentage of all vs. all
        sns.scatterplot(data=df, x='pos_emb_frac', y='all', hue='label', hue_order=label_names, style='PE-label', style_order=["left", "right", "no"], markers=["<", ">", "o"], palette=palette, ax=axs[0,i])
        axs[0,i].set_xlim(-0.05, 1.05)

        # # Scatterplot with identical limits on x and y axes
        # sns.scatterplot(data=df, x='pos_emb', y='image + bias', hue='label', style='uses-PE', style_order=[True, False], ax=axs[1,i])
        # axs[1,i].set_title(f"Seed {seed}")
        # # Set axs[i] axes to same range
        # limit = max(axs[1,i].get_xlim()[1], axs[1,i].get_ylim()[1])
        # axs[1,i].set_xlim(0, limit)
        # axs[1,i].set_ylim(0, limit)

        # # Same plot with fitted axes, for better visual discrimination
        # sns.scatterplot(data=df, x='pos_emb', y='image + bias', hue='label', style='uses-PE', style_order=[True, False], ax=axs[2,i])

        # If the average fraction of pos_emb is higher for both PE-using classes than for both PE-not-using classes, mark this in the name and on the plot
        class_0_mean = df[df['label'] == 0]['pos_emb_frac'].mean()
        class_1_mean = df[df['label'] == 1]['pos_emb_frac'].mean()
        class_2_mean = df[df['label'] == 2]['pos_emb_frac'].mean()
        class_3_mean = df[df['label'] == 3]['pos_emb_frac'].mean()
        sens_success = False
        if acc < 1.0:
            axs[0,i].set_title(f"Seed {seed} (TR. FAIL: {acc:.2f}) - FG - PE \% of total")
        if class_0_mean > class_2_mean and class_1_mean > class_2_mean and class_0_mean > class_3_mean and class_1_mean > class_3_mean:
            if not acc < 1.0:
                axs[0,i].set_title(f"Seed {seed} (SUCCESS) - FG - PE \% of total")
            sens_success = True
        else:
            axs[0,i].set_title(f"Seed {seed} - FG - PE \% of total")

        # Save results for further analysis
        results[i]['sens_df'] = df
        results[i]['acc'] = acc
        results[i]['train_success'] = acc == 1.0
        results[i]['sens_class_0_mean'] = class_0_mean
        results[i]['sens_class_1_mean'] = class_1_mean
        results[i]['sens_class_2_mean'] = class_2_mean
        results[i]['sens_class_3_mean'] = class_3_mean
        results[i]['sens_success'] = sens_success

    return results

def shap_scatterplots(image_shap_values, pe_shap_values, seeds, axs, n_classes, test_labels, start_row=0, plot_pe_shap_sum=False, plot_pe_sum_frac=True, plot_label_sum_frac=False, label_names=['R/L', 'R/R', 'G', 'B'], pos_labels=[0, 1]):
    results = []
    for i, seed in enumerate(seeds):
        seed_results = {}

        seed_image_shaps = image_shap_values[i]
        seed_pe_shaps = pe_shap_values[i]

        # Detect whether SHAP values are given for spatial dimensions or not.
        spatial_outputs = image_shap_values.dim() == 5

        df = []
        for c in range(n_classes):
            class_mask = test_labels == c
            if spatial_outputs:
                class_shaps = seed_image_shaps[:,:,:,:,c]
            else:
                class_shaps = seed_image_shaps[:,:,c]
            class_shaps = class_shaps[class_mask]
            # class_shaps = (N_c, Cin, H, W) or (N_c, Cin)

            if spatial_outputs:
                class_pe_shaps = seed_pe_shaps[:,:,:,:,c]
            else:
                class_pe_shaps = seed_pe_shaps[:,:,c]
            class_pe_shaps = class_pe_shaps[class_mask]
            class_pe_shaps = class_pe_shaps.sum(dim=1)
            # class_pe_shaps = (N_c, H, W) or (N_c)

            if spatial_outputs:
                image_shap_max = class_shaps.abs().max(dim=3)[0].max(dim=2)[0].max(dim=1)[0]
                image_shap_sum = class_shaps.abs().sum(dim=3).sum(dim=2).sum(dim=1)
                r_shap_max = class_shaps[:,0].abs().max(dim=2)[0].max(dim=1)[0]
                g_shap_max = class_shaps[:,1].abs().max(dim=2)[0].max(dim=1)[0]
                b_shap_max = class_shaps[:,2].abs().max(dim=2)[0].max(dim=1)[0]
                pe_shap_max = class_pe_shaps.abs().max(dim=2)[0].max(dim=1)[0]
                pe_shap_sum = class_pe_shaps.abs().sum(dim=2).sum(dim=1)
            else:
                image_shap_max = class_shaps.abs().max(dim=1)[0]
                image_shap_sum = class_shaps.abs().sum(dim=1)
                r_shap_max = class_shaps[:,0].abs()
                g_shap_max = class_shaps[:,1].abs()
                b_shap_max = class_shaps[:,2].abs()
                pe_shap_max = class_pe_shaps.abs()
                pe_shap_sum = class_pe_shaps.abs()
            total_shap_max = torch.stack([image_shap_max, pe_shap_max], dim=1).max(dim=1)[0]
            total_shap_sum = image_shap_sum + pe_shap_sum

            # label_shap_max: maximum SHAP value for the image channel that
            # corresponds to the label
            label = torch.full((len(class_shaps),), c)
            c_label = 0 if c < 2 else c - 1
            c_nonlabel = list(set(range(3)) - {c_label})
            if spatial_outputs:
                label_shap_max = class_shaps[:,c_label].abs().max(dim=2)[0].max(dim=1)[0]
                label_shap_sum = class_shaps[:,c_label].abs().sum(dim=2).sum(dim=1)
                nonlabel_shap_sum = torch.stack([class_shaps[:,nc].abs().sum(dim=2).sum(dim=1) for nc in c_nonlabel], dim=1).sum(dim=1)
            else:
                label_shap_max = class_shaps[:,c_label].abs()
                label_shap_sum = class_shaps[:,c_label].abs()
                nonlabel_shap_sum = torch.stack([class_shaps[:,nc].abs() for nc in c_nonlabel], dim=1).sum(dim=1)

            # data = (N_c, 5)
            data = torch.stack([total_shap_max, total_shap_sum, image_shap_max, image_shap_sum, r_shap_max, g_shap_max, b_shap_max, pe_shap_max, pe_shap_sum, label_shap_max, label_shap_sum, nonlabel_shap_sum, label], dim=1)
            df.append(data)

        df = torch.cat(df, dim=0)
        df = pd.DataFrame(df.numpy(), columns=['total_shap_max', 'total_shap_sum', 'image_shap_max', 'image_shap_sum', 'r_shap_max', 'g_shap_max', 'b_shap_max', 'pe_shap_max', 'pe_shap_sum', 'label_shap_max', 'label_shap_sum', 'nonlabel_shap_sum', 'label'])

        # Convert columns from tensor to float
        df['PE-label'] = df['label'].apply(lambda x: "left" if x == 0 else ("right" if x == 1 else "no"))
        df['label_max_frac'] = df['label_shap_max'] / df['total_shap_max']
        df['label_sum_frac'] = df['label_shap_sum'] / df['total_shap_sum']
        df['nonlabel_sum_frac'] = df['nonlabel_shap_sum'] / df['image_shap_sum']
        df['pe_max_frac'] = df['pe_shap_max'] / df['total_shap_max']
        df['pe_sum_frac'] = df['pe_shap_sum'] / df['total_shap_sum']
        df['label'] = df['label'].apply(lambda x: int(x))
        df['uses-PE'] = df['label'].apply(lambda x: x in pos_labels)

        # Detect "bias" class: class with lowest total_shap_max
        class_total_shap_max = df.groupby('label')['total_shap_max'].mean()
        bias_class = class_total_shap_max.idxmin()

        # Palette: colors to match colors in the task (red, red, green, blue)
        palette = sns.color_palette("tab10")
        palette = [palette[3], palette[3], palette[2], palette[0]]
        # Mark bias_class with gray
        palette[bias_class] = (0.5, 0.5, 0.5)

        df['label'] = df['label'].apply(lambda x: label_names[x])

        row = start_row
        if plot_pe_shap_sum:
            # PE SHAP max vs. overall SHAP max
            sns.scatterplot(data=df, x='pe_shap_sum', y='total_shap_sum', hue='label', hue_order=label_names, style='PE-label', style_order=["left", "right", "no"], markers=["<", ">", "o"], ax=axs[row,i], palette=palette)
            axs[row,i].plot([0, df['pe_shap_sum'].max()], [0, df['total_shap_sum'].max()], color='gray', linestyle='--', linewidth=1., alpha=0.5)
            if i == 0:
                axs[row,i].set_title(f"Seed {seed} - SHAP - PE vs. total")
            else:
                axs[row,i].set_title(f"Seed {seed}")
            # axs[row,i].set_xlim(-0.05, 1.05)
            row += 1

        # PE SHAP frac vs. overall SHAP max
        if plot_pe_sum_frac:
            sns.scatterplot(data=df, x='pe_sum_frac', y='total_shap_sum', hue='label', style='PE-label', style_order=["left", "right", "no"], markers=["<", ">", "o"], ax=axs[row,i], palette=palette)
            if i == 0:
                axs[row,i].set_title(f"Seed {seed} - SHAP - PE \% of total")
            else:
                axs[row,i].set_title(f"Seed {seed}")
            axs[row,i].set_xlim(-0.05, 1.05)
            row += 1

        # Correct label SHAP max vs. overall SHAP max
        if plot_label_sum_frac:
            sns.scatterplot(data=df, x='label_sum_frac', y='image_shap_sum', hue='label', style='PE-label', style_order=["left", "right", "no"], markers=["<", ">", "o"], ax=axs[row,i], palette=palette)
            if i == 0:
                axs[row,i].set_title(f"Seed {seed} - SHAP - channel \% of image, for right channel")
            else:
                axs[row,i].set_title(f"Seed {seed}")
            axs[row,i].set_xlim(-0.05, 1.05)
            row += 1

        # # Incorrect label SHAP sum vs. overall SHAP sum
        # sns.scatterplot(data=df, x='nonlabel_sum_frac', y='image_shap_sum', hue='label', style='PE-label', style_order=["left", "right", "no"], markers=["<", ">", "o"], ax=axs[row,i], palette=palette)
        # axs[row,i].set_xlim(-0.05, 1.05)
        # if i == 0:
        #     axs[row,i].set_title(f"Seed {seed} - channel \% of image, for wrong channels")
        # else:
        #     axs[row,i].set_title(f"Seed {seed}")

        class_0_mean = df[df['label'] == label_names[0]]['pe_sum_frac'].mean()
        class_1_mean = df[df['label'] == label_names[1]]['pe_sum_frac'].mean()
        class_2_mean = df[df['label'] == label_names[2]]['pe_sum_frac'].mean()
        class_3_mean = df[df['label'] == label_names[3]]['pe_sum_frac'].mean()

        seed_results['shap_success'] = False
        if class_0_mean > class_2_mean and class_1_mean > class_2_mean and class_0_mean > class_3_mean and class_1_mean > class_3_mean:
            seed_results['shap_success'] = True

        seed_results['shap_df'] = df
        seed_results['shap_class_0_mean'] = class_0_mean
        seed_results['shap_class_1_mean'] = class_1_mean
        seed_results['shap_class_2_mean'] = class_2_mean
        seed_results['shap_class_3_mean'] = class_3_mean

        results.append(seed_results)

    return results

def run_and_get_tokens(model, images, labels, batch_size):
    """
    Returns:
        image_tokens (torch.Tensor): Image tokens (N, C, H, W);
        pe_tokens (torch.Tensor): PE tokens (C_pe, H, W);
        all_labels (torch.Tensor): All labels (N).
    """
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images, labels), batch_size=batch_size)

    cuda = False
    if torch.cuda.is_available():
        cuda = True
        model.cuda()
        print("Using CUDA.")

    model.eval()
    model.store_tokens = True

    image_tokens = []
    pe_tokens = None
    all_labels = []
    for batch in test_loader:
        images, labels = batch
        if cuda:
            images = images.cuda()
            labels = labels.cuda()

        _ = model(images)

        image_tokens.append(model.image_tokens)
        all_labels.append(labels)

        batch_pe_tokens = model.pe_tokens
        assert torch.equal(batch_pe_tokens, model.pe_tokens), "PE tokens are not the same for all batches"
        pe_tokens = batch_pe_tokens

    # image_tokens = (N, C, H, W)
    image_tokens = torch.cat(image_tokens, dim=0)
    # pe_tokens = (C_pe, H, W)
    # all_labels = (N)
    all_labels = torch.cat(all_labels, dim=0)

    model.store_tokens = False

    return image_tokens, pe_tokens, all_labels