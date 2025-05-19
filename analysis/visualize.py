import math
import torch
from matplotlib.patches import Rectangle, Patch
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def colorbar(ax, im):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

def visualize_gradbased_attribution_wrt_attn_matrix(
        images, labels, attributions, weights, task_name,
        head=0,
        query_rule=None,
        highlight_query_rules=None,
        highlight_key_rules=None,
        shared_domain_scores=False,
        show_sample=True,
        show_original_weights=True,
        show_factor_scores=True,
        show_merged=False,
        show_weighed_top1=False,
        weight_factor_by_attn=True,
        max_attribution=None,
    ):
    """
    Given grad-based attributions for an attention matrix, visualize the
    attributions. By default, shows only sample, attributions and attribution
    scores.
    """
    N = len(images)
    n_cols = int(show_sample) + int(show_original_weights) + int(show_factor_scores) * 2 + int(show_merged) + int(show_weighed_top1)
    fig, axs = plt.subplots(N, n_cols, figsize=(2 + n_cols * 2, N * 2), dpi=120)
    if N == 1:
        axs = axs.reshape((1, n_cols))

    for b in range(N):
        j = 0

        image = images[b].detach()
        label = labels[b].detach()
        attn_weights = weights[b].detach()
        image_attr, pos_attr = attributions[b]

        if show_sample:
            axs[b,j].imshow(image.permute((1, 2, 0)))
            axs[b,j].set_title(f"Sample (label: {label})")
            axs[b,j].axis('off')
            j += 1

        num_heads = attn_weights.shape[0]
        num_tokens = attn_weights.shape[1]
        fm_size = int(math.sqrt(attn_weights.shape[1]))
        attn_weights = attn_weights[head].detach().cpu()

        query = None
        if query_rule is not None:
            queries = query_rule(image, label)
            if len(queries) > 1:
                raise ValueError(f"Query rule cannot return multiple queries. Got {len(queries)} queries.")
            if len(queries) == 0:
                raise ValueError(f"Query rule returned no queries.")
            query = queries[0]

            if highlight_query_rules is not None:
                print("WARNING: Cannot highlight query tokens when filtering attention weights by query. Ignoring highlight_query_rules.")

        im_scores = image_attr[head].detach()
        pos_scores = pos_attr[head].detach()
        head_max_attribution = max_attribution[head] if max_attribution is not None else None

        vis_ss_scores = im_scores
        vis_sp_scores = pos_scores

        vmin = vmax = None
        if head_max_attribution is not None:
            vmin = 0.0
            vmax = head_max_attribution
        elif shared_domain_scores:
            vmin = torch.min(torch.stack([vis_ss_scores, vis_sp_scores]))
            vmax = torch.max(torch.stack([vis_ss_scores, vis_sp_scores]))
            largest_v = max(abs(vmin), abs(vmax))
            vmin = -largest_v
            vmax = largest_v

        #
        # Flat
        #

        def plot_query_by_key(ax, image, label, highlight_query_rules, highlight_key_rules):
            if highlight_query_rules is not None:
                for (color, rule) in highlight_query_rules:
                    rows = rule(image, label)
                    for row in rows:
                        rect = Rectangle((- 0.5, row - 0.5), num_tokens, 1, linewidth=1.5, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

            if highlight_key_rules is not None:
                for (color, rule) in highlight_key_rules:
                    columns = rule(image, label)
                    for column in columns:
                        rect = Rectangle((column - 0.5, - 0.5), 1, num_tokens, linewidth=1.5, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

        def plot_key(ax, image, label, highlight_key_rules):
            if highlight_key_rules is not None:
                for (color, rule) in highlight_key_rules:
                    indices = rule(image, label)
                    for index in indices:
                        row, column = torch.div(index, 5, rounding_mode='floor'), index % 5
                        rect = Rectangle((column - 0.5, row - 0.5), 1, 1, linewidth=1.5, edgecolor=color, facecolor='none')
                        ax.add_patch(rect)

        def plot_tokens(ax, image, label, highlight_query_rules=None, highlight_key_rules=None):
            if query is not None:
                # if highlight_query_rules is not None:
                #     print("WARNING: Cannot highlight query tokens when filtering attention weights by query. Ignoring highlight_query_rules.")
                plot_key(ax, image, label, highlight_key_rules)
            else:
                plot_query_by_key(ax, image, label, highlight_query_rules, highlight_key_rules)

        def to_query_weights(weights, query):
            if query is None:
                return weights
            weights = weights[query].reshape(num_tokens, fm_size, -1)
            return weights.squeeze()


        if show_original_weights:
            im = axs[b,j].imshow(to_query_weights(attn_weights, query), cmap='viridis')
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Attn. matrix" + (f" ({query})" if query is not None else ""))
            plot_tokens(axs[b,j], image, label, highlight_query_rules, highlight_key_rules)
            j += 1

        if show_factor_scores:
            # TODO
            # ss_measure, sp_measure, ps_measure, pp_measure = \
            #     compute_decomposed_attention_measures(attn_weights, all_weights, im_scores, pos_scores, ps_scores, pp_scores)

            if weight_factor_by_attn:
                vis_ss_scores = vis_ss_scores * attn_weights
            im = axs[b,j].imshow(to_query_weights(vis_ss_scores, query), cmap='viridis', vmin=vmin, vmax=vmax)
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Image attribution ({vis_ss_scores.max()*100./head_max_attribution:.2f}\%)")
            # axs[b,j].set_title(f"Image attribution")
            # print(f"Image attribution: {vis_ss_scores.sum()}")
            plot_tokens(axs[b,j], image, label, highlight_query_rules, highlight_key_rules)
            j += 1

            if weight_factor_by_attn:
                vis_sp_scores = vis_sp_scores * attn_weights
            im = axs[b,j].imshow(to_query_weights(vis_sp_scores, query), cmap='viridis', vmin=vmin, vmax=vmax)
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Position attribution ({vis_sp_scores.max()*100./head_max_attribution:.2f}\%)")
            # axs[b,j].set_title(f"Position attribution")
            # print(f"Position attribution: {vis_sp_scores.sum()}")
            plot_tokens(axs[b,j], image, label, highlight_query_rules, highlight_key_rules)
            j += 1

        if show_merged:
            merged_image = torch.stack([im_scores, pos_scores, torch.zeros_like(im_scores)], dim=0).permute(1,2,0)
            low, high = torch.min(merged_image), torch.max(merged_image)
            normed_merged = (merged_image - low) / (high - low)
            # normed_merged = merged_image / high
            weighted_merged = normed_merged * attn_weights.unsqueeze(2)
            im = axs[b,j].imshow(to_query_weights(weighted_merged, query))
            # colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Average (im,pos,none)")
            plot_tokens(axs[b,j], image, label, highlight_query_rules, highlight_key_rules)
            j += 1

        if show_weighed_top1:
            merged_image = torch.stack([im_scores, pos_scores], dim=0).permute(1,2,0)
            argmax = torch.argmax(merged_image, dim=2, keepdim=True)
            indices = torch.stack([torch.ones((num_tokens,num_tokens)) * i for i in range(3)]).permute(1,2,0)
            top1 = (argmax == indices).float()
            weighted_top1 = top1 * attn_weights.unsqueeze(2)
            im = axs[b,j].imshow(to_query_weights(weighted_top1, query))
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Top1 (im,pos,none)")
            plot_tokens(axs[b,j], image, label, highlight_query_rules, highlight_key_rules)
            j += 1

    query_display = ""
    if query_rule is not None:
        query_display = f" / query: {query_rule.__name__}"
    fig.suptitle(f"Toy example: Grad-based attribution for {task_name}{query_display} (head {head+1} of {num_heads})")
    plt.tight_layout()


def visualize_gradbased_attribution_wrt_pred_class(
        images, labels, attributions, task_name,
        shared_domain_scores=False,
        show_sample=True,
        show_factor_scores=True,
        show_merged=False,
        max_attribution=None,
        abs_pos_maps=None,
    ):
    """
    Given grad-based attributions for an attention matrix, visualize the
    attributions. By default, shows only sample, attributions and attribution
    scores.
    """
    N = len(images)
    n_cols = int(show_sample) + int(show_factor_scores) * 2 + int(show_merged) + int(abs_pos_maps is not None)
    fig, axs = plt.subplots(N, n_cols, figsize=(2 + n_cols * 2, N * 2), dpi=120)
    if N == 1:
        axs = axs.reshape((1, n_cols))

    for b in range(N):
        j = 0

        image = images[b].detach()
        label = labels[b].detach()
        image_attr, pos_attr = attributions[b]

        if show_sample:
            axs[b,j].imshow(image.permute((1, 2, 0)))
            axs[b,j].set_title(f"Sample (label: {label})")
            axs[b,j].axis('off')
            j += 1

        im_scores = image_attr.detach()
        pos_scores = pos_attr.detach()

        vis_ss_scores = im_scores
        vis_sp_scores = pos_scores

        vmin = vmax = None
        if max_attribution is not None:
            vmin = 0.0
            vmax = max_attribution
        elif shared_domain_scores:
            vmin = torch.min(torch.stack([vis_ss_scores, vis_sp_scores]))
            vmax = torch.max(torch.stack([vis_ss_scores, vis_sp_scores]))
            largest_v = max(abs(vmin), abs(vmax))
            vmin = -largest_v
            vmax = largest_v

        if show_factor_scores:
            # TODO
            # ss_measure, sp_measure, ps_measure, pp_measure = \
            #     compute_decomposed_attention_measures(attn_weights, all_weights, im_scores, pos_scores, ps_scores, pp_scores)

            im = axs[b,j].imshow(vis_ss_scores, cmap='viridis', vmin=vmin, vmax=vmax)
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Image attributions ({vis_ss_scores.max()*100./max_attribution:.2f}\%)")
            # axs[b,j].set_title(f"Image attribution")
            j += 1

            im = axs[b,j].imshow(vis_sp_scores, cmap='viridis', vmin=vmin, vmax=vmax)
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Position attribution ({vis_sp_scores.max()*100./max_attribution:.2f}\%)")
            # axs[b,j].set_title(f"Position attribution")
            j += 1

        if show_merged:
            merged_image = torch.stack([im_scores, pos_scores, torch.zeros_like(im_scores)], dim=0).permute(1,2,0)
            low, high = torch.min(merged_image), torch.max(merged_image)
            normed_merged = (merged_image - low) / (high - low)
            # normed_merged = merged_image / high
            im = axs[b,j].imshow(normed_merged)
            # colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Average (im,pos,none)")
            j += 1

        if abs_pos_maps is not None:
            im = axs[b,j].imshow(abs_pos_maps[b], cmap='viridis', vmin=0.)
            colorbar(axs[b,j], im)
            axs[b,j].axis('off')
            axs[b,j].set_title(f"Absolute position ({abs_pos_maps[b].min()})")
            j += 1

    fig.suptitle(f"Toy example: Grad-based attribution for {task_name} (whole network)")
    plt.tight_layout()

def visualize_gradbased_attribution_wrt_feature_map(images, labels, attributions, *pargs, max_attribution=None, **kwargs):
    # Remove first dimension from each tensor in each list of attributions
    attributions = [[attribution[0] for attribution in attributions_i] for attributions_i in attributions]
    max_attribution = max_attribution[0] if max_attribution is not None else None
    visualize_gradbased_attribution_wrt_loss(images, labels, attributions, *pargs, max_attribution=max_attribution, **kwargs)


plt.style.use(['seaborn'])

BG_COLOR = "#eaeaf2ff"

COLORS = sns.color_palette("deep")
# BIAS_COLOR = 'white'
BIAS_COLOR = COLORS[8]
APPEARANCE_COLOR = COLORS[0]
POSITION_COLOR = COLORS[1]
RELPOS_COLOR = COLORS[3]

TEXT_COLORS = sns.color_palette("dark")
# BIAS_TEXT_COLOR = 'gray'
BIAS_TEXT_COLOR = TEXT_COLORS[8]
APPEARANCE_TEXT_COLOR = TEXT_COLORS[0]
POSITION_TEXT_COLOR = TEXT_COLORS[1]
RELPOS_TEXT_COLOR = TEXT_COLORS[3]


def plot_position_biases(ax, biases, allow_missing=False, annotate_percentages=True, size=1.0):
    """
    TODO:
    - Plot percentages in the middle of each wedge, if they fit
    - Fix width of the top black bar
    """
    appearance = float(biases['appearance'])
    if 'position' in biases:
        position = float(biases['position'])
    else:
        position = 0.
        print("WARNING: position not in biases, assuming 0.0")
    bias = float(biases['bias'])

    relative_position = 0.
    if 'relative_position' in biases:
        relative_position = float(biases['relative_position'])
    else:
        print("WARNING: relative_position not in biases, assuming 0.0")

    learned_relative_position = 0.
    if 'learned_relative_position' in biases:
        learned_relative_position = float(biases['learned_relative_position'])

    missing = 1 - (appearance + position + bias + relative_position)
    if not allow_missing and not math.isclose(missing, 0., rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(f"Biases do not sum to 1: {appearance} + {position} + {bias} + {relative_position} = {appearance + position + bias + relative_position}")

    import seaborn as sns
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    linewidth = 2.5 / size

    # Plot vspans for bias, appearance and position in order
    ax.axvspan(0, 1, 1 - missing, 1 - missing - bias, color=BIAS_COLOR, alpha=1.0, linewidth=linewidth)
    ax.axvspan(0, 1, 1 - missing - bias, 1 - missing - bias - appearance, color=APPEARANCE_COLOR, alpha=1.0, linewidth=linewidth)
    ax.axvspan(0, 1, 1 - missing - bias - appearance, 1 - missing - bias - appearance - position, color=POSITION_COLOR, alpha=1.0, linewidth=linewidth)

    # For each of the previously plotted axvspan's, show the percentage in the middle of the axvspan in the same color but a bit darker for contrast
    if annotate_percentages and bias > 0.1:
        ax.text(0.5, 1 - missing - bias/2, f"{bias*100:.0f}\%", ha='center', va='center', color=BIAS_TEXT_COLOR, fontsize=12)
    if annotate_percentages and appearance > 0.1:
        ax.text(0.5, 1 - missing - bias - appearance/2, f"{appearance*100:.0f}\%", ha='center', va='center', color=APPEARANCE_TEXT_COLOR, fontsize=12)
    if annotate_percentages and position > 0.1:
        ax.text(0.5 + learned_relative_position/2, 1 - missing - bias - appearance - position/2, f"{position*100:.0f}\%", ha='center', va='center', color=POSITION_TEXT_COLOR, fontsize=12)

    # Hatch a left-offset area for learned relative position
    ax.axvspan(0, learned_relative_position, 1 - missing - bias - appearance, 1 - missing - bias - appearance - position, facecolor='none', edgecolor=RELPOS_COLOR, hatch='/', alpha=1.0)
    stroke = Rectangle((0, relative_position), learned_relative_position, position, fill=False, edgecolor=RELPOS_COLOR, linewidth=linewidth)
    ax.add_patch(stroke)

    if annotate_percentages and position > 0.1 and learned_relative_position > 0.1:
        ax.text(learned_relative_position/2, 1 - missing - bias - appearance - position/2, f"({learned_relative_position*100:.0f}\%)", ha='center', va='center', color=RELPOS_TEXT_COLOR, fontsize=12)

    # Plot vspan for relative position at the bottom
    ax.axvspan(0, 1, 0, relative_position, color=RELPOS_COLOR, alpha=1.0, linewidth=linewidth)

    if annotate_percentages and relative_position > 0.1:
        ax.text(0.5, relative_position/2, f"{relative_position*100:.0f}\%", ha='center', va='center', color=RELPOS_TEXT_COLOR, fontsize=12)


    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.grid(False)

    stroke = Rectangle((0, 0), 1, 1, fill=False, edgecolor='gray', linewidth=linewidth)
    ax.add_patch(stroke)

    if missing < 0.98:
        stroke = Rectangle((0, 0), 1, 1. - missing, fill=False, edgecolor='black', linewidth=linewidth)
        ax.add_patch(stroke)

    return {
        'hatch.linewidth': 10.0,
        "text.usetex": True,
        "font.family": "sans-serif",
        'mathtext.fontset': 'stix',
        "font.sans-serif": ["Palatino"],
        'font.family': 'STIXGeneral'
    }

def plot_position_biases_legend(axs):
    axs.legend(
        [
            Patch(facecolor=BG_COLOR, edgecolor='gray', linewidth=2.0, label='Color Patch'),
            Patch(facecolor=BIAS_COLOR, edgecolor='black', linewidth=2.0, label='Color Patch'),
            Patch(facecolor=APPEARANCE_COLOR, edgecolor='black', linewidth=2.0, label='Color Patch'),
            Patch(facecolor=POSITION_COLOR, edgecolor='black', linewidth=2.0, label='Color Patch'),
            Patch(facecolor=RELPOS_COLOR, edgecolor='black', linewidth=2.0, label='Color Patch'),
        ],
        [
            'No contribution',
            'Bias',
            'Appearance',
            'Position',
            'Relative position'],
        loc='upper left',
    )