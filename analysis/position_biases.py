import torch
import argparse

import sys
sys.path.append('..')
import analysis
from analysis.learned_relative_position import learned_relative_position_measure
from analysis.saliency import gradbased_attribution, aggregate_saliency


def position_biases(activations, gradients, shape, labels, patch_size,
                    num_classes, attribution_method='input_gradient',
                    aggregate_fn='sum',
                    relpos_dispersion_method='variance_ratio',
                    normalize_method='sum', mean_over_samples=True,
                    lrp=True, exclude_bias=False):
    """
    Measure bias to bias, appearance, position, learned relative position and
    relative position given activations and gradients of the image and position
    embeddings w.r.t some target (e.g. predicted class logit, attention head
    activations).

    To elaborate: we posit that the network output of a ViT is determined by a
    combination of the following factors: the image input to the network
    (appearance bias), the position embeddings (position bias) which can learn
    to compute functions of relative position (learned relative position bias),
    the bias terms from the relative position embeddings integrated into
    self-attention (relative position bias), and finally the bias terms in all
    operations in the network (the bias bias).

    This method computes the bias w.r.t. each factor by measuring the ratio
    between the saliency maps of each factor. This goes for all factors except
    learned relative position bias, which is expressed as a fraction of the
    position bias, measured by shifting input images and computing dispersion
    along unshifted network outputs.

    See the paper for more information.

    TODO: learned relative position can be negative, because the F-statistic is
    not technically limited to yield a p-value between 0 and 0.5.

    Args: TODO
        activations (dict(dict([torch.Tensor]))): for each bias type, for each
            (y,x) shift, contains a list of Tensor inputs used in the network.
            If `relative_position` is not included, the relative position bias
            is not computed;
        gradients (dict(dict([torch.Tensor]))): for each bias type, for each
            (y,x) shift, contains a list of Tensor gradients w.r.t the target.
            If `relative_position` is not included, the relative position bias
            is not computed;
        shape (str): shape of the saliency maps; supported: 'scalar' means the
            target is a scalar (e.g. logit), 'head' means the target is a
            vector (e.g. attention head activations);
        patch_size (int): size of the patches used to compute the position
            embeddings;
        num_classes (int);
        method (str): method to use to compute the bias measures; supported:
            'input_gradient' (default);
        relpos_dispersion_method (str): method to use to compute the relative
            position dispersion; supported: 'variance_ratio' (default).
        normalize_method (str): how to normalize attributions; supported:
            'sum' (default), 'max'.
        mean_over_samples (bool, `True`): if `True`, average the per-sample
            computed bias measures over all samples.
        lrp (bool, `True`): compute learning relative position.
        exclude_bias (bool, `False`): if `True`, exclude the bias bias from the
            computed biases.

    Returns:
        overall_bias (dict(float)): scalar-valued bias for each factor;
    """
    if exclude_bias and 'bias' in activations:
        del activations['bias']
        del gradients['bias']

    # Compute saliency maps for each source
    saliency = {}
    for source in activations:
        saliency[source] = {shift: [] for shift in activations[source]}

        # targets: {(s_y, s_x): saliency map}
        for shift in activations[source]:
            # For each (target, shift, sample), compute the saliency maps
            for i in range(len(activations[source][shift])):
                # For anything but pos. embs, we only need to compute attribution for the zero-shift
                if source == 'pos_emb' or shift == (0, 0):
                    saliency[source][shift].append(gradbased_attribution(activations[source][shift][i], gradients[source][shift][i], method=attribution_method).detach())


    #
    # COMPUTE MEASURES
    #

    # Given saliency maps for zero shift, compute the bias measures
    zeroshift_saliency = {s: saliency[s][(0,0)] for s in saliency}
    # overall_bias, class_bias = \
    overall_bias = \
        gradbased_attribution_measures(zeroshift_saliency, labels, num_classes,
                                       shape=shape,
                                       aggregate_fn=aggregate_fn,
                                       normalize=normalize_method,
                                       mean_over_samples=mean_over_samples)

    # Given position salience maps, compute the relative position bias
    if lrp and 'pos_emb' in saliency:
        # relative_position_bias, class_relative_position_bias = \
        relative_position_bias = \
            learned_relative_position_measure(saliency['pos_emb'], patch_size, labels,
                                              num_classes,
                                              shape=shape, method=relpos_dispersion_method,
                                              mean_over_samples=mean_over_samples)
        overall_bias['learned_relative_position'] = relative_position_bias
        # class_bias['learned_relative_position'] = class_relative_position_bias

    # return overall_bias, class_bias
    return overall_bias


def gradbased_attribution_measures(saliency_maps, labels, num_classes, shape='scalar', aggregate_fn='sum', normalize='sum', mean_over_samples=True):
    """
    Compute bias measures for each class, given saliency maps of shape [].

    Args:
        image_saliency_maps (list of torch.Tensor);
        pos_emb_saliency_maps (list of torch.Tensor);
        labels;
        shape (str):
          'scalar' = [C_im or C_pos, H_im or H_pos, W_im or W_pos] -> [];
          'head' = [num_heads, C_im or C_pos, H_im or H_pos, W_im or W_pos] -> [num_heads]
        normalize (str): normalization method for the attribution scores. Can be
          'sum', 'max' or 'l2'.
    """
    # Saliency maps -> attribution scores, aggregated over the desired dimensions
    attrs = {source: [] for source in saliency_maps}
    for source in saliency_maps:
        source_attrs = []
        for i in range(len(saliency_maps[source])):
            attr = saliency_maps[source][i]
            attr = aggregate_saliency(attr, shape=shape, method=aggregate_fn)
            source_attrs.append(attr)
        attrs[source] = torch.stack(source_attrs)

    if normalize == 'max':
        norm_term = torch.stack([attrs[source] for source in attrs]).max(dim=0).values()
    elif normalize == 'sum':
        norm_term = torch.stack([attrs[source] for source in attrs]).sum(dim=0)
    elif normalize == 'l2':
        norm_term = torch.stack([attrs[source] for source in attrs]).square().sum(dim=0).sqrt()

    attrs = {s: torch.where(norm_term != 0., attrs[s] / norm_term, 0.) for s in attrs}

    overall_attrs = {s: attrs[s] for s in attrs}
    # class_attrs = {s: [] for s in attrs}
    # for c in range(num_classes):
    #     for s in attrs:
    #         class_attrs[s].append(attrs[s][labels == torch.tensor(c)])

    if mean_over_samples:
        overall_attrs = {s: overall_attrs[s].mean(dim=0) for s in overall_attrs}
        # for c in range(num_classes):
        #     for s in attrs:
        #         class_attrs[s][c] = class_attrs[s][c].mean(dim=0)

    # Rename
    if 'image' in overall_attrs:
        overall_attrs['appearance'] = overall_attrs['image']
        # class_attrs['appearance'] = class_attrs['image']
        del overall_attrs['image']
        # del class_attrs['image']
    if 'pos_emb' in overall_attrs:
        overall_attrs['position'] = overall_attrs['pos_emb']
        # class_attrs['position'] = class_attrs['pos_emb']
        del overall_attrs['pos_emb']
        # del class_attrs['pos_emb']
    if 'relpos' in overall_attrs:
        overall_attrs['relative_position'] = overall_attrs['relpos']
        # class_attrs['relative_position'] = class_attrs['relpos']
        del overall_attrs['relpos']
        # del class_attrs['relpos']

    # return overall_attrs, class_attrs
    return overall_attrs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments: filename (str), shape (str), attribution_method='input_gradient', relpos_dispersion_method='variance_ratio', normalize_method='sum'
    parser.add_argument('--filename', type=str)
    parser.add_argument('--shape', type=str, default='scalar')
    parser.add_argument('--attribution_method', type=str, default='input_gradient')
    parser.add_argument('--relpos_dispersion_method', type=str, default='variance_ratio')
    parser.add_argument('--normalize_method', type=str, default='sum')
    args = parser.parse_args()

    target = 'pred_class'

    # Load data
    print('Loading data...')
    data = torch.load(args.filename, map_location=torch.device('cpu'))
    activations = data['activations']
    gradients = data['gradients']
    test_labels = data['test_labels']
    patch_size = data['patch_size']

    # overall_biases, class_biases = \
    overall_biases = \
        position_biases(activations[target], gradients[target], args.shape,
                        test_labels, patch_size,
                        attribution_method=args.attribution_method,
                        relpos_dispersion_method=args.relpos_dispersion_method,
                        normalize_method=args.normalize_method)

    # Print the measures
    target_name = 'predicted class' if target == 'pred_class' else target
    method_name = 'input-gradient' if args.attribution_method == 'input_gradient' else '<not specified>'
    print(f"--- Position biases w.r.t. {target_name} using {method_name} attribution ---")

    if args.shape == 'scalar':
        print("Mean over classes:")
        print(f"\tBias: {overall_biases['bias']:.2f}")
        print(f"\tAppearance: {overall_biases['appearance']:.2f}")
        if 'position' in overall_biases:
            print(f"\tPosition: {overall_biases['position']:.2f}")
            print(f"\tLearned relative position: {overall_biases['learned_relative_position']:.2f}")
        if 'relative_position' in overall_biases:
            print(f"\tRelative position: {overall_biases['relative_position']:.2f}")
        # for c in range(2):
        #     print(f"Class {c}:")
        #     print(f"\tBias: {class_biases['bias'][c]:.2f}")
        #     print(f"\tAppearance: {class_biases['appearance'][c]:.2f}")
        #     if 'position' in class_biases:
        #         print(f"\tPosition: {class_biases['position'][c]:.2f}")
        #         print(f"\tLearned relative position: {class_biases['learned_relative_position'][c]:.2f}")
        #     if 'relative_position' in class_biases:
        #         print(f"\tRelative position: {class_biases['relative_position'][c]:.2f}")

    elif args.shape == 'head':
        print("Mean over classes:")
        print(f"\tBias: " + ", ".join([f"{a:.2f}" for a in overall_biases['bias']]))
        print(f"\tAppearance: " + ", ".join([f"{a:.2f}" for a in overall_biases['appearance']]))
        if 'position' in overall_biases:
            print(f"\tPosition: " + ", ".join([f"{a:.2f}" for a in overall_biases['position']]))
            print(f"\tLearned relative position: " + ", ".join([f"{a:.2f}" for a in overall_biases['learned_relative_position']]))
        if 'relative_position' in overall_biases:
            print(f"\tRelative position: " + ", ".join([f"{a:.2f}" for a in overall_biases['relative_position']]))
        # for c in range(2):
        #     print(f"Class {c}:")
        #     print(f"\tBias: " + ", ".join([f"{a:.2f}" for a in class_biases['bias'][c]]))
        #     print(f"\tAppearance: " + ", ".join([f"{a:.2f}" for a in class_biases['appearance'][c]]))
        #     if 'position' in class_biases:
        #         print(f"\tPosition: " + ", ".join([f"{a:.2f}" for a in class_biases['position'][c]]))
        #         print(f"\tLearned relative position: " + ", ".join([f"{a:.2f}" for a in class_biases['learned_relative_position'][c]]))
        #     if 'relative_position' in class_biases:
        #         print(f"\tRelative position: " + ", ".join([f"{a:.2f}" for a in class_biases['relative_position'][c]]))

    else:
        raise NotImplementedError()