import torch
import scipy


def learned_relative_position_measure(pos_emb_saliency_maps, patch_size, labels, num_classes, shape='scalar', method='variance_ratio', mean_over_samples=True):
    """
    From saliency maps of the image and position embeddings, compute the relative position measure.

    NOTE that this method has become a bit complicated, because it supports
    having an extra dimension D as the first dimension of the saliency maps, so
    that it can work both on network-wide and head-specific saliency maps.

    TODO: learned relative position can be negative, because the F-statistic is
    not technically limited to yield a p-value between 0 and 0.5.

    Args:
        pos_emb_saliency_maps (dict(torch.Tensor)): for each (y,x) shift,
            contains a list of saliency maps ([[D], C, H, W]) w.r.t the target
            for each position embeddings;
        patch_size (int): size of the patches; used to adjust shifts (in pixels)
            to tokens;
        labels (list(int)): labels of the samples;
        shape (str): shape of the saliency maps; supported: 'scalar' ([D] not
            included in pos_emb_saliency_maps), 'head' ([D] included in
            pos_emb_saliency_maps);
        mean_over_samples (bool, `True`): if `True`, average over all samples
            per class to get a single scalar value for the relative position.
    Returns:
        overall_relpos_attrs (list(float)): relative position measure for each
            sample;
    """
    # Stack all saliency maps into a single Tensor by applying the inverse shift
    # to the saliency maps
    num_maps = len(pos_emb_saliency_maps)
    num_samples = len(pos_emb_saliency_maps[(0,0)])
    max_padding = max([max(abs(shift[0]) // patch_size, abs(shift[1]) // patch_size) for shift in pos_emb_saliency_maps])
    H, W = pos_emb_saliency_maps[(0,0)][0].shape[-2:]
    # If saliency maps are given for each head, opt_D is [D], else it is []. We
    # will pass it as *opt_D to insert an optional dimension.
    opt_D = pos_emb_saliency_maps[(0,0)][0].shape[:-3]
    # Eventually we compute dispersion over position for each sample ([N]), and also
    # separately for each head, if salience maps are given for each head ([N, D])
    pos_dispersion = torch.zeros(num_samples, 1, dtype=torch.float32)
    if len(opt_D) > 0:
        pos_dispersion = torch.zeros(num_samples, *opt_D, dtype=torch.float32)
    for i in range(num_samples):
        # Stack shape: [num_maps, [D], H, W].
        # All the code in this for-loop should work with and without the leading dimension [D]
        sample_pos_stack = torch.zeros(num_maps, *opt_D, H, W, dtype=torch.float32)
        for j, shift in enumerate(pos_emb_saliency_maps):
            # Adjust shift (in image coords) to pos. embs. (in patches/tokens)
            token_shift_y, token_shift_x = (shift[0] // patch_size, shift[1] // patch_size)
            # Get/aggregate saliency map: shift, sample in dataset, (keep batch
            # dim if it exists), sum over channels -> [[D], H, W]
            salmap = pos_emb_saliency_maps[shift][i].sum(dim=-3)
            unshifted_map = shift_image(salmap, (-token_shift_y, -token_shift_x))
            sample_pos_stack[j] = unshifted_map

        # If we don't have head scores, "fake" a single head, then return just
        # that singleton score at the end of the method
        if sample_pos_stack.dim() == 3:
            sample_pos_stack = sample_pos_stack[:,None]

        # Crop the stack to make sure we only include image/pos. embedding
        # "pixels" that were not affected by padding (where we convert padding
        # from pixels to tokens again)
        sample_pos_stack = sample_pos_stack[:,:,max_padding:-max_padding,max_padding:-max_padding]

        for j in range(sample_pos_stack.shape[1]):
            pos_dispersion[i,j] = image_stack_dispersion(sample_pos_stack[:,j], method=method)

    # Average (per head) over all samples, or class-specific samples
    all_class_relpos = pos_dispersion
    if mean_over_samples:
        all_class_relpos = all_class_relpos.mean(dim=0)
    # class_relpos = []
    # for c in range(num_classes):
    #     class_disperson = pos_dispersion[labels == torch.tensor(c)]
    #     if mean_over_samples:
    #         class_disperson = class_disperson.mean(dim=0)
    #     class_relpos.append(class_disperson)

    if shape == 'scalar':
        if not mean_over_samples:
            # return all_class_relpos[0], [c[:,0] for c in class_relpos]
            return all_class_relpos[0]
        else:
            # return all_class_relpos[0], [c[0] for c in class_relpos]
            return all_class_relpos[0]
    elif shape == 'head':
        # return all_class_relpos, class_relpos
        return all_class_relpos
    else:
        raise NotImplementedError()


def shift_image(batched_image, shift):
    """
    Shift and crop a batch of images, retaining the original width and height.
    The underlying Torch Tensor is cloned, so that gradients can be computed for
    the shifted image instead of the original image.
    """
    # Pad to keep original image dimensions, then crop with shift
    H, W = batched_image.shape[-2:]
    SY, SX = shift
    # Padding needed is the maximum absolute shift
    pad_size = max(abs(SY), abs(SX))
    im_padded = torch.nn.functional.pad(batched_image, (pad_size, pad_size, pad_size, pad_size))
    # Coordinates are outer image limits (0 to SY/SX), plus shift, plus
    # padding
    y1 = pad_size + 0 + SY
    y2 = pad_size + H + SY
    x1 = pad_size + 0 + SX
    x2 = pad_size + W + SX
    # Slice and clone, to ensure that gradients are taken w.r.t. the
    # shifted image
    if im_padded.dim() == 4:
        return im_padded[:,:,y1:y2,x1:x2].clone()
    elif im_padded.dim() == 3:
        return im_padded[:,y1:y2,x1:x2].clone()
    elif im_padded.dim() == 2:
        return im_padded[y1:y2,x1:x2].clone()
    else:
        raise NotImplementedError()


def image_stack_dispersion(stack, method='variance_ratio'):
    """
    Compute dispersion along the first dimension of an image stack.

    Args:
        stack (torch.Tensor): [N, H, W];
        method (str): supported methods: 'variance_ratio';

    Returns:
        dispersion (torch.Tensor): [];
    """
    if method == 'variance_ratio':
        if stack.dim() != 3:
            raise NotImplementedError(f"Does not support stacks for multiple "
                                      f"heads. Please run method for each head "
                                      f"separately.")

        # Compute variance for regular stack
        stack = stack.reshape((stack.shape[0], -1))
        variance = torch.var(stack, dim=0, unbiased=True)
        # Compute variance along random permutation (per shift) of stack
        _, D = stack.shape
        random_stack = torch.zeros_like(stack)
        for i in range(stack.shape[0]):
            random_stack[i] = stack[i][torch.randperm(D)]
        random_variance = torch.var(random_stack, dim=0, unbiased=True)

        # Replace zeros in denominator of ratio below with small value, to avoid
        # division by zero
        zeros = torch.isclose(random_variance, torch.zeros_like(random_variance), atol=1e-14)
        if torch.any(zeros):
            random_variance[zeros] = 1e-14

        # Express as cdf of the F-distribution, as in an F-test of equality of
        # variances
        # (https://en.wikipedia.org/wiki/F-test_of_equality_of_variances)
        cdf = scipy.stats.f.cdf(variance / random_variance,
                               [stack.shape[0]] * stack.shape[1],
                               [random_stack.shape[0]] * random_stack.shape[1])
        # Turn into a one-tailed test by ruling out that variance >
        # random_variance, which is simplest to do by multiplying the cdf by 2
        sf = (1. - (cdf * 2.))

        return sf.mean()
    else:
        raise NotImplementedError(f"Method \"{method}\" not implemented")