import numpy as np
import torch
from torch import nn
import tqdm
import math

from analysis.learned_relative_position import shift_image
from analysis.sources import get_attribute_from_name_or_callable


def gradient_batched(model, target, input, allow_unused=False):
    """
    Compute gradient of input w.r.t. target, treating input as a "batch" of
    Tensors, indexed by the first dimension. This is used to efficiently compute
    gradients for several channels, heads, images etc. at once.

    Args:
        target (Tensor): The target to compute the gradient w.r.t. Assumed to be
            in shape [D, ...]. Dimensions [...] will be consumed.
        input (Tensor): The input to compute the gradient for.

    Returns:
        gradients (Tensor): gradients of input w.r.t batch dimension of target,
            in shape [D, *input_shape].
    """
    # We need to explicitly zero out the gradients, otherwise they accumulate
    # from earlier calls to gradient_batched
    model.zero_grad()

    D = target.shape[0]
    # The initial gradient is an identity matrix so that each
    # "batch" (index in first dimension of target) only computes the
    # gradients wrt that `target[i]`. The rest of the dimensions
    # need to match the original target shape.
    grad_out = torch.eye(D)
    grad_out = grad_out.view(D, D, *([1] * (target.dim() - 1)))
    grad_out = grad_out.expand(D, *target.shape)
    # Use retain_graph, so we can call gradient_batches multiple times without needing to rerun inference
    return torch.autograd.grad(target, input, grad_outputs=[grad_out], is_grads_batched=True, allow_unused=allow_unused, retain_graph=True)

def inference_to_gradbased_analysis(model, test_images, test_labels, sources, targets=['pred_class'], patch_size_y=None, patch_size_x=None, importance=False, seed=0, n_samples=None, only_correct=False, progress=True, lrp=True):
    """
    Extracts all the necessary ingredients (activations and gradients w.r.t
    targets) to do grad-based analysis of appearance and position. Gradients can
    be computed with respect to a specific target (predicted class or a specific
    head in a specific layer).

    Args:
        model: The model to analyze.
        test_images: The images to analyze.
        test_labels: The labels for the images.
        sources (dict(str: str or callable)): Each entry is a "source" of
            bias/information, with an associated str or callable. In case
            of a string, the source is the attribute of `model` with this name.
            In case of a callable, the callable returns a list of all Tensors
            that make up this source. Requires sources: `image`, `bias`.
            Optional sources: `pos_emb`, `rel_pos_emb`.
        targets (list(str), ["pred_class"]): If "pred_class", compute saliency
            w.r.t. the predicted class; else attempt to access the model
            attribution with this name and use it as the target.
        shift_y, shift_x (list(int), None): evaluate model for each provided 2D
            shift. If `None`, only evaluate for shift `(0,0)`. Note that the
            returned dictionaries still maintain the same format, just with only
            one entry for shifts.
        importance (bool, False): If True, additionally compute the gradient of
            the target w.r.t. the predicted class, to be used as an importance
            score in the measures.
        seed (int, 0): The random seed to use for reproducibility.
        n_samples (int, None): Limit on the number of samples to compute this
            analysis for.
        only_correct (bool, False): If True, only compute this analysis for the
            samples that the model correctly classifies.
        lrp (bool, `True`): do shifted inference for learned relative position.

    Returns:
        activations (dict(dict(dict([torch.Tensor])))): for each target, for
            each bias source, for each (y,x) shift, contains a list of Tensor
            inputs used in the network. If `relative_position` is not included,
            the relative position bias is not computed;
        gradients (dict(dict(dict([torch.Tensor])))): for each target, for each
            bias source, for each (y,x) shift, contains a list of Tensor
            gradients w.r.t the target. If `relative_position` is not included,
            the relative position bias is not computed;
    """
    if targets is None:
        raise NotImplementedError(f"No targets specified.")

    required_sources = ['image', 'bias']
    optional_sources = ['pos_emb', 'relpos']
    for source in sources:
        if source not in required_sources and source not in optional_sources:
            raise NotImplementedError(f"Source {source} not supported.")
        if source in required_sources:
            required_sources.remove(source)
    if len(required_sources) > 0:
        raise ValueError(f"Required sources {required_sources} not provided.")

    np.random.seed(seed)
    torch.manual_seed(seed)

    def inference(model, image, labels, image_source, shift=None):
        model.zero_grad()
        # Any sources that are not model parameters (only the image) need to
        # manually have their gradients reset
        if hasattr(image_source, 'grad') and image_source.grad is not None:
            image_source.grad.zero_()

        if shift is not None and shift != (0,0):
            image = shift_image(image.clone().detach(), shift)
        else:
            image = image.clone().detach()
        image.requires_grad_(True)
        # print('inference in is on', image.requires_grad)
        outputs = model(image)
        if hasattr(model, 'get_logits'):
            logits = model.get_logits(outputs)
        else:
            logits = outputs
        predictions = torch.argmax(logits, dim=1)
        return image, labels, logits, predictions

    # Construct set of shifts to apply. By default, test only a single patch
    # shift
    if lrp:
        shift_y = range(-patch_size_y, 2 * patch_size_y, patch_size_y)
        shift_x = range(-patch_size_x, 2 * patch_size_x, patch_size_x)
        shifts = [(y,x) for y in shift_y for x in shift_x]
    else:
        shifts = [(0,0)]

    # DEBUG
    # available_sources = []
    # for source in sources:
    #     source_tensor = get_attribute_from_name_or_callable(model, sources[source])
    #     if source_tensor is not None or (isinstance(source_tensor, list) and len(source_tensor) > 0):
    #         available_sources.append(source)
    # print(f"Sources: " + ", ".join(sources.keys()))
    # print(f"Available sources: " + ", ".join(available_sources))

    # Run samples through model to get attention weights
    total_samples = test_images.shape[0]
    i = 0
    activations = {t: {so: {s: [] for s in shifts} for so in list(sources.keys())} for t in targets}
    grads = {t: {so: {s: [] for s in shifts} for so in list(sources.keys()) + ['target']} for t in targets}
    pbar = range(total_samples)
    if progress:
        pbar = tqdm.tqdm(pbar, total=total_samples)
    for sample_i in pbar:
        # Get image and labels with leading dimension for num_samples=1, for
        # compatibility with model that expects a batch
        ims = test_images[sample_i:sample_i+1]
        labs = test_labels[sample_i:sample_i+1]

        # Convert multi-label to single label
        multi_label = False
        if labs.dim() == 2:
            multi_label = True
            all_labs = labs
            labs = torch.argmax(labs, dim=1)

        for shift in shifts:
            for target in targets:
                target_has_pos_emb = False
                target_has_rpe = False

                if target in ['pred_class', 'all_pred_class', 'gt_class', 'all_gt_class', 'loss', 'loss_all_classes', 'logits_all_class']:
                    # DEBUG: testing if inference works without setting
                    # requires_grad on position embedding, because it could be
                    # overwritten anyway so it should just be set in the model
                    # itself...
                    # if 'pos_emb' in sources:
                    #     pos_emb_source = get_attribute_from_name_or_callable(model, sources['pos_emb'])
                    #     pos_emb_source.requires_grad_(True)

                    image_source = get_attribute_from_name_or_callable(model, sources['image'])
                    _, labs, logits, predictions = inference(model, ims, labs, image_source, shift=shift)

                    if target == 'pred_class':
                        logits[0,predictions[0]].backward()
                    elif target == 'all_pred_class':
                        torch.sum(logits[0]).backward()
                    elif target == 'gt_class':
                        logits[0,labs[0]].backward()
                    elif target == 'all_gt_class':
                        if multi_label:
                            torch.sum(logits[0] * all_labs.type(torch.float32)).backward()
                        else:
                            # Just the same as gt_class
                            logits[0,labs[0]].backward()
                    elif target == 'loss':
                        loss = nn.CrossEntropyLoss()(logits, labs)
                        (-loss).backward()
                    elif target == 'logits_all_class':
                        pred_class = predictions[0]
                        pos_logit = logits[0,pred_class]
                        one_hot = torch.nn.functional.one_hot(labs[0], num_classes=logits.shape[1])
                        neg_one_hot = (~one_hot.type(torch.bool)).type(torch.float32)
                        neg_logits = logits[0] * neg_one_hot
                        (pos_logit - torch.sum(neg_logits)).backward()
                    elif target == 'loss_all_classes':
                        pos_loss = nn.CrossEntropyLoss()(logits, labs)
                        one_hot = torch.nn.functional.one_hot(labs[0], num_classes=logits.shape[1])
                        neg_one_hot = (~one_hot.type(torch.bool)).type(torch.float32)
                        neg_loss = nn.CrossEntropyLoss()(logits[0], neg_one_hot)
                        (neg_loss - pos_loss).backward()

                    image_source = get_attribute_from_name_or_callable(model, sources['image'])
                    image_activations = image_source[0].clone()
                    image_grads = image_source.grad[0].clone()

                    if 'pos_emb' in sources:
                        pos_emb_source = get_attribute_from_name_or_callable(model, sources['pos_emb'])
                        if pos_emb_source.grad is not None:
                            target_has_pos_emb = True
                            pos_emb_activations = pos_emb_source[0].clone()
                            pos_emb_grads = pos_emb_source.grad[0].clone()

                    bias_activations = get_attribute_from_name_or_callable(model, sources['bias'])
                    bias_grads = [b.grad for b in bias_activations]

                    if 'relpos' in sources:
                        rel_pos_emb_activations = get_attribute_from_name_or_callable(model, sources['relpos'])
                        if len(rel_pos_emb_activations) == 0:
                            rel_pos_emb_activations = None
                        else:
                            target_has_rpe = True
                            rel_pos_emb_grads = [rpe.grad for rpe in rel_pos_emb_activations]

                # Target: any attribute of the model, index on the first element
                # (because of batch size = 1)
                else:
                    if 'pos_emb' in sources:
                        pos_emb_source = get_attribute_from_name_or_callable(model, sources['pos_emb'])
                        pos_emb_source.requires_grad_(True)

                    image_source = get_attribute_from_name_or_callable(model, sources['image'])
                    ims, labs, logits, predictions = inference(model, ims, labs, image_source, shift=shift)

                    image_source = get_attribute_from_name_or_callable(model, sources['image'])
                    image_activations = image_source[0]
                    # [num_heads, N=1, C_im, H_im, W_im] -> [num_heads, C_im, H_im, W_im]
                    image_grads = gradient_batched(model, getattr(model, target)[0], image_source)[0][:,0]

                    if 'pos_emb' in sources and pos_emb_source.requires_grad:
                        target_has_pos_emb = True
                        pos_emb_activations = pos_emb_source[0]
                        # [num_heads, N=1, C_pos, H_pos, W_pos] -> [num_heads, C_pos, H_pos, W_pos]
                        pos_emb_grads = gradient_batched(model, getattr(model, target)[0], pos_emb_source)[0][:,0]

                    bias_activations = get_attribute_from_name_or_callable(model, sources['bias'])
                    bias_grads = gradient_batched(model, getattr(model, target)[0], bias_activations, allow_unused=True)

                    if 'relpos' in sources:
                        rel_pos_emb_activations = get_attribute_from_name_or_callable(model, sources['relpos'])
                        if len(rel_pos_emb_activations) == 0:
                            rel_pos_emb_activations = None
                        else:
                            target_has_rpe = True
                            rel_pos_emb_grads = gradient_batched(model, getattr(model, target)[0], rel_pos_emb_activations, allow_unused=True)

                # If pos_emb_activations.shape[0] is larger than the number of
                # tokens, then the first embedding is for the class token. We
                # want to ignore that one.
                num_patches = (image_activations.shape[1] // patch_size_y) * (image_activations.shape[2] // patch_size_x)
                # Class token
                if target_has_pos_emb and pos_emb_activations.shape[0] == num_patches + 1:
                    pos_emb_activations = pos_emb_activations[1:]
                    pos_emb_grads = pos_emb_grads[1:]
                # Class + distillation token
                elif target_has_pos_emb and pos_emb_activations.shape[0] == num_patches + 2:
                    pos_emb_activations = pos_emb_activations[1:-1]
                    pos_emb_grads = pos_emb_grads[1:-1]

                #
                # Bunch of filtering and reshaping specifically for biases.
                # Basically, biases can be 1D (just a value for each channel) or
                # 2D, where the *second* dimension is some kind of batch
                # dimension. In this latter case, we want to concatenate on that
                # second dimension. BUT: not all biases may have it, so we need
                # to check if *any* biases have this, and if so, cast all of
                # them to a 2D shape with a batch dimension.
                #

                # Remove biases with None gradient, as they were not used in the forward pass for this target
                has_grad = [bg is not None for bg in bias_grads]
                bias_activations = [b for b, hg in zip(bias_activations, has_grad) if hg]
                bias_grads = [bg for bg, hg in zip(bias_grads, has_grad) if hg]
                # Concat, not stack, b/c the bias terms are not equal size
                # Concat on the dimension pertaining to the bias, not the
                # (optional) target dimension
                if bias_grads[0].dim() > bias_activations[0].dim():
                    bias_grads = torch.concat(bias_grads, dim=1)
                else:
                    bias_grads = torch.concat(bias_grads)
                bias_activations = torch.concat(bias_activations)

                if target_has_rpe:
                    # Remove RPEs with None gradient, as they were not used in the forward pass for this target
                    has_grad = [rpeg is not None for rpeg in rel_pos_emb_grads]
                    rel_pos_emb_activations = [rpe for rpe, hg in zip(rel_pos_emb_activations, has_grad) if hg]
                    rel_pos_emb_grads = [rpeg for rpeg, hg in zip(rel_pos_emb_grads, has_grad) if hg]

                    # [num_1d_embs, emb_dim_0, emb_dim_1]
                    if rel_pos_emb_activations[0].dim() == 2:
                        rel_pos_emb_activations = torch.concat(rel_pos_emb_activations, dim=1)
                    else:
                        rel_pos_emb_activations = torch.concat(rel_pos_emb_activations)
                    # [num_1d_embs, [target dim], emb_dim_0, emb_dim_1]
                    if rel_pos_emb_grads[0].dim() == 2:
                        rel_pos_emb_grads = torch.concat(rel_pos_emb_grads, dim=1)
                    else:
                        rel_pos_emb_grads = torch.concat(rel_pos_emb_grads)

                    # If there is a target dim, permute to move it to the front
                    if rel_pos_emb_grads.dim() > rel_pos_emb_activations.dim():
                        other_dims = range(2, rel_pos_emb_grads.dim())
                        rel_pos_emb_grads = rel_pos_emb_grads.permute(1, 0, *other_dims)

                    # print('rel_pos_emb', (rel_pos_emb_activations * rel_pos_emb_grads).max())
                    # print('image', (image_activations * image_grads).max())

                # Gradient of target w.r.t. predicted class
                target_grad = None
                if importance:
                    model.zero_grad()
                    target_grad = torch.autograd.grad(logits[0,predictions[0]], getattr(model, target)[0], retain_graph=True)

                if not only_correct or labs[0] == predictions[0]:
                    # Move all to CPU
                    image_activations = image_activations.cpu()
                    image_grads = image_grads.cpu()
                    bias_activations = bias_activations.cpu()
                    bias_grads = bias_grads.cpu()
                    if target_has_pos_emb:
                        pos_emb_activations = pos_emb_activations.cpu()
                        pos_emb_grads = pos_emb_grads.cpu()
                    if target_has_rpe:
                        rel_pos_emb_activations = rel_pos_emb_activations.cpu()
                        rel_pos_emb_grads = rel_pos_emb_grads.cpu()
                    if target_grad is not None:
                        target_grad = target_grad[0].cpu()

                    activations[target]['image'][shift].append(image_activations)
                    activations[target]['bias'][shift].append(bias_activations)
                    grads[target]['image'][shift].append(image_grads)
                    grads[target]['bias'][shift].append(bias_grads)
                    grads[target]['target'][shift].append(target_grad)

                    if target_has_pos_emb:
                        if pos_emb_activations.dim() == 2:
                            T, C = pos_emb_activations.shape
                            P = int(math.sqrt(T))
                            pos_emb_activations = pos_emb_activations.reshape(P, P, C)
                            pos_emb_grads = pos_emb_grads.reshape(*pos_emb_grads.shape[:-2], P, P, C)
                        activations[target]['pos_emb'][shift].append(pos_emb_activations)
                        grads[target]['pos_emb'][shift].append(pos_emb_grads)

                    if target_has_rpe:
                        activations[target]['relpos'][shift].append(rel_pos_emb_activations)
                        grads[target]['relpos'][shift].append(rel_pos_emb_grads)

                    i += 1
                    if n_samples is not None and i >= n_samples:
                        break

    def remove_source_if_empty(activations, source):
        # Remove relpos for any targets for which it is not used
        for target in targets:
            if target not in activations:
                continue
            if source not in activations[target]:
                continue

            all_empty = True
            for shift in shifts:
                rpe_length = len(activations[target][source][shift])
                if rpe_length != 0:
                    all_empty = False
                    break
            if all_empty:
                del activations[target][source]

        return activations

    remove_source_if_empty(activations, 'relpos')
    remove_source_if_empty(activations, 'pos_emb')

    return activations, grads