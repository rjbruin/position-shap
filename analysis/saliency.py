import torch


def gradbased_attribution(activation, gradient, method='input_gradient'):
    """
    Gradient-based attribution.

    Arguments:
        - activation (torch.Tensor): activation of the input for which the
          saliency is computed.
        - gradient (torch.Tensor): gradient of the input for which the saliency
          is computed;
        - method (str: 'input_gradient'): attribution method to use. Can be
          'input_gradient', 'input_gradient_withnegative', 'input_gradient_signed'.;
    """
    if method == 'input_gradient':
        return input_gradient(activation, gradient, norm=False)
    elif method == 'input_gradient_withnegative':
        return input_gradient(activation, gradient, norm=False, positive=False)
    elif method == 'input_gradient_signed':
        return input_gradient(activation, gradient, norm=False, positive=False, signed=True)
    else:
        raise NotImplementedError(f"Method \"{method}\"")


def input_gradient(activation, gradient, norm=False, positive=True, signed=False):
    """
    Compute saliency map using input-gradient attribution [1].

    [1] Simonyan et al. 2013. Deep Inside Convolutional Networks: Visualising
    Image Classification Models and Saliency Maps.

    Arguments:
        - activation (torch.Tensor): activation of the input for which the
          saliency is computed;
        - gradient (torch.Tensor): gradient of the input for which the saliency
          is computed;
        - norm (bool: False): whether to normalize the saliency map by the
          square of the L1 norm of the inputs;
    """
    if positive and signed:
        raise ValueError("Positive and signed cannot be used together.")

    if not torch.is_tensor(activation):
        raise TypeError(f"activation must be a torch.Tensor, not {type(activation)}")
    if not torch.is_tensor(gradient):
        raise TypeError(f"gradient must be a torch.Tensor, not {type(gradient)}")

    # Add singleton dimensions for shape match
    for _ in range(len(gradient.shape) - len(activation.shape)):
        activation = activation.unsqueeze(0)

    # Input-gradient attribution
    saliency_map = activation * gradient
    if norm:
        saliency_map = saliency_map / saliency_map.square().sum()
    if positive:
        return torch.maximum(torch.zeros_like(saliency_map), saliency_map)
    elif signed:
        return saliency_map
    else:
        return torch.absolute(saliency_map)


def aggregate_saliency(saliency_map, shape=None, method='max'):
    """
    Aggregate saliency scores in a tensor to desired shape, to create a saliency
    map.

    Arguments:
        - saliency_map (torch.Tensor): saliency map to aggregate;
        - shape (str: None): shape of the saliency map. Can be 'scalar' or
          'layer_head'.
        - aggregate (str: 'max'): aggregation method for the saliency map. Can
          be 'max' or 'sum';
    """
    # Generalize aggregation to prevent code duplication
    def aggregate_fn(data, dim=None):
        if method == 'max' and dim is None:
            return torch.max(data)
        elif method == 'max':
            return torch.max(data, dim=dim)[0]
        elif method == 'sum' and dim is None:
            return torch.sum(data)
        elif method == 'sum':
            return torch.sum(data, dim=dim)
        else:
            raise NotImplementedError()

    #
    # Shape to provided shape, then aggregate. Shapes should be
    # self-explanatory.
    #

    if shape is None:
        return saliency_map

    elif shape in ['loss', 'scalar']:
        return aggregate_fn(saliency_map)

    elif shape == 'head':
        # [num_heads, C_im, H_im, W_im] -> [num_heads]
        num_heads = saliency_map.shape[0]
        return aggregate_fn(saliency_map.reshape(num_heads, -1), dim=-1)

    # elif shape == 'attn_probs_score':
    #     # [num_heads, Q, K, C_im, H_im, W_im] -> [num_heads, Q, K]
    #     num_heads, Q, K, _, _, _ = saliency_map.shape
    #     return aggregate_fn(saliency_map.reshape(num_heads, Q, K, -1), dim=-1)

    # elif shape == 'feature_map_shape_to_attn_probs':
    #     # [H, W, C, C_im, H_im, W_im] -> [fake_num_heads=1, H, W]
    #     H, W, _, _, _, _ = saliency_map.shape
    #     return aggregate_fn(saliency_map.reshape(1, H, W, -1), dim=-1)

    # elif shape == 'feature_map_for_abs_vs_rel':
    #     # [H, W, C, C_im, H_im, W_im] -> [H, W, H_im, W_im]
    #     H, W, _, _, H_im, W_im = saliency_map.shape
    #     return aggregate_fn(saliency_map.permute(0, 1, 4, 5, 2, 3).reshape(H, W, H_im, W_im, -1), dim=-1)

    # elif shape == 'loss_spatial':
    #     # [C_im, H_im, W_im] -> [H_im, W_im]
    #     return aggregate_fn(saliency_map, dim=0)

    else:
        raise NotImplementedError(f"Shape \"{shape}\"")