import torch
import os

import sys
sys.path.append('..')
from analysis.saliency import gradbased_attribution


def save_raw_attributions(filepath, activations, gradients, report=True, **extra_data):
    """
    Save the raw attributions to a pickled file for later analysis.

    `activations` and `gradients` are in a particular structure for legacy
    reasons. The structure is as follows: `{source: {shift: [Tensor]}}`.
    `source` refers to either "image", "pos_emb", or "bias". Each source may be
    represented by zero or more Tensors in the analyzed network. Typically,
    there is one "image" source, one "pos_emb" source (or none if no PE is used)
    and many "bias" sources. These sources are the Tensors in the inner list.
    The `shift` key in this data structure is included because the legacy code
    would analyze the network for different shifts of the input image, to do
    some analysis related to relative position. For the purposes of this method,
    we only need the shift (0,0) (i.e. no shift of the input image) for each
    source.

    The saved file is a pickle with the ".pt"-suffix. The saved data structure
    is a dictionary of `source`s to lists of Tensors, matching the structure of
    the input `activations` and `gradients`, except for the removal of the
    `shift` index.

    Saved saliency maps are the result of applying the 'input-gradient' method
    to each pair of activations and gradients, without any further
    post-processing (corresponding to the "input_gradient_signed" setting in
    `analysis.saliency:gradbased_attribution`).

    Args:
        filepath (str): path to the file where the saliency maps will be saved.
            ".pt" will be appended if not present. Directory must exist;
        activations (dict(dict([torch.Tensor]))): for each "bias type"/"source",
            for each q(y,x) shift, contains a list of Tensor inputs used in the
            network.
        gradients (dict(dict([torch.Tensor]))): for each "bias type"/"source",
            for each (y,x) shift, contains a list of Tensor gradients w.r.t the
            target.
        **extra_data (any): any extra data to save along with the saliency maps.

    Returns:
        str: the path where the saliency maps were saved, or False if the
            saving failed.
    """
    # Input validation
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")
    directory = os.path.split(filepath)[0]
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")

    # Compute saliency maps for each source, but only for shift [0,0]
    saliency = {}
    for source in activations:
        saliency[source] = []

        shift = (0, 0)
        if shift not in activations[source]:
            raise ValueError(f"Shift {shift} not found in activations for source {source}")
        if shift not in gradients[source]:
            raise ValueError(f"Shift {shift} not found in gradients for source {source}")

        # For each (source, shift=(0,0), tensor), compute the saliency maps
        for i in range(len(activations[source][shift])):
            saliency[source].append(gradbased_attribution(activations[source][shift][i], gradients[source][shift][i], method='input_gradient_signed').detach())

    data = {'saliency': saliency, **extra_data}

    # Save saliency maps
    if not filepath.endswith('.pt'):
        filepath += '.pt'
    try:
        torch.save(data, filepath)
        if report:
            print(f"Saved saliency maps to {filepath}")
        return filepath
    except Exception as e:
        print(f"Failed to save saliency maps to {filepath}")
        print(f"Raised exception: {e}")
        return False


def load_raw_attributions(filepath):
    """
    Load the raw attributions from a pickled file.

    Args:
        filepath (str): path to the file where the saliency maps are saved.

    Returns:
        dict(dict([torch.Tensor])): the saliency maps loaded from the file.
    """
    # Input validation
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, not {type(filepath)}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} does not exist")

    # Load saliency maps
    saliency = torch.load(filepath, map_location=torch.device('cpu'))
    return saliency

def discover_batched_attributions(directory):
    # Input validation
    if not isinstance(directory, str):
        raise TypeError(f"directory must be a string, not {type(directory)}")
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"{directory} is not a directory")

    # Find batch files and sort
    batch_files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    batch_files_with_batch_idx = [(int(f.split('_')[-1].split('.')[0]), os.path.join(directory, f)) for f in batch_files]
    batch_files_with_batch_idx.sort(key=lambda x: x[0])

    return list(zip(*batch_files_with_batch_idx))[1]

def concat_any_type(data, batch_data):
    """Helper function to concatenate data, whether in list, dict or Tensor
    form."""
    if type(data) != type(batch_data):
        raise ValueError(f"Data types do not match: {type(data)} and {type(batch_data)}")

    if isinstance(batch_data, list):
        data += batch_data
        return data
    elif isinstance(batch_data, torch.Tensor):
        data = torch.cat([data, batch_data], dim=0)
        return data
    elif isinstance(batch_data, dict):
        for key in batch_data:
            data[key] = concat_any_type(data[key], batch_data[key])
        return data
    else:
        raise ValueError(f"Unexpected type: {type(batch_data)}")

def load_batched_attributions(directory):
    batch_files = discover_batched_attributions(directory)

    # Load first batch to set up variables
    data = load_raw_attributions(batch_files[0])

    # Load per batch and concatenate
    for batch_file in batch_files[1:]:
        batch_data = load_raw_attributions(batch_file)
        data = concat_any_type(data, batch_data)

    # Return concatenated saliency maps
    return data