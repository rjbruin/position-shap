# Dataset policies

A *dataset policy* is just code for implementing a set of datasets and their
data augmentations. Different models may require different data setups.
Therefore, we provide some general data policies here. For published models, one
can specify a dataset policy as a (static) method of the `BasePublishedModel`
subclass, in which case that policy will be used.

## How to implement a general dataset policy

*NOTE*: if you want to implement data loading for a specific published model,
it's easier to do it as a method of the published model. See the `Example`
published model for details.

- Add a Python module for your policy and implement the data setup, using the
  variables `experiment`, corresponding to an instance of `train.py:Experiment`,
  and `args` corresponding to the parsed flags from `train.py`. Assign the data
  loaders to `Experiment`. See the policy `datasets/own_vit.py` for an example.
- Add your policy to the factory method `datasets/datasets.py:setup_datasets()`.
  This ensures `train.py` can find your policy.