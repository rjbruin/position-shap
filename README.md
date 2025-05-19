# Learning to Adapt to Position Bias in Vision Transformer Classifiers

*Anonymous.*

## Usage

TODO

## Installation

Use Anaconda to create a virtual environment with all the Python dependencies:

```bash
conda env create -f environment.yml
```

Alternatively, install PyTorch according to the [PyTorch installation instructions](https://pytorch.org/), then follow these commands:

```bash
conda install jupyter matplotlib munch numpy scipy
conda install pytorch-lightning einops pandas timm torchinfo wandb torchmetrics -c conda-forge
pip install ml_collections
```

## Experiments

### Toy experiments

See [toy-experiments](toy-experiments).

### Image classification experiments

Train using `train.py`. To detail arguments:

```bash
python train.py --help
```