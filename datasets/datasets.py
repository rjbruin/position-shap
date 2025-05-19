import os
import torch
import torchvision
import torch.nn as nn
import random
import torchxrayvision as xrv
import sklearn.model_selection
from torch.utils.data import Subset
import pytorch_lightning as pl

import toy_experiments.datasets as toy_datasets
import datasets.own_vit


def setup_datasets(experiment, args):
    # If we are using a toy dataset, use the corresponding dataset policy
    if args.dataset == 'toy':
        transform_train, transform_val = toy_datasets.get_toy_dataset_transforms(args.toy_dataset)
        return setup_loaders(experiment, args, transform_train, transform_val)

    # If we are using a published model, and the model has a dataset policy, use
    # that policy
    if args.model != 'own-vit' and hasattr(experiment.net, 'dataset_policy'):
        return experiment.net.dataset_policy(experiment, args)

    # Otherwise, use a dataset policy from the datasets folder
    if args.dataset_policy == 'own-vit':
        return datasets.own_vit.own_vit_policy(experiment, args)

    raise NotImplementedError(f"Dataset policy {args.dataset_policy} not implemented")

def setup_loaders(experiment, args, transform_train, transform_val, sampler=None):
    """
    Sets up the train and val dataset loaders for the experiment. This is called
    from the dataset policy of any published model, as the dataset policies
    typically only differ in the transforms.
    """
    # Set dataset seed directly before initializing datasets
    pl.seed_everything(args.data_seed)

    if args.dataset == 'cifar100':
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
    elif args.dataset == 'cifar10_debug':
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_set = Subset(train_set, range(0, 300))
        val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)
        val_set = Subset(val_set, range(0, 300))
    elif args.dataset == 'oxfordpets':
        train_set = torchvision.datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform_train)
        val_set = torchvision.datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform_val)
    elif args.dataset == 'flowers102':
        train_set = torchvision.datasets.Flowers102(root='./data', split='train', download=True, transform=transform_train)
        val_set = torchvision.datasets.Flowers102(root='./data', split='test', download=True, transform=transform_val)
    elif args.dataset == 'imagenet' or args.dataset == 'imagenette':
        train_set = datasets.ImageFolderWithIndex(root=os.path.join(args.data_root, 'train'), transform=transform_train)
        val_set = datasets.ImageFolderWithIndex(root=os.path.join(args.data_root, 'val'), transform=transform_val)
    elif args.dataset == 'eurosat':
        data = torchvision.datasets.ImageFolder(root=args.data_root, transform=None)
        # data = torchvision.datasets.DatasetFolder(root=args.data_root, loader=iloader, transform=None, extensions = 'tif')
        train_size = int(len(data) * 0.8)
        val_size = len(data) - train_size
        train_set, val_set = torch.utils.data.random_split(data, [train_size, val_size])
        train_set.dataset.transform = transform_train
        val_set.dataset.transform = transform_val
    elif args.dataset in ['nih', 'nih_google']:
        if args.dataset == 'nih':
            dataset = xrv.datasets.NIH_Dataset(imgpath=os.path.join(args.data_root, "images-224"), views=["PA","AP"], unique_patients=False)
        else:
            dataset = xrv.datasets.NIH_Google_Dataset(imgpath=os.path.join(args.data_root, "images-224"), views=["PA","AP"], unique_patients=False)

        # Train/val splits from torchxrayvision training script
        # https://github.com/mlmed/torchxrayvision/blob/master/scripts/train_model.py
        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]

        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=args.seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_set = xrv.datasets.SubsetDataset(dataset, train_inds)
        val_set = xrv.datasets.SubsetDataset(dataset, test_inds)
        train_set.dataset.transform = transform_train
        val_set.dataset.transform = transform_val
    elif args.dataset == 'toy':
        train_images, train_labels, val_images, val_labels, n_classes, pos_labels = toy_datasets.get_toy_dataset(args.toy_dataset, size=args.toy_size)
        train_set = torch.utils.data.TensorDataset(train_images, train_labels)
        val_set = torch.utils.data.TensorDataset(val_images, val_labels)
        # Save pos_labels in experiment to be able to use it during SHAP analysis
        experiment.pos_labels = pos_labels
        experiment.args.num_classes = n_classes
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    if sampler is not None:
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, sampler=sampler(train_set), num_workers=args.num_workers, pin_memory=True)
    else:
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    # Shuffle val to make SHAP estimates more stable
    experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers, pin_memory=True)
    experiment.n_samples = len(train_set)

    # Reset seed to global seed
    pl.seed_everything(args.seed)


class EuroSATTrainTransform(nn.Module):
    """
    Source: https://github.com/ahmadmughees/EuroSat-PyTorch

    Adapted to replace skimage dependency with PIL.Image, for loading images.
    """

    def __init__(self):
        super().__init__()

    def forward(self, data):
        data = torch.rot90(data, random.randint(-3,3), dims=random.choice([[2,1],[1,2]]))
        if random.random() > 0.75:
            data = torch.flip(data, dims = random.choice([[1,],[2,],[1,2]]))
        pixmis = torch.empty_like(data).random_(data.shape[-1])
        pixmis = torch.where(pixmis > (data.shape[-1] / 8), torch.ones_like(data), torch.zeros_like(data))
        return data * pixmis