import os
import numpy as np
import torch
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Subset
import torchxrayvision as xrv
import sklearn.model_selection

from timm.data import Mixup, rand_augment_transform

from augment import CIFAR10Policy, ImageNetPolicy
from utils import get_subset
from datasets.position_datasets import CIFAR10Position, CIFAR10Shortcut


class PermuteNumpy:
    def __init__(self, permutation):
        self.permutation = permutation

    def __call__(self, img):
        return img.transpose(self.permutation)

class TileNumpy:
    def __init__(self, n):
        self.n = n

    def __call__(self, img):
        return np.tile(img, (self.n, 1, 1))

def own_vit_policy(experiment, args):
    """
    Originally implemented directly in train.py.
    """
    size = experiment.img_size
    experiment.cutmix = None

    #
    # Transforms: building blocks for position-related augmentations
    #

    pos_augs_small = [
        transforms.RandomCrop(size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    pos_augs_small_no_crop = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    pos_augs_large = [
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
    ]
    aa_transforms_no_geometric = ['AutoContrast', 'Equalize', 'Invert', 'PosterizeIncreasing', 'SolarizeIncreasing', 'SolarizeAdd', 'ColorIncreasing', 'ContrastIncreasing', 'BrightnessIncreasing', 'SharpnessIncreasing']
    aa_no_pos = rand_augment_transform('rand-m9-mstd0.5-inc1', {}, transforms=aa_transforms_no_geometric)
    aa_pos = rand_augment_transform('rand-m9-mstd0.5-inc1', {})
    random_erase = [transforms.RandomErasing(p=args.random_erasing, scale=(0.02, 0.25))] \
                    if args.random_erasing > 0.0 \
                    else []
    # normalize = [transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
    #                                   std=[0.2023, 0.1994, 0.2010])] \
    if args.imagenet_normalization_huggingface:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    normalize = [transforms.Normalize(mean=mean, std=std)] \
                 if args.model_pretrained == 'imagenet' or args.dataset in ['imagenet', 'imagenet-tiny', 'imagenette'] \
                 else []

    #
    # Transforms: general
    #

    transform_train_rcrop_horflip_norm = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ])

    transform_test_norm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])

    transform_test_nonorm = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
    ])

    #
    # Transforms: dataset-specific
    #

    # Oxford Flowers
    transform_train_flower = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_flower_pos = transforms.Compose([
        *pos_augs_small,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_flower_adv = transforms.Compose([
        aa_no_pos,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_train_flower_adv_pos = transforms.Compose([
        aa_pos,
        *pos_augs_small,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_test_flower = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        *normalize,
    ])

    # SVHN
    transform_train_svhn = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_svhn_adv = transforms.Compose([
        aa_no_pos,
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    svhn_pos_augs = pos_augs_small_no_crop if size > 40 else pos_augs_small
    transform_train_svhn_pos = transforms.Compose([
        *svhn_pos_augs,
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_svhn_adv_pos = transforms.Compose([
        aa_pos,
        *svhn_pos_augs,
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_test_svhn = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])

    # Stanford Cars
    transform_train_stanfordcars = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_stanfordcars_adv = transforms.Compose([
        aa_no_pos,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_train_stanfordcars_pos = transforms.Compose([
        *pos_augs_small,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_stanfordcars_adv_pos = transforms.Compose([
        aa_pos,
        *pos_augs_small,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_test_stanfordcars = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])

    # ImageNet Tiny
    transform_train_imagenet_tiny = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_test_imagenet_tiny = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])

    # ImageNet
    transform_train_imagenet = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_test_imagenet = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        *normalize,
    ])

    # ImageNette
    transform_train_imagenette = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_imagenette_pos = transforms.Compose([
        *pos_augs_large,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_imagenette_adv = transforms.Compose([
        aa_no_pos,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_imagenette_adv_pos = transforms.Compose([
        *pos_augs_large,
        aa_pos,
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_test_imagenette = transforms.Compose([
        transforms.Resize((size, size)),
        # transforms.CenterCrop(size),
        transforms.ToTensor(),
        *normalize,
    ])

    # EuroSAT
    transform_train_eurosat = torchvision.transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_eurosat_pos = torchvision.transforms.Compose([
        *pos_augs_small,
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_eurosat_adv = torchvision.transforms.Compose([
        aa_no_pos,
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_train_eurosat_adv_pos = torchvision.transforms.Compose([
        aa_pos,
        *pos_augs_small,
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_test_eurosat = transform_test_nonorm

    # NIH Chest X-ray
    transform_train_nih = torchvision.transforms.Compose([
        transforms.ToTensor(),
        TileNumpy(3),
        PermuteNumpy([1, 2, 0]),
        transforms.Resize(size),
        *normalize,
    ])
    transform_train_nih_pos = torchvision.transforms.Compose([
        transforms.ToTensor(),
        TileNumpy(3),
        PermuteNumpy([1, 2, 0]),
        *pos_augs_small,
        transforms.Resize(size),
        *normalize,
    ])
    transform_train_nih_adv = torchvision.transforms.Compose([
        transforms.ToTensor(),
        TileNumpy(3),
        PermuteNumpy([1, 2, 0]),
        aa_no_pos,
        transforms.Resize(size),
        *random_erase,
        *normalize,
    ])
    transform_train_nih_adv_pos = torchvision.transforms.Compose([
        transforms.ToTensor(),
        TileNumpy(3),
        PermuteNumpy([1, 2, 0]),
        aa_pos,
        *pos_augs_small,
        transforms.Resize(size),
        *random_erase,
        *normalize,
    ])
    transform_test_nih = transforms.Compose([
        TileNumpy(3),
        PermuteNumpy([1, 2, 0]),
        transforms.ToTensor(),
        transforms.Resize(size),
        *normalize,
    ])

    # CIFAR-10
    transform_train_c10 = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_c10_pos = transforms.Compose([
        *pos_augs_small,
        transforms.Resize(size),
        transforms.ToTensor(),
        *normalize,
    ])
    transform_train_c10_adv = transforms.Compose([
        aa_no_pos,
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_train_c10_adv_pos = transforms.Compose([
        aa_pos,
        *pos_augs_small,
        transforms.RandomCrop(size, padding=4),
        transforms.Resize(size),
        transforms.ToTensor(),
        *random_erase,
        *normalize,
    ])
    transform_test_c10 = transform_test_nonorm



    #
    # Dataset setup
    #

    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    if args.dataset in ['cifar10', 'cifar10_position', 'cifar10_shortcut']:
        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_c10_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_c10_adv
        elif args.pos_augmentations:
            training_transform = transform_train_c10_pos
        else:
            training_transform = transform_train_c10

        testing_transform = transform_test_c10
        if args.dataset == 'cifar10':
            train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=training_transform)
            val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=testing_transform)
        elif args.dataset == 'cifar10_position':
            pos_settings = CIFAR10Position.set_pos_classes(scale=args.c10pos_scale,
                                                           pos_classes=args.c10pos_pos_classes,
                                                           pos_per_class=args.c10pos_pos_per_class,
                                                           shuffle_classes=args.c10pos_shuffle_classes)
            train_set = CIFAR10Position('./data', *pos_settings, train=True, download=True, transform=training_transform, scale=args.c10pos_scale)
            val_set = CIFAR10Position('./data', *pos_settings, train=False, download=True, transform=testing_transform, scale=args.c10pos_scale)
            experiment.pos_labels = train_set.pos_labels
        elif args.dataset == 'cifar10_shortcut':
            pos_settings = CIFAR10Shortcut.set_pos_classes(cut_classes=args.c10cut_cut_classes,
                                                           shuffle_classes=args.c10cut_shuffle_classes)
            train_set = CIFAR10Shortcut('./data', *pos_settings, train=True, download=True, transform=training_transform)
            val_set = CIFAR10Shortcut('./data', *pos_settings, train=False, download=True, transform=testing_transform)
            experiment.pos_labels = train_set.pos_labels

        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')

        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'cifar100':
        if args.adv_augmentations:
            raise NotImplementedError('Advanced augmentations not implemented for CIFAR-100')
        training_transform = transform_train_c10_pos if args.pos_augmentations else transform_train_c10
        testing_transform = transform_test_nonorm
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=training_transform)
        val_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=testing_transform)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'flowers102':
        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_flower_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_flower_adv
        elif args.pos_augmentations:
            training_transform = transform_train_flower_pos
        else:
            training_transform = transform_train_flower
        testing_transform = transform_test_flower

        # data_dir = './data/flower_data/'
        data_dir = args.data_root
        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=training_transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'valid'), transform=testing_transform)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')

        experiment.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, pin_memory=False, num_workers=args.num_workers, collate_fn=collate_fn)
        experiment.val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, pin_memory=False, num_workers=args.num_workers, collate_fn=collate_fn)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'svhn':
        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_svhn_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_svhn_adv
        elif args.pos_augmentations:
            training_transform = transform_train_svhn_pos
        else:
            training_transform = transform_train_svhn
        testing_transform = transform_test_svhn

        train_set = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=training_transform)
        val_set = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=testing_transform)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'stanfordcars':
        # Get dataset from:
        # https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616

        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_stanfordcars_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_stanfordcars_adv
        elif args.pos_augmentations:
            training_transform = transform_train_stanfordcars_pos
        else:
            training_transform = transform_train_stanfordcars
        testing_transform = transform_test_stanfordcars

        train_set = torchvision.datasets.StanfordCars(root=args.data_root, split='train', download=False, transform=training_transform)
        val_set = torchvision.datasets.StanfordCars(root=args.data_root, split='test', download=False, transform=testing_transform)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers, collate_fn=collate_fn)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'imagenet-tiny':
        if args.adv_augmentations:
            raise NotImplementedError('Advanced augmentations not implemented for ImageNet Tiny')
        if args.pos_augmentations:
            raise NotImplementedError('Position augmentations not implemented for ImageNet Tiny')
        # data_dir = './data/tiny-imagenet'
        data_dir = args.data_root
        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train_imagenet_tiny)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test_imagenet_tiny)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'imagenet':
        if args.adv_augmentations:
            raise NotImplementedError('Advanced augmentations not implemented for ImageNet')
        if args.pos_augmentations:
            raise NotImplementedError('Position augmentations not implemented for ImageNet')
        data_dir = args.data_root
        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform_train_imagenet)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform_test_imagenet)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers, drop_last=True)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'imagenette':
        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_imagenette_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_imagenette_adv
        elif args.pos_augmentations:
            training_transform = transform_train_imagenette_pos
        else:
            training_transform = transform_train_imagenette
        testing_transform = transform_test_imagenette

        data_dir = args.data_root
        train_set = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=training_transform)
        val_set = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=testing_transform)
        if args.num_per_class != 'all':
            print('Building sub dataset')
            train_set = get_subset(train_set, int(args.num_per_class))
            print('Finish building sub dataset')
        experiment.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset == 'eurosat':
        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_eurosat_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_eurosat_adv
        elif args.pos_augmentations:
            training_transform = transform_train_eurosat_pos
        else:
            training_transform = transform_train_eurosat
        testing_transform = transform_test_eurosat

        data = torchvision.datasets.ImageFolder(root=args.data_root, transform=None)
        train_size = int(len(data) * 0.8)
        val_size = len(data) - train_size
        train_set, val_set = torch.utils.data.random_split(data, [train_size, val_size])
        train_set.dataset.transform = training_transform
        val_set.dataset.transform = testing_transform
        experiment.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    elif args.dataset in ['nih', 'nih_google']:
        if args.dataset == 'nih':
            dataset = xrv.datasets.NIH_Dataset(imgpath=os.path.join(args.data_root, "images-224"), views=["PA","AP"], unique_patients=False)
        elif args.dataset == 'nih_google':
            dataset = xrv.datasets.NIH_Google_Dataset(imgpath=os.path.join(args.data_root, "images-224"), views=["PA","AP"], unique_patients=False)
        else:
            raise NotImplementedError()

        # Train/val splits from torchxrayvision training script
        # https://github.com/mlmed/torchxrayvision/blob/master/scripts/train_model.py
        # give patientid if not exist
        if "patientid" not in dataset.csv:
            dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]

        if args.adv_augmentations and args.pos_augmentations:
            training_transform = transform_train_nih_adv_pos
        elif args.adv_augmentations:
            training_transform = transform_train_nih_adv
        elif args.pos_augmentations:
            training_transform = transform_train_nih_pos
        else:
            training_transform = transform_train_nih
        testing_transform = transform_test_nih

        gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=args.data_seed)
        train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
        train_set = xrv.datasets.SubsetDataset(dataset, train_inds)
        val_set = xrv.datasets.SubsetDataset(dataset, test_inds)
        train_set.dataset.transform = training_transform
        val_set.dataset.transform = testing_transform
        experiment.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        experiment.val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
        experiment.n_samples = len(train_set)

    # elif args.dataset == 'chexpert':
    #     dataset = xrv.datasets.CheX_Dataset(imgpath=os.path.join(args.data_root, "images-224"), views=["PA","AP"], unique_patients=False)

    #     # Train/val splits from torchxrayvision training script
    #     # https://github.com/mlmed/torchxrayvision/blob/master/scripts/train_model.py
    #     # give patientid if not exist
    #     if "patientid" not in dataset.csv:
    #         dataset.csv["patientid"] = ["{}-{}".format(dataset.__class__.__name__, i) for i in range(len(dataset))]

    #     if args.adv_augmentations and args.pos_augmentations:
    #         training_transform = transform_train_nih_adv_pos
    #     elif args.adv_augmentations:
    #         training_transform = transform_train_nih_adv
    #     elif args.pos_augmentations:
    #         training_transform = transform_train_nih_pos
    #     else:
    #         training_transform = transform_train_nih
    #     testing_transform = transform_test_nih

    #     gss = sklearn.model_selection.GroupShuffleSplit(train_size=0.8, test_size=0.2, random_state=args.data_seed)
    #     train_inds, test_inds = next(gss.split(X=range(len(dataset)), groups=dataset.csv.patientid))
    #     train_set = xrv.datasets.SubsetDataset(dataset, train_inds)
    #     val_set = xrv.datasets.SubsetDataset(dataset, test_inds)
    #     train_set.dataset.transform = training_transform
    #     val_set.dataset.transform = testing_transform
    #     experiment.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    #     experiment.val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=args.shuffle_val, num_workers=args.num_workers)
    #     experiment.n_samples = len(train_set)