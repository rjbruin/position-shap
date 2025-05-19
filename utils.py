# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import random

import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torchvision import datasets


def get_git_revision():
    '''Get the current git revision.'''
    try:
        import subprocess
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()[:7]
    except Exception as e:
        print(f"Error getting git revision: {e}")
        return "Unknown"

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
    _, term_width = os.popen('stty size', 'r').read().split()
except:
    term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def get_subset(dataset, num_per_class):

    indices_dict = {}
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class in indices_dict.keys():
            indices_dict[current_class].append(i)
        else:
            indices_dict[current_class] = [i]
    print("*********Key")
    print(indices_dict.keys())
    indices_list = []
    for k in indices_dict.keys():
        current_class_list = indices_dict[k]
        random.shuffle(current_class_list)
        indices_list += current_class_list[:num_per_class]
    new_dataset = Subset(dataset, indices_list)
    return new_dataset


def get_class_subset(dataset, subset, classes=None):
    if subset == 'finegrained-10':
        classes = [4, 30, 55, 72, 95, 1, 32, 67, 73, 91]
    elif subset == 'finegrained-10-v2':
        classes = [54, 62, 70, 82, 92, 9, 10, 16, 28, 61]
    elif subset == 'finegrained-10-v3':
        classes = [0, 51, 53, 57, 83, 22, 39, 40, 86, 87]
    elif subset == 'coarse-10':
        classes = [4, 1, 54, 9, 0, 22, 5, 6, 3, 12]
    elif subset == 'coarse-10-v2':
        classes = [23, 15, 34, 26, 2, 27, 36, 47, 8, 41]
    elif subset == 'coarse-10-v3':
        classes = [30, 62, 51, 20, 42, 33, 63, 11, 50, 13]
    else:
        try:
            subset = int(subset)
        except ValueError:
            raise ValueError('Unknown subset: {}'.format(subset))

    indices_dict = {}
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class in indices_dict.keys():
            indices_dict[current_class].append(i)
        else:
            indices_dict[current_class] = [i]

    if classes is None:
        classes = sorted(list(indices_dict.keys()))
        in_classes = len(classes)
        random.shuffle(classes)
        classes = classes[:subset]
    else:
        in_classes = len(classes)

    indices_list = []
    class_mapping = {}
    for i, k in enumerate(classes):
        current_class_list = indices_dict[k]
        indices_list += current_class_list
        class_mapping[k] = i
    new_dataset = Subset(dataset, indices_list)

    print(f"Selected {subset} classes from dataset of {in_classes} classes:")
    print(",".join(map(str, classes)))

    class_mapping_fn = lambda x: class_mapping[x]
    if new_dataset.dataset.target_transform is not None:
        raise NotImplementedError("Cannot override non-None target_transform")
    new_dataset.dataset.target_transform = torchvision.transforms.Lambda(class_mapping_fn)

    return new_dataset, classes


def split_train_val(dataset, ratio=0.9):
    indices_dict = {}
    for i in range(len(dataset)):
        current_class = dataset[i][1]
        if current_class in indices_dict.keys():
            indices_dict[current_class].append(i)
        else:
            indices_dict[current_class] = [i]
    print("*********Key")
    print(indices_dict.keys())
    train_indices_list = []
    val_indices_list = []
    for k in indices_dict.keys():
        current_class_list = indices_dict[k]
        random.shuffle(current_class_list)
        train_indices_list += current_class_list[:int(len(current_class_list) * ratio)]
        val_indices_list += current_class_list[int(len(current_class_list) * ratio):]
    return train_indices_list, val_indices_list


if __name__ == '__main__':
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
    for i in range(100):
        print(str(trainset[i][1]) + ' ' + str(valset[i][1]))
    '''
    trainset, valset = split_train_val(trainset, 0.8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=32, shuffle=False)
    count = [0 for _ in range(10)]
    for data in trainloader:
        img, target = data
        for i in target:
            count[i] += 1
    print(count)
    count = [0 for _ in range(10)]
    for data in valloader:
        img, target = data
        for i in target:
            count[i] += 1
    print(count)
    '''