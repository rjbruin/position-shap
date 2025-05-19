#
# APPEARANCE
#

import torch
import torchvision

def dataset_appearance(size=[5,5], colors=[0, 1]):
    """Create a dataset where each image has one pixel, and we classify the color of the pixel."""
    images = []
    labels = []
    for t, c in enumerate(colors):
        for y1 in range(size[0]):
            for x1 in range(size[1]):
                image = torch.rand([3] + size) * 0.01
                image[c, y1, x1] = 1.0

                images.append(image)
                labels.append(t)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, len(colors)

def split_dataset(images, labels, train_fraction=1.0):
    train_split = (0.0, train_fraction)
    if train_fraction == 1.0:
        test_split = (0.0, 1.0)
    else:
        test_split = (train_fraction, 1.0)

    indices = torch.randperm(images.shape[0])
    train_indices = indices[int(train_split[0] * indices.shape[0]):int(train_split[1] * indices.shape[0])]
    test_indices = indices[int(test_split[0] * indices.shape[0]):int(test_split[1] * indices.shape[0])]
    train_images = images[train_indices]
    train_labels = labels[train_indices]
    test_images = images[test_indices]
    test_labels = labels[test_indices]
    analysis_images = images
    analysis_labels = labels

    print(f"{train_images.shape[0]} training samples")
    print(f"{test_images.shape[0]} test samples")
    print(f"{analysis_images.shape[0]} analysis samples")

    return train_images, train_labels, test_images, test_labels, analysis_images, analysis_labels


#
# ABSOLUTE POSITION
#

def dataset_absolute_position(size=[6,6], color=[0]):
    """Create a dataset where each image has a colored pixel, and we classify whether the colored pixel is on the left or the right half of the image."""
    images = []
    labels = []
    for y1 in range(size[0]):
        for x1 in range(size[1]):
            if x1 < size[1] // 2:
                label = 0
            else:
                label = 1

            # image = torch.rand([3] + size) * 0.1
            image = torch.rand([3] + size) * 0.0
            image[color, y1, x1] = 1.0

            images.append(image)
            labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, 2

#
# RELATIVE POSITION
#

def dataset_relative_position(size=[5,5]):
    """Create a dataset where each image has two red pixels, which is
        - class 0 if x1 - x2 = -2 and y1 - y2 = 0
        - class 1 if x1 - x2 == -1 and y1 - y2 = 0
    """
    images = []
    labels = []
    for y1 in range(size[0]):
        for x1 in range(size[1]):
            for y2 in range(size[0]):
                for x2 in range(size[1]):
                    hor_dist = x1 - x2
                    ver_dist = y1 - y2
                    if not (hor_dist == -2 or hor_dist == -1):
                        continue
                    if ver_dist != 0:
                        continue

                    label = torch.tensor(hor_dist == -2).long()

                    # image = torch.rand([3] + size) * 0.1
                    image = torch.zeros([3] + size)
                    image[0, y1, x1] = 1.0
                    image[0, y2, x2] = 1.0

                    images.append(image)
                    labels.append(label)

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, 2

#
# MIXED POSITION
#

import torch

COLORS = [
    # ([0.0, 0.0, 0.0], 'black'),
    (0, 'red'),
    (1, 'green'),
    (2, 'blue'),
    (3, '???'),
    (4, '???'),
    (5, '???'),
    (6, '???'),
]

def dataset_mixed_position(n_mixed, n_appearance, size=[6,6], noise=0.0, input_d=7):
    """Create a dataset where each image has a red pixel, and we classify whether the red pixel is on the left or the right half of the image."""
    if n_mixed + n_appearance > len(COLORS):
        raise ValueError("n_mixed + n_appearance must be less than the number of colors")

    images = []
    labels = []
    label_i = 0
    sort_groups = []
    for t, (val, c) in enumerate(COLORS[:n_mixed+n_appearance]):
        mixed_label = t < n_mixed

        for y1 in range(size[0]):
            for x1 in range(size[1]):
                image = torch.rand([input_d] + size) * noise
                image[val, y1, x1] = 1.0

                if mixed_label:
                    if x1 < size[1] // 2:
                        pos_label = 0
                    else:
                        pos_label = 1
                    label = label_i + pos_label
                else:
                    label = label_i

                images.append(image)
                labels.append(label)

        if mixed_label:
            sort_groups.append([label_i, label_i+1])
            label_i += 2
        else:
            sort_groups.append([label_i])
            label_i += 1

    images = torch.stack(images)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels, label_i, sort_groups

#
# MIXING APPEARANCE AND ABSOLUTE POSITION
#

def dataset_appearance_absolute_position(size=[6,6], colors=['red', 'green'], pos_colors=['red'], split_quadrants=False):
    """Create a dataset where each image has either 1) a red pixel in the left
    half of the image, 2) a red pixel in the right half of the image, or 3) a
    green pixel, no matter where."""
    train_images = []
    train_labels = []

    if split_quadrants:
        quadrants = {}
        test_images = []
        test_labels = []

    l = 0
    for t, c in enumerate(colors):
        pos_label = c in pos_colors
        for y1 in range(size[0]):
            for x1 in range(size[1]):
                image = torch.rand([3] + size) * 0.01
                image[t, y1, x1] = 1.0

                if not pos_label:
                    label = l
                elif x1 < size[1] // 2:
                    label = l
                else:
                    label = l + 1

                if not split_quadrants:
                    train_images.append(image)
                    train_labels.append(label)
                else:
                    # Find quadrant coordinates
                    qx = x1 // 2
                    qy = y1 // 2
                    if (qx, qy, c) not in quadrants:
                        quadrants[(qx, qy, c)] = []
                    quadrants[(qx, qy, c)].append((image, label))
        l += 1
        if pos_label:
            l += 1

    if split_quadrants:
        for _, images_labels in quadrants.items():
            test_idx = torch.randint(0, len(images_labels), ())
            for i, (image, label) in enumerate(images_labels):
                if i == test_idx:
                    test_images.append(image)
                    test_labels.append(label)
                else:
                    train_images.append(image)
                    train_labels.append(label)

    if not split_quadrants:
        train_images = torch.stack(train_images)
        train_labels = torch.tensor(train_labels, dtype=torch.long)
        return train_images, train_labels, 1 + len(colors)
    else:
        # Apply random permutation
        inds = torch.randperm(len(train_images))
        train_images = torch.stack(train_images)[inds]
        train_labels = torch.tensor(train_labels, dtype=torch.long)[inds]
        inds = torch.randperm(len(test_images))
        test_images = torch.stack(test_images)[inds]
        test_labels = torch.tensor(test_labels, dtype=torch.long)[inds]

        print(f"{train_images.shape[0]} training samples")
        print(f"{test_images.shape[0]} test samples")

        return train_images, train_labels, test_images, test_labels, l

def dataset_appearance_absolute_position_three_colors(size=[6,6], split_quadrants=False):
    return dataset_appearance_absolute_position(size=size, colors=['red', 'green', 'blue'], split_quadrants=split_quadrants)

def get_toy_dataset(name, size=6):
    if name == 'appearance_absolute_position_three_colors':
        size = [size, size]
        train_images, train_labels, test_images, test_labels, l = dataset_appearance_absolute_position_three_colors(size=size, split_quadrants=True)
        pos_labels = [0, 1]
        return train_images, train_labels, test_images, test_labels, l, pos_labels
    else:
        raise ValueError(f'Unknown toy dataset {name}')

def get_toy_dataset_transforms(name):
    if name == 'appearance_absolute_position_three_colors':
        transform_train = torchvision.transforms.ToTensor()
        transform_val = torchvision.transforms.ToTensor()
        return transform_train, transform_val
    else:
        raise ValueError(f'Unknown toy dataset {name}')