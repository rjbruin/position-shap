import torch
import torchvision
from torchvision.datasets import CIFAR10
from PIL.Image import Image


class CIFAR10Derivative(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

    def __len__(self):
        return super().__len__()

    def is_pos_class(self, class_index):
        return self.own_class_is_pos_class[class_index]

    def get_c10_label(self, class_index):
        for j in self.c10_class_to_own_class:
            if isinstance(self.c10_class_to_own_class[j], list):
                if class_index in self.c10_class_to_own_class[j]:
                    return j
            elif self.c10_class_to_own_class[j] == class_index:
                return j
        return None


class CIFAR10Position(CIFAR10Derivative):
    def __init__(self, root, nr_classes, c10_class_to_own_class, c10_class_to_positions, c10_class_is_pos_class, own_class_is_pos_class, scale=-1, train=True, transform=None, target_transform=None, download=False):
        """Variant of CIFAR-10 where some classes are designated as "position
        "classes", samples of which appear in multiple positions in the image,
        each with a different class label.

        Args:
            root (str): Root directory of dataset where
                ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            nr_classes (int): The number of classes in the dataset, after
                assigning positions;
            c10_class_to_own_class: A dictionary mapping CIFAR-10 classes to
                classes in this dataset;
            c10_class_to_positions: A dictionary mapping CIFAR-10 classes to
                positions in the image;
            c10_class_is_pos_class: A dictionary mapping CIFAR-10 classes to
                whether they are position classes or not.
            own_class_is_pos_class: A dictionary mapping classes in this dataset
                to whether they are position classes or not.
            train (bool, optional): If True, creates dataset from training.pt,
                otherwise from test.pt.
            transform (callable, optional): A function/transform that  takes in
                an PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
            scale (int, optional): The scale determines how many resized images
                fit in the original image size: `image_width = image_width *
                2^scale`. A scale of 0 means that the image is not resized.
        """
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.scale = scale
        self.nr_classes = nr_classes
        self.c10_class_to_own_class = c10_class_to_own_class
        self.c10_class_to_positions = c10_class_to_positions
        self.c10_class_is_pos_class = c10_class_is_pos_class
        self.own_class_is_pos_class = own_class_is_pos_class

        self.pos_labels = []
        for c in range(10):
            if self.c10_class_is_pos_class[c]:
                self.pos_labels.extend(self.c10_class_to_own_class[c])

        print(str(self))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        # Special case: no scaling, so also no position classes
        # Still applies shuffling of classes if set
        if self.scale == 0:
            new_target = self.c10_class_to_own_class[target]
            return img, new_target, index

        # From Pillow to Torch
        pillow = False
        if isinstance(img, Image):
            pillow = True
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.float() / 255.0

        # (C, H, W)
        frame = torch.zeros_like(img)
        width = img.shape[-1]
        height = img.shape[-2]

        # Resize image
        img = torch.nn.functional.interpolate(img[None], scale_factor=2 ** self.scale, mode='bilinear', align_corners=False)[0]

        # Get target and position for this sample
        if self.c10_class_is_pos_class[target]:
            # Random sample from assigned classes and their positions
            n_options = len(self.c10_class_to_own_class[target])
            choice = torch.randint(0, n_options, (1,)).item()
            own_target = self.c10_class_to_own_class[target][choice]
            position = self.c10_class_to_positions[target][choice]
        else:
            # Given target and random sample from assigned (=all) positions
            own_target = self.c10_class_to_own_class[target]
            n_options = len(self.c10_class_to_positions[target])
            choice = torch.randint(0, n_options, (1,)).item()
            position = self.c10_class_to_positions[target][choice]

        # Place image in frame at position (one of the corners)
        blocks_per_row = 2 ** -self.scale
        block_width = width // blocks_per_row
        block_height = height // blocks_per_row
        row = position // blocks_per_row
        col = position % blocks_per_row
        frame[:, row * block_height:(row + 1) * block_height, col * block_width:(col + 1) * block_width] = img

        # Back to Pillow
        if pillow:
            frame = (frame * 255.).byte()
            frame = torchvision.transforms.functional.to_pil_image(frame)

        return frame, own_target, index

    @staticmethod
    def set_pos_classes(scale=-1, pos_classes=4, pos_per_class=2, shuffle_classes=False, shuffle_positions=True):
        """

        Args:
            scale (int, optional): The scale determines how many resized images
                fit in the original image size: `image_width = image_width *
                2^scale`. A scale of 0 means that the image is not resized.
            pos_classes (int, optional): The number of classes that are position
                classes.
            pos_per_class (int, optional): The number of positions per
                class.
            shuffle_classes (bool, optional): If True, the classes are shuffled
                before assigning position classes. If False, the first
                `pos_classes` classes are assigned as position classes.
            shuffle_positions (bool, optional): If True, the positions are
                shuffled before assigning them to classes. If False, positions
                are always assigned in column-row order.
        """
        assert scale <= 0, "scale must be less than or equal to 0"

        # The scale determines how many resized images fit in the original image size
        max_pos_per_class = (2 ** -scale) ** 2

        assert pos_classes >= 0, "pos_classes must be greater than or equal to 0"
        assert pos_classes <= 10, "pos_classes must be less than or equal the number of classes in CIFAR10"
        if pos_classes > 0:
            assert pos_per_class <= max_pos_per_class, f"scale {scale} allows for a maximum of {max_pos_per_class} positions per class, but {pos_per_class} were requested"

        # We keep all ten classes, and create extra classes for each position
        # above 1 in each position class
        nr_classes = CIFAR10Position.get_nr_classes(pos_classes, pos_per_class)

        # Assign CIFAR-10 classes to use position or not
        cifar10_classes = torch.arange(10)
        if shuffle_classes:
            cifar10_classes = cifar10_classes[torch.randperm(10)]

        c10_class_to_own_class = {}
        c10_class_to_positions = {}
        c10_class_is_pos_class = {}
        own_class_is_pos_class = {}
        j = 0
        for i in range(10):
            c = cifar10_classes[i].item()
            positions = torch.arange(max_pos_per_class)

            if i < pos_classes:
                c10_class_is_pos_class[c] = True
                c10_class_to_own_class[c] = [j + k for k in range(pos_per_class)]
                for k in range(pos_per_class):
                    own_class_is_pos_class[j + k] = True

                if shuffle_positions:
                    positions = positions[torch.randperm(len(positions))]
                c10_class_to_positions[c] = positions[:pos_per_class]

                j += pos_per_class
            else:
                c10_class_is_pos_class[c] = False
                c10_class_to_own_class[c] = j
                own_class_is_pos_class[j] = False

                c10_class_to_positions[c] = positions

                j += 1
        assert j == nr_classes

        return nr_classes, c10_class_to_own_class, c10_class_to_positions, c10_class_is_pos_class, own_class_is_pos_class

    def __str__(self):
        return f"CIFAR10Position({self.c10_class_to_own_class}, {self.c10_class_to_positions}"

    @staticmethod
    def get_nr_classes(pos_classes, pos_per_class):
        return pos_classes * pos_per_class + (10 - pos_classes)


class CIFAR10Shortcut(CIFAR10Derivative):
    def __init__(self, root, nr_classes, c10_class_to_own_class, c10_class_to_shortcuts, c10_class_is_cut_class, own_class_is_cut_class, train=True, transform=None, target_transform=None, download=False):
        """Variant of CIFAR-10 where some classes are designated as "shortcut
        classes", samples of which contain a visually discriminative artifact,
        to simulate conditions where such artifacts are undesirable and do not
        appear in o.o.d. data.

        Args:
            root (str): Root directory of dataset where
                ``CIFAR10/processed/training.pt``
                and  ``CIFAR10/processed/test.pt`` exist.
            nr_classes (int): The number of classes in the dataset, after
                assigning shortcuts;
            c10_class_to_own_class: A dictionary mapping CIFAR-10 classes to
                classes in this dataset;
            c10_class_to_shortcuts: A dictionary mapping CIFAR-10 classes to
                shortcuts in the image;
            c10_class_is_cut_class: A dictionary mapping CIFAR-10 classes to
                whether they are shortcut classes or not.
            own_class_is_cut_class: A dictionary mapping classes in this dataset
                to whether they are shortcut classes or not.
            train (bool, optional): If True, creates dataset from training.pt,
                otherwise from test.pt.
            transform (callable, optional): A function/transform that  takes in
                an PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.nr_classes = nr_classes
        self.c10_class_to_own_class = c10_class_to_own_class
        self.c10_class_to_positions = c10_class_to_shortcuts
        self.c10_class_is_pos_class = c10_class_is_cut_class
        self.own_class_is_pos_class = own_class_is_cut_class
        self.ood_mode = False

        self.pos_labels = []
        for c in range(10):
            if self.c10_class_is_pos_class[c]:
                self.pos_labels.append(self.c10_class_to_own_class[c])

        print(str(self))

    def __getitem__(self, index):
        img, target = super().__getitem__(index)

        if self.ood_mode:
            return img, target, index

        # From Pillow to Torch
        pillow = False
        if isinstance(img, Image):
            pillow = True
            img = torchvision.transforms.functional.pil_to_tensor(img)
            img = img.float() / 255.0

        # (C, H, W)
        shortcut = self.c10_class_to_positions[target]
        if shortcut is not None:
            img = shortcut.draw(img)

        # Back to Pillow
        if pillow:
            img = (img * 255.).byte()
            img = torchvision.transforms.functional.to_pil_image(img)

        return img, target, index

    @staticmethod
    def set_pos_classes(cut_classes=4, shuffle_classes=False, shuffle_shortcuts=True):
        """

        Args:
            cut_classes (int, optional): The number of classes that are shortcut
                classes.
            shuffle_classes (bool, optional): If True, the classes are shuffled
                before assigning position classes. If False, the first
                `pos_classes` classes are assigned as position classes.
            shuffle_shortcuts (bool, optional): If True, the shortcuts are
                shuffled before assigning them to classes. If False, shortcuts
                are always assigned in the same order.
        """
        assert cut_classes >= 0, "pos_classes must be greater than or equal to 0"
        assert cut_classes <= len(SHORTCUTS), "pos_classes must be less than or equal the number of shortcuts implemented"
        assert cut_classes <= 10, "pos_classes must be less than or equal the number of classes in CIFAR-10"

        nr_classes = 10

        # Assign CIFAR-10 classes to use position or not
        cifar10_classes = torch.arange(10)
        if shuffle_classes:
            cifar10_classes = cifar10_classes[torch.randperm(10)]

        if shuffle_shortcuts:
            shortcuts = []
            for c in torch.randperm(len(SHORTCUTS))[:cut_classes]:
                shortcuts.append(SHORTCUTS[c])
        else:
            shortcuts = SHORTCUTS[:cut_classes]

        c10_class_to_own_class = {}
        c10_class_to_positions = {}
        c10_class_is_pos_class = {}
        own_class_is_pos_class = {}
        j = 0
        for i in range(10):
            c = cifar10_classes[i].item()

            if i < cut_classes:
                c10_class_is_pos_class[c] = True
                c10_class_to_own_class[c] = j
                own_class_is_pos_class[j] = True
                c10_class_to_positions[c] = shortcuts[j]
            else:
                c10_class_is_pos_class[c] = False
                c10_class_to_own_class[c] = j
                own_class_is_pos_class[j] = False
                c10_class_to_positions[c] = None

            j += 1
        assert j == nr_classes

        return nr_classes, c10_class_to_own_class, c10_class_to_positions, c10_class_is_pos_class, own_class_is_pos_class

    def __str__(self):
        return f"CIFAR10Shortcut({self.c10_class_to_own_class}, {','.join([str(self.c10_class_to_positions[i]) for i in range(10)]):s}"

    @staticmethod
    def get_nr_classes():
        return 10


class Shortcut:
    def __init__(self):
        pass

    def draw(self, image):
        pass


class ColoredSquareShortcut(Shortcut):
    def __init__(self, color=(1.0, 0, 0), size=3, position=0, pos_noise=1, img_size=32, noise=0.05):
        self.color = torch.tensor(color, dtype=torch.float32)
        self.noise = noise
        self.pos_noise = pos_noise
        self.size = size
        from_edge = 4
        positions = [
            (from_edge, from_edge),
            (img_size - from_edge, from_edge),
            (from_edge, img_size - from_edge),
            (img_size - from_edge, img_size - from_edge)
        ]
        self.position = positions[position]

    def draw(self, image):
        """
        Arguments:
            image: Tensor of shape (img_size, img_size, 3).
        """
        noise = torch.rand_like(self.color) * self.noise

        image = image.clone()
        xnoise = torch.randint(-self.pos_noise, self.pos_noise + 1, (1,)).item()
        ynoise = torch.randint(-self.pos_noise, self.pos_noise + 1, (1,)).item()
        for i in range(-self.size // 2, self.size // 2):
            for j in range(-self.size // 2, self.size // 2):
                image[:, self.position[0] + i + xnoise, self.position[1] + j + ynoise] = torch.clamp(self.color + noise, min=0., max=1.)
        return image

    def __str__(self):
        return f"ColoredSquareShortcut(color={self.color}, size={self.size}, position={self.position}, pos_noise={self.pos_noise}, noise={self.noise})"

SHORTCUTS = [
    ColoredSquareShortcut(color=(1.0, 0, 0), size=3, position=0),
    ColoredSquareShortcut(color=(0, 1.0, 0), size=3, position=1),
    ColoredSquareShortcut(color=(0, 0, 1.0), size=3, position=2),
    ColoredSquareShortcut(color=(1.0, 1.0, 0), size=3, position=3),
]