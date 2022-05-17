# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import torchvision.transforms
from torchvision.transforms import transforms


class ImageTransformation(object):
    """
    Random image transformation for training data augmentation.
    """
    def __call__(self, x):
        transform = transforms.Compose([
            transforms.RandomErasing(0.4, scale=(0.02, 0.05)),
            transforms.RandomAffine(20),
            transforms.RandomPerspective(
                distortion_scale=0.5,
                p=0.4,
                interpolation=torchvision.transforms.InterpolationMode.NEAREST
            )
        ])
        x = transform(x)
        return x
