import math
from typing import Union
from torchvision import transforms as T

from .simple_transforms import ResizeCrop
from .normalize import Normalize


def normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    return T.Normalize(mean=mean, std=std)


def train_transforms(image_size, train_img_scale=(0.35, 1)):
    """
    The standard imagenet transforms: random crop, resize to self.image_size, flip.
    Scale factor by default as at fast.ai example train script.
    """
    preprocessing = T.Compose(
        [
            T.RandomResizedCrop(image_size, scale=train_img_scale),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            Normalize(),
        ]
    )

    return preprocessing


def val_transforms(image_size: int, extra_size: Union[int, None] = None, xtra_pct: float = 0.14):
    """
    The standard imagenet transforms for validation: central crop, resize to self.image_size.
    """
    if extra_size is None:
        extra_size = math.ceil(image_size * xtra_pct / 8) * 8
    else:
        extra_size = extra_size

    preprocessing = T.Compose(
        [
            ResizeCrop(image_size, extra_size),
            T.ToTensor(),
            Normalize(),
        ]
    )
    return preprocessing
