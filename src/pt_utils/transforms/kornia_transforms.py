import torch

from kornia.augmentation import (
    RandomHorizontalFlip,
    RandomResizedCrop,
    Normalize,
    CenterCrop,
)
from kornia.geometry import Resize

from .accimage_transforms import AccimageImageToTensorNN


def train_transforms(
    image_size,
    train_img_scale=(0.35, 1),
    normalize: bool = True,
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
):
    """Transforms for train augmentation with Kornia."""

    transforms = [
        AccimageImageToTensorNN(),
        RandomResizedCrop((image_size, image_size), train_img_scale, keepdim=True),
        RandomHorizontalFlip(keepdim=True),
    ]
    if normalize:
        transforms.append(Normalize(mean=std, std=std, keepdim=True))
    return torch.nn.Sequential(*transforms)


def val_transforms(
    image_size,
    extra_size=32,
    normalize: bool = True,
    mean=torch.tensor([0.485, 0.456, 0.406]),
    std=torch.tensor([0.229, 0.224, 0.225]),
):
    """
    The standard imagenet transforms for validation with Kornia: central crop, resize to self.image_size.
    """
    transforms = [
        AccimageImageToTensorNN(),
        Resize(image_size + extra_size),
        CenterCrop(image_size, keepdim=True),
    ]
    if normalize:
        transforms.append(Normalize(mean=std, std=std, keepdim=True))
    return torch.nn.Sequential(*transforms)
