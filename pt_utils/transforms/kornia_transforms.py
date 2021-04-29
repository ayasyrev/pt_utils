from typing import Tuple
from typing import Tuple, Union
import torch
import numpy as np
import accimage
# import kornia
from kornia.augmentation import RandomHorizontalFlip, RandomResizedCrop, Normalize, CenterCrop
from kornia.geometry import Resize


class AccimageImageToTensor(torch.nn.Module):

    def forward(self, img: accimage.Image):
        nppic = np.empty([img.channels, img.height, img.width], dtype=np.float32)
        img.copyto(nppic)
        return torch.from_numpy(nppic)

    def __repr__(self):
        return self.__class__.__name__


def train_transforms(image_size, train_img_scale=(0.35, 1),
    normalize: bool = True, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
    transforms = [
        AccimageImageToTensor(),
        RandomResizedCrop((image_size, image_size), train_img_scale, keepdim=True),
        RandomHorizontalFlip(keepdim=True)]
    if normalize:
        transforms.append(Normalize(mean=std, std=std, keepdim=True))
    return torch.nn.Sequential(*transforms)


def val_transforms(image_size, extra_size=32,
    normalize: bool = True, mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225])):
    """
    The standard imagenet transforms for validation: central crop, resize to self.image_size.
    """
    transforms = [
        AccimageImageToTensor(),
        Resize(image_size + extra_size),
        CenterCrop(image_size, keepdim=True)        ]
    if normalize:
        transforms.append(Normalize(mean=std, std=std, keepdim=True))
    return torch.nn.Sequential(*transforms)
