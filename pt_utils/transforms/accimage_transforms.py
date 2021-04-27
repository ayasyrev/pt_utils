from torchvision import transforms as T
import torch
import accimage
import numpy as np


class AccimageImageToTensor(object):

    def __call__(self, img: accimage.Image) -> torch.Tensor:
        nppic = np.empty([img.channels, img.height, img.width], dtype=np.float32)
        img.copyto(nppic)
        return torch.from_numpy(nppic)

    def __repr__(self):
        return self.__class__.__name__


class Normalize(object):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 inplace=False,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        mean = torch.as_tensor(mean, dtype=dtype, device=device)
        self.mean = mean[:, None, None]
        std = torch.as_tensor(std, dtype=dtype, device=device)
        self.std = std[:, None, None]
        self.inplace = inplace

    def __call__(self, tensor: torch.Tensor):
        if not self.inplace:
            tensor = tensor.clone()
        return tensor.sub_(self.mean).div_(self.std)

    def __repr__(self):
        return f'{self.__class__.__name__} mean={self.mean}, std={self.std})'

normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


def train_transforms(image_size, train_img_scale=(0.35, 1)):
    """
    The standard imagenet transforms: random crop, resize to self.image_size, flip.
    Scale factor by default as at fast.ai example train script.
    """
    preprocessing = T.Compose([
        T.RandomResizedCrop(image_size, scale=train_img_scale),
        T.RandomHorizontalFlip(),
        AccimageImageToTensor(),
        normalize,
    ])

    return preprocessing


def train_transform_no_norm(image_size, train_img_scale=(0.35, 1)):
    """
    The standard imagenet transforms without normalize: random crop, resize to self.image_size, flip.
    Scale factor by default as at fast.ai example train script.
    """
    preprocessing = T.Compose([
        T.RandomResizedCrop(image_size, scale=train_img_scale),
        T.RandomHorizontalFlip(),
        AccimageImageToTensor()
    ])

    return preprocessing


def val_transforms(image_size, extra_size=32):
    """
    The standard imagenet transforms for validation: central crop, resize to self.image_size.
    """
    preprocessing = T.Compose([
        T.Resize(image_size + extra_size),
        T.CenterCrop(image_size),
        AccimageImageToTensor(),
        normalize,
    ])
    return preprocessing


def val_transforms_simple(image_size, extra_size=32):
    """
    Simplified version of val_transform - for presized images.
    Only to tensor and normalize.
    """
    preprocessing = T.Compose([
        AccimageImageToTensor(),
        normalize,
    ])
    return preprocessing
