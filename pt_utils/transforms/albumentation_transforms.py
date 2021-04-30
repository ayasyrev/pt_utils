from typing import Callable, List, Union
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbumentationTransforms:
    """Base class for transforms"""
    def __init__(self, transforms: List[Callable], normalize: bool = True, to_tensor: bool = True) -> None:
        if normalize:
            transforms.append(A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]))
        if to_tensor:
            transforms.append(ToTensorV2())
        self.transforms = A.Compose(transforms)

    def __call__(self, image):
        return self.transforms(image=image)['image']


class AlbumentationTransformsTrain(AlbumentationTransforms):
    """Transforms for train augmentation"""

    def __init__(self, size: int, scale=(0.35, 1), transforms: Union[List[Callable], None] = None,
                 normalize: bool = True, to_tensor: bool = True) -> None:
        if transforms is None:
            transforms = [A.RandomResizedCrop(size, size, scale=scale),
                          A.HorizontalFlip()]
        super().__init__(transforms, normalize, to_tensor)


class AlbumentationTransformsVal(AlbumentationTransforms):
    """Transforms for train augmentation"""

    def __init__(self, size: int, extra_size: int = 32, transforms: Union[List[Callable], None] = None,
                 normalize: bool = True, to_tensor: bool = True) -> None:
        if transforms is None:
            transforms = [A.Resize(size, size),
                          A.CenterCrop(size, size)]
        super().__init__(transforms, normalize, to_tensor)
