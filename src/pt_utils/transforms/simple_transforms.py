import warnings
from typing import Sequence

import torch
from torchvision.transforms import functional as F


# forked from torchvision - simplified version of Resize + CenterCrop
class ResizeCrop(torch.nn.Module):
    """Resize and Crop for validation transforms.
    """

    def __init__(
        self,
        size: int,
        extra_size: int,
        interpolation=F.InterpolationMode.BILINEAR,
        max_size=None,
        antialias=None,
        ):
        super().__init__()
        # _log_api_usage_once(self)
        if not isinstance(size, (int, Sequence)):
            raise TypeError(f"Size should be int or sequence. Got {type(size)}")
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size
        self.extra_size = extra_size
        self.resize = (size + extra_size, size + extra_size)
        self.max_size = max_size
        self.crop_top = int(round(extra_size / 2.))

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = F._interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        return F.crop(
            F.resize(img, self.resize, self.interpolation, self.max_size, self.antialias),
            self.crop_top, self.crop_top, self.size, self.size,
        )

    def __repr__(self) -> str:
        detail = f"(size={self.size}, resize={self.resize}, extra={self.extra_size}, antialias={self.antialias})"
        return f"{self.__class__.__name__}{detail}"
