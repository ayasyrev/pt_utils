import torch
from torchvision.utils import _log_api_usage_once


class Normalize(torch.nn.Module):
    """Normalize simplified from torchvision
    """

    def __init__(
        self,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        inplace=False,
        dtype=torch.float32,
        ):
        super().__init__()
        _log_api_usage_once(self)
        mean = torch.as_tensor(mean, dtype=dtype)
        std = torch.as_tensor(std, dtype=dtype)
        if mean.ndim == 1:
            mean = mean.view(-1, 1, 1)
        if std.ndim == 1:
            std = std.view(-1, 1, 1)
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not self.inplace:
            tensor = tensor.clone()
        return tensor.sub_(self.mean).div_(self.std)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
