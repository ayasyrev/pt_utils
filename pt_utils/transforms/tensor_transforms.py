import torch
import torch.nn as nn


class Normalize(nn.Module):
    '''Normalize nn.Module. As at pytorch, simplified'''

    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 inplace=False,
                 dtype=torch.float32,
                 ):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=dtype)
        self.mean = nn.Parameter(mean[:, None, None], requires_grad=False)
        std = torch.as_tensor(std, dtype=dtype)
        self.std = nn.Parameter(std[:, None, None], requires_grad=False)
        self.inplace = inplace

    def forward(self, tensor: torch.Tensor):
        if not self.inplace:
            tensor = tensor.clone()
        return tensor.sub_(self.mean).div_(self.std)

    def __repr__(self):
        return f'{self.__class__.__name__} mean={self.mean}, std={self.std})'
