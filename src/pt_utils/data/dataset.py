from pathlib import Path
from typing import Callable, List, Union

from pydantic import BaseModel, validator
from torchvision import transforms as T

from .image_folder_dataset import ImageFolderDataset
from .img_loaders import img_libs_available, img_libs_supported, loaders


class DataCfg(BaseModel):
    train_data_path: Union[str, Path]
    val_data_path: Union[str, Path]
    train_tfms: List[Callable]
    val_tfms: List[Callable]
    batch_size: int = 64
    num_workers: int = 4
    dataset_func = ImageFolderDataset
    image_backend: str = "PIL"  # 'accimage'
    limit_dataset: Union[bool, int] = False
    pin_memory: bool = True
    shuffle: bool = True
    shuffle_val: bool = False  # need it?
    drop_last: bool = True
    drop_last_val: bool = False   # need it?
    persistent_workers: bool = False

    @validator("image_backend")
    def check_supported(cls, v):
        if v not in img_libs_supported:
            raise ValueError(f"{v} is not supported")
        elif v not in img_libs_available:
            raise ValueError(f"{v} is supported but not available")
        return v


def get_dataset(cfg: DataCfg, train: bool):
    return cfg.dataset_func(
        root=cfg.train_data_path if train else cfg.val_data_path,
        transform=T.Compose(cfg.train_tfms if train else cfg.val_tfms),
        loader=loaders[cfg.image_backend],
        limit_dataset=cfg.limit_dataset,
    )


def get_datasets(cfg: DataCfg):
    return get_dataset(cfg, train=True), get_dataset(cfg, train=False)
