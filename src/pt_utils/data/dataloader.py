from pathlib import PosixPath
from typing import Callable, List, Union

from pt_utils.data.image_folder_dataset import ImageFolderDataset
from pydantic import BaseConfig
from torch.utils.data import DataLoader
from torchvision import set_image_backend
from torchvision import transforms as T
from torchvision.datasets.folder import default_loader


class DlsCfg(BaseConfig):
    train_data_path: Union[str, PosixPath]
    val_data_path: Union[str, PosixPath]
    train_tfms: List[Callable]
    val_tfms: List[Callable]
    batch_size: int = 64
    num_workers: int = 4
    dataset_func: Callable = ImageFolderDataset
    loader: Callable = default_loader
    image_backend: str = "PIL"  # 'accimage'
    limit_dataset: Union[bool, int] = False
    pin_memory: bool = True
    shuffle: bool = True
    shuffle_val: bool = False
    drop_last: bool = True
    drop_last_val: bool = False
    persistent_workers: bool = False


def get_dataloaders(
    cfg=DlsCfg
):
    set_image_backend(cfg.image_backend)
    train_tfms = T.Compose(cfg.train_tfms)
    val_tfms = T.Compose(cfg.val_tfms)
    train_ds = cfg.dataset_func(
        root=cfg.train_data_path,
        transform=train_tfms,
        loader=cfg.loader,
        limit_dataset=cfg.limit_dataset,
    )
    val_ds = cfg.dataset_func(
        root=cfg.val_data_path,
        transform=val_tfms,
        loader=cfg.loader,
        limit_dataset=cfg.limit_dataset,
    )

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle,
        drop_last=cfg.drop_last,
        persistent_workers=cfg.persistent_workers,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle_val,
        drop_last=cfg.drop_last_val,
        persistent_workers=cfg.persistent_workers,
    )
    return train_loader, val_loader
