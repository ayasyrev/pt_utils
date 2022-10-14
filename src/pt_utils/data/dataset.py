from torchvision import transforms as T

from .data_config import DataCfg
from .img_loaders import loaders
from ..transforms.get_transforms import get_transforms


def get_dataset(cfg: DataCfg, train: bool):
    tfms = getattr(cfg, "train_tfms" if train else "val_tfms")
    if tfms is None:
        tfms = get_transforms(cfg, train)
    else:
        tfms = T.Compose(tfms)
    return cfg.dataset_func(
        root=cfg.train_data_path if train else cfg.val_data_path,
        transform=tfms,
        loader=loaders[cfg.image_backend],
        limit_dataset=cfg.limit_dataset,
    )


def get_datasets(cfg: DataCfg):
    return get_dataset(cfg, train=True), get_dataset(cfg, train=False)
