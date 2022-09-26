from ..data.data_config import DataCfg
from .base_transforms import train_transforms, val_transforms


def get_transforms(cfg: DataCfg, train: bool):
    if train:
        return train_transforms(
            image_size=cfg.size,
            train_img_scale=(cfg.scale_min, cfg.scale_max),
        )
    else:
        return val_transforms(
            image_size=cfg.size,
            extra_size=16,
        )
