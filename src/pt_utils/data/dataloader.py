from typing import Optional
from torch.utils.data import DataLoader, Dataset

from .dataset import DataCfg, get_dataset


def get_dataloader(
    cfg: DataCfg,
    train: bool,
    dataset: Optional[Dataset] = None,
) -> DataLoader:
    if dataset is None:
        dataset = get_dataset(cfg, train)
    return DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        shuffle=cfg.shuffle if train else False,
        drop_last=cfg.drop_last if train else False,
        persistent_workers=cfg.persistent_workers,
    )


def get_dataloaders(cfg: DataCfg) -> tuple[DataLoader, DataLoader]:
    return get_dataloader(cfg, train=True), get_dataloader(cfg, train=False)
