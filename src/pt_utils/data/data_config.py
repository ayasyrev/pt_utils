from pathlib import Path
from typing import Callable, List, Optional, Union

from pydantic import BaseModel, validator

from .image_folder_dataset import ImageFolderDataset
from .img_loaders import img_libs_available, img_libs_supported


class DataCfg(BaseModel):
    train_data_path: Union[str, Path, None] = None
    val_data_path: Union[str, Path, None] = None
    train_tfms: Optional[List[Callable]] = None
    val_tfms: Optional[List[Callable]] = None
    size: int = 224
    extra_size: int = 32
    batch_size: int = 64
    num_workers: int = 4
    scale_min: float = 0.8
    scale_max: float = 1.
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
