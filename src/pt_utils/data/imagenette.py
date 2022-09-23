import os
from pathlib import Path, PosixPath
from typing import Union

from torchvision.datasets.utils import download_and_extract_archive

DATADIR = Path("/Data/")

imagenette_urls = {
    "imagenette2": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
    "imagenette2-320": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz",
    "imagewoof2": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz",
    "imagewoof2-320": "https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2-320.tgz",
}

imagenette_len = {
    "imagenette2": {"train": 9469, "val": 3925},
    "imagenette2-320": {"train": 9469, "val": 3925},
    "imagewoof2": {"train": 9025, "val": 3929},
    "imagewoof2-320": {"train": 9025, "val": 3929},
}

imagenette_md5 = {
    "imagenette2": "fe2fc210e6bb7c5664d602c3cd71e612",
    "imagenette2-320": "3df6f0d01a2c9592104656642f5e78a3",
    "imagewoof2": "9aafe18bcdb1632c4249a76c458465ba",
    "imagewoof2-320": "0f46d997ec2264e97609196c95897a44",
}


def check_data_exists(data_dir: Union[PosixPath, str, Path], name: str) -> bool:
    """Verify data at data_dir / name and return True if len of images is Ok."""
    ds_path = Path(data_dir) / name

    num_classes = 10
    if not ds_path.exists():
        return False

    for split in ["train", "val"]:
        split_path = Path(ds_path, split)
        if not split_path.exists():
            return False

        classes_dirs = [
            dir_entry for dir_entry in os.scandir(split_path) if dir_entry.is_dir()
        ]
        if num_classes != len(classes_dirs):
            return False

        num_samples = 0
        for dir_entry in classes_dirs:
            num_samples += len([fn for fn in os.scandir(dir_entry) if fn.is_file()])

        if num_samples != imagenette_len[name][split]:
            return False

    return True


def prepare_data(
    data_dir: Union[PosixPath, str, Path] = DATADIR, name: str = "imagenette2"
) -> Union[PosixPath, Path]:
    """Download data if no data. data_dir: path for dir with datasets."""
    ds_path = Path(data_dir) / name

    if check_data_exists(data_dir, name):
        print(f"Data already exist, path: {ds_path}")
    else:
        dataset_url = imagenette_urls[name]
        download_and_extract_archive(
            url=dataset_url, download_root=str(data_dir), md5=imagenette_md5[name]
        )
        print(f"Downloaded, path: {data_dir}")
    return ds_path


def prepare_woof(data_dir: Union[PosixPath, Path] = DATADIR) -> Union[PosixPath, Path]:
    """Prepare imagewoof2 dataset"""
    return prepare_data(data_dir, "imagewoof2")
