import os
from pathlib import Path, PosixPath
from typing import Union

from torchvision.datasets.utils import download_and_extract_archive

DATADIR = Path('/Data/')

imagenette_urls = {'imagenette2': 'https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz',
                   'imagewoof2': 'https://s3.amazonaws.com/fast-ai-imageclas/imagewoof2.tgz'}

imagenette_len = {'imagenette2': {'train': 9469, 'val': 3925},
                  'imagewoof2': {'train': 9025, 'val': 3929}
                  }

imagenette_md5 = {'imagenette2': '43b0d8047b7501984c47ae3c08110b62',
                  'imagewoof2': '5eaf5bbf4bf16a77c616dc6e8dd5f8e9'}


def check_data_exists(data_dir: Union[PosixPath, str], name: str) -> bool:
    ''' Verify data at data_dir / name and return True if len of images is Ok.
    '''
    ds_path = Path(data_dir) / name

    num_classes = 10
    if not ds_path.exists():
        return False

    for split in ['train', 'val']:
        split_path = Path(ds_path, split)
        if not split_path.exists():
            return False

        classes_dirs = [dir_entry for dir_entry in os.scandir(split_path)
                        if dir_entry.is_dir()]
        if num_classes != len(classes_dirs):
            return False

        num_samples = 0
        for dir_entry in classes_dirs:
            num_samples += len([fn for fn in os.scandir(dir_entry)
                                if fn.is_file()])

        if num_samples != imagenette_len[name][split]:
            return False

    return True


def prepare_data(data_dir: Union[PosixPath, str] = DATADIR, name: str = 'imagenette2') -> PosixPath:
    """ Download data if no data. data_dir: path for dir with datasets.
    """
    ds_path = Path(data_dir) / name

    if check_data_exists(data_dir, name):
        print(f"Data alredy exist, path: {ds_path}")
    else:
        # dataset_url = imagenette_urls[name]
        # download_and_extract_archive(url=dataset_url, download_root=ds_path, md5=imagenette_md5[name])
        # print(f"Downloaded, path: {data_dir}")
        print('not_exist')
    return ds_path


def prepare_woof(data_dir: PosixPath = DATADIR) -> None:
    '''Prepare imagewoof2 dataset'''
    return prepare_data(data_dir, 'imagewoof2')
