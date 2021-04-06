from pathlib import Path, PosixPath
from typing import List, Union


IMG_EXT = [".JPEG", '.JPG', '.jpeg', '.jpg']


def get_img_files(data_dir: Union[str, PosixPath], num_samples: int = 0) -> List[str]:
    """Return list of num_samples image filenames from data_dir.
    If num_samples == 0 return list of ALL images.

    Args:
        data_dir (str):
        num_samples (int, optional): Number of samples to return. Defaults to 0.
        If num_samples == 0 return list of ALL images.


    Returns:
        List[str]: List of filnames
    """
    image_filenames = [Path(fn) for fn in Path(data_dir).rglob("*.*") if fn.suffix in IMG_EXT]
    if num_samples != 0:
        image_filenames = image_filenames[:num_samples]
    return image_filenames
