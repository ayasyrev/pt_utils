from pathlib import Path, PosixPath
from typing import List, Union


IMG_EXT = [".JPEG", ".JPG", ".jpeg", ".jpg", ".PNG", ".png"]


def get_img_files(
    data_dir: Union[str, PosixPath, Path], num_samples: int = 0
) -> List[Path]:
    """Return list of num_samples image filenames from data_dir.
    If num_samples == 0 return list of ALL images.

    Args:
        data_dir (str | PosixPath | Path):
        num_samples (int, optional): Number of samples to return. Defaults to 0.
        If num_samples == 0 return list of ALL images.


    Returns:
        List[Path]: List of filnames
    """
    image_filenames = [
        Path(fn) for fn in Path(data_dir).rglob("*.*") if fn.suffix in IMG_EXT
    ]
    if num_samples != 0:
        image_filenames = image_filenames[:num_samples]
    return image_filenames
