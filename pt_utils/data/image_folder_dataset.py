from pt_utils.data.get_image_files import get_img_files
from typing import Callable, Tuple, Union
from pathlib import Path, PosixPath

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader


class ImageFolderDataset(Dataset):
    """Image Dataset from folders, classes by folders"""
    def __init__(self, root: Union[str, PosixPath],
                 transform: Union[Callable, None] = None,
                 loader: Callable = default_loader,
                 limit_dataset: Union[bool, int] = False):
        """Create dataset from folder structured data.

        Args:
            root (Union[str, PosixPath]): Data directory
            transform (Union[Callable, None], optional): Transform for samples. Defaults to None.
            loader (Callable, optional): Func for read images. Defaults to default_loader.
            limit_dataset (Union[bool, int], optional): If set dataset will be limited to this number.
                                                        Defaults to False.
        """
        self.root = Path(root)
        self.transform = transform
        self.loader = loader
        filenames = get_img_files(self.root, num_samples=limit_dataset)  # ? sorted
        self.classes = sorted(list(set(fn.parent.name for fn in filenames)))
        self.class_to_idx = {item: num for num, item in enumerate(self.classes)}
        self.idx_to_class = {num: item for num, item in enumerate(self.classes)}
        self.samples = [(str(filename), self.class_to_idx[filename.parent.name]) for filename in filenames]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


class ImageFolderDatasetMemory(Dataset):
    """Image Dataset from folders, classes by folders, loaded to memory"""
    def __init__(self, root: Union[str, PosixPath],
                 transform: Union[Callable, None] = None,
                 loader: Callable = default_loader,
                 limit_dataset: Union[bool, int] = False):
        """Create dataset from folder structured data.
        Samples wil be in memory.

        Args:
            root (Union[str, PosixPath]): Data directory
            transform (Union[Callable, None], optional): Transform for samples. Defaults to None.
            loader (Callable, optional): Func for read images. Defaults to default_loader.
            limit_dataset (Union[bool, int], optional): If set dataset will be limited to this number.
                                                        Defaults to False.
        """
        self.root = Path(root)
        self.transform = transform
        self.loader = loader
        filenames = get_img_files(self.root, num_samples=limit_dataset)  # ? sorted
        self.classes = sorted(list(set(fn.parent.name for fn in filenames)))
        self.class_to_idx = {item: num for num, item in enumerate(self.classes)}
        self.idx_to_class = {num: item for num, item in enumerate(self.classes)}
        self.samples = [(loader(str(filename)), self.class_to_idx[filename.parent.name]) for filename in filenames]
        self.filenames = filenames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple:
        sample, target = self.samples[index]
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target
