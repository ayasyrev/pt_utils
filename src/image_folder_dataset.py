from pt_utils.get_image_files import get_img_files
from typing import Callable, Tuple, Union
from pathlib import Path, PosixPath

from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from torchvision import transforms as T


normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])


def train_transforms(image_size, train_img_scale=(0.35, 1)):
    """
    The standard imagenet transforms: random crop, resize to self.image_size, flip.
    Scale factor by default as at fast.ai example train script.
    """
    preprocessing = T.Compose([
        T.RandomResizedCrop(image_size, scale=train_img_scale),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])

    return preprocessing


def val_transforms(image_size, extra_size=32):
    """
    The standard imagenet transforms for validation: central crop, resize to self.image_size.
    """
    preprocessing = T.Compose([
        T.Resize(image_size + extra_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        normalize,
    ])
    return preprocessing


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
            limit_dataset (Union[bool, int], optional): If set< dataset will be limited to this number. Defaults to False.
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
            limit_dataset (Union[bool, int], optional): If set< dataset will be limited to this number. Defaults to False.
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
