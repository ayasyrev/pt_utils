# adopted from torchvision examples
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from PIL.Image import Image
from torch import Tensor


def set_plt(
    # plt,
    figsize: tuple[int, int] = (20, 12),
    bbox: str = "tight",
    dark: bool = True,
):
    # global plt
    # plt = globals().get("plt")
    if dark:
        plt.style.use("dark_background")
    else:
        plt.style.use("classic")
    plt.rcParams["savefig.bbox"] = bbox
    plt.rcParams["figure.figsize"] = figsize


def plot(
    imgs: Image | Tensor | List[Image | Tensor] | List[List[Image | Tensor]],
    original: Image | Tensor | None = None,
    row_title: List[str] | None = None,
    titles=None,
    **imshow_kwargs,
):
    """Plot Images or Tensor images.
    Single item, list of items or list or list.
    """
    if isinstance(imgs, (Image, Tensor)):
        imgs = [[imgs]]

    if isinstance(imgs[0], (Image, Tensor)):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]
    with_orig = 1 if original is not None else 0
    num_rows = len(imgs)
    num_cols = len(imgs[0]) + with_orig
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        row = [original] + row if with_orig else row
        for col_idx, img in enumerate(row):
            if isinstance(img, Tensor):
                # img = img.detach()
                img = F.to_pil_image(img.detach())
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if col_idx < num_cols:  # clear axis for empty axs
            for ax_id in range(col_idx, num_cols):
                axs[row_idx, ax_id].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if with_orig:
        axs[0, 0].set(title="Original image")
        # axs[0, 0].title.set_size(8)
    if titles is not None:  # check for sizes, 2d case.
        # offset = 1 if with_orig else 0
        for num, val in enumerate(titles):
            # axs[0, num + offset].set(title=val)
            axs[0, num + with_orig].set(title=val)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()


def show_tensors(imgs: Tensor | List[Tensor]):
    """Plot tensor images from list or single one."""
    plt = globals().get("plt")

    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def make_2d_list(src_list: List, cols: int, num_samples: int | None = None) -> List:
    """Create 2d list from given list."""
    num_samples = num_samples or len(src_list)
    src_list = src_list[:num_samples]
    rows = num_samples // cols
    if num_samples % cols != 0:
        rows += 1
    res_list = []
    for row in range(rows):
        res_list.append(src_list[row * cols:row * cols + cols])
    return res_list
