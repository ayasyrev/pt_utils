# copied from torchvision.datasets.folder
import accimage
from .img_loader_pil import loader_pil


def loader_accimage(path: str) -> accimage.Image:

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return loader_pil(path)
