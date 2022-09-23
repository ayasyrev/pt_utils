# copied from torchvision.datasets.folder
import accimage
from pt_utils.data.loader_pil  import pil_loader


def accimage_loader(path: str) -> accimage.Image:

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)
