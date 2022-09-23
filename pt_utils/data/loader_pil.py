# copied from torchvision.datasets.folder
from PIL import Image


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as fh:
        img = Image.open(fh)
        return img.convert("RGB")
