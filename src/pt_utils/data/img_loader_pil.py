# copied from torchvision.datasets.folder
from PIL import Image


def loader_pil(path: str) -> Image.Image:
    with open(path, "rb") as fh:
        img = Image.open(fh)
        return img.convert("RGB")
