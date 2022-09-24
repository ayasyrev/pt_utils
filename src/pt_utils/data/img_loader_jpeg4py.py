import jpeg4py
from PIL import Image


def loader_jpeg4py(img_path: str) -> Image.Image:
    return Image.fromarray(jpeg4py.JPEG(img_path).decode())
