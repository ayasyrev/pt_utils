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
