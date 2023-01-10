import albumentations as A
from doctr import transforms as DT
from torchvision import transforms as T

MEAN = (0.694, 0.695, 0.693)
STD = (0.299, 0.296, 0.301)


train_transform_list = [
    A.ISONoise(),
    A.JpegCompression(75),
    A.RandomBrightnessContrast(),
    A.Affine(
        scale=(0.5, 1.2),
        translate_percent=(-0.1, 0.1),
        rotate=(-10, 10),
        shear=(-10, 10),
        p=0.8,
    ),
    A.InvertImg(p=0.1),
    A.ColorJitter(),
    T.ToTensor(),
    T.AugMix(),
    DT.Resize((32, 32 * 4), preserve_aspect_ratio=True),
    T.Normalize(MEAN, STD),
]

val_transform_list = [
    T.ToTensor(),
    DT.Resize((32, 32 * 4), preserve_aspect_ratio=True),
    T.Normalize(MEAN, STD),
]

train_transform = T.Compose(train_transform_list)
val_transform = T.Compose(val_transform_list)
