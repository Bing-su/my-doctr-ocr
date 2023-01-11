import albumentations as A
import cv2
from doctr import transforms as DT
from torchvision import transforms as T

MEAN = (0.694, 0.695, 0.693)
STD = (0.299, 0.296, 0.301)


class MyCompose(T.Compose):
    def __call__(self, img):
        for t in self.transforms:
            if isinstance(t, A.BasicTransform):
                img = t(image=img)["image"]
            else:
                img = t(img)
        return img


train_transform_list = [
    A.ISONoise(),
    A.ImageCompression(75),
    A.Affine(
        scale=(0.5, 1.2),
        translate_percent=(-0.07, 0.07),
        rotate=(-7, 7),
        shear=(-7, 7),
        interpolation=cv2.INTER_CUBIC,
        p=0.5,
    ),
    A.InvertImg(p=0.1),
    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
    A.HueSaturationValue(180, 180, 180),
    T.ToPILImage(),
    T.AugMix(),
    T.ToTensor(),
    DT.Resize((32, 32 * 4), preserve_aspect_ratio=True),
    T.Normalize(MEAN, STD),
]

val_transform_list = [
    T.ToTensor(),
    DT.Resize((32, 32 * 4), preserve_aspect_ratio=True),
    T.Normalize(MEAN, STD),
]

train_transform = MyCompose(train_transform_list)
val_transform = MyCompose(val_transform_list)
