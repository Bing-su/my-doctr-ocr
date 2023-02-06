import argparse
from pathlib import Path

from doctr import transforms as T
from doctr.datasets import DetectionDataset
from munch import Munch
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, Compose, Normalize


def val_sample_transform(args: argparse.Namespace):
    return T.SampleCompose(
        (
            [
                T.Resize(
                    (args.input_size, args.input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                )
            ]
            if not args.rotation or args.eval_straight
            else []
        )
        + (
            [
                T.Resize(
                    args.input_size, preserve_aspect_ratio=True
                ),  # This does not pad
                T.RandomRotate(90, expand=True),
                T.Resize(
                    (args.input_size, args.input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                ),
            ]
            if args.rotation and not args.eval_straight
            else []
        )
    )


def train_img_transform():
    return Compose(
            [
                # Augmentations
                T.RandomApply(T.ColorInversion(), 0.1),
                ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.02),
            ]
        )


def train_sample_transform(args: argparse.Namespace):
    return T.SampleCompose(
        (
            [
                T.Resize(
                    (args.input_size, args.input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                )
            ]
            if not args.rotation
            else []
        )
        + (
            [
                T.Resize(args.input_size, preserve_aspect_ratio=True),
                T.RandomRotate(90, expand=True),
                T.Resize(
                    (args.input_size, args.input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                ),
            ]
            if args.rotation
            else []
        )
    )


yaml_path = Path(__file__).parent.parent / ".det_csv.yaml"
paths = Munch.fromYAML(yaml_path.read_text("utf-8"))


def train_dataset(args: argparse.Namespace):
    train_dss = [
        DetectionDataset(
            img_folder=Path(paths.train[i], "images").as_posix(),
            label_path=Path(paths.train[i], "labels.json").as_posix(),
            img_transforms=train_img_transform(),
            sample_transforms=train_sample_transform(args),
            use_polygons=args.rotation,
        )
        for i in range(len(paths.train))
    ]
    return ConcatDataset(train_dss)


def val_dataset(args: argparse.Namespace):
    val_dss = [
        DetectionDataset(
            img_folder=Path(paths.validation[i], "images").as_posix(),
            label_path=Path(paths.validation[i], "labels.json").as_posix(),
            sample_transforms=val_sample_transform(args),
            use_polygons=args.rotation and not args.eval_straight,
        )
        for i in range(len(paths.validation))
    ]
    return ConcatDataset(val_dss)
