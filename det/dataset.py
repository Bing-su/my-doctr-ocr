from __future__ import annotations

import time
from pathlib import Path

import pytorch_lightning as pl
from loguru import logger
from munch import Munch
from doctr.datasets import DetectionDataset
from doctr import transforms as T
from torch.utils.data import ConcatDataset, DataLoader, RandomSampler, SequentialSampler
from torchvision.transforms import ColorJitter, Compose, Normalize


def val_sample_transform(
    input_size: int, rotation: bool = False, eval_straight: bool = False
):
    return T.SampleCompose(
        (
            [
                T.Resize(
                    (input_size, input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                )
            ]
            if not rotation or eval_straight
            else []
        )
        + (
            [
                T.Resize(input_size, preserve_aspect_ratio=True),  # This does not pad
                T.RandomRotate(90, expand=True),
                T.Resize(
                    (input_size, input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                ),
            ]
            if rotation and not eval_straight
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


def train_sample_transform(input_size: int, rotation: bool = False):
    return T.SampleCompose(
        (
            [
                T.Resize(
                    (input_size, input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                )
            ]
            if not rotation
            else []
        )
        + (
            [
                T.Resize(input_size, preserve_aspect_ratio=True),
                T.RandomRotate(90, expand=True),
                T.Resize(
                    (input_size, input_size),
                    preserve_aspect_ratio=True,
                    symmetric_pad=True,
                ),
            ]
            if rotation
            else []
        )
    )


yaml_path = Path(__file__).parent.parent / ".det_csv.yaml"
paths = Munch.fromYAML(yaml_path.read_text("utf-8"))


def train_dataset(input_size: int, rotation: bool = False):
    train_dss = [
        DetectionDataset(
            img_folder=Path(paths.train[i], "images").as_posix(),
            label_path=Path(paths.train[i], "labels.json").as_posix(),
            img_transforms=train_img_transform(),
            sample_transforms=train_sample_transform(input_size, rotation),
            use_polygons=rotation,
        )
        for i in range(len(paths.train))
    ]
    return ConcatDataset(train_dss)


def val_dataset(input_size: int, rotation: bool = False, eval_straight: bool = False):
    val_dss = [
        DetectionDataset(
            img_folder=Path(paths.validation[i], "images").as_posix(),
            label_path=Path(paths.validation[i], "labels.json").as_posix(),
            sample_transforms=val_sample_transform(input_size, rotation, eval_straight),
            use_polygons=rotation and not eval_straight,
        )
        for i in range(len(paths.validation))
    ]
    return ConcatDataset(val_dss)


class DetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_size: int = 1024,
        rotation: bool = True,
        eval_straight: bool = False,
        batch_size: int = 4,
        num_workers: int = 0,
    ):
        super().__init__()
        self.input_size = input_size
        self.rotation = rotation
        self.eval_straight = eval_straight
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        start = time.time()
        logger.info("Loading datasets...")
        self.train_ds = train_dataset(self.input_size, self.rotation)
        logger.info(f"Train dataset loaded in {time.time() - start:.2f}s")
        self.val_ds = val_dataset(self.input_size, self.rotation, self.eval_straight)
        logger.info(f"Validation dataset loaded in {time.time() - start:.2f}s")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.train_ds[0].collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.val_ds[0].collate_fn,
            pin_memory=True,
            persistent_workers=True,
        )
