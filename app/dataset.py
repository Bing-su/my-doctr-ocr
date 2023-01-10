from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pls
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

BatchOutput = tuple[torch.Tensor, str]
TransformType = Callable[[npt.NDArray[np.uint8]], torch.Tensor]


class RecDataset(Dataset):
    "https://github.com/mindee/doctr/blob/main/doctr/datasets/datasets/base.py"

    def __init__(self, csv: str, transform: TransformType):
        self.path = Path(csv).parent / "images"
        self.df = pls.read_csv(csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> BatchOutput:
        row: tuple[str, str] = self.df.row(idx)
        image_path, text = row
        pil_image = Image.open(self.path / image_path).convert("RGB")
        array = np.asarray(pil_image)
        return self.transform(array), text


def collate_fn(batch: list[BatchOutput]) -> tuple[torch.Tensor, list[str]]:
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


class RecDataModule(pl.LightningDataModule):
    def __init__(
        self, transform: TransformType, batch_size: int = 32, num_workers: int = 8
    ):
        super().__init__()
        self.train_csv = []
        self.val_csv = []
        self.transform = transform
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_datasets = [RecDataset(csv, self.transform) for csv in self.train_csv]
        val_datasets = [RecDataset(csv, self.transform) for csv in self.val_csv]
        self.train_dataset = ConcatDataset(train_datasets)
        self.val_dataset = ConcatDataset(val_datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            shuffle=True,  # True for sampling
            pin_memory=True,
            persistent_workers=True,
        )
