from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pls
import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from .util import train_transform, val_transform

BatchOutput = tuple[torch.Tensor, str]
TransformType = Callable[[npt.NDArray[np.uint8]], torch.Tensor]


class RecDataset(Dataset):
    "https://github.com/mindee/doctr/blob/main/doctr/datasets/datasets/base.py"

    def __init__(self, csv: str | Path, transform: TransformType):
        self.path = Path(csv).parent / "images"
        self.df = pls.read_csv(csv)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> BatchOutput:
        row: tuple[str, str] = self.df.row(idx)
        img_name, text = row
        img_path = self.path / img_name
        if not text or not img_path.exists():
            j = np.random.randint(0, len(self))
            return self[j]
        pil_image = Image.open(img_path).convert("RGB")
        array = np.array(pil_image)
        return self.transform(array), text


def collate_fn(batch: list[BatchOutput]) -> tuple[torch.Tensor, list[str]]:
    images, texts = zip(*batch)
    return torch.stack(images), list(texts)


class RecDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: Iterable[str | Path],
        val_csv: Iterable[str | Path],
        batch_size: int = 32,
        num_workers: int = 8,
    ):
        super().__init__()
        self.train_csv = list(train_csv)
        self.val_csv = list(val_csv)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        train_datasets = [RecDataset(csv, train_transform) for csv in self.train_csv]
        val_datasets = [RecDataset(csv, val_transform) for csv in self.val_csv]
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
            num_workers=max(6, self.num_workers),
            collate_fn=collate_fn,
            shuffle=True,  # True for sampling
            pin_memory=True,
            persistent_workers=True,
        )
