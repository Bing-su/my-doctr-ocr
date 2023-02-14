from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from doctr.models import detection
from doctr.file_utils import CLASS_NAME
from doctr.utils.metrics import LocalizationConfusion
from timm.optim import create_optimizer_v2
from torchvision.transforms import Normalize


class DetModule(pl.LightningModule):
    def __init__(
        self,
        arch: str = "small",
        lr: float = 1e-3,
        optimizer: str = "adamw",
        weight_decay: float = 0.01,
        rotation: bool = True,
        eval_straight: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        arch = arch.lower()
        self.model = detection.__dict__[arch](
            pretrained=True,
            assume_straight_pages=not rotation,
            class_names=[CLASS_NAME],
        )

        self.batch_transform = Normalize(
            mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)
        )
        self.metric = LocalizationConfusion(
            use_polygons=rotation and not eval_straight, use_broadcasting=False
        )

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets)
        self.log("train_loss", output["loss"])
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets, return_preds=True)
        self.log("val_loss", output["loss"])

        loc_preds = output["preds"]
        for target, loc_pred in zip(targets, loc_preds):
            for boxes_gt, boxes_pred in zip(target.values(), loc_pred.values()):
                if self.hparams.rotation and self.hparams.eval_straight:
                    # Convert pred to boxes [xmin, ymin, xmax, ymax]  N, 4, 2 --> N, 4
                    boxes_pred = np.concatenate(
                        (boxes_pred.min(axis=1), boxes_pred.max(axis=1)), axis=-1
                    )
                self.metric.update(gts=boxes_gt, preds=boxes_pred[:, :4])

    def on_before_batch_transfer(self, batch, dataloader_idx):
        images, targets = batch
        images = self.batch_transform(images)
        return images, targets

    def on_validation_epoch_start(self):
        self.metric.reset()

    def on_validation_epoch_end(self):
        recall, precision, mean_iou = self.metric.summary()
        result = {
            "val_recall": recall,
            "val_precision": precision,
            "val_mean_iou": mean_iou,
        }
        self.log_dict(result)

    def configure_optimizers(self):
        kwargs = {}
        if self.hparams.optimizer in ("adam", "adamw"):
            kwargs["capturable"] = True

        optimizer = create_optimizer_v2(
            self.model,
            opt=self.hparams.optimizer,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            **kwargs,
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return [optimizer], [scheduler_config]

    def save(self, path: str | None = None):
        save_path = Path(path) if path else Path.cwd() / "model.pt"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        state_dict = self.model.state_dict()
        config = self.model.cfg
        torch.save(state_dict, save_path)
        if config:
            config_path = save_path.with_suffix(".json")
            with config_path.open("w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=False, indent=2)
