import json
from pathlib import Path

import pytorch_lightning as pl
import torch
from doctr.datasets import VOCABS
from doctr.models import vitstr_base, vitstr_small
from timm.optim import create_optimizer_v2
from torchmetrics import CharErrorRate, MatchErrorRate


class RecModule(pl.LightningModule):
    def __init__(
        self,
        size: str = "small",
        lr: float = 1e-3,
        optimizer: str = "adamw",
        weight_decay: float = 0.01,
    ):
        super().__init__()
        self.save_hyperparameters()
        size = size.lower()
        if size == "small":
            self.model = vitstr_small(pretrained=True, vocab=VOCABS["korean"])
        elif size == "base":
            self.model = vitstr_base(pretrained=True, vocab=VOCABS["korean"])
        else:
            raise ValueError("size must be either 'small' or 'base'")

        self.cer = CharErrorRate()
        self.mer = MatchErrorRate()

    def training_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets)
        self.log("train_loss", output["loss"])
        return output["loss"]

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        output = self.model(images, targets, return_preds=True)
        if output["preds"]:
            preds, _ = zip(*output["preds"])
        else:
            preds = []

        self.cer(preds, targets)
        self.mer(preds, targets)

        self.log("val_loss", output["loss"])
        self.log("val_cer", self.cer)
        self.log("val_mer", self.mer)

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
