from datetime import datetime
from pathlib import Path

import pytorch_lightning as pl
from doctr.models import push_to_hf_hub
from loguru import logger
from munch import Munch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger
from rich.traceback import install

from rec.dataset import RecDataModule
from rec.module import RecModule

install(show_locals=True, suppress=["torch"])


def main():
    cfg: Munch = Munch.fromYAML(Path("config.yaml").read_text("utf-8"))
    data: Munch = Munch.fromYAML(Path(".csv.yaml").read_text("utf-8"))

    logger.info("Start training")

    module = RecModule(
        size=cfg.model.size,
        optimizer=cfg.model.optimizer,
        lr=cfg.model.lr,
        weight_decay=cfg.model.weight_decay,
    )

    datamodule = RecDataModule(
        train_csv=data.train,
        val_csv=data.validation,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    checkpoints = ModelCheckpoint(
        dirpath="checkpoints", save_top_k=3, monitor="val_loss", save_last=True
    )

    callbacks = [
        checkpoints,
        RichProgressBar(),
        LearningRateMonitor(),
    ]

    now = datetime.now().strftime("%y%m%d_%H%M%S")
    wandb_name = f"vitstr_{cfg.model.size}_{now}"

    trainer = pl.Trainer(
        accelerator="auto",
        precision=cfg.trainer.precision,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_epochs=cfg.trainer.max_epochs,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        callbacks=callbacks,
        logger=WandbLogger(project="doctr", name=wandb_name),
        limit_val_batches=cfg.trainer.limit_val_batches,
        fast_dev_run=cfg.trainer.fast_dev_run,
    )

    logger.info("Start fit")
    trainer.fit(module, datamodule=datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path
    logger.info(f"Best model path: {best_model_path}")
    if best_model_path and Path(best_model_path).is_file():
        logger.info(f"Load best model: {best_model_path}")
        best_model = RecModule.load_from_checkpoint(best_model_path)
        best_model.save("save/best_model.pt")
        model = best_model.model

        logger.info(f"Push to HuggingFace Hub: {wandb_name}")
        push_to_hf_hub(
            model,
            model_name=wandb_name,
            task="recognition",
            arch=f"vitstr_{cfg.model.size}",
        )


if __name__ == "__main__":
    main()
