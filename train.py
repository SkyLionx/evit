from utils import LogImagesCallback, KerasProgressBar, ColabSaveCallback, is_using_colab
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import json
import torch
from typing import Dict, Any


def train_model(
    model: pl.LightningModule,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    train_batch,
    valid_batch,
    PARAMS: Dict[str, Any],
):
    callbacks = []
    callbacks.append(LogImagesCallback(train_batch, valid_batch, n=5, n_epochs=5))
    callbacks.append(KerasProgressBar())
    if is_using_colab():
        dst_path = "/content/drive/MyDrive/Master Thesis"
        colab_cb = ColabSaveCallback(
            "Exp.zip", dst_path, 60 * 60, ["zip -r Exp.zip Experiments"]
        )
        callbacks.append(colab_cb)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    callbacks.append(checkpoint_callback)

    logger = pl.loggers.TensorBoardLogger(
        PARAMS["EXPERIMENTS_DIR"], version=PARAMS["TRAINING_PARAMS"]["comment"]
    )
    log_every = 50
    if len(train_dataloader) < log_every:
        log_every = 1

    params_markdown = (
        "```json\n" + json.dumps(PARAMS, indent=2).replace("\n", "  \n") + "\n```"
    )
    logger.experiment.add_text("params", params_markdown)

    trainer = pl.Trainer(
        max_epochs=PARAMS["TRAINING_PARAMS"]["n_epochs"],
        callbacks=callbacks,
        accelerator="gpu",
        profiler=None,
        logger=logger,
        log_every_n_steps=log_every,
    )
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
        ckpt_path=PARAMS["TRAINING_PARAMS"].get("ckpt_path", None),
    )
