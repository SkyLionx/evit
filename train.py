from my_utils import (
    LogImagesCallback,
    KerasProgressBar,
    ColabSaveCallback,
    is_using_colab,
)
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import json
import torch
from typing import Dict, Any


def train_model(
    model: pl.LightningModule,
    train_dataloader: torch.utils.data.DataLoader,
    valid_dataloader: torch.utils.data.DataLoader,
    images_train_batch: torch.Tensor,
    images_valid_batch: torch.Tensor,
    PARAMS: Dict[str, Any],
):
    """
    Train the model with the specified parameters.

    Args:
        model (pl.LightningModule): model which needs to be trained.
        train_dataloader (torch.utils.data.DataLoader): dataloader for training dataset.
        valid_dataloader (torch.utils.data.DataLoader): dataloader for validation dataset.
        images_train_batch (torch.Tensor): input batch which will be used to show how the model performs on training images.
        images_valid_batch (_type_): input batch which will be used to show how the model performs on validation images.
        PARAMS (Dict[str, Any]): training parameters. Recognized keys are "n_epochs", "comment" and "ckpt_path".
    """
    callbacks = []
    # Log learning rate on TensorBoard
    callbacks.append(LearningRateMonitor())
    # Log train and valid images on TensorBoard
    callbacks.append(
        LogImagesCallback(images_train_batch, images_valid_batch, n=5, n_epochs=5)
    )
    # Use textual progress bar
    callbacks.append(KerasProgressBar())

    # Save experiment every hour so if Colab dies, there is a backup
    if is_using_colab():
        dst_path = "/content/drive/MyDrive/Master Thesis"
        colab_cb = ColabSaveCallback(
            "Exp.zip", dst_path, 60 * 60, ["zip -r Exp.zip Experiments"]
        )
        callbacks.append(colab_cb)

    # Save best and last checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor="val_LPIPS", mode="min", save_last=True
    )
    checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"
    callbacks.append(checkpoint_callback)

    logger = pl.loggers.TensorBoardLogger(
        PARAMS["EXPERIMENTS_DIR"], version=PARAMS["TRAINING_PARAMS"]["comment"]
    )
    log_every = 50
    if len(train_dataloader) < log_every:
        log_every = 1

    # Save params as text on TensorBoard
    params_markdown = (
        "```json\n" + json.dumps(PARAMS, indent=2).replace("\n", "  \n") + "\n```"
    )
    logger.experiment.add_text("params", params_markdown)

    trainer = pl.Trainer(
        max_epochs=PARAMS["TRAINING_PARAMS"]["n_epochs"],
        callbacks=callbacks,
        accelerator=PARAMS["DEVICE"],
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
