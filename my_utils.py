import datetime
import time
import os
from typing import Sequence
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.progress.base import ProgressBarBase

from IPython import get_ipython


def is_using_colab() -> bool:
    """Returns whether if running on the Colab platform."""
    return "google.colab" in str(get_ipython())


def format_current_date() -> str:
    """
    Return the current date using the format YY-MM-DD hh-mm-ss
    which is Windows-friendly for folder names.
    """
    today = datetime.datetime.today()
    return today.strftime("%Y-%m-%d %H-%M-%S")


class ColabSaveCallback(pl.Callback):
    """
    Callback that saves files periodically to avoid loosing progress on Colab.
    """

    def __init__(
        self,
        src_files: str,
        dst_folder: str,
        every_seconds: int,
        pre_commands: Sequence[str] = None,
    ):
        """
        Copy some files to another folder periodically.

        Args:
            src_files (str): file(s) to be copied.
            The string can be any format accepted by the Linux cp command.
            dst_folder (str): destination folder.
            every_seconds (int): copy the files every these seconds.
            pre_commands (Sequence[str], optional): Sequence of commands to execute before copying files.
            They can be any Linux commands. Defaults to None.
        """
        super().__init__()
        self.src_files = src_files
        self.dst_folder = dst_folder
        self.every_seconds = every_seconds
        self.pre_commands = pre_commands

        self.last_time = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        # Initialize first time
        if self.last_time is None:
            self.last_time = time.time()

    def _execute_commands(self):
        if self.pre_commands:
            for command in self.pre_commands:
                exit_code = os.system(command)
                if exit_code != 0:
                    print(
                        "Warning, there was an error executing the command:",
                        command,
                    )

        command = f'cp "{self.src_files}" "{self.dst_folder}"'
        exit_code = os.system(command)
        if exit_code != 0:
            print("Warning, there was an error executing the command:", command)

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        now = time.time()
        elapsed_seconds = now - self.last_time
        if elapsed_seconds >= self.every_seconds:
            self.last_time = now
            self._execute_commands()

    def on_train_end(self, trainer, pl_module) -> None:
        self._execute_commands()


def compare_gt_plot(
    predict: torch.Tensor,
    gt: torch.Tensor,
    permute_predict=True,
    permute_gt=False,
) -> plt.Figure:
    """
    Create a plot to compare the predicted image with the ground truth.

    Args:
        predict (torch.Tensor): predicted image.
        gt (torch.Tensor): ground truh.
        permute_predict (bool, optional): permute prediction from CHW to HWC. Defaults to True.
        permute_gt (bool, optional): permute ground truth from CHW to HWC. Defaults to False.

    Returns:
        plt.Figure: matplotlib generated figure.
    """
    if permute_predict:
        predict = predict.permute(1, 2, 0)
    if permute_gt:
        gt = gt.permute(1, 2, 0)

    predict = predict.detach().cpu().numpy()
    gt = gt.detach().cpu().numpy()

    fig = plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("Model Output")
    plt.axis("off")
    plt.imshow(predict.squeeze(), cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.tight_layout()
    plt.imshow(gt.squeeze(), cmap="gray")

    return fig


class LogImagesCallback(pl.Callback):
    """
    Callback to be used with the TensorBoard logger in order to save training and
    validations images outputs during training.
    """

    def __init__(
        self,
        train_batch: torch.tensor,
        valid_batch: torch.tensor,
        n: int = 5,
        n_epochs: int = 5,
    ):
        """
        Log `n` images from the `train_batch` and `valid_batch` after every `n_epochs`.
        """
        super().__init__()
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.n = n
        self.n_epochs = n_epochs

    def on_train_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        if epoch % self.n_epochs != 0:
            return

        train_X, train_y = self.train_batch
        valid_X, valid_y = self.valid_batch

        train_X = train_X[: self.n].to(device=pl_module.device)
        train_y = train_y[: self.n].to(device=pl_module.device)
        valid_X = valid_X[: self.n].to(device=pl_module.device)
        valid_y = valid_y[: self.n].to(device=pl_module.device)

        if hasattr(pl_module, "predict_images"):
            train_out = pl_module.predict_images((train_X, train_y))
            valid_out = pl_module.predict_images((valid_X, valid_y))
        else:
            train_out = pl_module((train_X, train_y))
            valid_out = pl_module((valid_X, valid_y))

        train_figs = [compare_gt_plot(train_out[i], train_y[i]) for i in range(self.n)]
        valid_figs = [compare_gt_plot(valid_out[i], valid_y[i]) for i in range(self.n)]

        trainer.logger.experiment.add_figure(f"train", train_figs, epoch)
        trainer.logger.experiment.add_figure(f"valid", valid_figs, epoch)

        for fig in train_figs + valid_figs:
            fig.clear()


class KerasProgressBar(ProgressBarBase):
    """
    Textual progress bar that emulates the Keras progress bar
    """

    def __init__(self, hide_v_num=True):
        super().__init__()

        self.hide_v_num = hide_v_num

        self.enable = True

    def _get_bar(self, value, total, len=20):
        progress = value * len // total
        remaining = len - progress
        progress_str = "=" * progress
        remaining_str = " " * remaining
        return "[" + progress_str + remaining_str + "]"

    def disable(self):
        self.enable = False

    def get_metrics(self, trainer, pl_model):
        items = super().get_metrics(trainer, pl_model)

        if self.hide_v_num:
            items.pop("v_num", None)

        return items

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)

        self._start = time.time()

        print(f"Epoch {trainer.current_epoch}/{trainer.max_epochs}")

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)

        if self.train_batch_idx == 0:
            self._time_after_first_step = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)

        now = time.time()
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        progress = self._get_bar(self.train_batch_idx, self.total_train_batches)

        time_per_unit = (now - self._time_after_first_step) / self.train_batch_idx
        eta = time_per_unit * (self.total_train_batches - self.train_batch_idx)

        if eta > 3600:
            eta_format = "%d:%02d:%02d" % (eta // 3600, (eta % 3600) // 60, eta % 60)
        elif eta > 60:
            eta_format = "%d:%02d" % (eta // 60, eta % 60)
        else:
            eta_format = "%ds" % eta

        metrics = " - ".join(
            f"{key}: {float(value):.4f}"
            for key, value in self.get_metrics(trainer, pl_module).items()
        )

        print(
            f"\r{self.train_batch_idx}/{self.total_train_batches} {progress} - ETA: {eta_format} - {metrics}",
            end="",
        )

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        now = time.time()
        percent = (self.train_batch_idx / self.total_train_batches) * 100
        progress = self._get_bar(self.train_batch_idx, self.total_train_batches)
        elapsed_time = now - self._start
        time_per_unit = (now - self._time_after_first_step) / self.train_batch_idx

        if time_per_unit >= 1 or time_per_unit == 0:
            formatted = " %.0fs/step" % time_per_unit
        elif time_per_unit >= 1e-3:
            formatted = " %.0fms/step" % (time_per_unit * 1e3)
        else:
            formatted = " %.0fus/step" % (time_per_unit * 1e6)

        metrics = " - ".join(
            f"{key}: {float(value):.4f}"
            for key, value in self.get_metrics(trainer, pl_module).items()
        )

        print(
            f"\r{self.train_batch_idx}/{self.total_train_batches} {progress} - {elapsed_time:.0f}s {formatted} - {metrics}"
        )
