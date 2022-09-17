import datetime
import time

from IPython import get_ipython
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt
from pytorch_lightning.callbacks.progress.base import ProgressBarBase


def is_using_colab() -> bool:
    return "google.colab" in str(get_ipython())


def format_current_date() -> str:
    today = datetime.datetime.today()
    return today.strftime("%Y-%m-%d %H-%M-%S")


class LogImagesCallback(pl.Callback):
    def __init__(
        self,
        train_batch: torch.tensor,
        valid_batch: torch.tensor,
        n: int = 5,
        n_epochs: int = 5,
    ):
        """
        Uses the TensorBoard logger to save `n` images after every `n_epochs`
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

        train_out = pl_module(train_X[: self.n].to(device=pl_module.device))
        valid_out = pl_module(valid_X[: self.n].to(device=pl_module.device))

        train_figs = [self.create_plot(train_out[i], train_y[i]) for i in range(self.n)]
        valid_figs = [self.create_plot(valid_out[i], valid_y[i]) for i in range(self.n)]

        trainer.logger.experiment.add_figure(f"train", train_figs, epoch)
        trainer.logger.experiment.add_figure(f"valid", valid_figs, epoch)

        for fig in train_figs + valid_figs:
            plt.close(fig)

    def create_plot(self, out, gt):
        out = torch.permute(out, (1, 2, 0)).detach().cpu().numpy()
        gt = gt.detach().cpu().numpy()

        fig = plt.figure()
        plt.subplot(1, 2, 1)
        plt.title("Model Output")
        plt.axis("off")
        plt.imshow(out)
        plt.subplot(1, 2, 2)
        plt.title("Ground Truth")
        plt.axis("off")
        plt.tight_layout()
        plt.imshow(gt)

        return fig


class KerasProgressBar(ProgressBarBase):
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
