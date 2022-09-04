import datetime

from IPython import get_ipython
import pytorch_lightning as pl
import torch
import matplotlib.pyplot as plt

def is_using_colab() -> bool:
    return "google.colab" in str(get_ipython())

def format_current_date() -> str:
    today = datetime.datetime.today()
    return today.strftime("%Y-%m-%d %H-%M-%S")

class LogImagesCallback(pl.Callback):
    def __init__(self, train_batch: torch.tensor, valid_batch: torch.tensor, n: int = 5, n_epochs: int = 5):
        """
        Uses the TensorBoard logger to save `n` images after every `n_epochs`
        """
        super().__init__()
        self.train_batch = train_batch
        self.valid_batch = valid_batch
        self.n = n
        self.n_epochs = n_epochs
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if epoch % self.n_epochs != 0:
            return
        
        train_X, train_y = self.train_batch
        valid_X, valid_y = self.valid_batch

        train_out = pl_module(train_X[:self.n].to(device=pl_module.device))
        valid_out = pl_module(valid_X[:self.n].to(device=pl_module.device))

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