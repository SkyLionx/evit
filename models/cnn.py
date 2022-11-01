import torch
import pytorch_lightning as pl


class BasicCNN(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

        self.time_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, 3, padding="same"),
            torch.nn.ReLU(),
        )

        self.color_cnn = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 3, 3, padding="same"),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_cnn(x)
        x = self.color_cnn(x)
        return x

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = torch.einsum("bhwc -> bchw", y)
        out = self(x)
        loss = torch.nn.functional.mse_loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        x, y = batch
        y = torch.einsum("bhwc -> bchw", y)
        out = self(x)
        loss = torch.nn.functional.mse_loss(out, y)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    model = BasicCNN(lr=0.001)
    from torchinfo import summary

    summary(model, input_size=(8, 10, 128, 128))
