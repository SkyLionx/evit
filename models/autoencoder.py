import pytorch_lightning as pl
import torch


class EventEncoder(torch.nn.Module):
    def __init__(self, input_features: int, n_filters: list):
        super().__init__()

        self.enc_blocks = torch.nn.ModuleList()

        in_features = input_features
        for i in range(len(n_filters)):
            out_features = n_filters[i]

            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_features, out_features, (3, 3), padding="same"),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_features, out_features, (3, 3), padding="same"),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2),
            )
            self.enc_blocks.append(block)

            in_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.enc_blocks:
            x = block(x)
        return x


class EventDecoder(torch.nn.Module):
    def __init__(self, input_features: int, n_filters: list):
        super().__init__()

        self.dec_blocks = torch.nn.ModuleList()

        in_features = input_features
        for i in range(len(n_filters)):
            out_features = n_filters[i]

            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_features, out_features, (2, 2), stride=2),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_features, out_features, (3, 3), padding="same"),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
            )
            self.dec_blocks.append(block)

            in_features = out_features

        self.last_conv = torch.nn.Conv2d(out_features, 1, (3, 3), padding="same")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.dec_blocks:
            x = block(x)

        x = self.last_conv(x)
        x = self.sigmoid(x)

        return x


class EventAutoEncoder(pl.LightningModule):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module, lr: float):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        return self.decoder(code)

    def training_step(self, train_batch: torch.Tensor, batch_idx):
        X, _ = train_batch
        X = X[:, 0].unsqueeze(1)
        X_hat = self(X)

        criterion = torch.nn.MSELoss()
        loss = criterion(X, X_hat)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, valid_batch: torch.Tensor, batch_idx):
        X, _ = valid_batch
        X = X[:, 0].unsqueeze(1)
        X_hat = self(X)

        criterion = torch.nn.MSELoss()
        loss = criterion(X, X_hat)
        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
