from typing import Tuple

import pytorch_lightning as pl
import torch
from models.transformer import PatchExtractor, PositionalEncoding

import matplotlib.pyplot as plt


class ResBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class ResBlockTranspose(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(
                in_channels, out_channels, 3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(
                out_channels, out_channels, 3, padding=1, bias=False
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        return x + self.block(x)


class Teacher(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.lr = lr

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

    def _build_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            ResBlock(32, 32),
            torch.nn.Conv2d(32, 32, 2, stride=2, padding=0, bias=False),
            ResBlock(32, 32),
            torch.nn.Conv2d(32, 64, 2, stride=2, padding=0, bias=False),
            ResBlock(64, 64),
            torch.nn.Conv2d(64, 64, 2, stride=2, padding=0, bias=False),
            ResBlock(64, 64),
            torch.nn.Conv2d(64, 128, 2, stride=2, padding=0, bias=False),
            ResBlock(128, 128),
        )

    def _build_decoder(self):
        return torch.nn.Sequential(
            ResBlockTranspose(128, 128),
            torch.nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, bias=False),
            ResBlockTranspose(64, 64),
            torch.nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, bias=False),
            ResBlockTranspose(64, 64),
            torch.nn.ConvTranspose2d(64, 32, 2, stride=2, padding=0, bias=False),
            ResBlockTranspose(32, 32),
            torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, bias=False),
            ResBlockTranspose(32, 32),
            torch.nn.Conv2d(32, 3, 3, bias=False, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        code = self.encoder(x)
        y = self.decoder(code)
        return y, code

    def training_step(self, batch, batch_idx):
        events, images = batch
        images = torch.einsum("bhwc -> bchw", images)
        y_hat, code = self(images)
        loss = torch.nn.functional.mse_loss(y_hat, images)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        events, images = batch
        images = torch.einsum("bhwc -> bchw", images)
        y_hat, code = self(images)
        loss = torch.nn.functional.mse_loss(y_hat, images)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class StudentA(pl.LightningModule):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        enc_size: int,
        heads: int,
        num_layers: int,
        features_weight: float,
        images_weight: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.teacher = teacher
        self.features_weight = features_weight
        self.images_weight = images_weight
        self.lr = lr

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        self.embeddings = torch.nn.Embedding(128, 64)

        self.pos_enc = PositionalEncoding(enc_size, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            enc_size, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            enc_size, heads, batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer, num_layers
        )

    def forward(self, x):
        x = self.patch_extractor(x)

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)

        emb = self.embeddings(torch.arange(128, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)

        x = x.reshape(x.shape[0], 128, 8, 8)
        features = x

        x = self.teacher.decoder(x)

        return x, features

    def training_step(self, train_batch, train_idx):
        events, images = train_batch
        images = torch.einsum("bhwc -> bchw", images)
        teach_rgb, teach_features = self.teacher(images)
        student_rgb, student_features = self(events)

        features_loss = torch.nn.functional.mse_loss(teach_features, student_features)
        self.log("train_features_loss", features_loss)
        image_loss = torch.nn.functional.mse_loss(teach_rgb, student_rgb)
        self.log("train_image_loss", image_loss)
        loss = self.features_weight * features_loss + self.images_weight * image_loss
        self.log("train_loss", loss)
        return loss

    def validation_step(self, val_batch, val_idx):
        events, images = val_batch
        images = torch.einsum("bhwc -> bchw", images)
        teach_rgb, teach_features = self.teacher(images)
        student_rgb, student_features = self(events)

        features_loss = torch.nn.functional.mse_loss(teach_features, student_features)
        self.log("val_features_loss", features_loss)
        image_loss = torch.nn.functional.mse_loss(teach_rgb, student_rgb)
        self.log("val_image_loss", image_loss)
        loss = self.features_weight * features_loss + self.images_weight * image_loss
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


if __name__ == "__main__":
    from torchinfo import summary

    teacher = Teacher(0.001)
    # summary(
    #     teacher,
    #     input_size=(8, 3, 128, 128),
    #     col_names=["input_size", "output_size", "num_params"],
    # )

    PARAMS = {
        "patch_size": (8, 8),
        "enc_size": 64,
        "heads": 4,
        "num_layers": 3,
        "lr": 0.0001,
    }

    student = StudentA(teacher=teacher, **PARAMS)
    # student(torch.rand((8, 10, 128, 128)))
    summary(
        student,
        input_size=(8, 3, 128, 128),
        col_names=["input_size", "output_size", "num_params"],
    )
