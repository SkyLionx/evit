import pytorch_lightning as pl
import torch
from models.transformer import PatchExtractor, PositionalEncoding


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
            torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            torch.nn.Conv2d(32, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=2, padding=1, bias=False),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
        )

    def _build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, bias=False),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, bias=False),
            torch.nn.ConvTranspose2d(64, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, bias=False),
            torch.nn.ConvTranspose2d(32, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, bias=False, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
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


class Student(pl.LightningModule):
    def __init__(self, teacher: torch.nn.Module, lr: float):
        super().__init__()
        self.save_hyperparameters()
        self.teacher = teacher
        self.lr = lr

        self.patch_extractor = PatchExtractor()

        self.pos_enc = PositionalEncoding()
        transformer_enc_layer = torch.nn.TransformerEncoderLayer()
        self.transformer_encoder = torch.nn.TransformerEncoder()

    def forward(self, x):
        x = self.patch_extractor(x)

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)

        x = x.reshape()
        features = x

        x = self.teacher.decoder(x)

        return x, features

    def training_step(self, train_batch, train_idx):
        teach_rgb, teach_features = self.teacher(train_batch)
        student_rgb, student_features = self(train_batch)

        features_loss = torch.nn.functional.mse_loss(teach_features, student_features)
        image_loss = torch.nn.functional.mse_loss(teach_rgb, student_rgb)

        loss = self.features_weight * features_loss + self.image_weight * image_loss
        self.log("train_features_loss", features_loss)
        self.log("train_image_loss", image_loss)
        return loss

    def configure_optimizers(self):
        pass


if __name__ == "__main__":
    from torchinfo import summary

    teacher = Teacher(0.001)
    summary(teacher, input_size=(8, 3, 128, 128))
