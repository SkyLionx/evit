import torch
import pytorch_lightning as pl
import torchmetrics

from models.transformer import PatchExtractor, PositionalEncoding


class BaseModel(pl.LightningModule):
    def __init__(self, lr: float):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr

        # Normalize should be True if images are in [0, 1] (False -> [-1, +1])
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        y = torch.einsum("bhwc -> bchw", y)

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        loss = criterion(model_images, y)

        self.log("train_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.log("train_MSE", mse)
        self.log("train_SSIM", ssim)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        y = torch.einsum("bhwc -> bchw", y)

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        loss = criterion(model_images, y)

        self.log("val_loss", loss, prog_bar=True)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.lpips(model_images, y)
        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim, prog_bar=True)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_images(self, batch):
        events, images = batch
        return self(events)


class TestA(BaseModel):
    """Simple CNN"""

    def __init__(self, lr):
        super().__init__(lr)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(10, 128, 3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 3, 3, padding="same"),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        return self.conv(x)


class TestB(BaseModel):
    """Simple CNN with Patch Extractor"""

    def __init__(self, lr, input_shape, patch_size):
        super().__init__(lr)
        self.bins, self.h, self.w = input_shape
        self.pw, self.ph = patch_size
        self.n_patches_y = self.h // self.ph
        self.n_patches_x = self.w // self.pw
        self.n_patches = self.n_patches_y * self.n_patches_x
        self.patches = PatchExtractor(patch_size)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(10, 128, 3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 3, 3, padding="same"),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size, bins, h, w = x.shape
        x = self.patches(x)
        # x = (batch_size, bins * n_patches, ph * pw)
        x = x.reshape(batch_size, self.bins, self.n_patches, self.ph, self.pw)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batch_size * self.n_patches, self.bins, self.ph, self.pw)
        x = self.conv(x)
        x = x.reshape(
            batch_size, self.n_patches_y, self.n_patches_x, -1, self.ph, self.pw
        )
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(batch_size, -1, self.h, self.w)


class TestC(BaseModel):
    def __init__(self, lr, embed_dim, patch_size, nhead, num_layers):
        super().__init__(lr)

        self.patches = PatchExtractor(patch_size)

        pw, ph = patch_size
        self.pos_enc = PositionalEncoding(embed_dim)

        self.linear_proj = torch.nn.Linear(pw * ph, embed_dim)

        enc_layer = torch.nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True)
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = torch.nn.TransformerDecoderLayer(embed_dim, nhead, batch_first=True)
        self.transf_dec = torch.nn.TransformerDecoder(dec_layer, num_layers)

        self.inverse_proj = torch.nn.Linear(embed_dim, pw * ph)

    def forward(self, x: torch.Tensor):
        x = self.patches(x)
        # x = (batch_size, bins * n_patches, ph * pw)
        x = self.linear_proj(x)

        x = self.pos_enc(x)
        x = self.transf_enc(x)

        n_patches = 16
        zeros = torch.zeros(x.shape[0], 3 * n_patches, x.shape[2], device=x.device)
        pos = self.pos_enc(zeros)
        x = self.transf_dec(pos, x)

        x = self.inverse_proj(x)

        n_p_y = n_p_x = 4
        ph = pw = 32
        x = x.reshape(x.shape[0], 3, n_p_y, n_p_x, ph, pw)
        x = x.permute(0, 1, 2, 4, 3, 5)
        h = w = 128
        x = x.reshape(x.shape[0], 3, h, w)
        x = torch.nn.functional.sigmoid(x)
        return x


class TestD(BaseModel):
    def __init__(self, lr, nhead, num_layers):
        super().__init__(lr)

        self.conv_enc = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 128, 3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 3, padding="same"),
            torch.nn.Conv2d(128, 128, 2, 2, padding=0),
        )

        self.pos_enc = PositionalEncoding(128)

        enc_layer = torch.nn.TransformerEncoderLayer(128, nhead, batch_first=True)
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, num_layers)

        dec_layer = torch.nn.TransformerDecoderLayer(128, nhead, batch_first=True)
        self.transf_dec = torch.nn.TransformerDecoder(dec_layer, num_layers)

        self.conv_dec = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(128, 128, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size, bins, h, w = x.shape

        # Apply convolution channel-wise
        x = self.conv_enc(x)
        channels, new_h, new_w = x.shape[1:]

        x = x.reshape(batch_size, channels, new_h * new_w)
        x = x.transpose(1, 2)

        # print("Shape before pos enc:", x.shape)
        x = self.pos_enc(x)
        x = self.transf_enc(x)

        x = x.reshape(batch_size, new_h, new_w, channels)
        x = x.permute(0, 3, 1, 2)
        # print("Shape before conv dec:", x.shape)
        x = self.conv_dec(x)

        return x
