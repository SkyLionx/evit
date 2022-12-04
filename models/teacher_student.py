from typing import Tuple

import torch
import pytorch_lightning as pl
import torchmetrics

from models.modules import (
    PatchExtractor,
    PositionalEncoding,
    ResBlock,
    ResBlockTranspose,
)

from models.convit import CustomConViT, PatchEmbed


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

    def predict_images(self, batch: torch.Tensor):
        events, images = batch
        images = torch.einsum("bhwc -> bchw", images)
        out_images, codes = self(images)
        return out_images


class TeacherTanh(Teacher):
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
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.Tanh(),
        )


class StudentBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # Define metrics functions
        self.ssim_fn = torchmetrics.functional.structural_similarity_index_measure
        self.mse_fn = torchmetrics.functional.mean_squared_error
        self.lpips_fn = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )

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

        # Compute metrics
        ssim = self.ssim_fn(student_rgb, images)
        self.log("train_SSIM", ssim)
        mse = self.mse_fn(student_rgb, images)
        self.log("train_MSE", mse)

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
        self.log("val_loss", loss, prog_bar=True)

        # Compute metrics
        ssim = self.ssim_fn(student_rgb, images)
        self.log("val_SSIM", ssim, prog_bar=True)
        mse = self.mse_fn(student_rgb, images)
        self.log("val_MSE", mse)
        lpips = self.lpips_fn(student_rgb, images)
        self.log("val_LPIPS", lpips)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def predict_images(self, batch: torch.Tensor):
        events, images = batch
        rgbs, codes = self(events)
        return rgbs


class StudentA(StudentBase):
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


class StudentB(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
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

        self.output_size = output_size

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        fmaps_c, fmaps_h, fmaps_w = output_size
        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, fmaps_c)

        p_h, p_w = patch_size
        self.pos_enc = PositionalEncoding(p_h * p_w, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            p_h * p_w, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        self.linear_proj = torch.nn.Linear(p_h * p_w, fmaps_c)

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            fmaps_c, heads, batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer, num_layers
        )

    def forward(self, x):
        batch_size, bins, h, w = x.shape  # (bs, b, h, w)
        x = self.patch_extractor(x)  # (bs, b * p, ph * pw)
        # print("Patch extractor")

        x = self.pos_enc(x)  # (bs, b * p, ph * pw)
        x = self.transformer_encoder(x)  # (bs, b * p, ph * pw)
        # print("Transformer encoder")

        x = self.linear_proj(x)  # (bs, b * p, fc)
        # print("Linear projection")

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(
            torch.arange(fmaps_h * fmaps_w, device=x.device)
        )  # (fh * fw, fc)
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)  # (bs, fh * fw, fc)
        x = self.transformer_decoder(emb, x)  # (bs, fh * fw, fc)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, fmaps_c)  # (bs, fh, fw, fc)
        x = x.permute(0, 3, 1, 2)  # (bs, fc, fh, fw)
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentC(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
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

        self.output_size = output_size

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        fmaps_c, fmaps_h, fmaps_w = output_size
        self.linear_proj = torch.nn.Linear(p_h * p_w, fmaps_c)

        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, fmaps_c)

        self.pos_enc = PositionalEncoding(fmaps_c, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            fmaps_c, heads, batch_first=True, activation=torch.nn.functional.leaky_relu
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            fmaps_c, heads, batch_first=True, activation=torch.nn.functional.leaky_relu
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer,
            num_layers,
        )

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.linear_proj(x)
        # print("Linear projection")

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, fmaps_c).permute(0, 3, 1, 2)
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentD(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        features_weight: float,
        images_weight: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.teacher = teacher
        self.lr = lr
        self.features_weight = features_weight
        self.images_weight = images_weight

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding=1, bias=False),
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

    def forward(self, x):
        features = self.conv(x)
        x = features

        x = self.teacher.decoder(x)

        return x, features


class StudentE(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        heads: int,
        num_layers: int,
        features_weight: float,
        images_weight: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.teacher = teacher
        self.lr = lr
        self.features_weight = features_weight
        self.images_weight = images_weight

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding=1, bias=False),
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

        self.embeddings = torch.nn.Embedding(8 * 8, 128)

        self.pos_enc = PositionalEncoding(128, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            128, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            128, heads, batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer, num_layers
        )

    def forward(self, x):
        batch_size, bins, h, w = x.shape

        x = self.conv(x)
        # Output Shape = (batch_size, 128, 8, 8)

        x = x.reshape(batch_size, 128, 8 * 8)
        x = x.permute(0, 2, 1)
        # Output Shape = (batch_size, 64, 128)

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        emb = self.embeddings(torch.arange(8 * 8, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, 8, 8, 128).permute(0, 3, 1, 2)
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentF(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
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

        self.output_size = output_size

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        embed_dim = p_h * p_w
        self.pos_enc = PositionalEncoding(embed_dim, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            embed_dim, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        fmaps_c, fmaps_h, fmaps_w = output_size
        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, embed_dim)
        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            embed_dim, heads, batch_first=True
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer, num_layers
        )

        self.linear_proj = torch.nn.Linear(embed_dim, fmaps_c)

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = self.linear_proj(x)
        # print("Linear projection")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, fmaps_c).permute(0, 3, 1, 2)
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentG(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
        heads: int,
        num_layers: int,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.teacher = teacher
        self.lr = lr

        self.output_size = output_size

        self.conv_decoder = torch.nn.Sequential(
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

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        fmaps_c, fmaps_h, fmaps_w = output_size
        self.linear_proj = torch.nn.Linear(p_h * p_w, fmaps_c)

        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, fmaps_c)

        self.pos_enc = PositionalEncoding(fmaps_c, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            fmaps_c, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            fmaps_c, heads, batch_first=True, activation=torch.nn.functional.leaky_relu
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer,
            num_layers,
        )

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.linear_proj(x)
        # print("Linear projection")

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, fmaps_c).permute(0, 3, 1, 2)
        x = self.conv_decoder(x)

        return x


class StudentH(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
        heads: int,
        num_layers: int,
        features_weight: float,
        images_weight: float,
        lr: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])
        self.teacher = teacher
        self.output_size = output_size
        self.features_weight = features_weight
        self.images_weight = images_weight
        self.lr = lr

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.conv_decoder = torch.nn.Sequential(
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

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        fmaps_c, fmaps_h, fmaps_w = output_size
        self.linear_proj = torch.nn.Linear(p_h * p_w, fmaps_c)

        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, fmaps_c)

        self.pos_enc = PositionalEncoding(fmaps_c, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            fmaps_c, heads, batch_first=True
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            fmaps_c, heads, batch_first=True, activation=torch.nn.functional.leaky_relu
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer,
            num_layers,
        )

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.linear_proj(x)
        # print("Linear projection")

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)

        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, fmaps_c).permute(0, 3, 1, 2)
        features = x
        x = self.conv_decoder(x)

        return x, features


class StudentI(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
        heads: int,
        num_layers: int,
        encoder_dim: int,
        decoder_dim: int,
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

        self.output_size = output_size

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        fmaps_c, fmaps_h, fmaps_w = output_size
        self.encoder_proj = torch.nn.Linear(p_h * p_w, encoder_dim)

        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, decoder_dim)

        self.pos_enc = PositionalEncoding(encoder_dim, max_len=3000)
        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            encoder_dim,
            heads,
            batch_first=True,
            activation=torch.nn.functional.leaky_relu,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        self.decoder_proj = torch.nn.Linear(encoder_dim, decoder_dim)

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            decoder_dim,
            heads,
            batch_first=True,
            activation=torch.nn.functional.leaky_relu,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer,
            num_layers,
        )

        self.conv_proj = torch.nn.Linear(decoder_dim, fmaps_c)

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.encoder_proj(x)
        # print("Linear projection")

        x = self.pos_enc(x)
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        x = self.decoder_proj(x)

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, -1)

        x = self.conv_proj(x).permute(0, 3, 1, 2)
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentJ(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        input_size: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        output_size: Tuple[int, int, int],
        heads: int,
        num_layers: int,
        encoder_dim: int,
        decoder_dim: int,
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

        self.output_size = output_size

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.patch_extractor = PatchExtractor(patch_size)

        p_h, p_w = patch_size
        fmaps_c, fmaps_h, fmaps_w = output_size
        self.encoder_proj = torch.nn.Linear(p_h * p_w, encoder_dim)

        self.embeddings = torch.nn.Embedding(fmaps_h * fmaps_w, decoder_dim)

        bins, h, w = input_size
        n_p_y, n_p_x = h // p_h, w // p_w
        self.learnable_pos_enc = torch.nn.Embedding(bins * n_p_y * n_p_x, encoder_dim)

        transformer_enc_layer = torch.nn.TransformerEncoderLayer(
            encoder_dim,
            heads,
            batch_first=True,
            activation=torch.nn.functional.leaky_relu,
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            transformer_enc_layer, num_layers
        )

        self.decoder_proj = torch.nn.Linear(encoder_dim, decoder_dim)

        transformer_dec_layer = torch.nn.TransformerDecoderLayer(
            decoder_dim,
            heads,
            batch_first=True,
            activation=torch.nn.functional.leaky_relu,
        )
        self.transformer_decoder = torch.nn.TransformerDecoder(
            transformer_dec_layer,
            num_layers,
        )

        self.conv_net = torch.nn.Sequential(
            torch.nn.Conv2d(decoder_dim, fmaps_c // 2, 3, padding="same"),
            torch.nn.BatchNorm2d(fmaps_c // 2),
            torch.nn.LeakyReLU(fmaps_c // 2),
            torch.nn.Conv2d(fmaps_c // 2, fmaps_c, 3, padding="same"),
            torch.nn.BatchNorm2d(fmaps_c),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(fmaps_c, fmaps_c, 3, padding="same"),
            torch.nn.BatchNorm2d(fmaps_c),
            torch.nn.LeakyReLU(),
        )

        torch.nn.Linear(decoder_dim, fmaps_c)

    def forward(self, x):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)
        # print("Patch extractor")

        x = self.encoder_proj(x)
        # print("Linear projection")

        pos_enc = self.learnable_pos_enc(torch.arange(x.shape[-2], device=x.device))
        pos_enc = pos_enc.unsqueeze(0).repeat(batch_size, 1, 1)

        x = x + pos_enc
        x = self.transformer_encoder(x)
        # print("Transformer encoder")

        x = self.decoder_proj(x)

        fmaps_c, fmaps_h, fmaps_w = self.output_size
        emb = self.embeddings(torch.arange(fmaps_h * fmaps_w, device=x.device))
        emb = emb.unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = self.transformer_decoder(emb, x)
        # print("Transformer decoder")

        x = x.reshape(batch_size, fmaps_h, fmaps_w, -1)

        x = self.conv_net(x.permute(0, 3, 1, 2))
        features = x

        x = self.teacher.decoder(x)

        return x, features


class StudentK(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        input_shape: Tuple[int, int, int],
        num_heads: int,
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

        # Required in order to obtain 8x8=64 patches
        PATCH_SIZE = (16, 16)

        self.patch_extractor = PatchExtractor(PATCH_SIZE)

        bins, h, w = input_shape
        self.ph, self.pw = PATCH_SIZE
        self.num_patches_y = h // self.ph
        self.num_patches_x = w // self.pw
        self.num_patches = self.num_patches_y * self.num_patches_x

        self.patches = PatchExtractor(PATCH_SIZE)

        # Required by the teacher decoder a feature map of shape (128, 8, 8)
        EMBED_DIM = 128

        self.linear_proj = torch.nn.Linear(bins * self.ph * self.pw, EMBED_DIM)

        self.pos_enc = PositionalEncoding(EMBED_DIM)
        enc_layer = torch.nn.TransformerEncoderLayer(
            EMBED_DIM, num_heads, activation=torch.nn.LeakyReLU()
        )
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, x: torch.Tensor):
        batch_size, bins, h, w = x.shape

        x = self.patches(x)
        # x.shape = (batch_size, bins * num_patches, h * w)
        x = x.reshape(batch_size, bins, self.num_patches, self.ph * self.pw)
        x = x.transpose(1, 2)
        x = x.reshape(batch_size, self.num_patches, bins * self.ph * self.pw)

        x = self.linear_proj(x)

        x = self.pos_enc(x)
        x = self.transf_enc(x)

        x = x.reshape(batch_size, self.num_patches_y, self.num_patches_x, 128)
        x = x.permute(0, 3, 1, 2)
        features = x

        x = self.teacher.decoder(x)

        return x, features

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="max", verbose=True
            ),
            "interval": "epoch",
            "monitor": "val_SSIM",
        }
        return {"optimizer": optim, "lr_scheduler": lr_scheduler_config}


class StudentL(StudentBase):
    def __init__(
        self,
        teacher: torch.nn.Module,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        features_weight: float,
        images_weight: float,
        learning_rate: float,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])

        # Freeze teacher parameters
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.teacher = teacher
        self.input_shape = input_shape
        self.p_h, self.p_w = patch_size
        self.num_patch_y = input_shape[1] // self.p_h
        self.num_patch_x = input_shape[2] // self.p_w
        self.features_weight = features_weight
        self.images_weight = images_weight
        self.lr = learning_rate

        bins, h, w = input_shape
        self.convvit = CustomConViT(
            input_shape[1:],
            patch_size,
            bins,
            embed_dim,
            num_layers,
            num_heads,
        )

    def forward(self, x: torch.Tensor):
        x = self.convvit(x)
        # x.shape = (batch_size, num_patches, embed_dim)

        x = x.reshape(x.shape[0], self.num_patch_y, self.num_patch_x, -1)
        x = x.permute(0, 3, 1, 2)
        features = x
        x = self.teacher.decoder(x)
        return x, features


class TeacherTest(Teacher):
    def _build_encoder(self):
        return torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, 2, 2, padding=0, bias=False),
            torch.nn.Conv2d(16, 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, 2, 2, padding=0, bias=False),
            torch.nn.Conv2d(16, 16, 3, padding=1, bias=False),
            torch.nn.ReLU(),
        )
        # Feature map shape = (16, 32, 32)

    def _build_decoder(self):
        return torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, 2, 2, padding=0, bias=False),
            torch.nn.ConvTranspose2d(16, 16, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, 2, 2, padding=0, bias=False),
            torch.nn.ConvTranspose2d(16, 8, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 3, 3, padding=1, bias=False),
            torch.nn.Sigmoid(),
        )


class StudentTest(StudentBase):
    def __init__(
        self, teacher, num_heads, num_layers, features_weight, images_weight, lr
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["teacher"])

        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

        self.teacher = teacher

        self.lr = lr
        self.features_weight = features_weight
        self.images_weight = images_weight

        img_size = 128
        patch_size = 4
        in_chans = 10
        embed_dim = 16
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        transf_enc_layer = torch.nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transf_enc = torch.nn.TransformerEncoder(transf_enc_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        # x.shape = (batch_size, 32*32, 16)
        x = self.transf_enc(x)
        x = x.reshape(x.shape[0], 32, 32, 16)
        x = x.permute(0, 3, 1, 2)
        features = x

        with torch.no_grad():
            x = self.teacher.decoder(x)
        return x, features

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optim, mode="max", verbose=True
            ),
            "interval": "epoch",
            "monitor": "val_SSIM",
        }
        return {"optimizer": optim, "lr_scheduler": lr_scheduler_config}


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
