import math
from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl
import cv2
from torchvision.models import vgg19, VGG19_Weights, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from models.modules import PositionalEncoding, PatchExtractor, BayerDecomposer


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        encoding_size: int,
        heads: int,
        layers_number: int,
    ):
        super(TransformerModel, self).__init__()

        width, height, channels = output_shape

        self.output_shape = output_shape
        self.encoding_size = encoding_size
        self.heads = heads
        self.layers_number = layers_number

        self.linear_proj = torch.nn.Linear(width * height, encoding_size)
        self.activation = torch.nn.ReLU()
        self.linear_inv_proj = torch.nn.Linear(encoding_size, width * height * channels)
        self.sigmoid = torch.nn.Sigmoid()

        sos_tensor = torch.tensor(
            [1] + [0] * (encoding_size - 1), dtype=torch.float32, requires_grad=False
        ).reshape(1, 1, -1)
        self.sos_token = torch.nn.Parameter(sos_tensor)

        self.pe = PositionalEncoding(encoding_size, max_len=10)
        self.transformer = torch.nn.Transformer(
            d_model=encoding_size,
            nhead=heads,
            num_encoder_layers=layers_number,
            num_decoder_layers=layers_number,
            batch_first=True,
        )
        # self.lstm = torch.nn.LSTM(
        #   batch_first=True,
        #   input_size=encoding_size,
        #   num_layers=layers_number,
        #   hidden_size=encoding_size
        # )

    def forward(self, x):
        # print("x shape", x.shape, "output shape", self.output_shape)
        width, height, channels = self.output_shape
        batches, bins = x.shape[:2]

        x = x.reshape(batches, bins, height * width)
        # print("Shape before projection", x.shape)
        x = self.linear_proj(x)
        # print("Shape after projection", x.shape)
        x = self.activation(x)

        x = self.pe(x)
        # print("Shape after positional encoding", x.shape)

        batched_sos = torch.repeat_interleave(self.sos_token, batches, 0)
        # print("Transformer inputs", x.shape, batched_sos.shape)
        x = self.transformer(x, batched_sos)
        # x = self.lstm(x)[0][:, -1]
        # print("Transformer output", x.shape)

        x = self.linear_inv_proj(x)
        # print("Shape after inv. projection", x.shape)
        y = self.sigmoid(x)
        y = y.reshape(batches, channels, height, width)
        # print("Output shape", y.shape)
        return y


class EventEncoderTransformer(torch.nn.Module):
    def __init__(
        self,
        output_shape: Tuple[int, int, int],
        encoder: torch.nn.Module,
        encoding_size: int,
        heads: int,
        layers_number: int,
    ):
        super(EventEncoderTransformer, self).__init__()

        width, height, channels = output_shape

        self.output_shape = output_shape
        self.encoder = encoder
        self.encoding_size = encoding_size
        self.heads = heads
        self.layers_number = layers_number

        self.activation = torch.nn.ReLU()
        self.linear_inv_proj = torch.nn.Linear(encoding_size, width * height * channels)
        self.sigmoid = torch.nn.Sigmoid()

        sos_tensor = torch.tensor(
            [1] + [0] * (encoding_size - 1), dtype=torch.float32, requires_grad=False
        ).reshape(1, 1, -1)
        self.sos_token = torch.nn.Parameter(sos_tensor)

        self.pe = PositionalEncoding(encoding_size, max_len=10)
        self.transformer = torch.nn.Transformer(
            d_model=encoding_size,
            nhead=heads,
            num_encoder_layers=layers_number,
            num_decoder_layers=layers_number,
            batch_first=True,
        )

    def forward(self, x):
        width, height, channels = self.output_shape
        batches, bins = x.shape[:2]

        x = self.encoder(x.reshape(-1, 1, height, width))
        x = x.reshape(batches, bins, -1)

        x = self.pe(x)

        batched_sos = torch.repeat_interleave(self.sos_token, batches, 0)
        x = self.transformer(x, batched_sos)

        x = self.linear_inv_proj(x)
        y = self.sigmoid(x)

        y = y.reshape(batches, channels, height, width)
        return y


class VisionTransformer(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        encoding_size: int,
        heads: int,
        layers_number: int,
        use_linear_proj: bool,
        learning_rate: float,
        use_LPIPS: bool = False,
        vgg_layer: str = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.p_h, self.p_w = patch_size
        self.use_linear_proj = use_linear_proj
        self.lr = learning_rate

        self.use_LPIPS = use_LPIPS
        self.vgg_layer = vgg_layer
        if use_LPIPS and not vgg_layer:
            raise Exception(
                "In order to use the LPIPS loss, you need to specify a vgg_layer."
            )

        self.bins, self.h, self.w = self.input_shape
        self.n_patch_x = self.w // self.p_w
        self.n_patch_y = self.h // self.p_h

        self.token_dim = self.p_w * self.p_h

        self.patch_extractor = PatchExtractor(patch_size)

        if use_linear_proj:
            self.linear_proj = torch.nn.Linear(self.token_dim, encoding_size)
            self.token_dim = encoding_size

        self.pe = PositionalEncoding(
            self.token_dim, max_len=self.n_patch_x * self.n_patch_y * self.bins
        )
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=encoding_size, nhead=heads, batch_first=True
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        # dec_layer = torch.nn.TransformerDecoderLayer(
        #     d_model=encoding_size, nhead=heads, batch_first=True
        # )
        # self.dec = torch.nn.TransformerDecoder(dec_layer, layers_number)

        if use_linear_proj:
            self.inv_proj = torch.nn.Linear(self.token_dim, self.p_w * self.p_h)

        # n_convs = int(math.log2(self.w) - math.log2(self.n_patch_x))
        # self.transp_convs = torch.nn.ModuleList()
        # for i in range(n_convs):
        #     in_filters = self.token_dim // 2 ** (i)
        #     out_filters = self.token_dim // (2 ** (i + 1))
        #     tconv = torch.nn.ConvTranspose2d(in_filters, out_filters, (2, 2), stride=(2, 2))
        #     self.transp_convs.append(tconv)

        self.convolutions = torch.nn.Sequential(
            torch.nn.Conv2d(self.bins, 64, (3, 3), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, (3, 3), padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 3, (3, 3), padding="same"),
        )
        # self.final_conv = torch.nn.Conv2d(self.bins, 3, (3, 3), padding="same")
        self.sigmoid = torch.nn.Sigmoid()

        if use_LPIPS:
            vgg_weights = VGG19_Weights.IMAGENET1K_V1
            vgg = list(vgg19(weights=vgg_weights).children())[0]
            vgg.eval()

            # Freeze layers
            for param in vgg.parameters():
                param.requires_grad = False

            self.vgg_preprocess = vgg_weights.transforms()
            self.vgg_extractor = create_feature_extractor(vgg, [vgg_layer])

    def forward(self, x):
        batch, bins, h, w = x.shape

        x = self.patch_extractor(x)
        # x shape = (batch, bins * n_patches, p_h * p_w)

        if self.use_linear_proj:
            x = self.linear_proj(x)

        x = self.pe(x)

        x = self.enc(x)

        x = x.reshape(batch, bins, -1, self.token_dim)

        # Option 1, with decoder
        # x = torch.mean(x, dim=1)
        # x = self.dec(x)
        # x = self.inv_proj(x)
        #  return self.assemble_image(x)

        # Option 2, with traspose conv
        if self.use_linear_proj:
            x = self.inv_proj(x)
        x = x.reshape(batch, bins, self.n_patch_y, self.n_patch_x, self.p_h, self.p_w)
        x = torch.einsum("btyxhw -> btyhxw", x)
        x = x.reshape(batch, bins, self.h, self.w)

        x = self.convolutions(x)
        x = self.sigmoid(x)
        return x

    def assemble_image(self, x: torch.Tensor) -> torch.Tensor:
        batch, n, d = x.shape

        x = x.reshape(batch, self.h // self.p_h, self.w // self.p_w, self.p_h, self.p_w)
        x = x.swapaxes(2, 3)
        x = x.reshape(batch, self.h, self.w)

        return x

    def reconstruct_image(self, events: torch.Tensor):
        raw_img = self(events)[0]
        color_img = cv2.demosaicing(raw_img, cv2.COLOR_BayerRGGB2RGB)
        return color_img

    def _extract_features(self, x):
        return self.vgg_extractor(self.vgg_preprocess(x))[self.vgg_layer]

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        if self.use_LPIPS:
            model_features = self._extract_features(model_images)
            y_features = self._extract_features(y)

            loss = criterion(model_features, y_features)
        else:
            loss = criterion(model_images, y)

        self.log("train_loss", loss)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        if self.use_LPIPS:
            model_features = self._extract_features(model_images)
            y_features = self._extract_features(y)

            loss = criterion(model_features, y_features)
        else:
            loss = criterion(model_images, y)

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class VisionTransformerConv(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        heads: int,
        layers_number: int,
        learning_rate: float,
        image_loss_weight: int,
        feature_loss_weight: int,
        color_output: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.bins, self.h, self.w = self.input_shape
        self.p_h, self.p_w = patch_size
        self.lr = learning_rate
        self.image_loss_weight = image_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.color_output = color_output

        self.token_dim = self.p_w * self.p_h

        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.patch_extractor = PatchExtractor(patch_size)

        self.pe = PositionalEncoding(self.token_dim, max_len=2048)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, batch_first=True
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        out_filters = 3 if self.color_output else 1

        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, out_filters, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        # Normalize should be True if images are in [0, 1] (False -> [-1, +1])
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def forward(self, x: torch.Tensor):
        batch, bins, h, w = x.shape

        x = self.conv_encoder(x)
        x_features_pre = x
        # x shape = (batch, out_filters, new_h, new_w)
        # print("Encoder output shape:", x.shape)

        # Save the output shape for later
        _, out_filters, new_h, new_w = x.shape
        num_patches_x = new_w // self.p_w
        num_patches_y = new_h // self.p_h

        x = self.patch_extractor(x)
        # x shape = (batch, out_filters * n_patches, p_h * p_w)
        # print("Patch extractor output shape:", x.shape)

        x = self.pe(x)
        x = self.enc(x)
        # print("Encoder output shape:", x.shape)

        x = x.reshape(
            batch, out_filters, num_patches_y, num_patches_x, self.p_h, self.p_w
        )
        x = torch.einsum("btyxhw -> btyhxw", x)
        x = x.reshape(batch, out_filters, new_h, new_w)

        x_features_post = x

        x = self.conv_decoder(x)

        return x, x_features_pre, x_features_post

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images, pre, post = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)
        features_loss = criterion(pre, post)

        loss = image_loss + self.feature_loss_weight * features_loss

        self.log("train_image_loss", image_loss)
        self.log("train_features_loss", features_loss)
        self.log("train_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.log("train_MSE", mse)
        self.log("train_SSIM", ssim)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images, pre, post = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)
        features_loss = criterion(pre, post)

        loss = (
            self.image_loss_weight * image_loss
            + self.feature_loss_weight * features_loss
        )

        self.log("val_image_loss", image_loss)
        self.log("val_features_loss", features_loss)
        self.log("val_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)

        if not self.color_output:
            images_rgb = torch.repeat_interleave(model_images, 3, 1)
            y_rgb = torch.repeat_interleave(y, 3, 1)
            self.lpips(images_rgb, y_rgb)
        else:
            self.lpips(model_images, y)
        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_images(self, batch):
        events, images = batch
        return self(events)[0]


def predict_color_images(self, batch: torch.Tensor):
    events, images = batch
    out_images = []
    for event_grid, image in zip(events, images):
        r = event_grid[:, 0::2, 0::2]
        g = (event_grid[:, 0::2, 1::2] + event_grid[:, 1::2, 0::2]) / 2
        b = event_grid[:, 1::2, 1::2]
        x = torch.stack([r, g, b])

        up = torch.nn.Upsample(scale_factor=(2, 2), mode="bilinear")
        x = up(x)

        out_image = self(x)[0].squeeze()
        out_images.append(out_image)
    return torch.stack(out_images)


class ViTransformerConvSmall(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        heads: int,
        layers_number: int,
        dim_ff: int,
        learning_rate: float,
        image_loss_weight: int,
        feature_loss_weight: int,
        color_output: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.bins, self.h, self.w = self.input_shape
        self.p_h, self.p_w = patch_size
        self.lr = learning_rate
        self.image_loss_weight = image_loss_weight
        self.feature_loss_weight = feature_loss_weight
        self.color_output = color_output

        self.token_dim = self.p_w * self.p_h

        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(10, 16, 3, padding="same"),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 16, 3, padding="same"),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
        )

        self.patch_extractor = PatchExtractor(patch_size)

        self.pe = PositionalEncoding(self.token_dim, max_len=2048)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=heads,
            dim_feedforward=dim_ff,
            batch_first=True,
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        out_filters = 3 if self.color_output else 1

        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(16, 16, 3, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 16, 2, 2, padding=0),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, out_filters, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        # Normalize should be True if images are in [0, 1] (False -> [-1, +1])
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def forward(self, x: torch.Tensor):
        batch, bins, h, w = x.shape

        x = self.conv_encoder(x)
        x_features_pre = x

        _, out_filters, new_h, new_w = x.shape
        num_patches_x = new_w // self.p_w
        num_patches_y = new_h // self.p_h

        x = self.patch_extractor(x)

        x = self.pe(x)
        x = self.enc(x)

        x = x.reshape(
            batch, out_filters, num_patches_y, num_patches_x, self.p_h, self.p_w
        )
        x = torch.einsum("btyxhw -> btyhxw", x)
        x = x.reshape(batch, out_filters, new_h, new_w)

        x_features_post = x

        x = self.conv_decoder(x)

        return x, x_features_pre, x_features_post

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images, pre, post = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)
        features_loss = criterion(pre, post)

        loss = image_loss + self.feature_loss_weight * features_loss

        self.log("train_image_loss", image_loss)
        self.log("train_features_loss", features_loss)
        self.log("train_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.log("train_MSE", mse)
        self.log("train_SSIM", ssim)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images, pre, post = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)
        features_loss = criterion(pre, post)

        loss = (
            self.image_loss_weight * image_loss
            + self.feature_loss_weight * features_loss
        )

        self.log("val_image_loss", image_loss)
        self.log("val_features_loss", features_loss)
        self.log("val_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)

        if not self.color_output:
            images_rgb = torch.repeat_interleave(model_images, 3, 1)
            y_rgb = torch.repeat_interleave(y, 3, 1)
            self.lpips(images_rgb, y_rgb)
        else:
            self.lpips(model_images, y)
        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_images(self, batch):
        events, images = batch
        return self(events)[0]


class ViTTransformerConvSkip(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        heads: int,
        layers_number: int,
        learning_rate: float,
        color_output: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.bins, self.h, self.w = self.input_shape
        self.p_h, self.p_w = patch_size
        self.lr = learning_rate
        self.color_output = color_output

        self.token_dim = self.p_w * self.p_h

        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(10, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
        )

        self.patch_extractor = PatchExtractor(patch_size)

        self.pe = PositionalEncoding(self.token_dim, max_len=2048)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, batch_first=True
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        out_filters = 3 if self.color_output else 1

        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, out_filters, 3, padding=1),
            torch.nn.Sigmoid(),
        )

        # Normalize should be True if images are in [0, 1] (False -> [-1, +1])
        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def forward(self, x: torch.Tensor):
        batch, bins, h, w = x.shape

        x = self.conv_encoder(x)
        x_features_pre = x

        # Save the output shape for later
        _, out_filters, new_h, new_w = x.shape
        num_patches_x = new_w // self.p_w
        num_patches_y = new_h // self.p_h

        x = self.patch_extractor(x)

        x = self.pe(x)
        x = self.enc(x)

        x = x.reshape(
            batch, out_filters, num_patches_y, num_patches_x, self.p_h, self.p_w
        )
        x = torch.einsum("btyxhw -> btyhxw", x)
        x = x.reshape(batch, out_filters, new_h, new_w)

        x = x + x_features_pre

        x = self.conv_decoder(x)

        return x

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

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
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        loss = criterion(model_images, y)

        self.log("val_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)

        if not self.color_output:
            images_rgb = torch.repeat_interleave(model_images, 3, 1)
            y_rgb = torch.repeat_interleave(y, 3, 1)
            self.lpips(images_rgb, y_rgb)
        else:
            self.lpips(model_images, y)
        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def predict_images(self, batch):
        events, images = batch
        return self(events)


class TransformerBase(pl.LightningModule):
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


class TransformerA(TransformerBase):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        encoder_dim: int,
        decoder_dim: int,
        conv_dim: int,
        heads: int,
        layers_number: int,
        learning_rate: float,
    ):
        super().__init__(learning_rate)
        self.conv_dim = conv_dim

        self.pw, self.ph = patch_size
        self.bins, self.h, self.w = input_shape
        self.n_patches_x = self.w // self.pw
        self.n_patches_y = self.h // self.ph
        self.n_patches = self.n_patches_x * self.n_patches_y

        self.patch_extractor = PatchExtractor(patch_size)

        self.encoder_proj = torch.nn.Linear(self.pw * self.ph, encoder_dim)

        self.pos_enc = PositionalEncoding(
            encoder_dim, max_len=self.bins * self.n_patches
        )

        enc_layer = torch.nn.TransformerEncoderLayer(
            encoder_dim, heads, batch_first=True
        )
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        self.decoder_proj = torch.nn.Linear(encoder_dim, decoder_dim)
        self.dec_pos_emb = torch.nn.Embedding(8 * 8, decoder_dim)
        dec_layer = torch.nn.TransformerDecoderLayer(
            decoder_dim, heads, batch_first=True
        )
        self.transf_dec = torch.nn.TransformerDecoder(dec_layer, layers_number)

        self.conv_proj = torch.nn.Linear(decoder_dim, conv_dim)

        self.conv_dec = torch.nn.Sequential(
            torch.nn.Conv2d(conv_dim, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.Conv2d(32, 16, 3, padding="same"),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 16, 2, 2, padding=0),
            torch.nn.Conv2d(16, 16, 3, padding="same"),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(),
            torch.nn.ConvTranspose2d(16, 16, 2, 2, padding=0),
            torch.nn.Conv2d(16, 3, 3, padding="same"),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        x = self.patch_extractor(x)
        x = self.encoder_proj(x)

        x = self.pos_enc(x)
        x = self.transf_enc(x)

        x = self.decoder_proj(x)

        emb = self.dec_pos_emb(torch.arange(8 * 8, device=x.device))
        batch_size = x.shape[0]
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.transf_dec(emb, x)

        x = self.conv_proj(x)
        x = x.reshape(batch_size, 8, 8, self.conv_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_dec(x)

        return x


class TransformerB(TransformerBase):
    def __init__(
        self,
        num_heads: int,
        num_layers: int,
        lr: float,
    ):
        super().__init__(lr)
        self.save_hyperparameters()

        self.bayer_decomposer = BayerDecomposer()

        self.conv_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 3, padding="same"),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(32, 32, 2, 2, padding=0),
            torch.nn.Conv2d(32, 64, 3, padding="same"),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(64, 64, 2, 2, padding=0),
            torch.nn.Conv2d(64, 128, 3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(128, 128, 2, 2, padding=0),
            torch.nn.Conv2d(128, 128, 3, padding="same"),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(),
        )

        feature_map_shape = (8, 8)
        token_dim = 128

        self.pos_enc = PositionalEncoding(token_dim, max_len=2048)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=num_heads, batch_first=True
        )
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, num_layers)

        num_tokens = feature_map_shape[0] * feature_map_shape[1]
        self.dec_embeddings = torch.nn.Embedding(num_tokens, token_dim)
        dec_layer = torch.nn.TransformerDecoderLayer(
            d_model=token_dim, nhead=num_heads, batch_first=True
        )
        self.transf_dec = torch.nn.TransformerDecoder(dec_layer, num_layers)

        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(32, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.Conv2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch, bins, h, w = x.shape

        x = x.reshape(batch * bins, h, w)
        x = self.bayer_decomposer(x)
        # x.shape = (batch * bins, 4, h // 2, w // 2)

        x = self.conv_encoder(x)
        filters, new_h, new_w = x.shape[1:]

        x = x.reshape(batch, bins, filters, new_h, new_w)
        x = x.permute(0, 1, 3, 4, 2)
        x = x.reshape(batch, bins * new_h * new_w, filters)

        x = self.pos_enc(x)
        x = self.transf_enc(x)

        emb_indexes = torch.arange(new_h * new_w, device=x.device)
        decoder_emb = self.dec_embeddings(emb_indexes).unsqueeze(0).repeat(batch, 1, 1)
        x = self.transf_dec(decoder_emb, x)

        x = x.reshape(batch, new_h, new_w, filters)
        x = x.permute(0, 3, 1, 2)

        x = self.conv_decoder(x)

        return x


class TransformerC(TransformerBase):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        lr: float,
    ):
        super().__init__(lr)
        self.save_hyperparameters()

        bins, h, w = input_shape
        self.ph, self.pw = patch_size
        self.num_patches_y = h // self.ph
        self.num_patches_x = w // self.pw
        self.num_patches = self.num_patches_y * self.num_patches_x

        self.patches = PatchExtractor(patch_size)

        self.linear_proj = torch.nn.Linear(bins * self.ph * self.pw, embed_dim)

        self.pos_enc = PositionalEncoding(embed_dim)
        enc_layer = torch.nn.TransformerEncoderLayer(embed_dim, num_heads)
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, num_layers)

        self.conv_decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(embed_dim, embed_dim, 3, padding=1),
            torch.nn.BatchNorm2d(embed_dim),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(embed_dim, embed_dim // 2, 3, padding=1),
            torch.nn.BatchNorm2d(embed_dim // 2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(embed_dim // 2, embed_dim // 2, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(embed_dim // 2, embed_dim // 4, 3, padding=1),
            torch.nn.BatchNorm2d(embed_dim // 4),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(embed_dim // 4, embed_dim // 4, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(embed_dim // 4, embed_dim // 8, 3, padding=1),
            torch.nn.BatchNorm2d(embed_dim // 8),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(embed_dim // 8, embed_dim // 16, 2, 2, padding=0),
            torch.nn.Conv2d(embed_dim // 16, 3, 3, padding=1),
            torch.nn.Sigmoid(),
        )

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

        x = x.reshape(batch_size, self.num_patches_y, self.num_patches_x, -1)
        x = x.permute(0, 3, 1, 2)

        x = self.conv_decoder(x)

        return x


class TransformerD(TransformerBase):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        encoder_dim: int,
        decoder_dim: int,
        conv_dim: int,
        heads: int,
        layers_number: int,
        learning_rate: float,
        warmup_steps: int,
    ):
        super().__init__(learning_rate)
        self.warmup_steps = warmup_steps
        self.conv_dim = conv_dim

        self.pw, self.ph = patch_size
        self.bins, self.h, self.w = input_shape
        self.n_patches_x = self.w // self.pw
        self.n_patches_y = self.h // self.ph
        self.n_patches = self.n_patches_x * self.n_patches_y

        self.patch_extractor = PatchExtractor(patch_size)

        self.encoder_proj = torch.nn.Linear(self.bins * self.pw * self.ph, encoder_dim)

        self.pos_enc = PositionalEncoding(
            encoder_dim, max_len=self.bins * self.n_patches
        )

        enc_layer = torch.nn.TransformerEncoderLayer(
            encoder_dim, heads, batch_first=True
        )
        self.transf_enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        self.decoder_proj = torch.nn.Linear(encoder_dim, decoder_dim)
        self.dec_pos_emb = torch.nn.Embedding(8 * 8, decoder_dim)
        dec_layer = torch.nn.TransformerDecoderLayer(
            decoder_dim, heads, batch_first=True
        )
        self.transf_dec = torch.nn.TransformerDecoder(dec_layer, layers_number)

        self.conv_proj = torch.nn.Linear(decoder_dim, conv_dim)

        self.conv_dec = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 128, 3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 128, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(128, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 64, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(64, 32, 3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 32, 2, 2, padding=0),
            torch.nn.ConvTranspose2d(32, 3, 3, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor):
        batch_size, bins, h, w = x.shape
        x = self.patch_extractor(x)

        x = x.reshape(batch_size, bins, self.n_patches, self.ph * self.pw)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, self.n_patches, bins * self.ph * self.pw)
        x = self.encoder_proj(x)

        x = self.pos_enc(x)
        x = self.transf_enc(x)

        x = self.decoder_proj(x)

        emb = self.dec_pos_emb(torch.arange(8 * 8, device=x.device))
        batch_size = x.shape[0]
        emb = emb.unsqueeze(0).repeat(batch_size, 1, 1)
        x = self.transf_dec(emb, x)

        x = self.conv_proj(x)
        x = x.reshape(batch_size, 8, 8, self.conv_dim)
        x = x.permute(0, 3, 1, 2)
        x = self.conv_dec(x)

        return x
