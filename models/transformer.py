import math
from typing import Tuple
import torch
import torchmetrics
import pytorch_lightning as pl
import cv2
from torchvision.models import vgg19, VGG19_Weights, VGG16_Weights
from torchvision.models.feature_extraction import create_feature_extractor


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


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


class PatchExtractor(torch.nn.Module):
    def __init__(self, patch_size: Tuple[int, int]):
        super().__init__()
        self.p_w, self.p_h = patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, h, w = x.shape

        x = torch.einsum("bchw -> bhwc", x)
        x = x.reshape(
            (batch, h // self.p_h, self.p_h, w // self.p_w, self.p_w, channels)
        )

        x = x.swapaxes(2, 3)
        x = x.reshape((batch, -1, self.p_h, self.p_w, channels))
        x = torch.einsum("bnhwc -> bcnhw", x)

        # Merge channels and patches
        x = x.flatten(start_dim=1, end_dim=2)
        # Flatten patches
        x = x.flatten(start_dim=2)

        # output shape = (batch, channels * n_patches, h * w)
        return x


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
        feature_loss_weight: int,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_shape = input_shape
        self.bins, self.h, self.w = self.input_shape
        self.p_h, self.p_w = patch_size
        self.lr = learning_rate
        self.feature_loss_weight = feature_loss_weight

        self.token_dim = 1024

        self.patch_extractor = PatchExtractor(patch_size)

        self.pe = PositionalEncoding(self.token_dim, max_len=2560)
        enc_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.token_dim, nhead=heads, batch_first=True
        )
        self.enc = torch.nn.TransformerEncoder(enc_layer, layers_number)

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(10, 16, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 32, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32, 16, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 1, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(1),
        )

        self.color = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding="same"),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16, 3, 3, padding="same"),
            torch.nn.Sigmoid(),
        )

        self.lpips = torchmetrics.image.lpip.LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", weights=VGG16_Weights.DEFAULT
        )
        self.ssim = torchmetrics.functional.structural_similarity_index_measure
        self.mse = torchmetrics.functional.mean_squared_error

    def forward(self, x: torch.Tensor):
        # print("Input shape", x.shape)
        batch, bins, h, w = x.shape
        n_y_patches = h // self.p_h
        n_x_patches = w // self.p_w

        x = self.patch_extractor(x)

        # print("Shape after patches, before encoder", x.shape)

        x = self.pe(x)
        x = self.enc(x)

        x = x.reshape(batch, bins, n_y_patches, n_x_patches, self.p_h, self.p_w)
        x = torch.einsum("btyxhw -> btyhxw", x)
        x = x.reshape(batch, bins, n_y_patches * self.p_h, n_x_patches * self.p_w)

        # print("Shape before last conv", x.shape)
        x = self.conv(x)

        # print("Shape after last conv, before color", x.shape)
        x = self.color(x)

        # print("Output shape:", x.shape)
        return x

    def training_step(self, train_batch, batch_idx):
        X, y = train_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)

        loss = image_loss

        self.log("train_image_loss", image_loss)
        self.log("train_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.lpips(model_images, y)
        self.log("train_MSE", mse)
        self.log("train_SSIM", ssim)
        self.log("train_LPIPS", self.lpips)

        return loss

    def validation_step(self, val_batch, batch_idx):
        X, y = val_batch
        X = X[:, :, : self.h, : self.w]
        y = torch.einsum("bhwc -> bchw", y)[:, :, : self.h, : self.w]

        model_images = self(X)

        criterion = torch.nn.MSELoss()

        image_loss = criterion(model_images, y)

        loss = image_loss

        self.log("val_image_loss", image_loss)
        self.log("val_loss", loss)

        # Compute metrics
        mse = self.mse(model_images, y)
        ssim = self.ssim(model_images, y, data_range=1)
        self.lpips(model_images, y)
        self.log("val_MSE", mse)
        self.log("val_SSIM", ssim)
        self.log("val_LPIPS", self.lpips)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
