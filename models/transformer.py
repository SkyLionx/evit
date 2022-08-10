import math
from typing import Tuple
import torch
import cv2


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
        x = x + self.pe[: x.size(0)]
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
        x = x.reshape((batch, h // self.p_h, self.p_h, w // self.p_w, self.p_w, channels))

        x = x.swapaxes(2, 3)
        x = x.reshape((batch, -1, self.p_h, self.p_w, channels))
        x = torch.einsum("bnhwc -> bcnhw", x)

        # Merge channels and patches
        x = x.flatten(start_dim=1, end_dim=2)
        # Flatten patches
        x = x.flatten(start_dim=2)

        # output shape = (batch, channels * n_patches, h * w)
        return x

class VisionTransformer(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        patch_size: Tuple[int, int],
        encoding_size: int,
        heads: int,
        layers_number: int,
        use_linear_proj: bool,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.p_h, self.p_w = patch_size
        self.use_linear_proj = use_linear_proj

        self.bins, self.h, self.w = self.input_shape
        self.n_patch_x = self.w // self.p_w
        self.n_patch_y = self.h // self.p_h

        self.token_dim = self.p_w * self.p_h

        self.patch_extractor = PatchExtractor(patch_size)
        
        if use_linear_proj:
            self.linear_proj = torch.nn.Linear(self.token_dim, encoding_size)
            self.token_dim = encoding_size

        self.pe = PositionalEncoding(self.token_dim, max_len=self.n_patch_x * self.n_patch_y * self.bins)
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

        self.final_conv = torch.nn.Conv2d(self.bins, 3, (3, 3), padding="same")

    def forward(self, x):
        batch, bins, h, w = x.shape

        x = self.patch_extractor(x)
        # x shape = (batch, bins * n_patches, p_h * p_w)

        if self.use_linear_proj:
            x = self.linear_proj(x)

        x = x + self.pe(x)

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

        x = self.final_conv(x)
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