from typing import Tuple
import torch
import math

class EventsToImagesUNet(torch.nn.Module):
    def __init__(self, input_channels):
        super(EventsToImagesUNet, self).__init__()
        self.input_channels = input_channels

        self.conv64 = self.double_conv_block(self.input_channels, 64, 3)
        self.conv128 = self.double_conv_block(64, 128, 3)
        self.conv256 = self.double_conv_block(128, 256, 3)
        self.conv512 = self.double_conv_block(256, 512, 3)
        self.conv1024 = self.conv_block(512, 1024, 3)

        self.conv512_ = self.double_conv_block(512 + 1024, 512, 3)
        self.conv256_ = self.double_conv_block(256 + 512, 256, 3)
        self.conv128_ = self.double_conv_block(128 + 256, 128, 3)
        self.conv64_ = self.double_conv_block(64 + 128, 64, 3)

        self.max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.final_block = torch.nn.Sequential(
            torch.nn.Conv2d(64, 3, 1), torch.nn.BatchNorm2d(3), torch.nn.Sigmoid()
        )

    def forward(self, x):
        first_block = self.conv64(x)
        x = self.max_pool(first_block)

        second_block = self.conv128(x)
        x = self.max_pool(second_block)

        third_block = self.conv256(x)
        x = self.max_pool(third_block)

        last_block = self.conv512(x)
        x = self.max_pool(last_block)

        x = self.conv1024(x)

        x = self.upsample(x)
        x = torch.cat((last_block, x), dim=1)
        x = self.conv512_(x)

        x = self.upsample(x)
        x = torch.cat((third_block, x), dim=1)
        x = self.conv256_(x)

        x = self.upsample(x)
        x = torch.cat((second_block, x), dim=1)
        x = self.conv128_(x)

        x = self.upsample(x)
        x = torch.cat((first_block, x), dim=1)
        x = self.conv64_(x)

        x = self.final_block(x)

        return x

    def double_conv_block(self, in_channels, out_channels, size):
        return torch.nn.Sequential(
            self.conv_block(in_channels, out_channels, size),
            self.conv_block(out_channels, out_channels, size),
        )

    def conv_block(self, in_channels, out_channels, size):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, size, padding="same"),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
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
            [1] + [0] * (encoding_size-1), dtype=torch.float32, requires_grad=False
        ).reshape(1, 1, -1)
        self.sos_token = torch.nn.Parameter(sos_tensor)

        self.pe = PositionalEncoding(encoding_size, max_len=10)
        self.transformer = torch.nn.Transformer(
            d_model=encoding_size,
            nhead=heads, 
            num_encoder_layers=layers_number,
            num_decoder_layers=layers_number,
            batch_first=True
        )

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
        x = self.transformer.forward(x, batched_sos)
        # print("Transformer output", x.shape)
        x = self.linear_inv_proj(x)
        # print("Shape after inv. projection", x.shape)
        y = self.sigmoid(x)
        y = y.reshape(batches, channels, height, width)
        # print("Output shape", y.shape)
        return y
