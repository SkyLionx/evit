import torch
import math
from typing import Tuple


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


class BayerDecomposer(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposes an Bayer image into 4 RGGB channels.

        Args:
            x (torch.Tensor): raw images of shape [batch_size, h, w]

        Returns:
            torch.Tensor: RGGB channels downsapled of shape [batch_size, 4, h//2, w//2]
        """
        r = x[:, 0::2, 0::2]
        g = x[:, 0::2, 1::2]
        G = x[:, 1::2, 0::2]
        b = x[:, 1::2, 1::2]
        return torch.stack((r, g, G, b), dim=1)


if __name__ == "__main__":
    import numpy as np

    print(BayerDecomposer()(torch.zeros((16, 128, 128))).shape)
