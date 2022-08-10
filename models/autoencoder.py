from typing import Tuple
import torch

class EventEncoder(torch.nn.Module):
    def __init__(self, n_filters: list):
        super().__init__()

        self.enc_blocks = torch.nn.ModuleList()
        for i in range(len(n_filters)):

            if i == 0:
                in_features = 1
            out_features = n_filters[i]

            block = torch.nn.Sequential(
                torch.nn.Conv2d(in_features, out_features, (3, 3), padding="same"),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_features, out_features, (3, 3), padding="same"),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2)
            )
            self.enc_blocks.append(block)

            in_features = out_features
        
        self.last_conv = torch.nn.Conv2d(out_features, 1, (3, 3), padding="same")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.enc_blocks:
            x = block(x)
        x = self.last_conv(x)
        return x.flatten(start_dim=1)

class EventDecoder(torch.nn.Module):
    def __init__(self, output_shape: Tuple[int, int, int], n_filters: list):
        super().__init__()
        self.output_shape = output_shape
        self.dec_blocks = torch.nn.ModuleList()
        for i in range(len(n_filters)):

            if i == 0:
                in_features = 1
            out_features = n_filters[i]

            block = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_features, out_features, (2, 2), stride=2),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
                torch.nn.Conv2d(out_features, out_features, (3, 3), padding=1),
                torch.nn.BatchNorm2d(out_features),
                torch.nn.ReLU(),
            )
            self.dec_blocks.append(block)

            in_features = out_features
        
        self.last_conv = torch.nn.Conv2d(out_features, output_shape[-1], (3, 3), padding="same")
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        width = self.output_shape[0] // (2 ** len(self.dec_blocks))
        height = self.output_shape[1] // (2 ** len(self.dec_blocks))
        x = x.reshape(-1, 1, height, width)
        for block in self.dec_blocks:
            x = block(x)
        x = self.last_conv(x)
        return self.sigmoid(x)

class EventAutoEncoder(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        code = self.encoder(x)
        return self.decoder(code)