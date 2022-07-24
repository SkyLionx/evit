from typing import Tuple
import torch


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


class TransformerModel(torch.nn.Module):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        encoding_size: int,
        heads: int,
        layers_number: int,
    ):
        super(TransformerModel, self).__init__()

        width, height, channels = input_shape

        self.input_shape = input_shape
        self.encoding_size = encoding_size
        self.heads = heads
        self.layers_number = layers_number

        self.linear_proj = torch.nn.Linear(width * height, encoding_size)
        self.activation = torch.nn.ReLU()
        self.linear_inv_proj = torch.nn.Linear(encoding_size, width * height * channels)
        self.sigmoid = torch.nn.Sigmoid()

        sos_tensor = torch.tensor(
            [1] + [0] * 255, dtype=torch.float32, requires_grad=False
        ).reshape(1, 1, -1)
        self.sos_token = torch.nn.Parameter(sos_tensor)

        self.transformer = torch.nn.LSTM()

    def forward(self, x):
        # print("Input shape", x.shape)
        width, height, channels = self.input_shape
        batches, bins = x.shape[0], x.shape[1]

        x = x.reshape(-1, bins, height * width)
        # print("Shape before projection", x.shape)
        x = self.linear_proj(x)
        # print("Shape after projection", x.shape)
        x = self.activation(x)

        batched_sos = torch.repeat_interleave(self.sos_token, batches, 0)
        # print("Transformer inputs", x.shape, batched_sos.shape)
        x = self.transformer.forward(x, batched_sos)
        # print("Transformer output", x.shape)
        x = self.linear_inv_proj(x)
        # print("Shape after inv. projection", x.shape)
        y = self.sigmoid(x)
        y = y.reshape(-1, channels, height, width)
        # print("Output shape", y.shape)
        return y
