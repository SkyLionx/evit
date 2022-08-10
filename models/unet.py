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