import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvBlock, self).__init__()
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.convBlock(x)


class UNET(nn.Module):
    # Binary image segmentation with out_channels = 1
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.downsamplings = nn.ModuleList()
        self.upsamplings = nn.ModuleList()
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downsamplings.append(DoubleConvBlock(in_channels, feature))
            in_channels = feature   # loop all conv layers

        # Up part of UNET
        for feature in reversed(features):
            self.upsamplings.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2))
            self.upsamplings.append(DoubleConvBlock(feature * 2, feature))

        self.bottleneck = DoubleConvBlock(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for downsampling in self.downsamplings:
            x = downsampling(x)
            skip_connections.append(x)
            x = self.pooling(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for id in range(0, len(self.upsamplings), 2):  # pass the DoubleConvBlock
            x = self.upsamplings[id](x)
            # divide by 2 as range step is 2
            skip_connection = skip_connections[id // 2]

            if x.shape != skip_connection.shape:
                # just take H and W
                x = F.resize(x, size=skip_connection.shape[2:])

            # (N, C, H , W): add to channels
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # run through DoubleConvBlock
            x = self.upsamplings[id + 1](concat_skip)

        return self.final_conv(x)
