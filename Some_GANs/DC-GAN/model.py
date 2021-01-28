import torch.nn as nn
from torch.nn import init
from torch.nn.modules.activation import Sigmoid
from torch.nn.modules.batchnorm import BatchNorm3d


class Discriminator(nn.Module):
    def __init__(self, channel_dim, features_d):
        super().__init__()
        self.discriminator = nn.Sequential(
            # input: N x C x 64 x 64
            nn.Conv2d(channel_dim, features_d, 4, 2, 1),  # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16 x 16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8 x 8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4 x 4

            nn.Conv2d(features_d * 8, 1, 4, 2, 0),  # 1 x 1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, channel_img, features_g):
        super().__init__()
        self.generator = nn.Sequential(
            # input: N x latent_dim x 1 x 1
            self._block(latent_dim, features_g * 16, 4, 1,
                        0),  # N x features_g*16 x 4 x 4

            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8 x 8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16 x 16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32 x 32

            nn.ConvTranspose2d(features_g*2, channel_img, 4,
                               2, 1),  # N x channel_img x 64 x 64
            nn.Tanh()  # range that img was normalized: [-1, 1]

        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False),   # up-scale
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.generator(x)


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
