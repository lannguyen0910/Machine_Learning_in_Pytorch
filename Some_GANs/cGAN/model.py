import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU


class Discriminator(nn.Module):
    def __init__(self, in_channels, features_d, num_classes, img_size):
        super().__init__()
        self.discriminator = nn.Sequential(
            # input: N x in_channels + 1 (an additional channel for labels) x 64 x 64
            nn.Conv2d(in_channels + 1, features_d,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),

            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),

            # make into 1 x 1
            nn.Conv2d(features_d * 8, 1, 4, 2, 0)
        )
        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, self.img_size * self.img_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).view(x.reshape(
            labels.shape[0], 1, self.img_size, self.img_size))  # additional channel
        # N x C x img_size(H) x img_size(W)
        x = torch.cat([x, embedding], dim=1)
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, channel_noise, channels_img, features_g, num_classes, img_size, emb_size):
        super().__init__()
        self.generator = nn.Sequential(
            # input: N x channel_noise + embed_size (which label it should produce) x 1 x 1
            self._block(channel_noise + emb_size, features_g * \
                        16, 4, 1, 0),  # img_size: 4x4,
            self._block(features_g * 16, features_g * \
                        8, 4, 2, 1),  # img_size: 8x8
            self._block(features_g * 8, features_g * 4,
                        4,  2, 1),  # img_size: 16x16
            self._block(features_g * 4, features_g * 2,
                        4,  2, 1),  # img_size: 32x32

            nn.ConvTranspose2d(features_g * 2, channels_img,
                               kernel_size=4, stride=2, padding=1),

            # output: N x channels_img x 64 x 64
            nn.Tanh()
        )

        self.img_size = img_size
        self.embed = nn.Embedding(num_classes, emb_size)

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size,
                               stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x, labels):
        embedding = self.embed(labels).unsqueeze(2).unsqueeze(3)
        # N x C x img_size(H) x img_size(W)
        x = torch.cat([x, embedding], dim=1)
        return self.generator(x)
