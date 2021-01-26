import torch.nn as nn

# Simple model for GAN


class Discriminator(nn.Module):
    def __init__(self, img_dim):
        super().__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.LeakyReLU(0.1),     # better for GAN
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.discriminator(x)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_dim):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, 256),  # latent_dim = noise
            nn.LeakyReLU(0.1),
            nn.Linear(256, img_dim),
            nn.Tanh()  # normalize input
        )

    def forward(self, x):
        return self.generator(x)
