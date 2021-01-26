import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from model import Generator, Discriminator
from torch.utils.data import DataLoader, dataloader
from torch.utils.tensorboard import SummaryWriter


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)

# Hyperparameters (very sensitive when working with GANs)
lr = 3e-4
latent_dim = 64  # 64, 128, 256
image_dim = 28*28*1
batch_size = 32
num_epochs = 50

discriminator = Discriminator(image_dim).to(device)
generator = Generator(latent_dim, image_dim).to(device)
noise = torch.randn((batch_size, latent_dim)).to(
    device)   # torch.randn from gaussian distribution
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='dataset/', transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

optim_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
optim_g = torch.optim.Adam(generator.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_fake = SummaryWriter(f'runs/GAN_MNIST/fake')
writer_real = SummaryWriter(f'runs/GAN_MNIST/real')
step = 0

for epoch in range(num_epochs):
    for batch_id, (real_img, label) in enumerate(dataloader):
        real = real_img.view(-1, 784).to(device)
        batch_size = real.shape[0]
        # print('Batch_size: ', batch_size)

        # Train Discriminator max( log(D) + log(1 - D(G(latent))) )
        noise = torch.randn(batch_size, latent_dim).to(device)
        fake = generator(noise)

        discriminator_real = discriminator(real).view(-1)  # flatten
        lossD_real = criterion(
            discriminator_real, torch.ones_like(discriminator_real))

        discriminator_fake = discriminator(fake).view(-1)
        lossD_fake = criterion(
            discriminator_fake, torch.zeros_like(discriminator_fake))

        lossD = (lossD_real + lossD_fake)*0.5

        discriminator.zero_grad()
        # keep the discrimator's computational graph for later computing generator
        lossD.backward(retain_graph=True)
        optim_d.step()

        # Train Generator min log(1 - D(G(latent))) = max log(D(G(latent)))
        output = discriminator(fake).view(-1)
        lossG = criterion(output, torch.ones_like(output))
        generator.zero_grad()
        lossG.backward()
        optim_g.step()

        # Tensorboard
        if batch_id == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_id}/{len(dataloader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

            with torch.no_grad():
                fake = generator(noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(
                    fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(
                    data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1
