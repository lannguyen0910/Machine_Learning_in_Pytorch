import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from preprocessing import *


class CarvanaDataset(Dataset):
    def __init__(self, img_path, mask_path, dataframe=None, transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.transform = transform
        self.df = dataframe
        self.load_data()

    def load_data(self):
        self.fns = []
        for id, row in self.df.iterrows():
            img_path = os.path.join(self.img_path, row[0]).replace('\\', '/')
            mask_path = os.path.join(
                self.mask_path, row[0].replace('.jpg', '_mask.gif')).replace('\\', '/')
            self.fns.append([img_path, mask_path])

    def __getitem__(self, index):
        img_path, mask_path = self.fns[index]
        img = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentation = self.transform(image=img, mask=mask)
            img = augmentation['image']
            mask = augmentation['mask']

            img = ToTensor()(img)
            mask = torch.FloatTensor(mask)

        return {'img': img, 'mask': mask}

    def visualize_image(self, idx, figsize=(10, 10)):
        item = self.__getitem__(idx)
        img, mask = item['img'], item['mask']
        img = denormalize(img)
        mask = mask.squeeze(0)

        plt.figure(figsize=figsize)
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.subplot(1, 2, 2)
        plt.imshow(mask)

        plt.show()

    def visualize_batch(self, num_samples=2):
        images = [np.random.randint(0, len(self.fns) - num_samples)
                  for i in range(num_samples)]
        for i in images:
            self.visualize_image(i)

    def collate_fn(self, batch):
        imgs = torch.stack([i['img'] for i in batch])
        masks = torch.stack([i['mask'] for i in batch]).unsqueeze(1)
        return {
            'imgs': imgs,
            'masks': masks
        }

    def __len__(self):
        return len(self.fns)
