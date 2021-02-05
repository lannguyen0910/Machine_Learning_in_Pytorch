import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import albumentations as A

from torchvision import models
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
from dataset import *
from model import UNET
torch.backends.cudnn.benchmark = True

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 32
NUM_EPOCHS = 50
NUM_WORKERS = 100
PIN_MEMORY = True
IMAGE_HEIGHT = 160
IMAGE_WIDTH = 240
LOAD_MODEL = True


def train(loader, model, optimizer, criterion, scaler):
    loop = tqdm(loader)


def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
            A.Rotate(20, p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,

            )
        ]
    )

    val_transforms = A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,

        ),
    ])

    trainset = CarvanaDataset(img_path='train',
                              mask_path='train_masks', dataframe=train_df, transform=train_transform)
    valset = CarvanaDataset(img_path='train',
                            mask_path='train_masks', dataframe=val_df, transform=val_transforms)

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    valloader = DataLoader(
        valset, batch_size=BATCH_SIZE, collate_fn=valset.collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of validating samples: {len(valset)}")

    # trainset.visualize_batch()


if __name__ == '__main__':
    main()
