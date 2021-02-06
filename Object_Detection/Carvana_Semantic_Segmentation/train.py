import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import albumentations as A
import time

from torchvision import models
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from dataset import *
from utils import *
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
LOAD_MODEL = False
CHECKPOINT = 'my_checkpoint.pth'


def train(loader, model, optimizer, criterion, scaler):
    loop = tqdm(loader)

    for batch_id, batch in enumerate(loop):
        data = batch['imgs'].to(device=DEVICE)
        targets = batch['masks'].to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():  # reduce vram usage and faster training
            predictions = model(data)
            loss = criterion(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():
    train_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
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
                              mask_path='train_masks', dataframe=train_df, transform=train_transforms)
    valset = CarvanaDataset(img_path='train',
                            mask_path='train_masks', dataframe=val_df, transform=val_transforms)

    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, collate_fn=trainset.collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    valloader = DataLoader(
        valset, batch_size=BATCH_SIZE, collate_fn=valset.collate_fn, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    print(f"Number of training samples: {len(trainset)}")
    print(f"Number of validating samples: {len(valset)}")

    # trainset.visualize_batch()

    # if out_channels > 1 then use cross entropy loss for multiple classes
    model = UNET(in_channels=3, out_channels=1)
    criterion = nn.BCEWithLogitsLoss()  # not doing Sigmoid at output layer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()  # prevent underflow

    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT, model, optimizer)
        check_accuracy(valloader, model, device=DEVICE)

    else:
        for epoch in range(NUM_EPOCHS):
            train(trainloader, model, optimizer, criterion, scaler)

            checkpoint = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            save_checkpoint(checkpoint)

            check_accuracy(valloader, model, device=DEVICE)

            save_predictions_as_imgs(
                valloader, model, folder='saved_images/', device=DEVICE)


if __name__ == '__main__':
    main()
